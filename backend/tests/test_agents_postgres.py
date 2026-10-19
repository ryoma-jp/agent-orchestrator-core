import asyncio
import importlib
import os
import uuid
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
import pytest_asyncio
from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import func, select, text
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from app.agent_schemas import (
    ChecklistItem,
    ChecklistReviewResult,
    IntegratedAnswer,
    PlannedSubtask,
    SubtaskResult,
    TaskPlan,
)
from app.models import (
    Base,
    ConversationMessage,
    Event,
    Job,
    QualityReview,
    Subtask,
    SystemSettings,
    Task,
    User,
)
from app.openai_client import AgentExecutionError
from app.orchestrator import _reserve_subtask_tool_call, execute_task
from app.schemas import TaskCreate
from test_agents_runtime import registry, tool_call

pytestmark = pytest.mark.asyncio


@pytest_asyncio.fixture
async def database(monkeypatch):
    url = os.getenv("TEST_DATABASE_URL")
    if not url:
        pytest.skip(
            "Set TEST_DATABASE_URL to a dedicated PostgreSQL test database"
        )
    parsed = make_url(url)
    if parsed.get_backend_name() != "postgresql" or not (
        parsed.database or ""
    ).endswith("_test"):
        pytest.fail(
            "TEST_DATABASE_URL must name a PostgreSQL database ending in _test"
        )
    schema = "sdk_test_" + uuid.uuid4().hex
    admin = create_async_engine(url, poolclass=NullPool)
    async with admin.begin() as connection:
        await connection.execute(text(f'CREATE SCHEMA "{schema}"'))
    engine = create_async_engine(
        url,
        poolclass=NullPool,
        connect_args={"server_settings": {"search_path": schema}},
    )
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    monkeypatch.setattr("app.orchestrator.SessionFactory", sessions)
    try:
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
        yield SimpleNamespace(engine=engine, sessions=sessions)
    finally:
        await engine.dispose()
        async with admin.begin() as connection:
            await connection.execute(text(f'DROP SCHEMA "{schema}" CASCADE'))
        await admin.dispose()


async def seed(database, *, limit=2, with_subtask=True):
    async with database.sessions() as session:
        user = User(
            email=f"{uuid.uuid4().hex}@example.org",
            password_hash="unused",
            role="user",
        )
        session.add(user)
        await session.flush()
        task = Task(
            owner_id=user.id,
            instruction="Find facts",
            status="running",
            config_snapshot={
                "manager_model": "test-manager",
                "worker_model": "test-worker",
                "max_concurrent_agents": 2,
                "max_agent_attempts": 1,
                "max_agent_turns": 24,
                "max_tool_calls_per_subtask": limit,
                "max_quality_revisions_per_subtask": 0,
                "max_quality_revisions_per_task": 0,
            },
        )
        session.add(task)
        await session.flush()
        job = Job(
            task_id=task.id,
            status="leased",
            lease_owner="test-worker",
            lease_expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
        )
        session.add(job)
        session.add(
            ConversationMessage(
                task_id=task.id,
                sequence=1,
                role="user",
                content="Find facts",
            )
        )
        subtask = None
        if with_subtask:
            subtask = Subtask(
                task_id=task.id,
                title="Research",
                instruction="Find facts",
                model_id="test-worker",
                status="running",
                plan_version=1,
                artifact_goal="Find facts",
                dependencies=[],
                acceptance_checklist=[
                    ChecklistItem(
                        id="facts", description="Find facts"
                    ).model_dump()
                ],
            )
            session.add(subtask)
        await session.commit()
        return SimpleNamespace(user=user, task=task, job=job, subtask=subtask)


async def reserve(state, owner="test-worker"):
    await _reserve_subtask_tool_call(
        state.task.id,
        state.subtask.id,
        state.job.id,
        owner,
    )


async def test_new_instruction_at_sdk_boundary_requeues_once(database):
    from app.orchestrator import InstructionChanged, _agent_options

    state = await seed(database)
    async with database.sessions() as session:
        session.add(
            ConversationMessage(
                task_id=state.task.id,
                sequence=2,
                role="user",
                content="New requirement",
            )
        )
        await session.commit()
    check = _agent_options(
        state.task.id,
        state.job.id,
        "test-worker",
        state.task.config_snapshot,
        conversation_sequence=1,
    )["before_model_call"]
    outcomes = await asyncio.gather(check(), check(), return_exceptions=True)
    assert any(isinstance(outcome, InstructionChanged) for outcome in outcomes)
    async with database.sessions() as session:
        assert (await session.get(Task, state.task.id)).status == "queued"
        assert (await session.get(Job, state.job.id)).status == "completed"
        assert (
            await session.scalar(
                select(func.count(Job.id)).where(Job.status == "pending")
            )
            == 1
        )
        assert (
            await session.get(Subtask, state.subtask.id)
        ).tool_calls_used == 0


async def test_tool_budget_is_atomic_and_survives_reclaim(database):
    state = await seed(database)
    outcomes = await asyncio.gather(
        *(reserve(state) for _ in range(6)), return_exceptions=True
    )
    assert sum(outcome is None for outcome in outcomes) == 2
    assert all(
        outcome is None or isinstance(outcome, AgentExecutionError)
        for outcome in outcomes
    )
    async with database.sessions() as session:
        stored = await session.get(Subtask, state.subtask.id)
        assert stored.tool_calls_used == 2
        assert await session.scalar(select(func.count(Event.id))) == 2
        job = await session.get(Job, state.job.id)
        job.lease_owner = "replacement-worker"
        await session.commit()
    with pytest.raises(AgentExecutionError, match="tool_call_limit_exceeded"):
        await reserve(state, owner="replacement-worker")


@pytest.mark.parametrize("reason", ["owner", "expired", "cancelled"])
async def test_tool_budget_rejects_stale_execution(database, reason):
    state = await seed(database)
    async with database.sessions() as session:
        job = await session.get(Job, state.job.id)
        if reason == "owner":
            job.lease_owner = "other-worker"
        elif reason == "expired":
            job.lease_expires_at = datetime.now(timezone.utc) - timedelta(
                seconds=1
            )
        else:
            task = await session.get(Task, state.task.id)
            task.status = "cancelled"
            job.status = "completed"
        await session.commit()
    with pytest.raises(RuntimeError, match="job_lease_lost"):
        await reserve(state)
    async with database.sessions() as session:
        assert (
            await session.get(Subtask, state.subtask.id)
        ).tool_calls_used == 0
        assert await session.scalar(select(func.count(Event.id))) == 0


async def test_tool_budget_cannot_consume_another_tasks_subtask(database):
    first = await seed(database)
    second = await seed(database)
    with pytest.raises(RuntimeError, match="subtask_not_found"):
        await _reserve_subtask_tool_call(
            first.task.id, second.subtask.id, first.job.id, "test-worker"
        )


async def test_snapshot_freezes_server_limits_without_changing_admin_contract(
    database, monkeypatch
):
    import app.routers.tasks as task_router
    from app.config import Settings

    state = await seed(database)
    settings = Settings(
        manager_model="manager",
        worker_model="worker",
        max_agent_turns=12,
        max_tool_calls_per_subtask=4,
    )
    monkeypatch.setattr(task_router, "get_settings", lambda: settings)
    async with database.sessions() as session:
        session.add(
            SystemSettings(
                id=1,
                settings_json={
                    "max_agent_turns": 64,
                    "max_tool_calls_per_subtask": 100,
                },
            )
        )
        await session.commit()
        created = await task_router.create_task(
            TaskCreate(instruction="New task"), state.user, session
        )
        settings.max_agent_turns = 2
        settings.max_tool_calls_per_subtask = 0
        assert created.config_snapshot["max_agent_turns"] == 12
        assert created.config_snapshot["max_tool_calls_per_subtask"] == 4
        assert created.config_snapshot["agent_runtime"]["version"] == "0.17.8"


async def test_sdk_budget_migration_round_trip_preserves_existing_rows(
    database,
):
    state = await seed(database)
    migration = importlib.import_module(
        "backend.migrations.versions.0004_agent_sdk_limits"
    )

    def migrate(connection, operation):
        with Operations.context(MigrationContext.configure(connection)):
            getattr(migration, operation)()

    async with database.engine.begin() as connection:
        await connection.run_sync(migrate, "downgrade")
        await connection.run_sync(migrate, "upgrade")
    async with database.sessions() as session:
        assert (
            await session.get(Subtask, state.subtask.id)
        ).tool_calls_used == 0


@pytest.mark.parametrize("accepted", [True, False])
async def test_task_uses_sdk_for_plan_work_review_and_synthesis(
    database, sdk_transport, monkeypatch, accepted
):
    state = await seed(database, with_subtask=False)
    plan = TaskPlan(
        summary="Research facts",
        final_checklist=[
            ChecklistItem(id="answer", description="Answer the user")
        ],
        subtasks=[
            PlannedSubtask(
                title="Research",
                instruction="Find facts",
                artifact_goal="Find facts",
                acceptance_checklist=[
                    ChecklistItem(id="facts", description="Find facts")
                ],
                dependencies=[],
                skill_names=[],
            )
        ],
    )
    work = SubtaskResult(
        status="succeeded",
        summary="Research result",
        findings=["Fact"],
        sources=[],
        tool_runs=[],
        questions=[],
        error="",
    )
    review = ChecklistReviewResult(
        results=[
            {
                "item_id": "facts",
                "passed": accepted,
                "evidence": "Fact" if accepted else "",
                "deficiency": "" if accepted else "Missing source",
            }
        ]
    )
    final_review = ChecklistReviewResult(
        results=[
            {
                "item_id": "answer",
                "passed": True,
                "evidence": "Final answer",
                "deficiency": "",
            }
        ]
    )
    transport = sdk_transport(
        [
            plan.model_dump_json(),
            tool_call(),
            work.model_dump_json(),
            review.model_dump_json(),
            IntegratedAnswer(
                answer="Final answer", caveats=[]
            ).model_dump_json(),
            final_review.model_dump_json(),
        ]
    )
    calls = []
    monkeypatch.setattr("app.orchestrator.tool_registry", registry(calls))
    await execute_task(state.task.id, state.job.id, "test-worker")
    async with database.sessions() as session:
        task = await session.get(Task, state.task.id)
        subtask = await session.scalar(
            select(Subtask).where(Subtask.task_id == task.id)
        )
        assert task.status == (
            "succeeded" if accepted else "partially_completed"
        )
        assert task.error is None
        assert subtask.status == (
            "accepted" if accepted else "partially_accepted"
        )
        assert subtask.tool_calls_used == 1
        assert (await session.get(Job, state.job.id)).status == "completed"
        assert await session.scalar(select(func.count(QualityReview.id))) == 2
        assert (
            len(list(await session.scalars(select(ConversationMessage)))) == 2
        )
    assert calls == ["facts"]
    assert len(transport.requests) == 6
    assert len(transport.key_reads) == 6
    assert all(request["store"] is False for request in transport.requests)
