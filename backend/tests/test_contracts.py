import uuid
from types import SimpleNamespace

import pytest
import httpx
from fastapi import HTTPException
from pydantic import BaseModel, ValidationError
from starlette.requests import Request

from app.agent_schemas import (
    ChecklistItem,
    ChecklistReviewResult,
    IntegratedAnswer,
    PlannedSubtask,
    SubtaskResult,
    TaskPlan,
)
from app.dependencies import get_task_user, require_task_csrf
from app.models import (
    ApiAccessKey,
    ConversationMessage,
    Event,
    Job,
    QualityReview,
    Subtask,
    Task,
    User,
)
from app.openai_client import StructuredModelClient
from app.quality import (
    failed_required_checklist_items,
    normalize_checklist_results,
    required_checklist_items_passed,
)
from app.routers.tasks import add_message
from app.schemas import (
    AdminSettingsUpdate,
    MessageCreate,
    TaskCreate,
    TavilyKeyRequest,
)
from app.security import hash_password, token_digest, verify_password
from app.tools import (
    TavilySearchArguments,
    TavilySearchTool,
    ToolRegistration,
    ToolRegistry,
)


def test_task_instruction_is_bounded_and_nonempty():
    assert (
        TaskCreate(instruction="research this").instruction == "research this"
    )


def test_empty_task_instruction_is_rejected():
    try:
        TaskCreate(instruction="")
    except ValueError:
        return
    raise AssertionError("empty instruction was accepted")


def test_admin_quality_revision_limits_are_bounded():
    settings = AdminSettingsUpdate(
        manager_model="manager",
        worker_model="worker",
        max_concurrent_agents=4,
        max_agent_attempts=3,
        max_quality_revisions_per_subtask=2,
        max_quality_revisions_per_task=10,
    )
    assert settings.max_quality_revisions_per_subtask == 2
    assert settings.max_quality_revisions_per_task == 10
    with pytest.raises(ValidationError):
        AdminSettingsUpdate(
            manager_model="manager",
            worker_model="worker",
            max_concurrent_agents=4,
            max_agent_attempts=3,
            max_quality_revisions_per_subtask=6,
            max_quality_revisions_per_task=10,
        )


def test_tavily_api_key_request_rejects_empty_keys():
    with pytest.raises(ValidationError):
        TavilyKeyRequest(api_key="")


def test_write_tavily_api_key_uses_configured_vault_path(monkeypatch):
    import app.vault as vault_module

    writes = []

    class FakeKVv2:
        def create_or_update_secret(self, **kwargs):
            writes.append(kwargs)

    settings = SimpleNamespace(
        vault_api_role_id_file="api-role-id",
        vault_api_secret_id_file="api-secret-id",
        vault_kv_mount="secret",
        vault_tavily_key_path="custom/tavily",
    )
    client = SimpleNamespace(
        secrets=SimpleNamespace(kv=SimpleNamespace(v2=FakeKVv2()))
    )
    monkeypatch.setattr(vault_module, "get_settings", lambda: settings)
    monkeypatch.setattr(
        vault_module,
        "_authenticated_client",
        lambda role_id_file, secret_id_file: client,
    )

    vault_module.write_tavily_api_key("tavily-secret")

    assert writes == [
        {
            "mount_point": "secret",
            "path": "custom/tavily",
            "secret": {"api_key": "tavily-secret"},
        }
    ]


def test_message_content_is_bounded_and_nonempty():
    assert MessageCreate(content="follow-up").content == "follow-up"


def test_task_plan_accepts_a_valid_dependency_graph():
    plan = TaskPlan(
        summary="Research and synthesize",
        final_checklist=[
            ChecklistItem(
                id="answer",
                description="Provide the requested answer",
            )
        ],
        subtasks=[
            PlannedSubtask(
                title="Research",
                instruction="Find facts",
                artifact_goal="Provide source-backed facts",
                acceptance_checklist=[
                    ChecklistItem(
                        id="sources",
                        description="Include relevant sources",
                        verification_mode="source_present",
                    )
                ],
                dependencies=[],
                skill_names=[],
            ),
            PlannedSubtask(
                title="Synthesis",
                instruction="Combine facts",
                artifact_goal="Synthesize research into an answer",
                acceptance_checklist=[
                    ChecklistItem(
                        id="synthesis",
                        description="Use the research findings",
                    )
                ],
                dependencies=[0],
                skill_names=[],
            ),
        ],
    )
    assert plan.subtasks[1].dependencies == [0]


def test_task_plan_rejects_dependency_cycles():
    with pytest.raises(ValidationError, match="contains a cycle"):
        TaskPlan(
            summary="Invalid plan",
            final_checklist=[
                ChecklistItem(id="complete", description="Complete")
            ],
            subtasks=[
                PlannedSubtask(
                    title="A",
                    instruction="A",
                    artifact_goal="A",
                    acceptance_checklist=[
                        ChecklistItem(id="a", description="A")
                    ],
                    dependencies=[1],
                    skill_names=[],
                ),
                PlannedSubtask(
                    title="B",
                    instruction="B",
                    artifact_goal="B",
                    acceptance_checklist=[
                        ChecklistItem(id="b", description="B")
                    ],
                    dependencies=[0],
                    skill_names=[],
                ),
            ],
        )


def test_task_plan_rejects_duplicate_checklist_ids():
    with pytest.raises(
        ValidationError, match="checklist item ids must be unique"
    ):
        TaskPlan(
            summary="Plan",
            final_checklist=[
                ChecklistItem(id="answer", description="First"),
                ChecklistItem(id="answer", description="Duplicate"),
            ],
            subtasks=[
                PlannedSubtask(
                    title="Research",
                    instruction="Research",
                    artifact_goal="Evidence",
                    acceptance_checklist=[
                        ChecklistItem(id="source", description="Find source")
                    ],
                    dependencies=[],
                    skill_names=[],
                )
            ],
        )


def test_checklist_review_result_preserves_item_evidence():
    review = ChecklistReviewResult.model_validate(
        {
            "results": [
                {
                    "item_id": "sources",
                    "passed": False,
                    "evidence": "No source list was attached.",
                    "deficiency": "Add sources relevant to the claim.",
                }
            ]
        }
    )
    assert review.results[0].item_id == "sources"
    assert not review.results[0].passed


@pytest.mark.asyncio
async def test_subtask_quality_review_retries_with_only_failed_items(
    monkeypatch,
):
    import app.orchestrator as orchestrator_module

    task_id = uuid.uuid4()
    job_id = uuid.uuid4()
    subtask_id = uuid.uuid4()
    stored = SimpleNamespace(
        id=subtask_id,
        title="Research",
        status="running",
        result=None,
        error=None,
        quality_revision_count=0,
    )

    class FakeSession:
        def __init__(self):
            self.added = []

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def scalar(self, _statement):
            return sum(
                isinstance(item, QualityReview)
                and item.verdict == "revision_requested"
                for item in self.added
            )

        async def get(self, model, _identity):
            return stored if model is Subtask else None

        def add(self, value):
            self.added.append(value)

        async def commit(self):
            return None

    session = FakeSession()

    async def assert_lease(*_args):
        return None

    review_responses = [
        ChecklistReviewResult(
            results=[
                {
                    "item_id": "finding",
                    "passed": False,
                    "evidence": "The finding is too general.",
                    "deficiency": "Name the relevant version.",
                }
            ]
        ),
        ChecklistReviewResult(
            results=[
                {
                    "item_id": "finding",
                    "passed": True,
                    "evidence": "The finding names version 2.4.",
                    "deficiency": "",
                }
            ]
        ),
    ]

    async def generate(**_kwargs):
        return review_responses.pop(0)

    revision_feedback = []

    async def revise(*_args, **kwargs):
        revision_feedback.append(kwargs["revision_feedback"])
        return SubtaskResult(
            status="succeeded",
            summary="Version 2.4 is supported.",
            findings=["Version 2.4 is supported."],
            sources=[],
            tool_runs=[],
            questions=[],
            error="",
        )

    monkeypatch.setattr(orchestrator_module, "SessionFactory", lambda: session)
    monkeypatch.setattr(
        orchestrator_module, "_assert_lease_in_session", assert_lease
    )
    monkeypatch.setattr(orchestrator_module.model_client, "generate", generate)
    monkeypatch.setattr(orchestrator_module, "_execute_subtask", revise)

    review_subtask = orchestrator_module._review_subtask_result
    result, status, questions = await review_subtask(
        row=SimpleNamespace(
            id=subtask_id,
            title="Research",
            instruction="Find details",
            artifact_goal="Find a version-specific fact",
            acceptance_checklist=[
                {
                    "id": "finding",
                    "description": "Name the relevant version",
                    "required": True,
                    "verification_mode": "model",
                }
            ],
            dependencies=[],
            model_id="test-worker",
            quality_revision_count=0,
        ),
        initial_result=SubtaskResult(
            status="succeeded",
            summary="A generic finding.",
            findings=["A generic finding."],
            sources=[],
            tool_runs=[],
            questions=[],
            error="",
        ),
        plan=SimpleNamespace(summary="Research task"),
        results={},
        conversation=[],
        task_id=task_id,
        job_id=job_id,
        lease_owner="worker-test",
        plan_version=1,
        config={
            "manager_model": "test-manager",
            "max_agent_attempts": 1,
            "max_quality_revisions_per_subtask": 1,
            "max_quality_revisions_per_task": 1,
        },
        conversation_sequence=1,
    )

    assert status == "accepted"
    assert questions == []
    assert result["summary"] == "Version 2.4 is supported."
    assert revision_feedback == [
        "- Name the relevant version: Name the relevant version."
    ]
    assert stored.status == "accepted"
    assert stored.quality_revision_count == 1
    verdicts = [
        item.verdict
        for item in session.added
        if isinstance(item, QualityReview)
    ]
    assert verdicts == ["revision_requested", "accepted"]


@pytest.mark.asyncio
async def test_subtask_quality_review_stops_at_revision_limit(monkeypatch):
    import app.orchestrator as orchestrator_module

    task_id = uuid.uuid4()
    job_id = uuid.uuid4()
    subtask_id = uuid.uuid4()
    stored = SimpleNamespace(
        id=subtask_id,
        title="Research",
        status="running",
        result=None,
        error=None,
        quality_revision_count=0,
    )

    class FakeSession:
        def __init__(self):
            self.added = []

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def scalar(self, _statement):
            return 0

        async def get(self, model, _identity):
            return stored if model is Subtask else None

        def add(self, value):
            self.added.append(value)

        async def commit(self):
            return None

    session = FakeSession()

    async def assert_lease(*_args):
        return None

    async def generate(**_kwargs):
        return ChecklistReviewResult(
            results=[
                {
                    "item_id": "finding",
                    "passed": False,
                    "evidence": "No supporting evidence.",
                    "deficiency": "Add a source-backed finding.",
                }
            ]
        )

    async def should_not_revise(*_args, **_kwargs):
        raise AssertionError("revision budget is exhausted")

    monkeypatch.setattr(orchestrator_module, "SessionFactory", lambda: session)
    monkeypatch.setattr(
        orchestrator_module, "_assert_lease_in_session", assert_lease
    )
    monkeypatch.setattr(orchestrator_module.model_client, "generate", generate)
    monkeypatch.setattr(
        orchestrator_module, "_execute_subtask", should_not_revise
    )

    review_subtask = orchestrator_module._review_subtask_result
    result, status, questions = await review_subtask(
        row=SimpleNamespace(
            id=subtask_id,
            title="Research",
            instruction="Find facts",
            artifact_goal="Provide evidence",
            acceptance_checklist=[
                {
                    "id": "finding",
                    "description": "Provide a source-backed finding",
                    "required": True,
                    "verification_mode": "model",
                }
            ],
            dependencies=[],
            model_id="test-worker",
            quality_revision_count=0,
        ),
        initial_result=SubtaskResult(
            status="succeeded",
            summary="An unsupported finding.",
            findings=["An unsupported finding."],
            sources=[],
            tool_runs=[],
            questions=[],
            error="",
        ),
        plan=SimpleNamespace(summary="Research task"),
        results={},
        conversation=[],
        task_id=task_id,
        job_id=job_id,
        lease_owner="worker-test",
        plan_version=1,
        config={
            "manager_model": "test-manager",
            "max_agent_attempts": 1,
            "max_quality_revisions_per_subtask": 0,
            "max_quality_revisions_per_task": 0,
        },
        conversation_sequence=1,
    )

    assert status == "partially_accepted"
    assert questions == []
    assert result["quality_review"][0]["deficiency"] == (
        "Add a source-backed finding."
    )
    assert stored.status == "partially_accepted"
    assert [
        item.verdict
        for item in session.added
        if isinstance(item, QualityReview)
    ] == ["partially_accepted"]


def test_stored_plan_restores_subtask_order_and_dependencies():
    from app.orchestrator import _restore_plan_from_rows

    first_id = uuid.uuid4()
    second_id = uuid.uuid4()
    rows = [
        SimpleNamespace(
            id=first_id,
            title="Research",
            instruction="Find facts",
            artifact_goal="Find source-backed facts",
            acceptance_checklist=[
                {
                    "id": "sources",
                    "description": "Attach sources",
                    "required": True,
                    "verification_mode": "source_present",
                }
            ],
            dependencies=[],
            skill_names=[],
        ),
        SimpleNamespace(
            id=second_id,
            title="Synthesis",
            instruction="Combine results",
            artifact_goal="Answer the user",
            acceptance_checklist=[
                {
                    "id": "answer",
                    "description": "Answer the question",
                    "required": True,
                    "verification_mode": "model",
                }
            ],
            dependencies=[str(first_id)],
            skill_names=[],
        ),
    ]

    restored = _restore_plan_from_rows(
        SimpleNamespace(
            summary="Research and synthesize",
            final_checklist=[
                {
                    "id": "final",
                    "description": "Provide a useful answer",
                    "required": True,
                    "verification_mode": "model",
                }
            ],
        ),
        rows,
    )

    assert restored.subtasks[1].dependencies == [0]
    assert restored.subtasks[0].artifact_goal == "Find source-backed facts"


def test_deterministic_checklist_modes_override_model_review():
    checklist = [
        ChecklistItem(
            id="content",
            description="Include findings",
            verification_mode="non_empty",
        ),
        ChecklistItem(
            id="source",
            description="Attach a source",
            verification_mode="source_present",
        ),
        ChecklistItem(
            id="citation",
            description="Cite an attached source",
            verification_mode="markdown_citation",
        ),
    ]
    review = ChecklistReviewResult(results=[])
    artifact = {
        "summary": "A supported finding.",
        "findings": [],
        "sources": [{"url": "https://example.org/source"}],
    }

    results = normalize_checklist_results(
        checklist,
        review,
        {
            **artifact,
            "summary": (
                "A supported finding ([source](https://example.org/source))."
            ),
        },
    )

    assert required_checklist_items_passed(results)
    assert all(result["passed"] for result in results)


def test_missing_review_and_unsupported_citation_fail_closed():
    checklist = [
        ChecklistItem(id="support", description="Explain the claim"),
        ChecklistItem(
            id="citation",
            description="Link an attached source",
            verification_mode="markdown_citation",
        ),
    ]
    results = normalize_checklist_results(
        checklist,
        ChecklistReviewResult(results=[]),
        {
            "answer": "An unsupported claim ([source](https://evil.example)).",
            "sources": [{"url": "https://example.org/source"}],
        },
    )

    assert not required_checklist_items_passed(results)
    assert len(failed_required_checklist_items(results)) == 2


def test_subtask_result_requires_questions_for_user_input():
    result = SubtaskResult(
        status="needs_input",
        summary="More information is needed.",
        findings=[],
        sources=[],
        tool_runs=[],
        questions=["Which audience should I focus on?"],
        error="",
    )
    assert result.questions == ["Which audience should I focus on?"]
    with pytest.raises(ValidationError, match="requires a question"):
        SubtaskResult(
            status="needs_input",
            summary="More information is needed.",
            findings=[],
            sources=[],
            tool_runs=[],
            questions=[],
            error="",
        )


@pytest.mark.asyncio
async def test_user_questions_pause_task_and_are_saved_as_assistant_message(
    monkeypatch,
):
    import app.orchestrator as orchestrator_module

    task_id = uuid.uuid4()
    job_id = uuid.uuid4()
    lease_owner = "worker-test"
    task = Task(
        id=task_id,
        owner_id=uuid.uuid4(),
        instruction="Research options",
        status="running",
        result="stale result",
        error={"code": "stale"},
    )
    job = Job(
        id=job_id,
        task_id=task_id,
        status="leased",
        lease_owner=lease_owner,
        lease_expires_at=None,
    )

    class FakeSession:
        def __init__(self):
            self.scalar_values = [task, 1]
            self.added = []

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def scalar(self, _statement):
            return self.scalar_values.pop(0)

        async def get(self, model, _identity):
            return job if model is Job else None

        def add(self, value):
            self.added.append(value)

        async def flush(self):
            return None

        async def commit(self):
            return None

    session = FakeSession()

    async def assert_lease(_session, _job_id, _lease_owner):
        return None

    monkeypatch.setattr(orchestrator_module, "SessionFactory", lambda: session)
    monkeypatch.setattr(
        orchestrator_module, "_assert_lease_in_session", assert_lease
    )

    await orchestrator_module._pause_for_user_input(
        task_id,
        job_id,
        lease_owner,
        1,
        [
            " Which audience should I focus on? ",
            "Which audience should I focus on?",
        ],
    )

    message = next(
        item for item in session.added if isinstance(item, ConversationMessage)
    )
    assert task.status == "needs_attention"
    assert task.result is None
    assert task.error is None
    assert message.role == "assistant"
    assert message.content == "1. Which audience should I focus on?"
    assert job.status == "completed"
    assert job.lease_owner is None
    assert any(
        isinstance(item, Event)
        and item.type == "task.status_changed"
        and item.payload_json["status"] == "needs_attention"
        and item.payload_json["conversation_sequence"] == 1
        for item in session.added
    )


@pytest.mark.asyncio
async def test_finish_task_records_partial_outcome_without_error(monkeypatch):
    import app.orchestrator as orchestrator_module

    task_id = uuid.uuid4()
    job_id = uuid.uuid4()
    task = Task(
        id=task_id,
        owner_id=uuid.uuid4(),
        instruction="Research options",
        status="running",
    )
    job = Job(
        id=job_id,
        task_id=task_id,
        status="leased",
        lease_owner="worker-test",
        lease_expires_at=None,
    )

    class FakeSession:
        def __init__(self):
            self.scalar_values = [task, 2]
            self.added = []

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def scalar(self, _statement):
            return self.scalar_values.pop(0)

        async def get(self, model, _identity):
            return job if model is Job else None

        def add(self, value):
            self.added.append(value)

        async def flush(self):
            return None

        async def commit(self):
            return None

    session = FakeSession()

    async def assert_lease(*_args):
        return None

    monkeypatch.setattr(orchestrator_module, "SessionFactory", lambda: session)
    monkeypatch.setattr(
        orchestrator_module, "_assert_lease_in_session", assert_lease
    )
    report = {
        "plan_version": 1,
        "unmet_items": [{"item_id": "sources", "description": "Cite sources"}],
    }

    await orchestrator_module._finish_task(
        task_id,
        job_id,
        "worker-test",
        IntegratedAnswer(answer="Partial answer", caveats=[]),
        2,
        fully_completed=False,
        completion_report=report,
    )

    assert task.status == "partially_completed"
    assert task.result == "Partial answer"
    assert task.error is None
    assert task.completion_report == report
    assert job.status == "completed"
    assert any(
        isinstance(item, Event) and item.type == "task.partially_completed"
        for item in session.added
    )
    assert not any(
        isinstance(item, Event) and item.type == "task.failed"
        for item in session.added
    )


@pytest.mark.asyncio
async def test_subtask_receives_conversation_history(monkeypatch):
    import json

    import app.orchestrator as orchestrator_module

    messages = [
        ConversationMessage(role="assistant", content="Which industry?"),
        ConversationMessage(role="user", content="IT, embedded software."),
    ]
    captured = {}

    async def assert_lease(*_args):
        return None

    async def generate(**kwargs):
        captured.update(kwargs)
        kwargs["on_tool_result"](
            "tavily_web_search",
            '{"query":"public query","max_results":1}',
            {
                "retrieved_at": "2026-10-12T00:00:00+00:00",
                "results": [
                    {
                        "title": "Verified source",
                        "url": "https://example.org/source",
                        "content": "Supporting excerpt",
                        "score": 0.8,
                        "published_date": "2026-10-01",
                    }
                ],
            },
        )
        return SubtaskResult(
            status="succeeded",
            summary="Completed",
            findings=[],
            sources=[],
            tool_runs=[],
            questions=[],
            error="",
        )

    monkeypatch.setattr(orchestrator_module, "_assert_lease", assert_lease)
    monkeypatch.setattr(orchestrator_module.model_client, "generate", generate)

    result = await orchestrator_module._execute_subtask(
        SimpleNamespace(
            title="Clarify profile",
            instruction="Proceed",
            artifact_goal=(
                "Summarize the user profile using verified evidence."
            ),
            acceptance_checklist=[
                {
                    "id": "profile",
                    "description": "Mention the stated industry",
                    "required": True,
                    "verification_mode": "model",
                }
            ],
            dependencies=[],
            model_id="test-worker",
        ),
        SimpleNamespace(summary="Plan"),
        {},
        messages,
        uuid.uuid4(),
        "worker-test",
        1,
    )

    prompt = json.loads(captured["input_text"])
    assert prompt["conversation"] == [
        {"role": "assistant", "content": "Which industry?"},
        {"role": "user", "content": "IT, embedded software."},
    ]
    assert prompt["artifact_goal"] == (
        "Summarize the user profile using verified evidence."
    )
    assert prompt["acceptance_checklist"][0]["id"] == "profile"
    assert result.sources[0].provider == "tavily"
    assert result.sources[0].url == "https://example.org/source"
    assert result.sources[0].score == 0.8
    assert result.tool_runs[0].request_summary == "public query"
    assert result.tool_runs[0].source_count == 1
    assert "do not ask again" in captured["system_prompt"]
    assert "tavily_web_search" in captured["system_prompt"]
    assert "Do not use tools" not in captured["system_prompt"]


@pytest.mark.asyncio
async def test_subtask_progress_events_are_tagged(monkeypatch):
    import app.orchestrator as orchestrator_module

    task_id = uuid.uuid4()
    job_id = uuid.uuid4()
    row = SimpleNamespace(id=uuid.uuid4(), title="Research")
    stored = SimpleNamespace(status="running", result=None, error=None)

    class FakeSession:
        def __init__(self):
            self.added = []

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def get(self, _model, _identity):
            return stored

        def add(self, value):
            self.added.append(value)

        async def commit(self):
            return None

    session = FakeSession()

    async def assert_lease(*_args):
        return None

    monkeypatch.setattr(orchestrator_module, "SessionFactory", lambda: session)
    monkeypatch.setattr(
        orchestrator_module, "_assert_lease_in_session", assert_lease
    )

    await orchestrator_module._save_subtask_result(
        task_id,
        job_id,
        "worker-test",
        row,
        {"status": "succeeded", "summary": "Research complete"},
        None,
        4,
    )

    progress_events = [
        item for item in session.added if isinstance(item, Event)
    ]
    assert len(progress_events) == 2
    assert all(
        item.payload_json["conversation_sequence"] == 4
        for item in progress_events
    )


@pytest.mark.asyncio
async def test_user_reply_requeues_task_waiting_for_attention():
    task_id = uuid.uuid4()
    user = User(
        id=uuid.uuid4(),
        email="person@example.com",
        password_hash="hash",
        role="user",
        is_active=True,
        must_change_password=False,
    )
    task = Task(
        id=task_id,
        owner_id=user.id,
        instruction="Research options",
        status="needs_attention",
        result="stale result",
        error={"code": "needs_input"},
    )

    class FakeSession:
        def __init__(self):
            self.scalar_values = [task, 2]
            self.added = []

        async def scalar(self, _statement):
            return self.scalar_values.pop(0)

        def add(self, value):
            self.added.append(value)

        async def flush(self):
            return None

        async def commit(self):
            return None

        async def refresh(self, _value):
            return None

    session = FakeSession()
    await add_message(
        task_id,
        MessageCreate(content="Focus on first-time founders."),
        user,
        session,
    )

    assert task.status == "queued"
    assert task.result is None
    assert task.error is None
    assert any(
        isinstance(item, Job) and item.status == "pending"
        for item in session.added
    )
    queued_event = next(
        item
        for item in session.added
        if isinstance(item, Event) and item.type == "task.status_changed"
    )
    assert queued_event.payload_json["conversation_sequence"] == 3


@pytest.mark.asyncio
async def test_partial_task_reply_clears_report_and_requeues():
    task_id = uuid.uuid4()
    user = User(
        id=uuid.uuid4(),
        email="person@example.com",
        password_hash="hash",
        role="user",
        is_active=True,
        must_change_password=False,
    )
    task = Task(
        id=task_id,
        owner_id=user.id,
        instruction="Research options",
        status="partially_completed",
        result="Partial answer",
        completion_report={"unmet_items": [{"item_id": "sources"}]},
    )

    class FakeSession:
        def __init__(self):
            self.scalar_values = [task, 2]
            self.added = []

        async def scalar(self, _statement):
            return self.scalar_values.pop(0)

        def add(self, value):
            self.added.append(value)

        async def flush(self):
            return None

        async def commit(self):
            return None

        async def refresh(self, _value):
            return None

    session = FakeSession()
    await add_message(
        task_id,
        MessageCreate(content="Use official sources."),
        user,
        session,
    )

    assert task.status == "queued"
    assert task.result is None
    assert task.error is None
    assert task.completion_report is None
    assert any(
        isinstance(item, Job) and item.status == "pending"
        for item in session.added
    )


def test_passwords_are_hashed_and_verifiable():
    value = "a-long-example-password"
    hashed = hash_password(value)
    assert hashed != value
    assert verify_password(value, hashed)
    assert not verify_password("another-password", hashed)


@pytest.mark.asyncio
async def test_request_validation_does_not_echo_submitted_secrets(monkeypatch):
    monkeypatch.setenv("MANAGER_MODEL", "test-manager")
    monkeypatch.setenv("WORKER_MODEL", "test-worker")
    from app.main import app

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/api/auth/login",
            json={"email": "invalid", "password": "do-not-echo-this"},
        )
    assert response.status_code == 422
    assert response.json() == {"detail": "Request validation failed"}
    assert "do-not-echo-this" not in response.text


class FakeAccessKeySession:
    def __init__(self, api_key=None, user=None):
        self.api_key = api_key
        self.user = user

    async def scalar(self, statement):
        return self.api_key

    async def get(self, model, user_id):
        return self.user


def bearer_request(token: str) -> Request:
    return Request(
        {
            "type": "http",
            "method": "GET",
            "path": "/api/tasks",
            "headers": [(b"authorization", f"Bearer {token}".encode())],
            "query_string": b"",
        }
    )


@pytest.mark.asyncio
async def test_bearer_access_key_authenticates_its_owner():
    token = "issued-once-token"
    user_id = uuid.uuid4()
    user = User(
        id=user_id,
        email="person@example.com",
        password_hash="hash",
        role="user",
        is_active=True,
        must_change_password=False,
    )
    api_key = ApiAccessKey(
        id=uuid.uuid4(),
        user_id=user_id,
        key_hash=token_digest(token),
        revoked_at=None,
    )
    session = FakeAccessKeySession(api_key=api_key, user=user)

    authenticated = await get_task_user(bearer_request(token), db=session)

    assert authenticated is user


@pytest.mark.asyncio
async def test_invalid_bearer_key_is_rejected():
    session = FakeAccessKeySession()

    with pytest.raises(HTTPException) as error:
        await get_task_user(bearer_request("invalid-token"), db=session)

    assert error.value.status_code == 401


@pytest.mark.asyncio
async def test_bearer_write_does_not_require_browser_csrf_cookie():
    request = bearer_request("issued-once-token")

    await require_task_csrf(request, db=FakeAccessKeySession())


class SmallOutput(BaseModel):
    answer: str


@pytest.mark.asyncio
async def test_model_json_fallback_when_schema_mode_is_unsupported(
    sdk_transport,
):
    transport = sdk_transport(
        [
            httpx.Response(
                400, json={"error": {"message": "json_schema unsupported"}}
            ),
            '{"answer":"validated"}',
        ]
    )

    result = await StructuredModelClient().generate(
        model="test-model",
        schema_name="small_output",
        schema=SmallOutput,
        system_prompt="Return a short answer.",
        input_text="Answer the prompt.",
        max_attempts=2,
    )

    assert result.answer == "validated"
    assert str(transport.clients[0].base_url) == "https://api.openai.com/v1/"
    requests = transport.requests
    assert len(requests) == 2
    assert "text" in requests[0]
    assert not requests[1].get("text", {}).get("format")
    assert all(client.is_closed() for client in transport.clients)


@pytest.mark.asyncio
async def test_tool_call_is_validated_executed_and_returned_to_model(
    sdk_transport,
):
    calls = []

    class ToolArguments(BaseModel):
        query: str

    async def search(arguments):
        calls.append(arguments.query)
        return {
            "results": [{"title": "Example", "url": "https://example.org"}]
        }

    registry = ToolRegistry(
        [
            ToolRegistration(
                name="test_search",
                version="1",
                description="Search test data",
                risk_class="read_only",
                timeout_seconds=5,
                arguments_model=ToolArguments,
                handler=search,
            )
        ]
    )
    transport = sdk_transport(
        [
            [
                {
                    "id": "function-1",
                    "type": "function_call",
                    "name": "test_search",
                    "arguments": '{"query":"latest release"}',
                    "call_id": "call-1",
                }
            ],
            '{"answer":"Release notes found."}',
        ]
    )
    lease_checks = []
    captured_results = []

    async def check_lease():
        lease_checks.append(True)

    result = await StructuredModelClient().generate(
        model="test-model",
        schema_name="small_output",
        schema=SmallOutput,
        system_prompt="Use tools if needed.",
        input_text="Find recent release information.",
        tool_registry=registry,
        before_tool_call=check_lease,
        on_tool_result=lambda name, arguments, value: captured_results.append(
            (name, arguments, value)
        ),
    )

    assert result.answer == "Release notes found."
    assert calls == ["latest release"]
    assert len(lease_checks) == 1
    assert captured_results[0][0] == "test_search"
    requests = transport.requests
    assert requests[0]["tools"][0]["name"] == "test_search"
    assert not requests[1].get("previous_response_id")
    assert any(
        item.get("call_id") == "call-1"
        and item["type"] == "function_call_output"
        for item in requests[1]["input"]
    )
    assert transport.credentials == ["Bearer test-key-1", "Bearer test-key-2"]
    assert all(request["store"] is False for request in requests)
    assert all(client.max_retries == 0 for client in transport.clients)
    assert all(client.is_closed() for client in transport.clients)


@pytest.mark.asyncio
async def test_tool_registry_rejects_unknown_and_invalid_arguments():
    invoked = []

    async def handler(_arguments):
        invoked.append(True)
        return {"ok": True}

    registry = ToolRegistry([TavilySearchTool().registration()])

    invalid = await registry.execute(
        "tavily_web_search", '{"query":"", "max_results":99}'
    )
    sensitive = await registry.execute(
        "tavily_web_search",
        '{"query":"contact admin@example.org", "max_results":2}',
    )
    unknown = await registry.execute("unknown", "{}")

    assert invalid == {"error": "invalid_tool_arguments"}
    assert sensitive == {"error": "invalid_tool_arguments"}
    assert unknown == {"error": "tool_not_allowed"}
    assert invoked == []


@pytest.mark.asyncio
async def test_tool_registry_blocks_non_read_only_tools():
    invoked = []

    class ToolArguments(BaseModel):
        value: str

    async def handler(_arguments):
        invoked.append(True)
        return {"ok": True}

    registry = ToolRegistry(
        [
            ToolRegistration(
                name="write_tool",
                version="1",
                description="A future write tool",
                risk_class="writes_data",
                timeout_seconds=5,
                arguments_model=ToolArguments,
                handler=handler,
            )
        ]
    )

    result = await registry.execute("write_tool", '{"value":"change"}')

    assert registry.openai_definitions() == []
    assert result == {"error": "tool_requires_approval"}
    assert invoked == []


@pytest.mark.asyncio
async def test_tavily_search_uses_worker_key_and_filters_results(monkeypatch):
    import json

    import app.tools as tools_module

    payload = {
        "results": [
            {
                "title": "Useful result",
                "url": "https://example.org/article",
                "content": "A verified-length excerpt.",
                "score": 0.9,
                "published_date": "2026-10-12",
            },
            {
                "title": "Invalid local result",
                "url": "file:///etc/passwd",
                "content": "Must be excluded.",
                "score": 0.2,
            },
        ]
    }
    request_values = {}

    class FakeResponse:
        status_code = 200
        headers = {}

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def aiter_bytes(self):
            yield json.dumps(payload).encode()

    class FakeClient:
        def __init__(self, **kwargs):
            request_values["timeout"] = kwargs["timeout"]

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        def stream(self, method, url, **kwargs):
            request_values.update({"method": method, "url": url, **kwargs})
            return FakeResponse()

    class NoopLimiter:
        async def acquire(self):
            return None

    monkeypatch.setattr(tools_module.httpx, "AsyncClient", FakeClient)
    monkeypatch.setattr(
        tools_module, "read_tavily_api_key", lambda: "worker-secret"
    )
    monkeypatch.setattr(
        tools_module,
        "get_settings",
        lambda: SimpleNamespace(
            tavily_timeout_seconds=20,
            max_agent_attempts=1,
        ),
    )
    tool = TavilySearchTool()
    tool._limiter = NoopLimiter()

    result = await tool._search(
        TavilySearchArguments(query="  current news  ", max_results=2)
    )

    assert request_values["url"] == "https://api.tavily.com/search"
    assert request_values["headers"]["Authorization"] == (
        "Bearer worker-secret"
    )
    assert request_values["json"]["query"] == "current news"
    assert request_values["json"]["include_answer"] is False
    assert len(result["results"]) == 1
    assert result["results"][0]["url"] == "https://example.org/article"
