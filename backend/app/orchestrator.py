import asyncio
import json
import uuid
from datetime import datetime, timezone

from sqlalchemy import func, select

from app.agent_schemas import IntegratedAnswer, SubtaskResult, TaskPlan
from app.db import SessionFactory
from app.models import ConversationMessage, Event, Job, Subtask, Task
from app.openai_client import StructuredModelClient

model_client = StructuredModelClient()


async def execute_task(
    task_id: uuid.UUID, job_id: uuid.UUID, lease_owner: str
) -> None:
    async with SessionFactory() as session:
        task = await session.get(Task, task_id)
        if task is None:
            raise RuntimeError("task_not_found")
        messages = await session.scalars(
            select(ConversationMessage)
            .where(ConversationMessage.task_id == task_id)
            .order_by(ConversationMessage.sequence)
        )
        conversation = list(messages)
        config = task.config_snapshot
    planned_through_sequence = conversation[-1].sequence if conversation else 0
    await _assert_lease(job_id, lease_owner)

    plan = await model_client.generate(
        model=config["manager_model"],
        schema_name="task_plan",
        schema=TaskPlan,
        system_prompt=(
            "You are a task-planning manager. Make a concise plan of "
            "independently useful, read-only research and reasoning subtasks. "
            "Do not request shell, file, network-write, or other side-effect "
            "tools. Include dependencies by zero-based subtask index. "
            "No chain-of-thought; use user-facing summaries. Use no skills."
        ),
        input_text=_format_conversation(conversation),
        max_attempts=config["max_agent_attempts"],
    )
    if any(subtask.skill_names for subtask in plan.subtasks):
        raise RuntimeError("unavailable_skill_requested")
    if await _requeue_if_new_messages(
        task_id, job_id, lease_owner, planned_through_sequence
    ):
        return

    subtask_rows = [
        Subtask(
            id=uuid.uuid4(),
            task_id=task_id,
            title=item.title,
            instruction=item.instruction,
            status="pending",
            dependencies=[],
            model_id=config["worker_model"],
            skill_names=[],
        )
        for item in plan.subtasks
    ]
    for index, row in enumerate(subtask_rows):
        row.dependencies = [
            str(subtask_rows[dependency].id)
            for dependency in plan.subtasks[index].dependencies
        ]

    async with SessionFactory() as session:
        await _assert_lease_in_session(session, job_id, lease_owner)
        session.add_all(subtask_rows)
        await session.flush()
        session.add(
            Event(
                task_id=task_id,
                type="plan.created",
                payload_json={
                    "summary": plan.summary,
                    "subtask_count": len(subtask_rows),
                },
            )
        )
        task = await session.get(Task, task_id)
        task.status = "running"
        task.updated_at = datetime.now(timezone.utc)
        session.add(
            Event(
                task_id=task_id,
                type="task.status_changed",
                payload_json={"status": "running"},
            )
        )
        await session.commit()

    result_by_id: dict[str, dict] = {}
    pending = {str(row.id): row for row in subtask_rows}
    concurrency = min(max(config["max_concurrent_agents"], 1), 32)
    while pending:
        if await _requeue_if_new_messages(
            task_id, job_id, lease_owner, planned_through_sequence
        ):
            return
        ready = [
            row
            for row_id, row in pending.items()
            if all(
                dependency in result_by_id for dependency in row.dependencies
            )
        ][:concurrency]
        if not ready:
            raise RuntimeError("subtask_dependency_deadlock")

        await _mark_subtasks_running(task_id, job_id, lease_owner, ready)
        outcomes = await asyncio.gather(
            *[
                _execute_subtask(
                    row,
                    plan,
                    result_by_id,
                    job_id,
                    lease_owner,
                    config["max_agent_attempts"],
                )
                for row in ready
            ],
            return_exceptions=True,
        )
        for row, outcome in zip(ready, outcomes):
            if isinstance(outcome, BaseException):
                error = {
                    "code": "subtask_execution_failed",
                    "message": str(outcome)[:500],
                }
                result = {
                    "status": "failed",
                    "summary": "",
                    "findings": [],
                    "error": error["message"],
                }
                await _save_subtask_result(
                    task_id, job_id, lease_owner, row, result, error
                )
                raise RuntimeError("subtask_execution_failed") from outcome
            serialized = outcome.model_dump(mode="json")
            await _save_subtask_result(
                task_id, job_id, lease_owner, row, serialized, None
            )
            if outcome.status == "failed":
                raise RuntimeError("subtask_reported_failure")
            result_by_id[str(row.id)] = serialized
            pending.pop(str(row.id))

    if await _requeue_if_new_messages(
        task_id, job_id, lease_owner, planned_through_sequence
    ):
        return

    async with SessionFactory() as session:
        messages = await session.scalars(
            select(ConversationMessage)
            .where(ConversationMessage.task_id == task_id)
            .order_by(ConversationMessage.sequence)
        )
        latest_conversation = list(messages)
    if await _requeue_if_new_messages(
        task_id, job_id, lease_owner, planned_through_sequence
    ):
        return
    synthesis_input = json.dumps(
        {
            "conversation": [
                {"role": message.role, "content": message.content}
                for message in latest_conversation
            ],
            "plan_summary": plan.summary,
            "completion_criteria": plan.completion_criteria,
            "subtask_results": list(result_by_id.values()),
        },
        ensure_ascii=False,
    )
    answer = await model_client.generate(
        model=config["manager_model"],
        schema_name="integrated_answer",
        schema=IntegratedAnswer,
        system_prompt=(
            "Integrate subtask results into a useful user-facing answer. Do "
            "not claim unsupported facts. Address the latest user message. "
            "State whether criteria are met and include material caveats. "
            "Do not reveal hidden reasoning."
        ),
        input_text=synthesis_input,
        max_attempts=config["max_agent_attempts"],
    )
    await _finish_task(
        task_id, job_id, lease_owner, answer, planned_through_sequence
    )


def _format_conversation(messages: list[ConversationMessage]) -> str:
    return "\n\n".join(
        f"{message.role}: {message.content}" for message in messages
    )


async def _execute_subtask(
    row, plan, results, job_id, lease_owner, max_attempts
) -> SubtaskResult:
    dependency_results = [
        results[dependency] for dependency in row.dependencies
    ]
    prompt = json.dumps(
        {
            "task_title": row.title,
            "instruction": row.instruction,
            "plan_summary": plan.summary,
            "dependency_results": dependency_results,
        },
        ensure_ascii=False,
    )
    await _assert_lease(job_id, lease_owner)
    return await model_client.generate(
        model=row.model_id,
        schema_name="subtask_result",
        schema=SubtaskResult,
        system_prompt=(
            "Complete one informational subtask using only the supplied "
            "context and your knowledge. Do not use tools or perform side "
            "effects. Return concise findings. Mark failed if you cannot "
            "complete it; do not reveal hidden reasoning."
        ),
        input_text=prompt,
        max_attempts=max_attempts,
    )


async def _mark_subtasks_running(task_id, job_id, lease_owner, rows) -> None:
    async with SessionFactory() as session:
        await _assert_lease_in_session(session, job_id, lease_owner)
        for row in rows:
            stored = await session.get(Subtask, row.id)
            stored.status = "running"
            stored.attempt += 1
            session.add(
                Event(
                    task_id=task_id,
                    type="subtask.status_changed",
                    payload_json={
                        "subtask_id": str(row.id),
                        "status": "running",
                        "title": row.title,
                    },
                )
            )
        await session.commit()


async def _save_subtask_result(
    task_id, job_id, lease_owner, row, result, error
) -> None:
    async with SessionFactory() as session:
        await _assert_lease_in_session(session, job_id, lease_owner)
        stored = await session.get(Subtask, row.id)
        stored.status = result["status"]
        stored.result = result
        stored.error = error
        session.add(
            Event(
                task_id=task_id,
                type="subtask.status_changed",
                payload_json={
                    "subtask_id": str(row.id),
                    "status": stored.status,
                    "title": row.title,
                },
            )
        )
        session.add(
            Event(
                task_id=task_id,
                type="agent.progress",
                payload_json={
                    "summary": result["summary"],
                    "subtask_id": str(row.id),
                },
            )
        )
        await session.commit()


async def _finish_task(
    task_id,
    job_id,
    lease_owner,
    answer: IntegratedAnswer,
    through_sequence: int,
) -> None:
    async with SessionFactory() as session:
        await _assert_lease_in_session(session, job_id, lease_owner)
        task = await session.scalar(
            select(Task).where(Task.id == task_id).with_for_update()
        )
        sequence = await session.scalar(
            select(ConversationMessage.sequence)
            .where(ConversationMessage.task_id == task_id)
            .order_by(ConversationMessage.sequence.desc())
            .limit(1)
        )
        job = await session.get(Job, job_id)
        if (sequence or 0) > through_sequence:
            _queue_for_new_instruction(session, task, job)
            await session.commit()
            return
        task.status = "succeeded" if answer.completion_met else "failed"
        task.result = answer.answer
        task.error = (
            None
            if answer.completion_met
            else {
                "code": "completion_criteria_not_met",
                "message": (
                    "The response was produced with unmet completion "
                    "criteria."
                ),
            }
        )
        task.updated_at = datetime.now(timezone.utc)
        assistant_message = ConversationMessage(
            task_id=task_id,
            sequence=(sequence or 0) + 1,
            role="assistant",
            content=answer.answer,
        )
        session.add(assistant_message)
        await session.flush()
        job.status = "completed" if answer.completion_met else "failed"
        job.lease_owner = None
        job.lease_expires_at = None
        event_type = (
            "task.completed" if answer.completion_met else "task.failed"
        )
        session.add(
            Event(
                task_id=task_id,
                type=event_type,
                payload_json={"status": task.status},
            )
        )
        session.add(
            Event(
                task_id=task_id,
                type="task.status_changed",
                payload_json={"status": task.status},
            )
        )
        session.add(
            Event(
                task_id=task_id,
                type="conversation.message_added",
                payload_json={
                    "message_id": str(assistant_message.id),
                    "sequence": (sequence or 0) + 1,
                    "role": "assistant",
                },
            )
        )
        await session.commit()


async def _requeue_if_new_messages(
    task_id, job_id, lease_owner, through_sequence: int
) -> bool:
    async with SessionFactory() as session:
        await _assert_lease_in_session(session, job_id, lease_owner)
        task = await session.scalar(
            select(Task).where(Task.id == task_id).with_for_update()
        )
        latest_sequence = await session.scalar(
            select(
                func.coalesce(func.max(ConversationMessage.sequence), 0)
            ).where(ConversationMessage.task_id == task_id)
        )
        if latest_sequence <= through_sequence:
            return False
        job = await session.get(Job, job_id)
        _queue_for_new_instruction(session, task, job)
        await session.commit()
        return True


def _queue_for_new_instruction(session, task: Task, job: Job) -> None:
    task.status = "queued"
    task.updated_at = datetime.now(timezone.utc)
    task.result = None
    task.error = None
    job.status = "completed"
    job.lease_owner = None
    job.lease_expires_at = None
    session.add(Job(task_id=task.id, status="pending"))
    session.add(
        Event(
            task_id=task.id,
            type="task.status_changed",
            payload_json={"status": "queued"},
        )
    )
    session.add(
        Event(
            task_id=task.id,
            type="agent.progress",
            payload_json={"summary": "New instruction received; replanning."},
        )
    )


async def _assert_lease(job_id, lease_owner) -> None:
    async with SessionFactory() as session:
        await _assert_lease_in_session(session, job_id, lease_owner)


async def _assert_lease_in_session(session, job_id, lease_owner) -> None:
    job = await session.scalar(
        select(Job).where(
            Job.id == job_id,
            Job.lease_owner == lease_owner,
            Job.status == "leased",
            Job.lease_expires_at > datetime.now(timezone.utc),
        )
    )
    if job is None:
        raise RuntimeError("job_lease_lost")
