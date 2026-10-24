import uuid
from datetime import datetime, timezone

from sqlalchemy import func, select

from app.db import SessionFactory
from app.models import ConversationMessage, Event, Job, Subtask, Task
from app.quality import failed_required_checklist_items


async def pause_for_tool_budget_approval(
    *,
    task_id: uuid.UUID,
    job_id: uuid.UUID,
    lease_owner: str,
    row: Subtask,
    plan_version: int,
    conversation_sequence: int,
    subtask_result: dict[str, object],
    review_results: list[dict[str, object]],
    reviewer_model_id: str,
    additional_calls: int,
) -> bool:
    request_id = uuid.uuid4()
    failed_items = failed_required_checklist_items(review_results)
    reason = "\n".join(
        f"- {item['description']}: {item['deficiency']}"
        for item in failed_items
        if item.get("requires_additional_tool_calls")
    )[:2_000]
    async with SessionFactory() as session:
        await assert_lease_in_session(session, job_id, lease_owner)
        task = await session.scalar(
            select(Task).where(Task.id == task_id).with_for_update()
        )
        latest_sequence = await session.scalar(
            select(
                func.coalesce(func.max(ConversationMessage.sequence), 0)
            ).where(ConversationMessage.task_id == task_id)
        )
        job = await session.get(Job, job_id)
        if (latest_sequence or 0) > conversation_sequence:
            queue_for_new_instruction(
                session, task, job, latest_sequence or conversation_sequence
            )
            await session.commit()
            return False
        if task.pending_tool_approval is not None:
            raise RuntimeError("tool_budget_approval_already_pending")
        subtask = await session.scalar(
            select(Subtask)
            .where(Subtask.id == row.id, Subtask.task_id == task_id)
            .with_for_update()
        )
        if subtask is None:
            raise RuntimeError("subtask_not_found")
        pending_approval = {
            "request_id": str(request_id),
            "plan_version": plan_version,
            "subtask_id": str(subtask.id),
            "subtask_title": subtask.title,
            "additional_calls": additional_calls,
            "reason": reason,
            "reviewer_model_id": reviewer_model_id,
            "conversation_sequence": conversation_sequence,
        }
        subtask.status = "waiting_approval"
        subtask.result = subtask_result
        subtask.error = None
        task.pending_tool_approval = pending_approval
        task.status = "waiting_approval"
        task.result = None
        task.error = None
        task.completion_report = None
        task.updated_at = datetime.now(timezone.utc)
        job.status = "completed"
        job.lease_owner = None
        job.lease_expires_at = None
        session.add(
            Event(
                task_id=task_id,
                type="subtask.status_changed",
                payload_json={
                    "subtask_id": str(subtask.id),
                    "status": "waiting_approval",
                    "title": subtask.title,
                    "plan_version": plan_version,
                },
            )
        )
        session.add(
            Event(
                task_id=task_id,
                type="task.tool_budget_approval_requested",
                payload_json=pending_approval,
            )
        )
        session.add(
            Event(
                task_id=task_id,
                type="task.status_changed",
                payload_json={
                    "status": "waiting_approval",
                    "conversation_sequence": conversation_sequence,
                },
            )
        )
        await session.commit()
    return True


async def pause_for_user_input(
    task_id,
    job_id,
    lease_owner,
    through_sequence: int,
    questions: list[str],
) -> None:
    unique_questions = list(
        dict.fromkeys(
            question.strip() for question in questions if question.strip()
        )
    )
    content = "\n".join(
        f"{index}. {question}"
        for index, question in enumerate(unique_questions, start=1)
    )
    async with SessionFactory() as session:
        await assert_lease_in_session(session, job_id, lease_owner)
        task = await session.scalar(
            select(Task).where(Task.id == task_id).with_for_update()
        )
        latest_sequence = await session.scalar(
            select(
                func.coalesce(func.max(ConversationMessage.sequence), 0)
            ).where(ConversationMessage.task_id == task_id)
        )
        job = await session.get(Job, job_id)
        if (latest_sequence or 0) > through_sequence:
            queue_for_new_instruction(session, task, job, latest_sequence)
            await session.commit()
            return

        task.status = "needs_attention"
        task.result = None
        task.error = None
        task.completion_report = None
        task.updated_at = datetime.now(timezone.utc)
        assistant_message = ConversationMessage(
            task_id=task_id,
            sequence=(latest_sequence or 0) + 1,
            role="assistant",
            content=content,
        )
        session.add(assistant_message)
        await session.flush()
        job.status = "completed"
        job.lease_owner = None
        job.lease_expires_at = None
        session.add(
            Event(
                task_id=task_id,
                type="task.status_changed",
                payload_json={
                    "status": "needs_attention",
                    "conversation_sequence": through_sequence,
                },
            )
        )
        session.add(
            Event(
                task_id=task_id,
                type="conversation.message_added",
                payload_json={
                    "message_id": str(assistant_message.id),
                    "sequence": (latest_sequence or 0) + 1,
                    "role": "assistant",
                },
            )
        )
        await session.commit()


async def requeue_if_new_messages(
    task_id, job_id, lease_owner, through_sequence: int
) -> bool:
    async with SessionFactory() as session:
        await assert_lease_in_session(session, job_id, lease_owner)
        task = await session.scalar(
            select(Task).where(Task.id == task_id).with_for_update()
        )
        await assert_lease_in_session(session, job_id, lease_owner)
        latest_sequence = await session.scalar(
            select(
                func.coalesce(func.max(ConversationMessage.sequence), 0)
            ).where(ConversationMessage.task_id == task_id)
        )
        if latest_sequence <= through_sequence:
            return False
        job = await session.get(Job, job_id)
        queue_for_new_instruction(session, task, job, latest_sequence)
        await session.commit()
        return True


def queue_for_new_instruction(
    session, task: Task, job: Job, conversation_sequence
) -> None:
    task.status = "queued"
    task.updated_at = datetime.now(timezone.utc)
    task.result = None
    task.error = None
    task.completion_report = None
    job.status = "completed"
    job.lease_owner = None
    job.lease_expires_at = None
    session.add(Job(task_id=task.id, status="pending"))
    session.add(
        Event(
            task_id=task.id,
            type="task.status_changed",
            payload_json={
                "status": "queued",
                "conversation_sequence": conversation_sequence,
            },
        )
    )
    session.add(
        Event(
            task_id=task.id,
            type="agent.progress",
            payload_json={
                "summary": "New instruction received; replanning.",
                "conversation_sequence": conversation_sequence,
            },
        )
    )


async def assert_lease(job_id, lease_owner) -> None:
    async with SessionFactory() as session:
        await assert_lease_in_session(session, job_id, lease_owner)


async def assert_lease_in_session(session, job_id, lease_owner) -> None:
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
