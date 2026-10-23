"""Apply an owner-approved extension to one subtask's tool budget."""

import uuid
from datetime import datetime, timezone

from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Event, Job, QualityReview, Subtask, Task
from app.schemas import ToolBudgetApprovalDecision


async def resolve_tool_budget_approval(
    task_id: uuid.UUID,
    owner_id: uuid.UUID,
    decision: ToolBudgetApprovalDecision,
    db: AsyncSession,
) -> Task:
    task = await db.scalar(
        select(Task)
        .where(
            Task.id == task_id,
            Task.owner_id == owner_id,
            Task.deletion_requested_at.is_(None),
        )
        .with_for_update()
    )
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")

    pending = task.pending_tool_approval
    if (
        task.status != "waiting_approval"
        or not isinstance(pending, dict)
        or pending.get("request_id") != str(decision.request_id)
    ):
        raise HTTPException(
            status_code=409,
            detail="No matching tool-budget request is pending",
        )

    try:
        subtask_id = uuid.UUID(pending["subtask_id"])
        plan_version = int(pending["plan_version"])
        additional_calls = int(pending["additional_calls"])
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=409, detail="Invalid pending tool-budget request"
        ) from exc

    subtask = await db.scalar(
        select(Subtask)
        .where(
            Subtask.id == subtask_id,
            Subtask.task_id == task_id,
            Subtask.plan_version == plan_version,
        )
        .with_for_update()
    )
    if (
        subtask is None
        or subtask.status != "waiting_approval"
        or not isinstance(subtask.result, dict)
    ):
        raise HTTPException(
            status_code=409,
            detail="The requested subtask is no longer pending",
        )

    revision_round = subtask.quality_revision_count
    checklist_results = subtask.result.get("quality_review", [])
    if not isinstance(checklist_results, list):
        checklist_results = []
    verdict = "partially_accepted"
    granted_calls = 0
    if decision.approved:
        initial_limit = task.config_snapshot.get(
            "max_tool_calls_per_subtask", 20
        )
        current_limit = subtask.tool_calls_limit
        if current_limit is None:
            current_limit = initial_limit
        if (
            not isinstance(current_limit, int)
            or current_limit < 0
            or not isinstance(initial_limit, int)
            or not 0 <= initial_limit <= 100
            or initial_limit == 0
            or additional_calls != initial_limit
        ):
            raise HTTPException(
                status_code=409, detail="Tool-budget extension is invalid"
            )
        granted_calls = additional_calls
        subtask.tool_calls_limit = current_limit + granted_calls
        subtask.quality_revision_count += 1
        subtask.status = "revising"
        verdict = "revision_requested"
    else:
        subtask.status = "partially_accepted"

    reviewer_model = pending.get("reviewer_model_id")
    db.add(
        QualityReview(
            task_id=task_id,
            plan_version=plan_version,
            target_type="subtask",
            target_id=subtask.id,
            revision_round=revision_round,
            checklist_results=checklist_results,
            verdict=verdict,
            reviewer_model_id=(
                reviewer_model
                if isinstance(reviewer_model, str)
                else "unknown"
            ),
        )
    )
    task.pending_tool_approval = None
    task.status = "queued"
    task.result = None
    task.error = None
    task.completion_report = None
    task.updated_at = datetime.now(timezone.utc)
    db.add(Job(task_id=task_id, status="pending"))
    db.add(
        Event(
            task_id=task_id,
            type="quality.review_completed",
            payload_json={
                "plan_version": plan_version,
                "target_type": "subtask",
                "target_id": str(subtask.id),
                "revision_round": revision_round,
                "verdict": verdict,
            },
        )
    )
    db.add(
        Event(
            task_id=task_id,
            type="subtask.status_changed",
            payload_json={
                "subtask_id": str(subtask.id),
                "status": subtask.status,
                "title": subtask.title,
                "plan_version": plan_version,
            },
        )
    )
    db.add(
        Event(
            task_id=task_id,
            type="task.tool_budget_approval_resolved",
            payload_json={
                "request_id": str(decision.request_id),
                "subtask_id": str(subtask.id),
                "approved": decision.approved,
                "additional_calls": granted_calls,
            },
        )
    )
    db.add(
        Event(
            task_id=task_id,
            type="task.status_changed",
            payload_json={"status": "queued"},
        )
    )
    await db.commit()
    await db.refresh(task)
    return task
