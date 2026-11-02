import asyncio
import json
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from sqlalchemy import delete, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.db import get_session
from app.dependencies import require_ready_task_user, require_task_csrf
from app.models import (
    ConversationMessage,
    Event,
    ImageGenerationRevision,
    Job,
    Subtask,
    SystemSettings,
    Task,
    TaskDeletionJob,
    TaskFile,
    User,
)
from app.openai_client import SDK_VERSION
from app.routers.workspace import (
    _owned_task,
)
from app.schemas import (
    ImageGenerationRevisionView,
    MessageCreate,
    MessageView,
    TaskCreate,
    TaskDeletionView,
    TaskView,
    ToolBudgetApprovalDecision,
)
from app.tool_budget_approval import resolve_tool_budget_approval
from app.workspaces import task_workspace

router = APIRouter(prefix="/api/tasks", tags=["tasks"])


@router.post(
    "",
    response_model=TaskView,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(require_task_csrf)],
)
async def create_task(
    payload: TaskCreate,
    user: User = Depends(require_ready_task_user),
    db: AsyncSession = Depends(get_session),
):
    settings = get_settings()
    stored_settings = await db.get(SystemSettings, 1)
    execution_settings = {
        "manager_model_provider": settings.manager_model_provider,
        "manager_model": settings.manager_model,
        "worker_model_provider": settings.worker_model_provider,
        "worker_model": settings.worker_model,
        "worker_image_provider": settings.worker_image_provider,
        "worker_image_model": settings.worker_image_model,
        "max_concurrent_agents": settings.max_concurrent_agents,
        "max_agent_attempts": min(max(settings.max_agent_attempts, 1), 3),
        "max_quality_revisions_per_subtask": min(
            max(settings.max_quality_revisions_per_subtask, 0), 5
        ),
        "max_quality_revisions_per_task": min(
            max(settings.max_quality_revisions_per_task, 0), 25
        ),
    }
    if stored_settings:
        execution_settings.update(stored_settings.settings_json)
    for legacy_key in (
        "image_review_model_provider",
        "image_review_model",
        "image_generation_agent_id",
        "image_review_agent_id",
    ):
        execution_settings.pop(legacy_key, None)
    execution_settings.update(
        {
            "max_agent_turns": settings.max_agent_turns,
            "max_tool_calls_per_subtask": settings.max_tool_calls_per_subtask,
            "agent_runtime": {
                "sdk": "openai-agents",
                "version": SDK_VERSION,
                "definition_version": "1",
            },
        }
    )
    task = Task(
        owner_id=user.id,
        instruction=payload.instruction,
        status="queued",
        config_snapshot=execution_settings,
    )
    db.add(task)
    await db.flush()
    try:
        task_workspace(task.id, create=True)
    except OSError as exc:
        raise HTTPException(
            status_code=503, detail="Task workspace is unavailable"
        ) from exc
    message = ConversationMessage(
        task_id=task.id, sequence=1, role="user", content=payload.instruction
    )
    db.add(message)
    await db.flush()
    db.add(
        Event(
            task_id=task.id,
            type="task.status_changed",
            payload_json={"status": "queued", "conversation_sequence": 1},
        )
    )
    db.add(
        Event(
            task_id=task.id,
            type="conversation.message_added",
            payload_json={
                "message_id": str(message.id),
                "sequence": 1,
                "role": "user",
            },
        )
    )
    db.add(Job(task_id=task.id, status="pending"))
    await db.commit()
    await db.refresh(task)
    return task


@router.get("", response_model=list[TaskView])
async def list_tasks(
    user: User = Depends(require_ready_task_user),
    db: AsyncSession = Depends(get_session),
    limit: int = 50,
):
    limit = min(max(limit, 1), 100)
    result = await db.scalars(
        select(Task)
        .where(
            Task.owner_id == user.id,
            Task.deletion_requested_at.is_(None),
        )
        .order_by(Task.updated_at.desc())
        .limit(limit)
    )
    return list(result)


@router.get("/{task_id}", response_model=TaskView)
async def get_task(
    task_id: uuid.UUID,
    user: User = Depends(require_ready_task_user),
    db: AsyncSession = Depends(get_session),
):
    task = await db.scalar(
        select(Task).where(
            Task.id == task_id,
            Task.owner_id == user.id,
            Task.deletion_requested_at.is_(None),
        )
    )
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    return task


@router.get(
    "/{task_id}/image-revisions",
    response_model=list[ImageGenerationRevisionView],
)
async def list_image_revisions(
    task_id: uuid.UUID,
    user: User = Depends(require_ready_task_user),
    db: AsyncSession = Depends(get_session),
):
    task = await db.scalar(
        select(Task.id).where(
            Task.id == task_id,
            Task.owner_id == user.id,
            Task.deletion_requested_at.is_(None),
        )
    )
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")

    rows = await db.execute(
        select(ImageGenerationRevision, Subtask.title)
        .join(Subtask, Subtask.id == ImageGenerationRevision.subtask_id)
        .where(ImageGenerationRevision.task_id == task_id)
        .order_by(
            ImageGenerationRevision.created_at,
            ImageGenerationRevision.id,
        )
    )
    return [
        ImageGenerationRevisionView(
            id=revision.id,
            subtask_id=revision.subtask_id,
            subtask_title=subtask_title,
            plan_version=revision.plan_version,
            generation_no=revision.generation_no,
            trigger_quality_review_id=revision.trigger_quality_review_id,
            source_artifact_paths=revision.source_artifact_paths,
            artifact_path=revision.artifact_path,
            prompt=revision.prompt,
            negative_prompt=revision.negative_prompt,
            openai_prompt=revision.openai_prompt,
            model_id=revision.model_id,
            width=revision.width,
            height=revision.height,
            steps=revision.steps,
            seed=revision.seed,
            quality=revision.quality,
            unmet_items=revision.unmet_items,
            is_backfilled=revision.is_backfilled,
            created_at=revision.created_at,
        )
        for revision, subtask_title in rows.all()
    ]


@router.get("/{task_id}/messages", response_model=list[MessageView])
async def list_messages(
    task_id: uuid.UUID,
    user: User = Depends(require_ready_task_user),
    db: AsyncSession = Depends(get_session),
):
    await _owned_task(db, task_id, user.id)
    messages = await db.scalars(
        select(ConversationMessage)
        .where(ConversationMessage.task_id == task_id)
        .order_by(ConversationMessage.sequence)
    )
    return list(messages)


@router.post(
    "/{task_id}/cancel",
    response_model=TaskView,
    dependencies=[Depends(require_task_csrf)],
)
async def cancel_task(
    task_id: uuid.UUID,
    user: User = Depends(require_ready_task_user),
    db: AsyncSession = Depends(get_session),
):
    task = await db.scalar(
        select(Task)
        .where(
            Task.id == task_id,
            Task.owner_id == user.id,
            Task.deletion_requested_at.is_(None),
        )
        .with_for_update()
    )
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    if task.status in {
        "succeeded",
        "partially_completed",
        "failed",
        "cancelled",
    }:
        raise HTTPException(status_code=409, detail="Task is already terminal")

    now = datetime.now(timezone.utc)
    task.status = "cancelled"
    task.pending_tool_approval = None
    task.updated_at = now
    jobs = await db.scalars(
        select(Job)
        .where(Job.task_id == task_id, Job.status.in_({"pending", "leased"}))
        .with_for_update()
    )
    for job in jobs:
        job.status = "completed"
        job.lease_owner = None
        job.lease_expires_at = None
    db.add(
        Event(
            task_id=task_id,
            type="task.cancelled",
            payload_json={"status": "cancelled"},
        )
    )
    db.add(
        Event(
            task_id=task_id,
            type="task.status_changed",
            payload_json={"status": "cancelled"},
        )
    )
    await db.commit()
    await db.refresh(task)
    return task


@router.post(
    "/{task_id}/tool-budget-approval",
    response_model=TaskView,
    dependencies=[Depends(require_task_csrf)],
)
async def decide_tool_budget_approval(
    task_id: uuid.UUID,
    payload: ToolBudgetApprovalDecision,
    user: User = Depends(require_ready_task_user),
    db: AsyncSession = Depends(get_session),
):
    return await resolve_tool_budget_approval(task_id, user.id, payload, db)


@router.delete(
    "/{task_id}",
    response_model=TaskDeletionView,
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[Depends(require_task_csrf)],
)
async def delete_task(
    task_id: uuid.UUID,
    user: User = Depends(require_ready_task_user),
    db: AsyncSession = Depends(get_session),
):
    task = await db.scalar(
        select(Task)
        .where(Task.id == task_id, Task.owner_id == user.id)
        .with_for_update()
    )
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    if task.status not in {
        "succeeded",
        "partially_completed",
        "failed",
        "cancelled",
    }:
        raise HTTPException(
            status_code=409, detail="Only terminal tasks can be deleted"
        )
    deletion_job = await db.get(TaskDeletionJob, task_id)
    if task.deletion_requested_at is None:
        task.deletion_requested_at = datetime.now(timezone.utc)
        if deletion_job is None:
            deletion_job = TaskDeletionJob(task_id=task_id)
            db.add(deletion_job)
    await db.commit()
    return TaskDeletionView(
        task_id=task_id,
        status=deletion_job.status if deletion_job is not None else "pending",
    )


@router.post(
    "/{task_id}/messages",
    response_model=MessageView,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(require_task_csrf)],
)
async def add_message(
    task_id: uuid.UUID,
    payload: MessageCreate,
    user: User = Depends(require_ready_task_user),
    db: AsyncSession = Depends(get_session),
):
    task = await db.scalar(
        select(Task)
        .where(
            Task.id == task_id,
            Task.owner_id == user.id,
            Task.deletion_requested_at.is_(None),
        )
        .with_for_update()
    )
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    if task.status == "waiting_approval":
        raise HTTPException(
            status_code=409,
            detail="Resolve the pending tool-budget request explicitly",
        )
    if task.status == "cancelled":
        raise HTTPException(
            status_code=409, detail="This task cannot be continued"
        )
    sequence = await db.scalar(
        select(func.coalesce(func.max(ConversationMessage.sequence), 0)).where(
            ConversationMessage.task_id == task_id
        )
    )
    message = ConversationMessage(
        task_id=task_id,
        sequence=sequence + 1,
        role="user",
        content=payload.content,
    )
    db.add(message)
    await db.flush()
    db.add(
        Event(
            task_id=task_id,
            type="conversation.message_added",
            payload_json={
                "message_id": str(message.id),
                "sequence": message.sequence,
                "role": "user",
            },
        )
    )
    if task.status in {
        "succeeded",
        "partially_completed",
        "failed",
        "needs_attention",
    }:
        task.status = "queued"
        task.result = None
        task.error = None
        task.completion_report = None
        task.updated_at = datetime.now(timezone.utc)
        db.add(
            Event(
                task_id=task_id,
                type="task.status_changed",
                payload_json={
                    "status": "queued",
                    "conversation_sequence": message.sequence,
                },
            )
        )
        db.add(Job(task_id=task_id, status="pending"))
    await db.commit()
    await db.refresh(message)
    return message


@router.put(
    "/{task_id}/messages/{message_id}",
    response_model=MessageView,
    dependencies=[Depends(require_task_csrf)],
)
async def edit_message(
    task_id: uuid.UUID,
    message_id: uuid.UUID,
    payload: MessageCreate,
    user: User = Depends(require_ready_task_user),
    db: AsyncSession = Depends(get_session),
):
    task = await db.scalar(
        select(Task)
        .where(
            Task.id == task_id,
            Task.owner_id == user.id,
            Task.deletion_requested_at.is_(None),
        )
        .with_for_update()
    )
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    if task.status not in {
        "succeeded",
        "partially_completed",
        "failed",
        "needs_attention",
    }:
        raise HTTPException(
            status_code=409,
            detail="This task is not available for message editing",
        )

    message = await db.scalar(
        select(ConversationMessage)
        .where(
            ConversationMessage.id == message_id,
            ConversationMessage.task_id == task_id,
        )
        .with_for_update()
    )
    if message is None:
        raise HTTPException(status_code=404, detail="Message not found")
    if message.role != "user":
        raise HTTPException(
            status_code=409, detail="Only user messages can be edited"
        )

    first_user_message_id = await db.scalar(
        select(ConversationMessage.id)
        .where(
            ConversationMessage.task_id == task_id,
            ConversationMessage.role == "user",
        )
        .order_by(ConversationMessage.sequence)
        .limit(1)
    )
    current_max_sequence = (
        await db.scalar(
            select(func.max(ConversationMessage.sequence)).where(
                ConversationMessage.task_id == task_id
            )
        )
        or 0
    )
    if current_max_sequence >= 2_147_483_647:
        raise HTTPException(status_code=409, detail="Message history is full")

    previous_sequence = message.sequence
    new_sequence = current_max_sequence + 1
    await db.execute(
        delete(ConversationMessage).where(
            ConversationMessage.task_id == task_id,
            ConversationMessage.sequence > previous_sequence,
        )
    )
    await db.execute(
        update(TaskFile)
        .where(
            TaskFile.task_id == task_id,
            TaskFile.role == "input",
            TaskFile.conversation_sequence > previous_sequence,
        )
        .values(role="source")
    )
    now = datetime.now(timezone.utc)
    message.content = payload.content
    message.sequence = new_sequence
    message.created_at = now
    if message.id == first_user_message_id:
        task.instruction = payload.content
        task.title = None
        task.title_generated = False
    task.status = "queued"
    task.result = None
    task.error = None
    task.completion_report = None
    task.pending_tool_approval = None
    task.updated_at = now
    db.add(
        Event(
            task_id=task_id,
            type="conversation.rewound",
            payload_json={
                "message_id": str(message.id),
                "previous_sequence": previous_sequence,
                "sequence": new_sequence,
                "conversation_sequence": new_sequence,
            },
        )
    )
    db.add(
        Event(
            task_id=task_id,
            type="task.status_changed",
            payload_json={
                "status": "queued",
                "conversation_sequence": new_sequence,
            },
        )
    )
    db.add(Job(task_id=task_id, status="pending"))
    await db.commit()
    await db.refresh(message)
    return message


@router.get("/{task_id}/events")
async def task_events(
    task_id: uuid.UUID,
    request: Request,
    user: User = Depends(require_ready_task_user),
    db: AsyncSession = Depends(get_session),
):
    await _owned_task(db, task_id, user.id)
    try:
        last_event_id = max(0, int(request.headers.get("last-event-id", "0")))
    except ValueError as exc:
        raise HTTPException(
            status_code=400, detail="Invalid Last-Event-ID"
        ) from exc

    async def stream():
        cursor = last_event_id
        while not await request.is_disconnected():
            async with get_settings_session() as stream_db:
                active_task = await stream_db.scalar(
                    select(Task.id).where(
                        Task.id == task_id,
                        Task.owner_id == user.id,
                        Task.deletion_requested_at.is_(None),
                    )
                )
                if active_task is None:
                    break
                rows = await stream_db.scalars(
                    select(Event)
                    .where(Event.task_id == task_id, Event.id > cursor)
                    .order_by(Event.id)
                    .limit(200)
                )
                events = list(rows)
            for event in events:
                cursor = event.id
                data = json.dumps(event.payload_json, separators=(",", ":"))
                yield f"id: {event.id}\nevent: {event.type}\ndata: {data}\n\n"
            if not events:
                yield ": keep-alive\n\n"
                await asyncio.sleep(get_settings().session_poll_seconds)

    from app.db import SessionFactory as get_settings_session

    await db.close()
    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache"},
    )
