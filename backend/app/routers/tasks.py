import asyncio
import json
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.db import get_session
from app.dependencies import require_ready_task_user, require_task_csrf
from app.models import (
    ConversationMessage,
    Event,
    Job,
    SystemSettings,
    Task,
    User,
)
from app.schemas import MessageCreate, MessageView, TaskCreate, TaskView

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
        "manager_model": settings.manager_model,
        "worker_model": settings.worker_model,
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
    task = Task(
        owner_id=user.id,
        instruction=payload.instruction,
        status="queued",
        config_snapshot=execution_settings,
    )
    db.add(task)
    await db.flush()
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
        .where(Task.owner_id == user.id)
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
        select(Task).where(Task.id == task_id, Task.owner_id == user.id)
    )
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    return task


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
        .where(Task.id == task_id, Task.owner_id == user.id)
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


@router.delete(
    "/{task_id}",
    status_code=status.HTTP_204_NO_CONTENT,
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
    await db.delete(task)
    await db.commit()


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
        .where(Task.id == task_id, Task.owner_id == user.id)
        .with_for_update()
    )
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    if task.status in {"failed", "cancelled"}:
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


async def _owned_task(
    db: AsyncSession, task_id: uuid.UUID, owner_id: uuid.UUID
) -> Task:
    task = await db.scalar(
        select(Task).where(Task.id == task_id, Task.owner_id == owner_id)
    )
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    return task
