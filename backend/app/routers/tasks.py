import asyncio
import hmac
import json
import mimetypes
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Depends, Header, HTTPException, Request, status
from fastapi.responses import FileResponse, StreamingResponse
from sqlalchemy import delete, func, select, update
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
    TaskDeletionJob,
    TaskFile,
    User,
)
from app.schemas import (
    MessageCreate,
    MessageView,
    TaskCreate,
    TaskDeletionView,
    TaskFileView,
    TaskView,
    ToolBudgetApprovalDecision,
    WorkspaceDirectoryView,
)
from app.tool_budget_approval import resolve_tool_budget_approval
from app.workspaces import (
    InvalidWorkspacePath,
    list_workspace_entries,
    normalize_workspace_path,
    resolve_workspace_file,
    task_workspace,
    write_workspace_stream,
)

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
    execution_settings.update(
        {
            "max_agent_turns": settings.max_agent_turns,
            "max_tool_calls_per_subtask": settings.max_tool_calls_per_subtask,
            "agent_runtime": {
                "sdk": "openai-agents",
                "version": "0.17.8",
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
    return await resolve_tool_budget_approval(
        task_id, user.id, payload, db
    )


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


@router.get(
    "/{task_id}/workspace/files",
    response_model=list[TaskFileView | WorkspaceDirectoryView],
)
async def list_task_files(
    task_id: uuid.UUID,
    user: User = Depends(require_ready_task_user),
    db: AsyncSession = Depends(get_session),
):
    await _owned_task(db, task_id, user.id)
    metadata_rows = await db.scalars(
        select(TaskFile).where(TaskFile.task_id == task_id)
    )
    metadata = {row.workspace_rel_path: row for row in metadata_rows}
    items = []
    for file_info in list_workspace_entries(task_id):
        path = str(file_info["path"])
        if file_info["kind"] == "directory":
            items.append(WorkspaceDirectoryView(path=path))
            continue
        row = metadata.get(path)
        if row is not None:
            items.append(TaskFileView.model_validate(row))
            continue
        updated_at = datetime.fromtimestamp(
            float(file_info["updated_at"]), tz=timezone.utc
        )
        items.append(
            TaskFileView(
                id=uuid.uuid5(task_id, path),
                path=path,
                role="source" if path.startswith("source/") else "input",
                source_kind="workspace",
                source_ref=None,
                size_bytes=int(file_info["size_bytes"]),
                created_at=updated_at,
                updated_at=updated_at,
            )
        )
    return items


@router.post(
    "/{task_id}/workspace/files",
    response_model=TaskFileView,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(require_task_csrf)],
)
async def upload_task_file(
    task_id: uuid.UUID,
    request: Request,
    filename: str,
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
    filename = filename.replace("\\", "/").rsplit("/", 1)[-1]
    if not filename or filename in {".", ".."} or len(filename) > 255:
        raise HTTPException(status_code=400, detail="Invalid filename")
    relative_path = f"inputs/{filename}"
    existing = await db.scalar(
        select(TaskFile).where(
            TaskFile.task_id == task_id,
            TaskFile.workspace_rel_path == relative_path,
        )
    )
    if existing is not None:
        raise HTTPException(status_code=409, detail="File already exists")
    try:
        existing_path = resolve_workspace_file(
            task_id, relative_path, create_parents=True
        )
        if existing_path.exists():
            raise HTTPException(status_code=409, detail="File already exists")
        size_bytes = await write_workspace_stream(
            task_id, relative_path, request.stream()
        )
    except InvalidWorkspacePath as exc:
        raise HTTPException(
            status_code=400, detail="Invalid workspace path"
        ) from exc
    task_file = TaskFile(
        task_id=task_id,
        workspace_rel_path=relative_path,
        role="input",
        source_kind="user_upload",
        source_ref=filename,
        size_bytes=size_bytes,
    )
    db.add(task_file)
    await db.flush()
    sequence = await db.scalar(
        select(func.coalesce(func.max(ConversationMessage.sequence), 0)).where(
            ConversationMessage.task_id == task_id
        )
    )
    message = ConversationMessage(
        task_id=task_id,
        sequence=sequence + 1,
        role="user",
        content=f"Attached input file: {relative_path}",
    )
    task_file.conversation_sequence = message.sequence
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
    await db.refresh(task_file)
    return task_file


@router.get("/{task_id}/workspace/files/{relative_path:path}")
async def download_task_file(
    task_id: uuid.UUID,
    relative_path: str,
    user: User = Depends(require_ready_task_user),
    db: AsyncSession = Depends(get_session),
):
    await _owned_task(db, task_id, user.id)
    try:
        normalized_path = normalize_workspace_path(relative_path)
        path = resolve_workspace_file(task_id, normalized_path)
    except InvalidWorkspacePath as exc:
        raise HTTPException(
            status_code=400, detail="Invalid workspace path"
        ) from exc
    if not path.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    media_type = (
        mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    )
    return FileResponse(path, media_type=media_type, filename=path.name)


@router.put(
    "/{task_id}/workspace/files/{relative_path:path}",
    response_model=TaskFileView,
    dependencies=[Depends(require_task_csrf)],
)
async def write_task_file(
    task_id: uuid.UUID,
    relative_path: str,
    request: Request,
    role: str = "input",
    user: User = Depends(require_ready_task_user),
    db: AsyncSession = Depends(get_session),
):
    await _owned_task(db, task_id, user.id)
    if role not in {"input", "source", "intermediate", "final"}:
        raise HTTPException(status_code=422, detail="Invalid file role")
    try:
        normalized_path = normalize_workspace_path(relative_path)
        size_bytes = await write_workspace_stream(
            task_id, normalized_path, request.stream()
        )
    except InvalidWorkspacePath as exc:
        raise HTTPException(
            status_code=400, detail="Invalid workspace path"
        ) from exc
    task_file = await db.scalar(
        select(TaskFile).where(
            TaskFile.task_id == task_id,
            TaskFile.workspace_rel_path == normalized_path,
        )
    )
    if task_file is None:
        task_file = TaskFile(
            task_id=task_id,
            workspace_rel_path=normalized_path,
            role=role,
            source_kind="user_edit",
            size_bytes=size_bytes,
        )
        db.add(task_file)
    else:
        task_file.size_bytes = size_bytes
        task_file.source_kind = "user_edit"
        if role != "input":
            task_file.role = role
    await db.commit()
    await db.refresh(task_file)
    return task_file


@router.delete(
    "/{task_id}/workspace/files/{relative_path:path}",
    status_code=status.HTTP_204_NO_CONTENT,
    dependencies=[Depends(require_task_csrf)],
)
async def delete_task_file(
    task_id: uuid.UUID,
    relative_path: str,
    user: User = Depends(require_ready_task_user),
    db: AsyncSession = Depends(get_session),
):
    await _owned_task(db, task_id, user.id)
    try:
        normalized_path = normalize_workspace_path(relative_path)
        path = resolve_workspace_file(task_id, normalized_path)
    except InvalidWorkspacePath as exc:
        raise HTTPException(
            status_code=400, detail="Invalid workspace path"
        ) from exc
    path.unlink(missing_ok=True)
    task_file = await db.scalar(
        select(TaskFile).where(
            TaskFile.task_id == task_id,
            TaskFile.workspace_rel_path == normalized_path,
        )
    )
    if task_file is not None:
        await db.delete(task_file)
        await db.commit()


@router.put(
    "/{task_id}/workspace/internal/{relative_path:path}",
    response_model=TaskFileView,
)
async def write_internal_task_file(
    task_id: uuid.UUID,
    relative_path: str,
    request: Request,
    x_workspace_token: str | None = Header(default=None),
    role: str = "final",
    db: AsyncSession = Depends(get_session),
):
    expected = get_settings().workspace_internal_token
    if (
        expected is None
        or x_workspace_token is None
        or not hmac.compare_digest(
            x_workspace_token, expected.get_secret_value()
        )
    ):
        raise HTTPException(status_code=404, detail="Not found")
    if role not in {"intermediate", "final"}:
        raise HTTPException(status_code=422, detail="Invalid file role")
    task = await db.scalar(
        select(Task).where(
            Task.id == task_id,
            Task.deletion_requested_at.is_(None),
        )
    )
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    try:
        normalized_path = normalize_workspace_path(relative_path)
        size_bytes = await write_workspace_stream(
            task_id, normalized_path, request.stream()
        )
    except InvalidWorkspacePath as exc:
        raise HTTPException(
            status_code=400, detail="Invalid workspace path"
        ) from exc
    task_file = await db.scalar(
        select(TaskFile).where(
            TaskFile.task_id == task_id,
            TaskFile.workspace_rel_path == normalized_path,
        )
    )
    if task_file is None:
        task_file = TaskFile(
            task_id=task_id,
            workspace_rel_path=normalized_path,
            role=role,
            source_kind="agent",
            source_ref="task_result",
            size_bytes=size_bytes,
        )
        db.add(task_file)
    else:
        task_file.role = role
        task_file.source_kind = "agent"
        task_file.source_ref = "task_result"
        task_file.size_bytes = size_bytes
    await db.commit()
    await db.refresh(task_file)
    return task_file


@router.get("/{task_id}/workspace/internal/inputs")
async def read_internal_task_inputs(
    task_id: uuid.UUID,
    x_workspace_token: str | None = Header(default=None),
    db: AsyncSession = Depends(get_session),
):
    expected = get_settings().workspace_internal_token
    if (
        expected is None
        or x_workspace_token is None
        or not hmac.compare_digest(
            x_workspace_token, expected.get_secret_value()
        )
    ):
        raise HTTPException(status_code=404, detail="Not found")
    task = await db.scalar(
        select(Task).where(
            Task.id == task_id,
            Task.deletion_requested_at.is_(None),
        )
    )
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    files = await db.scalars(
        select(TaskFile)
        .where(TaskFile.task_id == task_id, TaskFile.role == "input")
        .order_by(TaskFile.workspace_rel_path)
    )
    allowed_extensions = {
        ".csv",
        ".html",
        ".json",
        ".md",
        ".py",
        ".rst",
        ".sql",
        ".text",
        ".toml",
        ".ts",
        ".tsx",
        ".txt",
        ".xml",
        ".yaml",
        ".yml",
    }
    excerpts = []
    total_chars = 0
    for task_file in files:
        if (
            Path(task_file.workspace_rel_path).suffix.lower()
            not in allowed_extensions
            or task_file.size_bytes > 1_000_000
        ):
            continue
        try:
            path = resolve_workspace_file(
                task_id, task_file.workspace_rel_path
            )
            text = await asyncio.to_thread(path.read_text, encoding="utf-8")
        except (OSError, UnicodeDecodeError, InvalidWorkspacePath):
            continue
        remaining = 100_000 - total_chars
        if remaining <= 0:
            break
        excerpt = text[:remaining]
        excerpts.append(
            {"path": task_file.workspace_rel_path, "content": excerpt}
        )
        total_chars += len(excerpt)
    return {"files": excerpts}


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
    current_max_sequence = await db.scalar(
        select(func.max(ConversationMessage.sequence)).where(
            ConversationMessage.task_id == task_id
        )
    ) or 0
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


async def _owned_task(
    db: AsyncSession, task_id: uuid.UUID, owner_id: uuid.UUID
) -> Task:
    task = await db.scalar(
        select(Task).where(
            Task.id == task_id,
            Task.owner_id == owner_id,
            Task.deletion_requested_at.is_(None),
        )
    )
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    return task
