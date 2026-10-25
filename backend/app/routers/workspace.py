import asyncio
import hmac
import mimetypes
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Depends, Header, HTTPException, Request, status
from fastapi.responses import FileResponse
from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.db import get_session
from app.dependencies import require_ready_task_user, require_task_csrf
from app.models import ConversationMessage, Event, Job, Task, TaskFile, User
from app.schemas import TaskFileView, WorkspaceDirectoryView
from app.workspaces import (
    InvalidWorkspacePath,
    list_workspace_entries,
    normalize_workspace_path,
    resolve_workspace_file,
    write_workspace_stream,
)

router = APIRouter(prefix="/api/tasks", tags=["task workspace"])


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
        updated_at = datetime.fromtimestamp(
            float(file_info["updated_at"]), tz=timezone.utc
        )
        row = metadata.get(path)
        if row is not None:
            file_view = TaskFileView.model_validate(row)
            file_view.updated_at = updated_at
            items.append(file_view)
            continue
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
    statement = (
        pg_insert(TaskFile)
        .values(
            task_id=task_id,
            workspace_rel_path=normalized_path,
            role=role,
            source_kind="agent",
            source_ref="task_result",
            size_bytes=size_bytes,
        )
        .on_conflict_do_update(
            constraint="uq_task_files_path",
            set_={
                "role": role,
                "source_kind": "agent",
                "source_ref": "task_result",
                "size_bytes": size_bytes,
                "updated_at": func.now(),
            },
        )
        .returning(TaskFile)
    )
    task_file = await db.scalar(statement)
    await db.commit()
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
