"""Filesystem operations constrained to a task's persistent workspace."""

import os
import shutil
import uuid
from collections.abc import AsyncIterable
from pathlib import Path

from app.config import get_settings


class InvalidWorkspacePath(ValueError):
    pass


def normalize_workspace_path(relative_path: str) -> str:
    if (
        not relative_path
        or relative_path.startswith("/")
        or "\\" in relative_path
        or "\0" in relative_path
    ):
        raise InvalidWorkspacePath("Invalid workspace path")
    parts = relative_path.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise InvalidWorkspacePath("Invalid workspace path")
    return "/".join(parts)


def task_workspace(task_id: uuid.UUID, *, create: bool = False) -> Path:
    root = Path(get_settings().workspace_root).resolve()
    if create:
        root.mkdir(mode=0o700, parents=True, exist_ok=True)
    workspace = root / str(task_id)
    if workspace.is_symlink():
        raise InvalidWorkspacePath("Invalid workspace root")
    if create:
        workspace.mkdir(mode=0o700, exist_ok=True)
    if not workspace.resolve().is_relative_to(root):
        raise InvalidWorkspacePath("Invalid workspace root")
    return workspace


def resolve_workspace_file(
    task_id: uuid.UUID, relative_path: str, *, create_parents: bool = False
) -> Path:
    normalized = normalize_workspace_path(relative_path)
    workspace = task_workspace(task_id, create=create_parents)
    candidate = workspace
    parts = normalized.split("/")
    for part in parts[:-1]:
        candidate = candidate / part
        if candidate.is_symlink():
            raise InvalidWorkspacePath("Symbolic links are not allowed")
        if create_parents:
            candidate.mkdir(mode=0o700, exist_ok=True)
    target = candidate / parts[-1]
    if target.is_symlink():
        raise InvalidWorkspacePath("Symbolic links are not allowed")
    if not target.resolve().is_relative_to(workspace.resolve()):
        raise InvalidWorkspacePath("Invalid workspace path")
    return target


async def write_workspace_stream(
    task_id: uuid.UUID,
    relative_path: str,
    chunks: AsyncIterable[bytes],
) -> int:
    target = resolve_workspace_file(
        task_id, relative_path, create_parents=True
    )
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    size = 0
    try:
        with temporary.open("xb") as file:
            os.chmod(temporary, 0o600)
            async for chunk in chunks:
                if chunk:
                    file.write(chunk)
                    size += len(chunk)
        os.replace(temporary, target)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return size


def list_workspace_entries(task_id: uuid.UUID) -> list[dict[str, object]]:
    workspace = task_workspace(task_id, create=True)
    items: list[dict[str, object]] = []
    for directory, directories, filenames in os.walk(
        workspace, followlinks=False
    ):
        base = Path(directory)
        directories[:] = [
            name
            for name in directories
            if name != ".git" and not (base / name).is_symlink()
        ]
        for name in directories:
            path = base / name
            items.append(
                {
                    "kind": "directory",
                    "path": path.relative_to(workspace).as_posix(),
                }
            )
        for filename in filenames:
            path = base / filename
            if path.is_symlink() or not path.is_file():
                continue
            stat = path.stat()
            items.append(
                {
                    "kind": "file",
                    "path": path.relative_to(workspace).as_posix(),
                    "size_bytes": stat.st_size,
                    "updated_at": stat.st_mtime,
                }
            )
    return sorted(
        items,
        key=lambda item: (str(item["path"]).casefold(), str(item["kind"])),
    )


def delete_task_workspace(task_id: uuid.UUID) -> None:
    root = Path(get_settings().workspace_root).resolve()
    workspace = root / str(task_id)
    if workspace.is_symlink():
        workspace.unlink(missing_ok=True)
    elif workspace.exists():
        if not workspace.resolve().is_relative_to(root):
            raise InvalidWorkspacePath("Invalid workspace root")
        shutil.rmtree(workspace)
