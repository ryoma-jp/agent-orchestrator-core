import uuid
import asyncio
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.db import get_session
from app.dependencies import require_admin, require_csrf
from app.models import SystemSettings, Task, User, WebSession
from app.schemas import (
    AdminSettingsUpdate,
    AdminSettingsView,
    AdminTaskSummary,
    CreateUserRequest,
    OpenAIKeyRequest,
    PasswordResetRequest,
    UserView,
)
from app.security import hash_password
from app.vault import write_openai_api_key

router = APIRouter(prefix="/api/admin", tags=["administration"])


def _default_settings() -> dict:
    settings = get_settings()
    return {
        "manager_model": settings.manager_model,
        "worker_model": settings.worker_model,
        "max_concurrent_agents": settings.max_concurrent_agents,
        "max_agent_attempts": min(max(settings.max_agent_attempts, 1), 3),
    }


async def _settings_row(db: AsyncSession) -> SystemSettings:
    row = await db.get(SystemSettings, 1)
    if row is None:
        row = SystemSettings(
            id=1, settings_json=_default_settings(), version=1
        )
        db.add(row)
        await db.flush()
    return row


@router.get("/settings", response_model=AdminSettingsView)
async def get_admin_settings(
    admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_session),
):
    row = await db.get(SystemSettings, 1)
    values = (
        _default_settings()
        if row is None
        else {**_default_settings(), **row.settings_json}
    )
    return AdminSettingsView(
        **values,
        openai_key_configured=bool(row and row.openai_key_configured),
        version=row.version if row else 1,
    )


@router.put(
    "/settings",
    response_model=AdminSettingsView,
    dependencies=[Depends(require_csrf)],
)
async def update_admin_settings(
    payload: AdminSettingsUpdate,
    admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_session),
):
    row = await _settings_row(db)
    row.settings_json = payload.model_dump()
    row.version += 1
    row.updated_by = admin.id
    await db.commit()
    return AdminSettingsView(
        **payload.model_dump(),
        openai_key_configured=row.openai_key_configured,
        version=row.version,
    )


@router.put(
    "/secrets/openai-api-key",
    status_code=status.HTTP_204_NO_CONTENT,
    dependencies=[Depends(require_csrf)],
)
async def set_openai_api_key(
    payload: OpenAIKeyRequest,
    admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_session),
):
    try:
        await asyncio.to_thread(
            write_openai_api_key, payload.api_key.get_secret_value()
        )
    except RuntimeError as exc:
        raise HTTPException(
            status_code=503, detail="Secret storage is unavailable"
        ) from exc
    row = await _settings_row(db)
    row.openai_key_configured = True
    row.updated_by = admin.id
    row.version += 1
    await db.commit()


@router.post(
    "/users",
    response_model=UserView,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(require_csrf)],
)
async def create_user(
    payload: CreateUserRequest,
    admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_session),
):
    email = str(payload.email).strip().lower()
    existing = await db.scalar(select(User.id).where(User.email == email))
    if existing:
        raise HTTPException(
            status_code=409, detail="Email is already registered"
        )
    user = User(
        email=email,
        password_hash=hash_password(payload.initial_password),
        role="user",
        is_active=True,
        must_change_password=True,
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)
    return user


@router.post(
    "/users/{user_id}/password-reset",
    response_model=UserView,
    dependencies=[Depends(require_csrf)],
)
async def reset_user_password(
    user_id: uuid.UUID,
    payload: PasswordResetRequest,
    admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_session),
):
    user = await db.get(User, user_id)
    if user is None or user.role != "user":
        raise HTTPException(status_code=404, detail="User not found")
    user.password_hash = hash_password(payload.initial_password)
    user.must_change_password = True
    await db.execute(
        update(WebSession)
        .where(WebSession.user_id == user.id)
        .values(revoked_at=datetime.now(timezone.utc))
    )
    await db.commit()
    await db.refresh(user)
    return user


@router.get("/operations/tasks", response_model=list[AdminTaskSummary])
async def list_operations_tasks(
    admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_session),
    limit: int = 100,
):
    rows = await db.execute(
        select(Task, User.email)
        .join(User, User.id == Task.owner_id)
        .order_by(Task.updated_at.desc())
        .limit(min(max(limit, 1), 200))
    )
    return [
        AdminTaskSummary(
            id=task.id,
            owner_email=email,
            status=task.status,
            created_at=task.created_at,
            updated_at=task.updated_at,
            error=task.error,
        )
        for task, email in rows
    ]
