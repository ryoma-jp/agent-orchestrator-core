import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_session
from app.dependencies import (
    get_current_session,
    get_current_user,
    require_csrf,
)
from app.models import ApiAccessKey, User, WebSession
from app.schemas import AccessKeyResponse, PasswordChangeRequest
from app.security import (
    hash_password,
    new_token,
    token_digest,
    verify_password,
)

router = APIRouter(prefix="/api/account", tags=["account"])


@router.post(
    "/password",
    status_code=status.HTTP_204_NO_CONTENT,
    dependencies=[Depends(require_csrf)],
)
async def change_password(
    payload: PasswordChangeRequest,
    user: User = Depends(get_current_user),
    web_session: WebSession = Depends(get_current_session),
    db: AsyncSession = Depends(get_session),
):
    if not verify_password(payload.current_password, user.password_hash):
        raise HTTPException(
            status_code=403, detail="Current password is incorrect"
        )
    user.password_hash = hash_password(payload.new_password)
    user.must_change_password = False
    await db.execute(
        update(WebSession)
        .where(WebSession.user_id == user.id, WebSession.id != web_session.id)
        .values(revoked_at=datetime.now(timezone.utc))
    )
    await db.commit()


@router.post(
    "/access-key",
    response_model=AccessKeyResponse,
    dependencies=[Depends(require_csrf)],
)
async def issue_access_key(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
):
    now = datetime.now(timezone.utc)
    current = await db.scalar(
        select(ApiAccessKey)
        .where(
            ApiAccessKey.user_id == user.id,
            ApiAccessKey.revoked_at.is_(None),
        )
        .with_for_update()
    )
    if current:
        current.revoked_at = now
        await db.flush()

    access_key = new_token()
    db.add(
        ApiAccessKey(
            id=uuid.uuid4(),
            user_id=user.id,
            key_hash=token_digest(access_key),
        )
    )
    await db.commit()
    return AccessKeyResponse(access_key=access_key, created_at=now)
