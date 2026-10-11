import hmac
from datetime import datetime, timedelta, timezone

from fastapi import Cookie, Depends, Header, HTTPException, Request, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.db import get_session
from app.models import ApiAccessKey, User, WebSession
from app.security import token_digest


async def _session(session_token: str | None, db: AsyncSession) -> WebSession:
    if not session_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
        )
    row = await db.scalar(
        select(WebSession).where(
            WebSession.session_token_hash == token_digest(session_token)
        )
    )
    now = datetime.now(timezone.utc)
    settings = get_settings()
    if (
        row is None
        or row.revoked_at is not None
        or row.absolute_expires_at <= now
        or row.last_seen_at
        <= now - timedelta(minutes=settings.session_idle_minutes)
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Session expired"
        )
    row.last_seen_at = now
    await db.commit()
    return row


async def get_current_session(
    session_token: str | None = Cookie(default=None, alias="agent_session"),
    db: AsyncSession = Depends(get_session),
) -> WebSession:
    return await _session(session_token, db)


async def get_current_user(
    web_session: WebSession = Depends(get_current_session),
    db: AsyncSession = Depends(get_session),
) -> User:
    user = await db.get(User, web_session.user_id)
    if user is None or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Account unavailable",
        )
    return user


async def get_task_user(
    request: Request,
    db: AsyncSession = Depends(get_session),
) -> User:
    authorization = request.headers.get("authorization")
    if authorization:
        scheme, separator, token = authorization.partition(" ")
        if scheme.lower() != "bearer" or not separator or not token:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authorization credentials",
            )
        api_key = await db.scalar(
            select(ApiAccessKey).where(
                ApiAccessKey.key_hash == token_digest(token),
                ApiAccessKey.revoked_at.is_(None),
            )
        )
        user = await db.get(User, api_key.user_id) if api_key else None
    else:
        web_session = await _session(request.cookies.get("agent_session"), db)
        user = await db.get(User, web_session.user_id)

    if user is None or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
        )
    return user


async def require_csrf(
    request: Request,
    web_session: WebSession = Depends(get_current_session),
    csrf_header: str | None = Header(default=None, alias="X-CSRF-Token"),
) -> None:
    _validate_csrf(request, web_session, csrf_header)


async def require_task_csrf(
    request: Request,
    db: AsyncSession = Depends(get_session),
    csrf_header: str | None = Header(default=None, alias="X-CSRF-Token"),
) -> None:
    authorization = request.headers.get("authorization", "")
    if authorization.lower().startswith("bearer "):
        return
    web_session = await _session(request.cookies.get("agent_session"), db)
    _validate_csrf(request, web_session, csrf_header)


def _validate_csrf(
    request: Request, web_session: WebSession, csrf_header: str | None
) -> None:
    csrf_cookie = request.cookies.get("agent_csrf")
    if (
        not csrf_cookie
        or not csrf_header
        or not hmac.compare_digest(csrf_cookie, csrf_header)
        or not hmac.compare_digest(
            token_digest(csrf_header), web_session.csrf_token_hash
        )
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="CSRF validation failed",
        )


async def require_ready_user(user: User = Depends(get_current_user)) -> User:
    if user.must_change_password:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Password change required",
        )
    return user


async def require_ready_task_user(
    user: User = Depends(get_task_user),
) -> User:
    if user.must_change_password:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Password change required",
        )
    return user


async def require_admin(user: User = Depends(require_ready_user)) -> User:
    if user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Administrator access required",
        )
    return user
