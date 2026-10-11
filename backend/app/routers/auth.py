from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, Response, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.db import get_session
from app.dependencies import (
    get_current_session,
    get_current_user,
    require_csrf,
)
from app.models import User, WebSession
from app.schemas import LoginRequest, LoginResponse, UserView
from app.security import (
    new_token,
    token_digest,
    verify_password,
)

router = APIRouter(prefix="/api/auth", tags=["authentication"])
SESSION_COOKIE = "agent_session"
CSRF_COOKIE = "agent_csrf"


@router.post("/login", response_model=LoginResponse)
async def login(
    payload: LoginRequest,
    response: Response,
    db: AsyncSession = Depends(get_session),
):
    email = str(payload.email).strip().lower()
    user = await db.scalar(select(User).where(User.email == email))
    if (
        user is None
        or not user.is_active
        or not verify_password(payload.password, user.password_hash)
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
        )

    now = datetime.now(timezone.utc)
    session_token = new_token()
    csrf_token = new_token()
    settings = get_settings()
    web_session = WebSession(
        user_id=user.id,
        session_token_hash=token_digest(session_token),
        csrf_token_hash=token_digest(csrf_token),
        last_seen_at=now,
        absolute_expires_at=now
        + timedelta(hours=settings.session_absolute_hours),
    )
    db.add(web_session)
    await db.commit()
    response.set_cookie(
        SESSION_COOKIE,
        session_token,
        httponly=True,
        secure=settings.session_cookie_secure,
        samesite="lax",
        max_age=settings.session_absolute_hours * 3600,
        path="/",
    )
    response.set_cookie(
        CSRF_COOKIE,
        csrf_token,
        httponly=False,
        secure=settings.session_cookie_secure,
        samesite="lax",
        max_age=settings.session_absolute_hours * 3600,
        path="/",
    )
    return LoginResponse(
        user=UserView.model_validate(user), csrf_token=csrf_token
    )


@router.post(
    "/logout",
    status_code=status.HTTP_204_NO_CONTENT,
    dependencies=[Depends(require_csrf)],
)
async def logout(
    response: Response,
    web_session: WebSession = Depends(get_current_session),
    db: AsyncSession = Depends(get_session),
):
    web_session.revoked_at = datetime.now(timezone.utc)
    await db.commit()
    response.delete_cookie(SESSION_COOKIE, path="/")
    response.delete_cookie(CSRF_COOKIE, path="/")


@router.get("/me", response_model=UserView)
async def me(user: User = Depends(get_current_user)):
    return user
