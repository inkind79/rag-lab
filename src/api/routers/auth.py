"""
Auth Router — registration, login, logout, check

Uses FastAPI-Users for user management with a custom username-based login.
"""

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel
from sqlalchemy import select

from fastapi_users.db import SQLAlchemyUserDatabase

from src.api.rate_limit import LOGIN_LIMIT, limiter
from src.api.users import (
    fastapi_users, auth_backend, UserRead, UserCreate, UserManager,
    current_active_user, get_jwt_strategy,
)
from src.api.db import User, get_async_session

router = APIRouter()

# ── Registration (FastAPI-Users built-in; rate-limited via PathRateLimitMiddleware) ──
router.include_router(
    fastapi_users.get_register_router(UserRead, UserCreate),
    prefix="/auth",
    tags=["auth"],
)

# ── Login/Logout (FastAPI-Users built-in, email-based) ──
router.include_router(
    fastapi_users.get_auth_router(auth_backend),
    prefix="/auth/cookie",
    tags=["auth"],
)


# ── Custom username-based login (our primary login endpoint) ──
class LoginRequest(BaseModel):
    username: str
    password: str


@router.post("/auth/login")
@limiter.limit(LOGIN_LIMIT)
async def login_by_username(
    request: Request,
    body: LoginRequest,
    response: Response,
    session=Depends(get_async_session),
):
    """Login using username + password (the primary login method for RAG Lab)."""
    from src.api.users import get_user_manager, get_user_db
    from fastapi_users.exceptions import UserNotExists

    # Look up user by username
    result = await session.execute(select(User).where(User.username == body.username))
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    # Verify password using FastAPI-Users password helper
    user_db = SQLAlchemyUserDatabase(session, User)
    manager = UserManager(user_db)
    verified, _updated_hash = manager.password_helper.verify_and_update(body.password, user.hashed_password)
    if not verified:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    if not user.is_active:
        raise HTTPException(status_code=401, detail="Account is inactive")

    # Generate JWT token and set cookie
    strategy = get_jwt_strategy()
    token = await strategy.write_token(user)
    response.set_cookie(
        key="auth_token",
        value=token,
        httponly=True,
        samesite="lax",
        max_age=7 * 24 * 3600,
        path="/",
    )
    return {"success": True, "data": {"user_id": user.username}}


@router.post("/auth/logout")
async def logout(response: Response):
    response.delete_cookie("auth_token", path="/")
    return {"success": True, "message": "Logged out"}


@router.get("/auth/check")
async def check_auth(user: User = Depends(current_active_user)):
    return {"success": True, "data": {"user_id": user.username}}
