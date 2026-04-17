"""
Auth Router — registration, login, logout, check

Uses FastAPI-Users for user management with a custom username-based login.
"""

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel
from sqlalchemy import select

from fastapi_users.db import SQLAlchemyUserDatabase

from src.api.users import (
    fastapi_users, auth_backend, UserRead, UserCreate, UserManager,
    current_active_user, get_jwt_strategy,
)
from src.api.db import User, get_async_session

router = APIRouter()

# ── Registration (FastAPI-Users built-in) ──
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


_INVALID_CREDENTIALS = "Invalid credentials"

# Pre-computed hash of a random string. We verify against this when the user
# isn't found so the timing of "user missing" matches "wrong password" — both
# pay one bcrypt verification round.
_DUMMY_HASH: str | None = None


def _get_dummy_hash(manager: 'UserManager') -> str:
    global _DUMMY_HASH
    if _DUMMY_HASH is None:
        import secrets as _secrets
        _DUMMY_HASH = manager.password_helper.hash(_secrets.token_hex(16))
    return _DUMMY_HASH


@router.post("/auth/login")
async def login_by_username(
    request: Request,
    body: LoginRequest,
    response: Response,
    session=Depends(get_async_session),
):
    """Login using username + password (the primary login method for RAG Lab).

    Always pays one bcrypt verification round and always returns the same
    "Invalid credentials" message regardless of failure mode (missing user,
    wrong password, inactive account) so the endpoint isn't an account
    enumeration oracle.
    """
    user_db = SQLAlchemyUserDatabase(session, User)
    manager = UserManager(user_db)

    result = await session.execute(select(User).where(User.username == body.username))
    user = result.scalar_one_or_none()

    # Verify against the real hash if we have one, else against the dummy hash.
    target_hash = user.hashed_password if user else _get_dummy_hash(manager)
    verified, _updated_hash = manager.password_helper.verify_and_update(body.password, target_hash)

    if not user or not verified or not user.is_active:
        raise HTTPException(status_code=401, detail=_INVALID_CREDENTIALS)

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
