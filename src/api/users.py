"""
FastAPI-Users configuration — JWT cookie auth, registration, user management.

Uses username-based auth (not email) since this is a local-first app.
The email field is still required by FastAPI-Users but serves as a secondary identifier.
"""

import uuid
from typing import Optional

from fastapi import Depends, Request
from fastapi_users import BaseUserManager, FastAPIUsers, UUIDIDMixin, schemas
from fastapi_users.authentication import AuthenticationBackend, CookieTransport, JWTStrategy
from pydantic import Field

from src.api.db import User, get_user_db
from src.api.config import JWT_SECRET, JWT_EXPIRY_HOURS
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ── Schemas ──

class UserRead(schemas.BaseUser[uuid.UUID]):
    username: str


class UserCreate(schemas.BaseUserCreate):
    username: str
    # email is required by FastAPI-Users but can be set to username@local for local apps
    email: str = Field(default="")

    def model_post_init(self, __context) -> None:
        # Auto-generate email from username if not provided
        if not self.email or self.email == "":
            self.email = f"{self.username}@raglab.app"


class UserUpdate(schemas.BaseUserUpdate):
    username: Optional[str] = None


# ── User Manager ──

class UserManager(UUIDIDMixin, BaseUserManager[User, uuid.UUID]):
    reset_password_token_secret = JWT_SECRET
    verification_token_secret = JWT_SECRET

    async def on_after_register(self, user: User, request: Optional[Request] = None):
        logger.info(f"User registered: {user.username} ({user.id})")


async def get_user_manager(user_db=Depends(get_user_db)):
    yield UserManager(user_db)


# ── Auth Backend (JWT in HttpOnly cookie) ──

cookie_transport = CookieTransport(
    cookie_name="auth_token",
    cookie_max_age=JWT_EXPIRY_HOURS * 3600,
    cookie_httponly=True,
    cookie_samesite="lax",
)


def get_jwt_strategy() -> JWTStrategy:
    return JWTStrategy(secret=JWT_SECRET, lifetime_seconds=JWT_EXPIRY_HOURS * 3600)


auth_backend = AuthenticationBackend(
    name="cookie",
    transport=cookie_transport,
    get_strategy=get_jwt_strategy,
)

# ── FastAPI-Users instance ──

fastapi_users = FastAPIUsers[User, uuid.UUID](get_user_manager, [auth_backend])

current_active_user = fastapi_users.current_user(active=True)
