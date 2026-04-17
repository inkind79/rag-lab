"""
FastAPI-Users configuration — JWT cookie auth, registration, user management.

Uses username-based auth (not email) since this is a local-first app.
The email field is still required by FastAPI-Users but serves as a secondary identifier.
"""

import re
import uuid
from typing import Optional

from fastapi import Depends, Request
from fastapi_users import BaseUserManager, FastAPIUsers, UUIDIDMixin, exceptions, schemas
from fastapi_users.authentication import AuthenticationBackend, CookieTransport, JWTStrategy
from pydantic import Field

from src.api.db import User, get_user_db
from src.api.config import JWT_SECRET, JWT_EXPIRY_HOURS
from src.utils.logger import get_logger

logger = get_logger(__name__)

PASSWORD_MIN_LENGTH = 8
PASSWORD_MAX_LENGTH = 128
_PASSWORD_HAS_LETTER = re.compile(r'[A-Za-z]')
_PASSWORD_HAS_DIGIT = re.compile(r'\d')

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

    async def validate_password(self, password: str, user) -> None:
        """Enforce minimum complexity. Called by fastapi-users on register + reset."""
        if not isinstance(password, str):
            raise exceptions.InvalidPasswordException(reason="Password must be a string")
        if len(password) < PASSWORD_MIN_LENGTH:
            raise exceptions.InvalidPasswordException(
                reason=f"Password must be at least {PASSWORD_MIN_LENGTH} characters"
            )
        if len(password) > PASSWORD_MAX_LENGTH:
            raise exceptions.InvalidPasswordException(
                reason=f"Password must be at most {PASSWORD_MAX_LENGTH} characters"
            )
        if not _PASSWORD_HAS_LETTER.search(password):
            raise exceptions.InvalidPasswordException(
                reason="Password must contain at least one letter"
            )
        if not _PASSWORD_HAS_DIGIT.search(password):
            raise exceptions.InvalidPasswordException(
                reason="Password must contain at least one digit"
            )
        # Reject password containing the username (case-insensitive substring)
        username = getattr(user, 'username', None) or getattr(user, 'email', '')
        if username and len(username) >= 4 and username.lower() in password.lower():
            raise exceptions.InvalidPasswordException(
                reason="Password must not contain the username"
            )


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
