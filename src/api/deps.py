"""
Shared FastAPI dependencies.

These are injected via Depends() into route handlers.
"""

import re
from typing import Optional
from fastapi import Header, HTTPException, Depends

from src.api.users import current_active_user
from src.api.db import User
from src.api import config
from src.utils.logger import get_logger

logger = get_logger(__name__)

_UUID_RE = re.compile(r'^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$', re.IGNORECASE)


async def get_current_user(user: User = Depends(current_active_user)) -> str:
    """Return the username string for backward compatibility with all routes."""
    return user.username


async def get_session_id(
    x_session_uuid: Optional[str] = Header(None, alias="X-Session-UUID"),
) -> str:
    """Extract and validate session UUID from X-Session-UUID header."""
    if not x_session_uuid:
        raise HTTPException(status_code=400, detail="X-Session-UUID header required")
    if not _UUID_RE.match(x_session_uuid):
        raise HTTPException(status_code=400, detail="Invalid session UUID format")
    return x_session_uuid


async def get_session_data(
    session_id: str = Depends(get_session_id),
    user_id: str = Depends(get_current_user),
) -> dict:
    """Load and validate session data. Raises 404/403 on failure."""
    from src.services.session_manager.manager import load_session
    session_data = load_session(config.SESSION_FOLDER, session_id)
    if not session_data:
        raise HTTPException(status_code=404, detail="Session not found")
    if session_data.get('user_id') != user_id:
        raise HTTPException(status_code=403, detail="Not authorized for this session")
    return session_data


def get_rag_models():
    """Get the global thread-safe model manager."""
    from src.utils.thread_safe_models import get_thread_safe_model_manager
    return get_thread_safe_model_manager()
