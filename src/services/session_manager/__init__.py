"""
Session manager service for handling session operations.

This service is responsible for creating, loading, saving, and managing sessions.
"""
from .manager import (
    create_session,
    load_session,
    save_session,
    get_all_sessions, # Re-added for multi-session per user
    delete_session_data
)

__all__ = [
    'create_session',
    'load_session',
    'save_session',
    'get_all_sessions', # Re-added
    'delete_session_data'
]
