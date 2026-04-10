"""
Sessions Router

GET    /api/v1/sessions
POST   /api/v1/sessions
GET    /api/v1/sessions/{uuid}
POST   /api/v1/sessions/{uuid}/activate
PUT    /api/v1/sessions/{uuid}/rename
DELETE /api/v1/sessions/{uuid}
DELETE /api/v1/sessions/all
"""

import os
import shutil
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from src.api.deps import get_current_user
from src.api import config
from src.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()


class CreateSessionRequest(BaseModel):
    name: Optional[str] = None


class RenameRequest(BaseModel):
    name: str


@router.get("/sessions")
async def list_sessions(user_id: str = Depends(get_current_user)):
    from src.services.session_manager.manager import get_all_sessions
    sessions_list = get_all_sessions(config.SESSION_FOLDER, user_id)
    return {"success": True, "data": {"sessions": sessions_list}}


@router.post("/sessions", status_code=201)
async def create_session(body: CreateSessionRequest, user_id: str = Depends(get_current_user)):
    from src.services.session_manager.manager import create_session as _create

    new_uuid, session_data = _create(config.SESSION_FOLDER, user_id, body.name)
    if not new_uuid:
        raise HTTPException(status_code=500, detail="Failed to create session")

    return {
        "success": True,
        "data": {
            "session_id": new_uuid,
            "session_name": session_data.get('session_name', ''),
        }
    }


@router.get("/sessions/{session_uuid}")
async def get_session(session_uuid: str, user_id: str = Depends(get_current_user)):
    from src.services.session_manager.manager import load_session
    session_data = load_session(config.SESSION_FOLDER, session_uuid)

    if not session_data:
        raise HTTPException(status_code=404, detail="Session not found")
    if session_data.get('user_id') != user_id:
        raise HTTPException(status_code=403, detail="Not authorized")

    return {"success": True, "data": session_data}


@router.post("/sessions/{session_uuid}/activate")
async def activate_session(session_uuid: str, user_id: str = Depends(get_current_user)):
    from src.services.session_manager.manager import load_session
    session_data = load_session(config.SESSION_FOLDER, session_uuid)

    if not session_data:
        raise HTTPException(status_code=404, detail="Session not found")
    if session_data.get('user_id') != user_id:
        raise HTTPException(status_code=403, detail="Not authorized")

    return {
        "success": True,
        "data": {
            "session_id": session_uuid,
            "session_name": session_data.get('session_name', ''),
        }
    }


@router.put("/sessions/{session_uuid}/rename")
async def rename_session(session_uuid: str, body: RenameRequest, user_id: str = Depends(get_current_user)):
    from src.services.session_manager.manager import load_session, save_session
    session_data = load_session(config.SESSION_FOLDER, session_uuid)

    if not session_data:
        raise HTTPException(status_code=404, detail="Session not found")
    if session_data.get('user_id') != user_id:
        raise HTTPException(status_code=403, detail="Not authorized")

    session_data['session_name'] = body.name.strip()
    save_session(config.SESSION_FOLDER, session_uuid, session_data)
    return {"success": True, "data": {"session_name": body.name.strip()}}


@router.delete("/sessions/all")
async def delete_all_sessions(user_id: str = Depends(get_current_user)):
    """Delete all sessions for the current user. Must be defined before the
    {session_uuid} route so FastAPI doesn't treat 'all' as a UUID."""
    from src.services.session_manager.manager import get_all_sessions

    sessions_list = get_all_sessions(config.SESSION_FOLDER, user_id)
    deleted = 0

    for sess in sessions_list:
        sess_uuid = sess.get('id', '')
        if not sess_uuid:
            continue
        try:
            session_file = os.path.join(config.SESSION_FOLDER, f"{sess_uuid}.json")
            if os.path.exists(session_file):
                os.remove(session_file)

            session_upload = os.path.join(config.UPLOAD_FOLDER, sess_uuid)
            if os.path.exists(session_upload):
                shutil.rmtree(session_upload, ignore_errors=True)

            session_images = os.path.join(config.STATIC_FOLDER, 'images', sess_uuid)
            if os.path.exists(session_images):
                shutil.rmtree(session_images, ignore_errors=True)

            try:
                from src.models.vector_stores.lancedb_manager import clear_lancedb_resources
                clear_lancedb_resources(sess_uuid)
            except Exception:
                pass

            deleted += 1
        except Exception as e:
            logger.error(f"Error deleting session {sess_uuid}: {e}")

    return {"success": True, "message": f"Deleted {deleted} sessions", "deleted_count": deleted}


@router.delete("/sessions/{session_uuid}")
async def delete_session(session_uuid: str, user_id: str = Depends(get_current_user)):
    from src.services.session_manager.manager import load_session
    session_data = load_session(config.SESSION_FOLDER, session_uuid)

    if not session_data:
        raise HTTPException(status_code=404, detail="Session not found")
    if session_data.get('user_id') != user_id:
        raise HTTPException(status_code=403, detail="Not authorized")

    # Delete session file
    session_file = os.path.join(config.SESSION_FOLDER, f"{session_uuid}.json")
    if os.path.exists(session_file):
        os.remove(session_file)

    # Delete uploaded documents
    session_upload = os.path.join(config.UPLOAD_FOLDER, session_uuid)
    if os.path.exists(session_upload):
        shutil.rmtree(session_upload, ignore_errors=True)

    # Delete static images
    session_images = os.path.join(config.STATIC_FOLDER, 'images', session_uuid)
    if os.path.exists(session_images):
        shutil.rmtree(session_images, ignore_errors=True)

    # Clean up LanceDB
    try:
        from src.models.vector_stores.lancedb_manager import clear_lancedb_resources
        clear_lancedb_resources(session_uuid)
    except Exception as e:
        logger.warning(f"Error cleaning LanceDB for {session_uuid}: {e}")

    return {"success": True, "message": f"Session {session_uuid} deleted"}
