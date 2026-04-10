"""
Scores Router

POST /api/v1/scores
GET  /api/v1/scores
POST /api/v1/scores/slope/toggle
PUT  /api/v1/scores/slope/params
"""

from typing import Any, Optional
from fastapi import APIRouter, Depends
from pydantic import BaseModel

from src.api.deps import get_session_id, get_session_data
from src.api import config
from src.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()


class SaveScoresRequest(BaseModel):
    scores: Any


class SlopeToggle(BaseModel):
    enabled: bool


class SlopeParams(BaseModel):
    rel_drop_threshold: Optional[float] = None
    abs_score_threshold: Optional[float] = None


@router.post("/scores")
async def save_scores(
    body: SaveScoresRequest,
    session_id: str = Depends(get_session_id),
    session_data: dict = Depends(get_session_data),
):
    from src.services.session_manager.manager import save_session
    session_data['similarity_scores'] = body.scores
    save_session(config.SESSION_FOLDER, session_id, session_data)
    return {"success": True, "message": "Scores saved"}


@router.get("/scores")
async def load_scores(
    session_id: str = Depends(get_session_id),
    session_data: dict = Depends(get_session_data),
):
    return {"success": True, "data": {"scores": session_data.get('similarity_scores')}}


@router.post("/scores/slope/toggle")
async def toggle_slope(
    body: SlopeToggle,
    session_id: str = Depends(get_session_id),
    session_data: dict = Depends(get_session_data),
):
    from src.services.session_manager.manager import save_session
    session_data['use_score_slope'] = body.enabled
    save_session(config.SESSION_FOLDER, session_id, session_data)
    return {"success": True, "data": {"use_score_slope": body.enabled}}


@router.put("/scores/slope/params")
async def update_slope_params(
    body: SlopeParams,
    session_id: str = Depends(get_session_id),
    session_data: dict = Depends(get_session_data),
):
    from src.services.session_manager.manager import save_session

    if body.rel_drop_threshold is not None:
        session_data['rel_drop_threshold'] = body.rel_drop_threshold
    if body.abs_score_threshold is not None:
        session_data['abs_score_threshold'] = body.abs_score_threshold

    save_session(config.SESSION_FOLDER, session_id, session_data)
    return {
        "success": True,
        "data": {
            "rel_drop_threshold": session_data.get('rel_drop_threshold'),
            "abs_score_threshold": session_data.get('abs_score_threshold'),
        }
    }
