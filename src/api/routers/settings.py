"""
Settings Router — PUT /api/v1/settings, PUT /api/v1/settings/ocr
"""

from typing import Optional, Dict, Any
from fastapi import APIRouter, Depends
from pydantic import BaseModel

from src.api.deps import get_session_id, get_session_data, get_current_user
from src.api import config
from src.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()


class SettingsUpdate(BaseModel):
    generation_model: Optional[str] = None
    indexer_model: Optional[str] = None
    retrieval_count: Optional[int] = None
    distance_metric: Optional[str] = None
    similarity_threshold: Optional[float] = None
    use_score_slope: Optional[bool] = None
    rel_drop_threshold: Optional[float] = None
    abs_score_threshold: Optional[float] = None
    use_ocr: Optional[bool] = None
    ocr_engine: Optional[str] = None
    resized_height: Optional[int] = None
    resized_width: Optional[int] = None
    cloud_history_limit: Optional[int] = None
    local_history_limit: Optional[int] = None
    show_score_viz: Optional[bool] = None
    retrieval_method: Optional[str] = None
    text_embedding_model: Optional[str] = None
    chunk_size: Optional[int] = None
    chunk_overlap: Optional[int] = None
    hybrid_visual_weight: Optional[float] = None
    model_params: Optional[Dict[str, Any]] = None


class OCRSettings(BaseModel):
    use_ocr: Optional[bool] = None
    ocr_engine: Optional[str] = None


@router.put("/settings")
async def update_settings(
    body: SettingsUpdate,
    session_id: str = Depends(get_session_id),
    session_data: dict = Depends(get_session_data),
):
    from src.services.session_manager.manager import save_session, _deep_merge_model_params

    updated = {}
    for field, value in body.model_dump(exclude_none=True).items():
        if field == 'model_params':
            session_data['model_params'] = _deep_merge_model_params(
                session_data.get('model_params', {}), value
            )
            updated['model_params'] = '(merged)'
        else:
            session_data[field] = value
            updated[field] = value

    save_session(config.SESSION_FOLDER, session_id, session_data)
    return {"success": True, "data": {"updated": updated}}


@router.put("/settings/ocr")
async def update_ocr_settings(
    body: OCRSettings,
    session_id: str = Depends(get_session_id),
    session_data: dict = Depends(get_session_data),
):
    from src.services.session_manager.manager import save_session

    if body.use_ocr is not None:
        session_data['use_ocr'] = body.use_ocr
    if body.ocr_engine is not None:
        session_data['ocr_engine'] = body.ocr_engine

    save_session(config.SESSION_FOLDER, session_id, session_data)
    return {
        "success": True,
        "data": {
            "use_ocr": session_data.get('use_ocr'),
            "ocr_engine": session_data.get('ocr_engine'),
        }
    }


class OllamaKeyUpdate(BaseModel):
    api_key: str = ""


@router.put("/settings/ollama-key")
async def update_ollama_key(body: OllamaKeyUpdate, user_id: str = Depends(get_current_user)):
    """Save or clear the Ollama Cloud API key."""
    from src.utils.ollama_client import set_ollama_api_key
    set_ollama_api_key(body.api_key.strip())
    return {"success": True, "data": {"has_key": bool(body.api_key.strip())}}


@router.get("/settings/ollama-key")
async def get_ollama_key_status(user_id: str = Depends(get_current_user)):
    """Check if an Ollama Cloud API key is configured (doesn't return the key itself)."""
    from src.utils.ollama_client import get_ollama_api_key
    return {"success": True, "data": {"has_key": bool(get_ollama_api_key())}}
