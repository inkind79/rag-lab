"""
Templates Router — Prompt template CRUD

GET    /api/v1/templates
POST   /api/v1/templates
GET    /api/v1/templates/{id}
PUT    /api/v1/templates/{id}
DELETE /api/v1/templates/{id}
POST   /api/v1/templates/{id}/set-default
POST   /api/v1/sessions/{uuid}/template
"""

from typing import Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from src.api.deps import get_current_user
from src.api.deps import get_session_id, get_session_data
from src.api import config
from src.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()


class TemplateData(BaseModel):
    name: Optional[str] = None
    template_type: Optional[str] = None
    description: Optional[str] = None
    system_prompt: Optional[str] = None
    query_prefix: Optional[str] = None
    query_suffix: Optional[str] = None


class SelectTemplateRequest(BaseModel):
    template_id: str


@router.get("/templates")
async def list_templates(user_id: str = Depends(get_current_user)):
    from src.models.prompt_templates import get_user_templates
    templates = get_user_templates(user_id)
    return {"success": True, "templates": templates}


@router.get("/templates/{template_id}")
async def get_template(template_id: str, user_id: str = Depends(get_current_user)):
    from src.models.prompt_templates import get_template_by_id
    template = get_template_by_id(user_id, template_id)
    if not template:
        raise HTTPException(status_code=404, detail="Template not found")
    return {"success": True, "template": template}


@router.post("/templates", status_code=201)
async def create_template(body: TemplateData, user_id: str = Depends(get_current_user)):
    from src.models.prompt_templates import create_template as _create
    template_data = body.model_dump(exclude_none=True)
    template_id = _create(user_id, template_data)
    if not template_id:
        raise HTTPException(status_code=500, detail="Failed to create template")
    return {"success": True, "template_id": template_id}


@router.put("/templates/{template_id}")
async def update_template(template_id: str, body: TemplateData, user_id: str = Depends(get_current_user)):
    from src.models.prompt_templates import update_template as _update
    template_data = body.model_dump(exclude_none=True)
    success = _update(user_id, template_id, template_data)
    if not success:
        raise HTTPException(status_code=404, detail="Template not found")
    return {"success": True}


@router.delete("/templates/{template_id}")
async def delete_template(template_id: str, user_id: str = Depends(get_current_user)):
    from src.models.prompt_templates import delete_template as _delete
    success = _delete(user_id, template_id)
    if not success:
        raise HTTPException(status_code=404, detail="Template not found")
    return {"success": True}


@router.post("/templates/{template_id}/set-default")
async def set_default_template(template_id: str, user_id: str = Depends(get_current_user)):
    from src.models.prompt_templates import set_default_template as _set_default
    success = _set_default(user_id, template_id)
    if not success:
        raise HTTPException(status_code=404, detail="Template not found")
    return {"success": True}


@router.post("/sessions/{session_uuid}/template")
async def select_template_for_session(
    session_uuid: str,
    body: SelectTemplateRequest,
    user_id: str = Depends(get_current_user),
):
    from src.services.session_manager.manager import load_session, save_session
    session_data = load_session(config.SESSION_FOLDER, session_uuid)

    if not session_data:
        raise HTTPException(status_code=404, detail="Session not found")
    if session_data.get('user_id') != user_id:
        raise HTTPException(status_code=403, detail="Not authorized")

    session_data['selected_template_id'] = body.template_id

    # Load the template to get its optimized_query if any
    from src.models.prompt_templates import get_template_by_id
    template = get_template_by_id(user_id, body.template_id)
    if template:
        session_data['optimized_query'] = template.get('optimized_query', '')

    save_session(config.SESSION_FOLDER, session_uuid, session_data)
    return {"success": True}
