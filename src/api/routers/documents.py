"""
Documents Router

GET    /api/v1/documents/{session_uuid}
POST   /api/v1/documents/upload
POST   /api/v1/documents/reindex
PUT    /api/v1/documents/selection
DELETE /api/v1/documents
"""

import os
from typing import List
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, Request
from pydantic import BaseModel

from src.api.deps import get_session_id, get_session_data, get_rag_models
from src.api.deps import get_current_user
from src.api.rate_limit import UPLOAD_LIMIT, limiter
from src.api import config
from src.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()


class SelectionUpdate(BaseModel):
    selected_docs: List[str] = []


class DeleteDocsRequest(BaseModel):
    filenames: List[str]


@router.get("/documents/{session_uuid}")
async def get_documents(session_uuid: str, user_id: str = Depends(get_current_user)):
    from src.services.session_manager.manager import load_session
    session_data = load_session(config.SESSION_FOLDER, session_uuid)

    if not session_data:
        raise HTTPException(status_code=404, detail="Session not found")
    if session_data.get('user_id') != user_id:
        raise HTTPException(status_code=403, detail="Not authorized")

    return {
        "success": True,
        "data": {
            "indexed_files": session_data.get('indexed_files', []),
            "selected_docs": session_data.get('selected_docs', []),
        }
    }


@router.post("/documents/upload")
@limiter.limit(UPLOAD_LIMIT)
async def upload_documents(
    request: Request,
    files: List[UploadFile] = File(...),
    session_id: str = Depends(get_session_id),
    session_data: dict = Depends(get_session_data),
):
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    from src.services.session_manager.manager import save_session
    from src.services.document_processor.processor import process_uploaded_files, build_document_metadata, index_documents

    upload_folder = os.path.join(config.UPLOAD_FOLDER, session_id)
    os.makedirs(upload_folder, exist_ok=True)

    # Save uploaded files to disk first (FastAPI UploadFile → disk)
    saved_werkzeug_files = []
    for upload_file in files:
        if not upload_file.filename:
            continue
        file_path = os.path.join(upload_folder, upload_file.filename)
        content = await upload_file.read()
        with open(file_path, 'wb') as f:
            f.write(content)
        saved_werkzeug_files.append(upload_file.filename)

    if not saved_werkzeug_files:
        raise HTTPException(status_code=400, detail="No valid files uploaded")

    # Use the same processor as Flask — handles page counting, metadata, etc.
    indexed_files = session_data.get('indexed_files', [])
    for fname in saved_werkzeug_files:
        ext = os.path.splitext(fname)[1].lower()
        file_type = 'pdf' if ext == '.pdf' else 'image' if ext in ('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif') else 'unknown'
        # Only add if not already indexed
        if not any(f.get('filename') == fname for f in indexed_files if isinstance(f, dict)):
            indexed_files.append({'filename': fname, 'file_type': file_type})

    session_data['indexed_files'] = indexed_files
    session_data['selected_docs'] = [f['filename'] for f in indexed_files if isinstance(f, dict)]

    all_metadata = build_document_metadata(saved_werkzeug_files, upload_folder, indexed_files)

    rag_models = get_rag_models()
    # Build a config dict matching what index_documents expects
    app_config = {
        'SESSION_FOLDER': config.SESSION_FOLDER,
        'UPLOAD_FOLDER': config.UPLOAD_FOLDER,
        'LANCEDB_FOLDER': config.LANCEDB_FOLDER,
        'RAG_models': rag_models,
    }

    success, message, _ = index_documents(
        session_id, upload_folder, all_metadata,
        session_data, app_config, rag_models,
    )

    save_session(config.SESSION_FOLDER, session_id, session_data)

    if success:
        return {
            "success": True,
            "data": {
                "indexed_files": indexed_files,
                "selected_docs": session_data['selected_docs'],
                "message": message,
            }
        }
    else:
        raise HTTPException(status_code=500, detail=f"Indexing failed: {message}")


@router.post("/documents/reindex")
async def reindex_documents(
    session_id: str = Depends(get_session_id),
    session_data: dict = Depends(get_session_data),
):
    from src.services.document_processor.processor import build_document_metadata, index_documents

    indexed_files = session_data.get('indexed_files', [])
    if not indexed_files:
        raise HTTPException(status_code=400, detail="No documents to reindex")

    upload_folder = os.path.join(config.UPLOAD_FOLDER, session_id)
    filenames = [f.get('filename') for f in indexed_files if isinstance(f, dict) and f.get('filename')]
    all_metadata = build_document_metadata(filenames, upload_folder, indexed_files)

    rag_models = get_rag_models()
    app_config = {
        'SESSION_FOLDER': config.SESSION_FOLDER,
        'UPLOAD_FOLDER': config.UPLOAD_FOLDER,
        'LANCEDB_FOLDER': config.LANCEDB_FOLDER,
        'RAG_models': rag_models,
    }

    success, message, _ = index_documents(
        session_id, upload_folder, all_metadata,
        session_data, app_config, rag_models,
    )

    if success:
        return {"success": True, "data": {"message": message}}
    else:
        raise HTTPException(status_code=500, detail=f"Reindex failed: {message}")


@router.put("/documents/selection")
async def update_document_selection(
    body: SelectionUpdate,
    session_id: str = Depends(get_session_id),
    session_data: dict = Depends(get_session_data),
):
    from src.services.session_manager.manager import save_session

    session_data['selected_docs'] = body.selected_docs
    save_session(config.SESSION_FOLDER, session_id, session_data)
    return {"success": True, "data": {"selected_docs": body.selected_docs}}


@router.delete("/documents")
async def delete_documents(
    body: DeleteDocsRequest,
    session_id: str = Depends(get_session_id),
    session_data: dict = Depends(get_session_data),
):
    from src.services.session_manager.manager import save_session

    upload_folder = os.path.join(config.UPLOAD_FOLDER, session_id)
    deleted = []

    for filename in body.filenames:
        file_path = os.path.join(upload_folder, filename)
        if os.path.exists(file_path):
            os.remove(file_path)
            deleted.append(filename)

    # Remove from indexed_files and selected_docs
    indexed_files = session_data.get('indexed_files', [])
    session_data['indexed_files'] = [f for f in indexed_files if isinstance(f, dict) and f.get('filename') not in body.filenames]
    session_data['selected_docs'] = [d for d in session_data.get('selected_docs', []) if d not in body.filenames]

    save_session(config.SESSION_FOLDER, session_id, session_data)

    return {"success": True, "data": {"deleted": deleted}}
