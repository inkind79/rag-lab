"""
RAG Lab FastAPI Application

Serves the SvelteKit frontend + JSON API.
Replaces Flask for the open-source release.

Usage:
    uvicorn fastapi_app:app --host 127.0.0.1 --port 8000 --loop uvloop
"""

import os
import sys

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Platform compatibility shim (Windows/Linux differences)
from src.utils.platform_shim import apply_platform_shim
apply_platform_shim()

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# Initialize Sentry before app construction so unhandled errors during
# startup (router import, lifespan) get reported. No-op when SENTRY_DSN unset.
from src.api.observability import init_sentry
init_sentry()

from src.api import config
from src.utils.logger import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown hooks."""
    # Ensure directories exist
    os.makedirs(config.SESSION_FOLDER, exist_ok=True)
    os.makedirs(config.UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(config.STATIC_FOLDER, exist_ok=True)
    os.makedirs(config.LANCEDB_FOLDER, exist_ok=True)

    # Initialize user database
    os.makedirs('data', exist_ok=True)
    from src.api.db import create_db_and_tables
    await create_db_and_tables()

    # Initialize retrievers
    from src.models.retriever_manager import initialize_retrievers
    initialize_retrievers()

    # Initialize default prompt templates
    from src.models.prompt_templates import update_system_default_template, update_all_user_templates
    update_system_default_template()
    update_all_user_templates()

    logger.info("FastAPI application started")
    yield

    # Shutdown cleanup
    try:
        from src.models.memory.memory_manager import memory_manager
        memory_manager.aggressive_cleanup()
    except Exception as e:
        logger.error(f"Shutdown cleanup error: {e}")
    logger.info("FastAPI application stopped")


app = FastAPI(
    title="RAG Lab API",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — SvelteKit dev server + production
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # SvelteKit dev
        "http://localhost:8000",  # Self
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "X-Session-UUID", "Authorization"],
)

# Standardized error responses
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"success": False, "error": exc.detail},
    )

# --- Register routers ---
from src.api.routers import auth, sessions, documents, chat, settings, system, scores, templates, feedback

# Auth routes (no /api/v1 prefix)
app.include_router(auth.router, tags=["auth"])

# API v1 routes
for r in (sessions, documents, chat, settings, system, scores, templates, feedback):
    app.include_router(r.router, prefix="/api/v1", tags=[r.__name__.rsplit('.', 1)[-1]])

# --- Static file mounts ---

# Serve uploaded document images
app.mount("/static", StaticFiles(directory=config.STATIC_FOLDER), name="static")

# Serve document files for viewing
from src.api.routers import documents as docs_router  # noqa: already imported
# Document viewing is handled by the /document/view endpoint below

@app.get("/document/view/{session_uuid}/{filename:path}")
async def view_document(session_uuid: str, filename: str):
    """Serve uploaded documents for viewing (PDF, images)."""
    file_path = os.path.join(config.UPLOAD_FOLDER, session_uuid, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path)

# --- SPA catch-all ---
# Serve SvelteKit build output. Must be LAST (after all API routes).
_svelte_build = os.path.join(os.path.dirname(__file__), "frontend", "build")
if os.path.isdir(_svelte_build):
    # Mount static assets from SvelteKit build
    _svelte_assets = os.path.join(_svelte_build, "_app")
    if os.path.isdir(_svelte_assets):
        app.mount("/_app", StaticFiles(directory=_svelte_assets), name="svelte-assets")

    # Serve favicon and other root-level static files
    _svelte_static = os.path.join(os.path.dirname(__file__), "frontend", "static")
    if os.path.isdir(_svelte_static):
        app.mount("/favicon", StaticFiles(directory=_svelte_static), name="svelte-static")

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        """Catch-all: serve SvelteKit index.html for client-side routing."""
        # Don't intercept API or static paths
        if full_path.startswith(("api/", "auth/", "static/", "document/", "_app/", "health")):
            raise HTTPException(status_code=404)
        index = os.path.join(_svelte_build, "index.html")
        if os.path.exists(index):
            return FileResponse(index)
        raise HTTPException(status_code=404, detail="Frontend not built")
else:
    logger.warning(f"SvelteKit build not found at {_svelte_build} — run 'cd frontend && npm run build'")


@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "rag-lab-api"}
