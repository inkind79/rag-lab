"""
RAG Lab FastAPI Application

Serves the SvelteKit frontend + JSON API.

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
from fastapi import Depends, FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

# Initialize Sentry before app construction so unhandled errors during
# startup (router import, lifespan) get reported. No-op when SENTRY_DSN unset.
from src.api.observability import init_sentry
init_sentry()

from src.api import config
from src.api.csrf import CSRFOriginMiddleware
from src.api.metrics import PrometheusMiddleware, render_latest
from src.api.rate_limit import (
    PathRateLimitMiddleware,
    limiter,
    path_limits,
    rate_limit_exceeded_handler,
)
from src.api.security_headers import SecurityHeadersMiddleware
from src.api.static_cache import StaticAssetCacheMiddleware
from src.utils.logger import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown hooks."""
    os.makedirs(config.SESSION_FOLDER, exist_ok=True)
    os.makedirs(config.UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(config.STATIC_FOLDER, exist_ok=True)
    os.makedirs(config.LANCEDB_FOLDER, exist_ok=True)

    os.makedirs('data', exist_ok=True)
    from src.api.db import create_db_and_tables
    await create_db_and_tables()

    from src.models.retriever_manager import initialize_retrievers
    initialize_retrievers()

    from src.models.prompt_templates import update_system_default_template, update_all_user_templates
    update_system_default_template()
    update_all_user_templates()

    logger.info("FastAPI application started")
    yield

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

# Middleware stack. Starlette runs these in reverse add-order on the response
# path, same-order on the request path. We want: first-request-to-touch is
# CORS (since preflight must always succeed) then CSRF, then the rest.

# CORS — env-driven via CORS_ORIGINS
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "X-Session-UUID", "Authorization"],
)

# CSRF: reject cross-origin state-changing requests that carry the auth cookie.
# Unauthenticated requests (login, register) pass through; CSRF_DISABLE=true disables.
app.add_middleware(CSRFOriginMiddleware)

# Security headers always-on (MIME / clickjacking / referrer / permissions);
# CSP opt-in via CSP_ENABLE=true.
app.add_middleware(SecurityHeadersMiddleware)

# Cache-Control per-path (long for hashed SvelteKit assets, no-store for /api/*).
app.add_middleware(StaticAssetCacheMiddleware)

# Prometheus HTTP request count + latency histogram.
app.add_middleware(PrometheusMiddleware)

# Rate limiting: slowapi reads @limiter.limit decorators; PathRateLimitMiddleware
# guards routes we don't own (fastapi-users /auth/register).
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)
app.add_middleware(PathRateLimitMiddleware, path_limits=path_limits())


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"success": False, "error": exc.detail},
    )

# --- Register routers ---
from src.api.routers import auth, sessions, documents, chat, settings, system, scores, templates, feedback

app.include_router(auth.router, tags=["auth"])
for r in (sessions, documents, chat, settings, system, scores, templates, feedback):
    app.include_router(r.router, prefix="/api/v1", tags=[r.__name__.rsplit('.', 1)[-1]])

# --- Static file mounts ---

app.mount("/static", StaticFiles(directory=config.STATIC_FOLDER), name="static")

from src.api.routers import documents as docs_router  # noqa: already imported


@app.get("/document/view/{session_uuid}/{filename:path}")
async def view_document(session_uuid: str, filename: str):
    """Serve uploaded documents for viewing (PDF, images)."""
    from src.api.deps import _UUID_RE
    from src.utils.path_safety import safe_filename, safe_join, UnsafePathError

    if not _UUID_RE.match(session_uuid):
        raise HTTPException(status_code=400, detail="Invalid session UUID")
    try:
        clean_name = safe_filename(filename)
        file_path = safe_join(os.path.join(config.UPLOAD_FOLDER, session_uuid), clean_name)
    except UnsafePathError:
        raise HTTPException(status_code=400, detail="Invalid filename")

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(str(file_path))


# --- SPA catch-all ---
_svelte_build = os.path.join(os.path.dirname(__file__), "frontend", "build")
if os.path.isdir(_svelte_build):
    _svelte_assets = os.path.join(_svelte_build, "_app")
    if os.path.isdir(_svelte_assets):
        app.mount("/_app", StaticFiles(directory=_svelte_assets), name="svelte-assets")

    _svelte_static = os.path.join(os.path.dirname(__file__), "frontend", "static")
    if os.path.isdir(_svelte_static):
        app.mount("/favicon", StaticFiles(directory=_svelte_static), name="svelte-static")

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        """Catch-all: serve SvelteKit index.html for client-side routing."""
        if full_path.startswith(("api/", "auth/", "static/", "document/", "_app/", "health", "metrics")):
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


from src.api.deps import get_current_user as _get_current_user  # noqa: E402


@app.get("/metrics")
async def prometheus_metrics(user_id: str = Depends(_get_current_user)):
    """Prometheus scrape endpoint. Admin-only — request paths can be sensitive."""
    if user_id != 'admin':
        raise HTTPException(status_code=403, detail="Admin access required")
    body, content_type = render_latest()
    return Response(content=body, media_type=content_type)
