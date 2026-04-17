"""
Application configuration for FastAPI server.
All paths are relative to the project root (cwd).
"""

import logging
import os
import secrets
from pathlib import Path

logger = logging.getLogger(__name__)

SESSION_FOLDER = os.environ.get('SESSION_FOLDER', 'sessions')
UPLOAD_FOLDER = os.environ.get('UPLOAD_FOLDER', 'uploaded_documents')
STATIC_FOLDER = os.environ.get('STATIC_FOLDER', 'static')
LANCEDB_FOLDER = os.environ.get('LANCEDB_FOLDER', os.path.join(os.getcwd(), '.lancedb'))
PROMPT_TEMPLATES_DIR = os.environ.get('PROMPT_TEMPLATES_DIR', 'data/prompt_templates')

MAX_UPLOAD_SIZE = 100 * 1024 * 1024  # 100MB

# CORS allowed origins — comma-separated env var, falls back to localhost dev defaults
_default_origins = "http://localhost:5173,http://localhost:8000"
CORS_ORIGINS = [o.strip() for o in os.environ.get('CORS_ORIGINS', _default_origins).split(',') if o.strip()]

# Ollama
OLLAMA_LOCAL_URL = os.environ.get('OLLAMA_BASE_URL', 'http://localhost:11434')
OLLAMA_CLOUD_URL = 'https://ollama.com'
OLLAMA_API_KEY = os.environ.get('OLLAMA_API_KEY', '')

# Production mode flag — when set, JWT_SECRET must be provided via env
APP_ENV = os.environ.get('APP_ENV', 'development').lower()
_IS_PRODUCTION = APP_ENV in ('production', 'prod')


def _load_or_create_jwt_secret() -> str:
    """Resolve the JWT signing secret.

    Resolution order:
      1. ``JWT_SECRET`` env var (always wins).
      2. ``data/.jwt_secret`` file (persists across dev restarts).
      3. Generate a new secret and write it to ``data/.jwt_secret``.

    In production (``APP_ENV=production``) only step 1 is allowed; missing
    env raises so a misconfigured deploy fails fast instead of silently
    invalidating every session on restart.
    """
    env_secret = os.environ.get('JWT_SECRET')
    if env_secret:
        return env_secret

    if _IS_PRODUCTION:
        raise RuntimeError(
            "JWT_SECRET must be set in the environment when APP_ENV=production"
        )

    secret_path = Path('data') / '.jwt_secret'
    if secret_path.exists():
        existing = secret_path.read_text().strip()
        if existing:
            return existing

    secret_path.parent.mkdir(parents=True, exist_ok=True)
    new_secret = secrets.token_hex(32)
    secret_path.write_text(new_secret)
    try:
        os.chmod(secret_path, 0o600)
    except OSError:
        pass  # best-effort on platforms without POSIX perms
    logger.warning(
        "Generated new JWT_SECRET at %s (dev only). Set JWT_SECRET env var for production.",
        secret_path,
    )
    return new_secret


JWT_SECRET = _load_or_create_jwt_secret()
JWT_ALGORITHM = 'HS256'
JWT_EXPIRY_HOURS = 24 * 7  # 7 days
AUTH_COOKIE = 'auth_token'
