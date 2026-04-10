"""
Application configuration for FastAPI server.
All paths are relative to the project root (cwd).
"""

import os
import secrets

SESSION_FOLDER = os.environ.get('SESSION_FOLDER', 'sessions')
UPLOAD_FOLDER = os.environ.get('UPLOAD_FOLDER', 'uploaded_documents')
STATIC_FOLDER = os.environ.get('STATIC_FOLDER', 'static')
LANCEDB_FOLDER = os.environ.get('LANCEDB_FOLDER', os.path.join(os.getcwd(), '.lancedb'))
PROMPT_TEMPLATES_DIR = os.environ.get('PROMPT_TEMPLATES_DIR', 'data/prompt_templates')

MAX_UPLOAD_SIZE = 100 * 1024 * 1024  # 100MB

# Ollama
OLLAMA_LOCAL_URL = os.environ.get('OLLAMA_BASE_URL', 'http://localhost:11434')
OLLAMA_CLOUD_URL = 'https://ollama.com'
OLLAMA_API_KEY = os.environ.get('OLLAMA_API_KEY', '')

# JWT — generate a random secret if not configured (safe for dev, set JWT_SECRET in .env for prod)
JWT_SECRET = os.environ.get('JWT_SECRET', secrets.token_hex(32))
JWT_ALGORITHM = 'HS256'
JWT_EXPIRY_HOURS = 24 * 7  # 7 days
AUTH_COOKIE = 'auth_token'
