"""
Ollama client helpers — resolves base URL, auth headers, and request timeouts
for local vs cloud models.

Cloud models (name contains ':cloud') route to https://ollama.com with Bearer auth.
Local models route to the configured OLLAMA_BASE_URL (default localhost:11434).

Timeouts are env-overridable via ``OLLAMA_CONNECT_TIMEOUT`` (default 10s) and
``OLLAMA_READ_TIMEOUT`` (default 1800s). A short connect timeout surfaces a
dead Ollama fast; the long read timeout accommodates large-model inference.
"""

import os
from typing import Tuple

import ollama

from src.api import config


def get_ollama_url(model_name: str) -> str:
    """Return the Ollama API base URL for a given model."""
    if ':cloud' in model_name:
        return config.OLLAMA_CLOUD_URL
    return config.OLLAMA_LOCAL_URL


def get_ollama_headers(model_name: str) -> dict:
    """Return request headers, including Bearer auth for cloud models."""
    headers = {}
    if ':cloud' in model_name:
        api_key = config.OLLAMA_API_KEY
        if api_key:
            headers['Authorization'] = f'Bearer {api_key}'
    return headers


def get_ollama_api_key() -> str:
    """Return the current Ollama API key (may be empty)."""
    return config.OLLAMA_API_KEY


def set_ollama_api_key(key: str):
    """Update the Ollama API key at runtime, keeping log redaction in sync."""
    from src.utils.log_redaction import register_secret, unregister_secret

    old = config.OLLAMA_API_KEY
    if old:
        unregister_secret(old)
    config.OLLAMA_API_KEY = key
    os.environ['OLLAMA_API_KEY'] = key
    if key:
        register_secret(key)
