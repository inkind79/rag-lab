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
    """Update the Ollama API key at runtime."""
    config.OLLAMA_API_KEY = key
    os.environ['OLLAMA_API_KEY'] = key


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def get_request_timeout() -> Tuple[float, float]:
    """Return ``(connect_timeout, read_timeout)`` for requests calls.

    Always a 2-tuple so callers pass it straight to ``requests.{get,post}``.
    """
    return (
        _env_float('OLLAMA_CONNECT_TIMEOUT', 10.0),
        _env_float('OLLAMA_READ_TIMEOUT', 1800.0),
    )


def get_ollama_client() -> ollama.Client:
    """Construct an ``ollama.Client`` with the configured read timeout.

    The ollama SDK takes a single ``timeout`` value (passed through to httpx
    as the read/write timeout). Use the longer half of our tuple.
    """
    _, read_timeout = get_request_timeout()
    return ollama.Client(timeout=read_timeout)
