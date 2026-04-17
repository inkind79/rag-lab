"""
Rate limiting for auth and upload endpoints.

Uses slowapi (starlette/FastAPI wrapper around the `limits` library).
Keys are per-IP via ``X-Forwarded-For`` when present, else socket address.

Limits are conservative defaults suited to a local-first app. Override by
setting env vars (``RATE_LIMIT_LOGIN``, ``RATE_LIMIT_REGISTER``,
``RATE_LIMIT_UPLOAD``) — values use the ``limits`` string syntax,
e.g. ``"5/minute"`` or ``"100/hour;20/minute"``.
"""

import os
from typing import Iterable

from limits import parse_many
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

LOGIN_LIMIT = os.environ.get('RATE_LIMIT_LOGIN', '5/minute')
REGISTER_LIMIT = os.environ.get('RATE_LIMIT_REGISTER', '3/minute;20/hour')
UPLOAD_LIMIT = os.environ.get('RATE_LIMIT_UPLOAD', '10/minute')


def _client_key(request: Request) -> str:
    """Prefer X-Forwarded-For for reverse-proxy deploys, else peer address."""
    fwd = request.headers.get('x-forwarded-for')
    if fwd:
        return fwd.split(',')[0].strip()
    return get_remote_address(request)


limiter = Limiter(key_func=_client_key, default_limits=[])


def rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    return JSONResponse(
        status_code=429,
        content={"success": False, "error": "Too many requests. Please slow down and try again."},
        headers={"Retry-After": "60"},
    )


class PathRateLimitMiddleware(BaseHTTPMiddleware):
    """Apply per-path rate limits to routes we can't decorate.

    Used for endpoints mounted from third-party routers (e.g.
    fastapi-users registration) where ``@limiter.limit`` isn't available.
    """

    def __init__(self, app, path_limits: dict[str, str]):
        super().__init__(app)
        self._limits = {path: parse_many(spec) for path, spec in path_limits.items()}

    async def dispatch(self, request: Request, call_next):
        specs = self._limits.get(request.url.path)
        if specs is not None:
            key = _client_key(request)
            for item in specs:
                # limiter.limiter is the underlying `limits` storage+strategy.
                if not limiter.limiter.hit(item, key, request.url.path):
                    return JSONResponse(
                        status_code=429,
                        content={
                            "success": False,
                            "error": "Too many requests. Please slow down and try again.",
                        },
                        headers={"Retry-After": "60"},
                    )
        return await call_next(request)


def path_limits() -> dict[str, str]:
    """Paths that need rate limits but can't be decorated directly."""
    return {"/auth/register": REGISTER_LIMIT}
