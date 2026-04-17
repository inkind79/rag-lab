"""
CSRF protection via Origin/Referer header validation.

The app authenticates with an HttpOnly JWT cookie + SameSite=lax. That
blocks the most common CSRF vectors in modern browsers, but:

- SameSite=lax still permits top-level GET navigations and (depending on
  browser version) some cross-site form POSTs.
- Older browsers or misconfigured proxies may weaken the guarantee.
- Defense-in-depth is cheap.

This middleware rejects *unsafe* requests (POST/PUT/DELETE/PATCH) that
carry the auth cookie when the ``Origin`` (or, as a fallback, ``Referer``)
header doesn't match an allowed origin. Unauthenticated requests —
including registration and login itself — are untouched, so a first-time
user can still get past the gate.

Allowed origins:
  - Pulled from the ``CSRF_ALLOWED_ORIGINS`` env var (comma-separated) if set.
  - Otherwise fall back to ``CORS_ORIGINS`` — same environments you already
    trust to talk to this API.

Set ``CSRF_DISABLE=true`` to turn the check off entirely (tests only).
"""

from __future__ import annotations

import logging
import os
from urllib.parse import urlparse

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

logger = logging.getLogger(__name__)

# HTTP methods considered state-changing.
UNSAFE_METHODS = {"POST", "PUT", "DELETE", "PATCH"}

# The auth backend cookie name. Must match src/api/config.AUTH_COOKIE.
_AUTH_COOKIE = "auth_token"


def _env_list(name: str) -> list[str]:
    raw = os.environ.get(name, "")
    return [v.strip() for v in raw.split(",") if v.strip()]


def _allowed_origins() -> list[str]:
    explicit = _env_list("CSRF_ALLOWED_ORIGINS")
    if explicit:
        return explicit
    # Fall back to the CORS allowlist — same origin-trust surface.
    return _env_list("CORS_ORIGINS") or [
        "http://localhost:5173",
        "http://localhost:8000",
    ]


def _origin_from_referer(referer: str) -> str:
    """Extract ``scheme://host[:port]`` from a Referer URL."""
    try:
        parsed = urlparse(referer)
        if not parsed.scheme or not parsed.netloc:
            return ""
        return f"{parsed.scheme}://{parsed.netloc}"
    except Exception:
        return ""


def _reject() -> JSONResponse:
    return JSONResponse(
        status_code=403,
        content={
            "success": False,
            "error": "CSRF check failed: Origin/Referer does not match an allowed host",
        },
    )


class CSRFOriginMiddleware(BaseHTTPMiddleware):
    """Block cross-origin unsafe requests that carry the auth cookie."""

    async def dispatch(self, request: Request, call_next):
        if os.environ.get("CSRF_DISABLE", "").lower() in ("true", "1", "yes"):
            return await call_next(request)

        if request.method not in UNSAFE_METHODS:
            return await call_next(request)

        # Only guard requests that actually have a session to attack. Login,
        # registration, and password-reset flows run without the auth cookie
        # and must be allowed through without Origin gymnastics.
        if _AUTH_COOKIE not in request.cookies:
            return await call_next(request)

        allowed = set(_allowed_origins())
        origin = request.headers.get("origin", "")
        referer_origin = _origin_from_referer(request.headers.get("referer", ""))

        # Prefer Origin when present (browsers send it for most cross-origin
        # fetches); fall back to the Referer's origin component.
        candidate = origin or referer_origin
        if not candidate:
            # Neither header present on an authenticated write request —
            # most likely a CSRF attempt from a stripped-origin channel, or
            # a non-browser client that should be using its own auth flow.
            logger.warning(
                "CSRF reject: authenticated %s %s with no Origin or Referer",
                request.method, request.url.path,
            )
            return _reject()

        if candidate not in allowed:
            logger.warning(
                "CSRF reject: authenticated %s %s from origin %r (allowed: %s)",
                request.method, request.url.path, candidate, sorted(allowed),
            )
            return _reject()

        return await call_next(request)
