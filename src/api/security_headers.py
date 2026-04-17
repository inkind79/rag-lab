"""
Security response headers.

Starlette ships none of these by default. This middleware adds the
low-risk ones unconditionally (MIME sniffing, clickjacking, referrer
leakage, permissions like camera/mic) and the higher-risk one — Content
Security Policy — only when opted in via ``CSP_ENABLE=true``.

CSP is opt-in because a strict policy will block the SvelteKit bundle
if ``'self'`` doesn't include your deployment's asset host. Operators
should turn it on after confirming their build works with the default
policy, or override the policy via ``CSP_POLICY``.

Configuration (env):
  - ``CSP_ENABLE``: ``true`` to emit ``Content-Security-Policy``.
  - ``CSP_POLICY``: override the default policy string.
  - ``FRAME_DENY``: defaults to ``true`` (``X-Frame-Options: DENY``).
    Set to ``false`` if you intentionally embed RAG Lab in an iframe.
"""

from __future__ import annotations

import os

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

# Default CSP. Includes the Google Fonts host the SvelteKit layout imports
# from. Operators who drop that @import can remove those sources.
DEFAULT_CSP = (
    "default-src 'self'; "
    "script-src 'self'; "
    "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
    "font-src 'self' https://fonts.gstatic.com data:; "
    "img-src 'self' data: blob:; "
    "connect-src 'self'; "
    "frame-ancestors 'none'; "
    "base-uri 'self'; "
    "form-action 'self';"
)


def _env_bool(name: str, default: bool) -> bool:
    return os.environ.get(name, str(default)).strip().lower() in ("true", "1", "yes", "on")


def _build_headers() -> dict[str, str]:
    """Resolve the header set from current env. Called on each dispatch so
    runtime env changes (tests, config reloads) take effect."""
    headers: dict[str, str] = {
        "X-Content-Type-Options": "nosniff",
        "Referrer-Policy": "strict-origin-when-cross-origin",
        # Locks down browser-level capabilities. Operators using speech-to-text
        # or geolocation features should override this.
        "Permissions-Policy": "camera=(), microphone=(), geolocation=(), interest-cohort=()",
    }

    if _env_bool("FRAME_DENY", True):
        headers["X-Frame-Options"] = "DENY"

    if _env_bool("CSP_ENABLE", False):
        headers["Content-Security-Policy"] = os.environ.get("CSP_POLICY", DEFAULT_CSP)

    return headers


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response: Response = await call_next(request)
        existing = {k.lower() for k in response.headers.keys()}
        for name, value in _build_headers().items():
            # Don't overwrite headers the handler set explicitly.
            if name.lower() not in existing:
                response.headers[name] = value
        return response
