"""
Cache-Control middleware for static and dynamic responses.

Default Starlette StaticFiles emits no Cache-Control header, so every
request — even for content-hashed SvelteKit chunks that will literally
never change — does a round-trip to the origin and re-downloads. Worse,
intermediate proxies have to guess at TTL.

This middleware sets explicit policies based on URL path:

- ``/_app/immutable/*`` (SvelteKit's content-hashed bundle): one year,
  ``immutable`` directive. The hash in the path means the URL changes
  whenever the contents change, so the cache can never serve stale data.
- ``/_app/*`` and ``/favicon/*`` (other static resources): one hour.
  Long enough to amortize repeat visits, short enough that operators
  pushing a fix don't have to wait.
- ``/static/images/*`` (per-session document thumbnails): five minutes,
  ``private`` so shared proxies don't fan content across users.
- ``/api/*``, ``/auth/*``: ``no-store``. JSON responses with session
  state must never be cached anywhere.

Existing Cache-Control headers from upstream handlers are not overridden
— only set if missing.
"""

from __future__ import annotations

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

# (prefix, header value). First match wins, so order matters: longer
# prefixes go before shorter ones.
_RULES: list[tuple[str, str]] = [
    ("/_app/immutable/", "public, max-age=31536000, immutable"),
    ("/_app/", "public, max-age=3600"),
    ("/favicon", "public, max-age=3600"),
    ("/static/images/", "private, max-age=300"),
    ("/static/", "public, max-age=3600"),
    ("/api/", "no-store"),
    ("/auth/", "no-store"),
    ("/document/view/", "private, max-age=300"),
]


def _policy_for(path: str) -> str | None:
    for prefix, value in _RULES:
        if path.startswith(prefix):
            return value
    return None


class StaticAssetCacheMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response: Response = await call_next(request)

        # Don't override an explicit Cache-Control set by the handler.
        if "cache-control" in {k.lower() for k in response.headers.keys()}:
            return response

        policy = _policy_for(request.url.path)
        if policy:
            response.headers["Cache-Control"] = policy
        return response
