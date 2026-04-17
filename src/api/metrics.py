"""
Prometheus metrics surface.

Defines the small set of counters / histograms RAG Lab exports, plus a
Starlette middleware that records HTTP request volume + latency. The
``/metrics`` endpoint (registered in fastapi_app) returns the standard
Prometheus text format and is admin-only — these counters carry low-
sensitivity operational data, but it's still better to gate them than
to leak request paths to anyone who finds the URL.

Default install does nothing surprising: prometheus_client is a small
pure-Python dep, the middleware adds a few microseconds per request,
and counters live in-process (no exporter daemon, no network egress).
"""

from __future__ import annotations

import time

from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

# ── HTTP metrics ─────────────────────────────────────────────────────────────

http_requests_total = Counter(
    "raglab_http_requests_total",
    "Total HTTP requests handled, labeled by method, route, and status class.",
    labelnames=("method", "route", "status_class"),
)

http_request_duration_seconds = Histogram(
    "raglab_http_request_duration_seconds",
    "HTTP request latency in seconds, labeled by method and route.",
    labelnames=("method", "route"),
    # Buckets tuned for an interactive UI: snappy under 1s, attentive to
    # multi-second LLM streaming, capped at the longest reasonable batch.
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 300.0),
)

# ── RAG-specific counters ────────────────────────────────────────────────────

cache_events_total = Counter(
    "raglab_cache_events_total",
    "Cache lookups by store and outcome (hit / miss / put / evict).",
    labelnames=("store", "outcome"),
)

llm_inference_duration_seconds = Histogram(
    "raglab_llm_inference_duration_seconds",
    "End-to-end LLM inference latency in seconds, labeled by provider and model.",
    labelnames=("provider", "model"),
    buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0, 600.0, 1800.0),
)

retrieval_duration_seconds = Histogram(
    "raglab_retrieval_duration_seconds",
    "Document retrieval latency in seconds, labeled by retriever method.",
    labelnames=("method",),
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
)


# ── Helpers ──────────────────────────────────────────────────────────────────


def render_latest() -> tuple[bytes, str]:
    """Render the current metric snapshot in Prometheus text format."""
    return generate_latest(), CONTENT_TYPE_LATEST


def record_cache_event(store: str, outcome: str) -> None:
    """Tiny wrapper so callers don't have to know the label schema."""
    cache_events_total.labels(store=store, outcome=outcome).inc()


# ── Middleware ───────────────────────────────────────────────────────────────


def _route_template(request: Request) -> str:
    """Use the matched route's template (e.g. ``/sessions/{uuid}``) instead of
    the raw URL so we don't blow up cardinality with one label per UUID."""
    route = request.scope.get("route")
    if route is not None and getattr(route, "path", None):
        return route.path
    # Fallback for unmatched routes (404s, OPTIONS preflight before routing).
    return request.url.path


def _status_class(status_code: int) -> str:
    return f"{status_code // 100}xx"


class PrometheusMiddleware(BaseHTTPMiddleware):
    """Record request count + duration for every HTTP request."""

    # Paths we never want to label (would dominate the time series): the
    # metrics endpoint itself plus static asset prefixes.
    _SKIP_PREFIXES = ("/metrics", "/_app", "/static", "/favicon")

    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        if any(path.startswith(p) for p in self._SKIP_PREFIXES):
            return await call_next(request)

        start = time.perf_counter()
        try:
            response: Response = await call_next(request)
            status = response.status_code
        except Exception:
            # Counted as 5xx so a crashed handler shows up in dashboards.
            elapsed = time.perf_counter() - start
            route = _route_template(request)
            http_requests_total.labels(
                method=request.method, route=route, status_class="5xx"
            ).inc()
            http_request_duration_seconds.labels(
                method=request.method, route=route
            ).observe(elapsed)
            raise

        elapsed = time.perf_counter() - start
        route = _route_template(request)
        http_requests_total.labels(
            method=request.method, route=route, status_class=_status_class(status)
        ).inc()
        http_request_duration_seconds.labels(
            method=request.method, route=route
        ).observe(elapsed)
        return response
