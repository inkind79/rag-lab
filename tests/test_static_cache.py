"""Cache-Control middleware: per-path policies + respects upstream override."""

import pytest

from src.api.static_cache import _policy_for


# ── Pure policy resolver ─────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "path,expected",
    [
        # SvelteKit hashed bundle: 1 year, immutable.
        ("/_app/immutable/chunks/abc.js", "public, max-age=31536000, immutable"),
        ("/_app/immutable/entry/start.js", "public, max-age=31536000, immutable"),
        # Other _app assets (not under immutable/): 1 hour.
        ("/_app/foo.js", "public, max-age=3600"),
        # Per-session document images: 5 min, private.
        ("/static/images/sess-uuid/page-1.png", "private, max-age=300"),
        # Other /static: 1 hour, public.
        ("/static/icon.svg", "public, max-age=3600"),
        # Document view (auth-gated): 5 min, private.
        ("/document/view/abc/file.pdf", "private, max-age=300"),
        # API + auth: never cache.
        ("/api/v1/sessions", "no-store"),
        ("/api/v1/chat/stream", "no-store"),
        ("/auth/login", "no-store"),
        ("/auth/register", "no-store"),
        # Favicon: 1 hour.
        ("/favicon/favicon.png", "public, max-age=3600"),
        # No policy for unrelated paths.
        ("/health", None),
        ("/", None),
        ("/random", None),
    ],
)
def test_policy_for_path(path, expected):
    assert _policy_for(path) == expected


def test_immutable_prefix_wins_over_app_prefix():
    """Order matters: /_app/immutable/ must match before /_app/."""
    p = _policy_for("/_app/immutable/x.js")
    assert "immutable" in p
    assert "max-age=31536000" in p


# Note: integration tests against the live FastAPI app live in the test-suite
# PR (chore/add-pytest-suite). Once that lands and provides the `client`
# fixture, end-to-end tests for "API responses get no-store" and "upstream
# Cache-Control isn't overridden" can be added here.
