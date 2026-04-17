"""CSRF middleware: Origin/Referer check on authenticated state-changing requests."""

from __future__ import annotations

import pytest

from src.api.csrf import (
    UNSAFE_METHODS,
    _allowed_origins,
    _origin_from_referer,
)


# ── Pure helpers ────────────────────────────────────────────────────────────


def test_unsafe_methods_covers_standard_writes():
    assert UNSAFE_METHODS == {"POST", "PUT", "DELETE", "PATCH"}


@pytest.mark.parametrize(
    "referer,expected",
    [
        ("https://app.example.com/dashboard", "https://app.example.com"),
        ("https://app.example.com:8443/page?q=1", "https://app.example.com:8443"),
        ("http://localhost:5173/login", "http://localhost:5173"),
        ("", ""),
        ("not-a-url", ""),
        ("/just/a/path", ""),
    ],
)
def test_origin_from_referer(referer, expected):
    assert _origin_from_referer(referer) == expected


# ── _allowed_origins env resolution ──────────────────────────────────────────


def test_allowed_origins_explicit_csrf_list(monkeypatch):
    monkeypatch.setenv("CSRF_ALLOWED_ORIGINS", "https://a.com,https://b.com")
    monkeypatch.setenv("CORS_ORIGINS", "https://ignored.com")
    assert _allowed_origins() == ["https://a.com", "https://b.com"]


def test_allowed_origins_falls_back_to_cors_origins(monkeypatch):
    monkeypatch.delenv("CSRF_ALLOWED_ORIGINS", raising=False)
    monkeypatch.setenv("CORS_ORIGINS", "https://foo.com")
    assert _allowed_origins() == ["https://foo.com"]


def test_allowed_origins_default_to_localhost(monkeypatch):
    monkeypatch.delenv("CSRF_ALLOWED_ORIGINS", raising=False)
    monkeypatch.delenv("CORS_ORIGINS", raising=False)
    assert _allowed_origins() == ["http://localhost:5173", "http://localhost:8000"]


def test_allowed_origins_strips_and_filters_empty(monkeypatch):
    """Whitespace and empty entries in the comma-separated list are dropped."""
    monkeypatch.setenv("CSRF_ALLOWED_ORIGINS", "  https://a.com , , https://b.com ,")
    assert _allowed_origins() == ["https://a.com", "https://b.com"]


# ── End-to-end via a minimal FastAPI app ─────────────────────────────────────


@pytest.fixture
def csrf_app(monkeypatch):
    """A tiny FastAPI app wired with just CSRFOriginMiddleware for testing."""
    monkeypatch.setenv("CSRF_ALLOWED_ORIGINS", "http://localhost:5173")
    monkeypatch.delenv("CSRF_DISABLE", raising=False)

    from fastapi import FastAPI

    from src.api.csrf import CSRFOriginMiddleware

    app = FastAPI()
    app.add_middleware(CSRFOriginMiddleware)

    @app.get("/read")
    def _read():
        return {"ok": True}

    @app.post("/write")
    def _write():
        return {"ok": True}

    @app.post("/auth/login")
    def _login():
        return {"ok": True}

    return app


def _client(app):
    from fastapi.testclient import TestClient
    return TestClient(app)


def test_get_is_never_blocked(csrf_app):
    """GET requests pass regardless of cookie or Origin."""
    client = _client(csrf_app)
    r = client.get("/read", cookies={"auth_token": "xyz"}, headers={"Origin": "https://evil.com"})
    assert r.status_code == 200


def test_unauthenticated_post_passes(csrf_app):
    """POST with no auth cookie is untouched by CSRF — login must not break."""
    client = _client(csrf_app)
    r = client.post("/write", headers={"Origin": "https://evil.com"})
    assert r.status_code == 200


def test_authenticated_post_with_good_origin_passes(csrf_app):
    client = _client(csrf_app)
    r = client.post(
        "/write",
        cookies={"auth_token": "xyz"},
        headers={"Origin": "http://localhost:5173"},
    )
    assert r.status_code == 200


def test_authenticated_post_with_bad_origin_rejected(csrf_app):
    client = _client(csrf_app)
    r = client.post(
        "/write",
        cookies={"auth_token": "xyz"},
        headers={"Origin": "https://evil.example.com"},
    )
    assert r.status_code == 403
    assert "CSRF" in r.json()["error"]


def test_authenticated_post_falls_back_to_referer(csrf_app):
    """If Origin is absent but Referer matches, the request passes."""
    client = _client(csrf_app)
    r = client.post(
        "/write",
        cookies={"auth_token": "xyz"},
        headers={"Referer": "http://localhost:5173/somewhere"},
    )
    assert r.status_code == 200


def test_authenticated_post_no_origin_or_referer_rejected(csrf_app):
    client = _client(csrf_app)
    r = client.post("/write", cookies={"auth_token": "xyz"})
    assert r.status_code == 403


def test_csrf_disable_flag_turns_middleware_off(monkeypatch):
    from fastapi import FastAPI

    from src.api.csrf import CSRFOriginMiddleware

    monkeypatch.setenv("CSRF_DISABLE", "true")

    app = FastAPI()
    app.add_middleware(CSRFOriginMiddleware)

    @app.post("/write")
    def _w():
        return {"ok": True}

    client = _client(app)
    # Even with a bad Origin and the auth cookie, the check is skipped.
    r = client.post(
        "/write",
        cookies={"auth_token": "xyz"},
        headers={"Origin": "https://evil.com"},
    )
    assert r.status_code == 200


def test_login_post_passes_without_cookie(csrf_app):
    """Login itself must work without a cookie — otherwise nobody can authenticate."""
    client = _client(csrf_app)
    r = client.post("/auth/login", headers={"Origin": "http://localhost:5173"})
    assert r.status_code == 200
