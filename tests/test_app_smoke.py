"""Boot-up sanity: the FastAPI app imports and serves /health."""

import pytest


@pytest.mark.integration
def test_app_imports(fastapi_app):
    """The app object exists and is a FastAPI instance with routes registered."""
    from fastapi import FastAPI

    assert isinstance(fastapi_app.app, FastAPI)
    paths = {route.path for route in fastapi_app.app.routes if hasattr(route, "path")}
    # Spot-check that the major routers landed.
    assert "/health" in paths
    assert "/auth/login" in paths


@pytest.mark.integration
def test_health_endpoint(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "service": "rag-lab-api"}


@pytest.mark.integration
def test_auth_check_unauthenticated(client):
    """An anonymous /auth/check returns 401, not 500."""
    r = client.get("/auth/check")
    assert r.status_code == 401
