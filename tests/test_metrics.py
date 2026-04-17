"""Prometheus metrics: middleware records, /metrics endpoint exposes, auth gates work."""

import pytest


def _register(client, username, password="Goodpass1"):
    return client.post(
        "/auth/register",
        json={"username": username, "email": f"{username}@x.com", "password": password},
    )


def _login(client, username, password="Goodpass1"):
    return client.post("/auth/login", json={"username": username, "password": password})


@pytest.mark.integration
def test_metrics_unauthenticated_returns_401(client):
    r = client.get("/metrics")
    assert r.status_code == 401


@pytest.mark.integration
def test_metrics_non_admin_returns_403(client):
    assert _register(client, "alice").status_code == 201
    assert _login(client, "alice").status_code == 200
    r = client.get("/metrics")
    assert r.status_code == 403


@pytest.mark.integration
def test_metrics_admin_returns_text_format(client):
    assert _register(client, "admin").status_code == 201
    assert _login(client, "admin").status_code == 200
    r = client.get("/metrics")
    assert r.status_code == 200
    assert "text/plain" in r.headers.get("content-type", "")
    body = r.text
    assert "raglab_http_requests_total" in body
    assert "raglab_http_request_duration_seconds" in body
    assert "raglab_cache_events_total" in body


@pytest.mark.integration
def test_request_counter_increments(client):
    """Hitting the same route should bump its counter."""
    assert _register(client, "admin").status_code == 201
    assert _login(client, "admin").status_code == 200

    # Exercise /health a few times, then read metrics
    for _ in range(3):
        client.get("/health")
    r = client.get("/metrics")
    body = r.text
    # The counter line for GET /health should appear at least once with value >= 3.
    health_lines = [
        line for line in body.splitlines()
        if line.startswith("raglab_http_requests_total{")
        and 'route="/health"' in line
        and 'method="GET"' in line
    ]
    assert health_lines, "expected at least one health counter line"
    # Pull the numeric value from "metric{labels} value"
    total = sum(float(line.rsplit(" ", 1)[-1]) for line in health_lines)
    assert total >= 3
