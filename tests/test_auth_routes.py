"""Auth router: registration, login, logout, /auth/check.

Tests target current behavior on main. As security hardening PRs land
(generic 401, password complexity, rate limiting), tests in those PRs
will tighten the assertions here.
"""

import pytest

REG_PATH = "/auth/register"
LOGIN_PATH = "/auth/login"


def _register(client, username="alice", email="alice@example.com", password="Goodpass1"):
    return client.post(
        REG_PATH,
        json={"username": username, "email": email, "password": password},
    )


@pytest.mark.integration
def test_registration_succeeds(client):
    r = _register(client)
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["username"] == "alice"


@pytest.mark.integration
def test_duplicate_registration_rejected(client):
    assert _register(client).status_code == 201
    r = _register(client)
    # FastAPI-Users returns 400 on dup email.
    assert r.status_code == 400


@pytest.mark.integration
def test_login_with_correct_credentials_sets_cookie(client):
    assert _register(client).status_code == 201
    r = client.post(LOGIN_PATH, json={"username": "alice", "password": "Goodpass1"})
    assert r.status_code == 200
    assert "auth_token" in r.cookies


@pytest.mark.integration
def test_login_with_wrong_password_returns_401(client):
    assert _register(client).status_code == 201
    r = client.post(LOGIN_PATH, json={"username": "alice", "password": "wrong-pass1"})
    assert r.status_code == 401


@pytest.mark.integration
def test_login_with_unknown_user_returns_401(client):
    r = client.post(LOGIN_PATH, json={"username": "ghost", "password": "anything1"})
    assert r.status_code == 401


@pytest.mark.integration
def test_check_after_login_returns_user(client):
    assert _register(client).status_code == 201
    login = client.post(LOGIN_PATH, json={"username": "alice", "password": "Goodpass1"})
    assert login.status_code == 200
    r = client.get("/auth/check")
    assert r.status_code == 200
    assert r.json()["data"]["user_id"] == "alice"


@pytest.mark.integration
def test_logout_clears_cookie(client):
    assert _register(client).status_code == 201
    client.post(LOGIN_PATH, json={"username": "alice", "password": "Goodpass1"})
    r = client.post("/auth/logout")
    assert r.status_code == 200
    # Subsequent /auth/check should be 401.
    r2 = client.get("/auth/check")
    assert r2.status_code == 401
