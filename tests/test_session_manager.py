"""Session manager: create / load / delete + ownership enforcement."""

import os
from pathlib import Path

import pytest

from src.services.session_manager.manager import (
    create_session,
    delete_session_data,
    get_all_sessions,
    load_session,
    save_session,
)


@pytest.fixture
def session_folder(tmp_path: Path) -> str:
    folder = tmp_path / "sessions"
    folder.mkdir()
    return str(folder)


def test_create_session_returns_uuid_and_data(session_folder):
    uid, data = create_session(session_folder, user_id="alice", session_name="Test")
    assert uid is not None
    assert data["user_id"] == "alice"
    assert data["session_name"] == "Test"
    assert os.path.exists(os.path.join(session_folder, f"{uid}.json"))


def test_create_session_requires_user_id(session_folder):
    uid, data = create_session(session_folder, user_id="", session_name="Test")
    assert uid is None
    assert data is None


def test_load_session_round_trip(session_folder):
    uid, data = create_session(session_folder, user_id="alice")
    loaded = load_session(session_folder, uid)
    assert loaded is not None
    assert loaded["user_id"] == "alice"


def test_load_session_missing_returns_none(session_folder):
    assert load_session(session_folder, "00000000-0000-0000-0000-000000000000") is None


def test_save_session_persists_changes(session_folder):
    uid, data = create_session(session_folder, user_id="alice")
    data["chat_history"].append({"role": "user", "content": "hi"})
    assert save_session(session_folder, uid, data) is True
    reloaded = load_session(session_folder, uid)
    assert len(reloaded["chat_history"]) == 1
    assert reloaded["chat_history"][0]["content"] == "hi"


def test_save_session_rejects_data_without_user_id(session_folder):
    uid, _ = create_session(session_folder, user_id="alice")
    bad = {"session_name": "no owner"}
    assert save_session(session_folder, uid, bad) is False


def test_get_all_sessions_filters_by_user(session_folder):
    a1, _ = create_session(session_folder, user_id="alice")
    a2, _ = create_session(session_folder, user_id="alice")
    b1, _ = create_session(session_folder, user_id="bob")

    alice_sessions = get_all_sessions(session_folder, "alice")
    bob_sessions = get_all_sessions(session_folder, "bob")

    alice_ids = {s["id"] for s in alice_sessions}
    bob_ids = {s["id"] for s in bob_sessions}
    assert alice_ids == {a1, a2}
    assert bob_ids == {b1}


def test_get_all_sessions_skips_invalid_files(session_folder, tmp_path):
    create_session(session_folder, user_id="alice")
    # A random non-JSON file shouldn't crash listing.
    (Path(session_folder) / "garbage.json").write_text("not json")
    sessions = get_all_sessions(session_folder, "alice")
    assert len(sessions) == 1


def test_delete_session_data_removes_file(session_folder, tmp_path):
    uid, _ = create_session(session_folder, user_id="alice")
    app_config = {
        "SESSION_FOLDER": session_folder,
        "UPLOAD_FOLDER": str(tmp_path / "uploads"),
        "STATIC_FOLDER": str(tmp_path / "static"),
        "LANCEDB_FOLDER": str(tmp_path / "lancedb"),
    }
    assert delete_session_data(uid, app_config) is True
    assert not os.path.exists(os.path.join(session_folder, f"{uid}.json"))
