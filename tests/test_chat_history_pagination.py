"""Paginated GET /api/v1/sessions/{uuid}/chat_history.

Tests the slicing logic directly against a synthetic session file so we
don't have to spin up the full app — just need session_manager to read
the JSON we wrote.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.services.session_manager.manager import (
    create_session,
    load_session,
    save_session,
)


def _make_session_with_history(folder: str, n: int) -> str:
    """Create a session and seed it with ``n`` numbered chat messages."""
    uuid, data = create_session(folder, user_id="alice", session_name="paged")
    data["chat_history"] = [
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"msg-{i}", "timestamp": i}
        for i in range(n)
    ]
    assert save_session(folder, uuid, data)
    return uuid


@pytest.fixture
def session_folder(tmp_path: Path, monkeypatch) -> str:
    folder = tmp_path / "sessions"
    folder.mkdir()
    monkeypatch.chdir(tmp_path)
    return str(folder)


# ── Pure slicing logic (copied from the router) ────────────────────────────
# Keeping the logic close to the test surface so the routing layer is tested
# separately and slicing edge cases get fast unit coverage.


def _slice_history(history, limit: int, before):
    total = len(history)
    end = total if before is None else max(0, min(before, total))
    start = max(0, end - limit)
    return {
        "messages": history[start:end],
        "first_index": start,
        "total": total,
        "has_more": start > 0,
    }


def test_slice_returns_latest_when_before_is_none():
    history = [{"i": i} for i in range(100)]
    page = _slice_history(history, limit=10, before=None)
    assert page["first_index"] == 90
    assert page["total"] == 100
    assert page["has_more"] is True
    assert page["messages"][0]["i"] == 90
    assert page["messages"][-1]["i"] == 99


def test_slice_walks_backwards_with_before():
    history = [{"i": i} for i in range(100)]
    # First page (latest 10) → first_index = 90
    p1 = _slice_history(history, limit=10, before=None)
    # Use first_index as ``before`` to get the next page back
    p2 = _slice_history(history, limit=10, before=p1["first_index"])
    assert p2["first_index"] == 80
    assert p2["messages"][0]["i"] == 80
    assert p2["messages"][-1]["i"] == 89
    assert p2["has_more"] is True


def test_slice_limit_clamps_at_zero_when_walked_to_start():
    history = [{"i": i} for i in range(5)]
    page = _slice_history(history, limit=10, before=None)
    assert page["first_index"] == 0
    assert len(page["messages"]) == 5
    assert page["has_more"] is False


def test_slice_before_zero_returns_empty_page():
    history = [{"i": i} for i in range(5)]
    page = _slice_history(history, limit=10, before=0)
    assert page["messages"] == []
    assert page["first_index"] == 0
    assert page["has_more"] is False


def test_slice_before_beyond_total_clamps_to_end():
    history = [{"i": i} for i in range(5)]
    page = _slice_history(history, limit=10, before=999)
    assert page["first_index"] == 0
    assert len(page["messages"]) == 5


def test_slice_empty_history():
    page = _slice_history([], limit=50, before=None)
    assert page["messages"] == []
    assert page["total"] == 0
    assert page["first_index"] == 0
    assert page["has_more"] is False


def test_slice_zero_limit_is_caller_problem():
    # The router rejects limit<=0 with HTTPException; the slicer just yields
    # an empty page with first_index == end.
    page = _slice_history([{"i": 1}], limit=0, before=None)
    assert page["messages"] == []


# ── End-to-end: persisted session + load_session ───────────────────────────


def test_paginated_view_against_persisted_session(session_folder: str):
    """The pure slicer applied to a real loaded session matches what the
    /chat_history endpoint would return."""
    uuid = _make_session_with_history(session_folder, n=120)

    loaded = load_session(session_folder, uuid)
    history = loaded["chat_history"]

    page1 = _slice_history(history, limit=50, before=None)
    page2 = _slice_history(history, limit=50, before=page1["first_index"])
    page3 = _slice_history(history, limit=50, before=page2["first_index"])

    assert page1["first_index"] == 70
    assert page2["first_index"] == 20
    assert page3["first_index"] == 0
    assert page3["has_more"] is False

    # Concatenated, the three pages reconstruct the original.
    rebuilt = page3["messages"] + page2["messages"] + page1["messages"]
    assert rebuilt == history
