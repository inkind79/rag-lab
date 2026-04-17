"""Behavior of src.utils.file_lock: serialization + atomic writes."""

import json
import threading
from pathlib import Path

import pytest

from src.utils.file_lock import file_lock, safe_json_read, safe_json_write
from tests.conftest import run_in_threads


@pytest.fixture(autouse=True)
def _isolated(reset_file_lock_registry):
    """Reset the in-process lock registry between tests."""
    yield


def test_safe_json_write_creates_file(tmp_path: Path):
    target = tmp_path / "session.json"
    assert safe_json_write(str(target), {"hello": "world"}) is True
    assert json.loads(target.read_text()) == {"hello": "world"}


def test_safe_json_read_missing_returns_default(tmp_path: Path):
    target = tmp_path / "missing.json"
    assert safe_json_read(str(target)) is None
    assert safe_json_read(str(target), default={"x": 1}) == {"x": 1}


def test_safe_json_round_trip(tmp_path: Path):
    target = tmp_path / "rt.json"
    payload = {"a": 1, "b": [1, 2, 3], "c": {"nested": True}}
    safe_json_write(str(target), payload)
    assert safe_json_read(str(target)) == payload


def test_atomic_write_no_partial_files_visible(tmp_path: Path):
    """The .tmp file used by safe_json_write must not be left behind on success."""
    target = tmp_path / "atomic.json"
    safe_json_write(str(target), {"x": 1})
    siblings = list(tmp_path.glob("*"))
    suffixes = {p.suffix for p in siblings}
    assert ".tmp" not in suffixes
    assert ".lock" not in suffixes


def test_concurrent_writes_dont_corrupt(tmp_path: Path):
    """20 threads each write 10 times; the file is always valid JSON afterward."""
    target = tmp_path / "concurrent.json"
    safe_json_write(str(target), {"counter": 0})

    def worker():
        for _ in range(10):
            data = safe_json_read(str(target)) or {}
            # Deliberate read-modify-write — lost-update is expected;
            # we only assert the file remains parseable.
            data.setdefault("threads", []).append(threading.get_ident())
            safe_json_write(str(target), data)

    errors = run_in_threads(worker, count=20)
    assert not errors, f"workers raised: {errors[:3]}"

    final = safe_json_read(str(target))
    assert isinstance(final, dict)
    assert isinstance(final.get("threads", []), list)
    # No leftover artifacts.
    assert not list(tmp_path.glob("*.tmp"))
    assert not list(tmp_path.glob("*.lock"))


def test_file_lock_releases_on_exception(tmp_path: Path):
    """If the body inside file_lock raises, the lock is still released."""
    target = tmp_path / "raise.json"
    lockfile = tmp_path / "raise.json.lock"

    with pytest.raises(RuntimeError):
        with file_lock(str(target)):
            raise RuntimeError("boom")

    assert not lockfile.exists()

    # And we can re-acquire immediately.
    with file_lock(str(target)):
        pass
