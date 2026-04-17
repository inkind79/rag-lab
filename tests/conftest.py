"""
Shared pytest fixtures.

Every test runs against an isolated working directory so concurrent test
sessions don't fight over ``data/users.db``, ``sessions/``, ``logs/``, etc.
The FastAPI app's lifespan loads the ColPali / OCR / template stack which
takes seconds and pulls in CUDA — we monkey-patch the heavy initializers
so integration tests can spin up the app in well under a second.
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path

import pytest

# Ensure repo root is importable regardless of where pytest is invoked.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


@pytest.fixture(autouse=True)
def isolated_workdir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Run every test from a fresh CWD with stable env defaults.

    Tests should never write into the developer's real ``data/``, ``sessions/``,
    or ``logs/`` directories — anything that uses relative paths now lands in
    a per-test tmp_path.
    """
    monkeypatch.chdir(tmp_path)
    # Predictable defaults so config import doesn't pick up the developer's env.
    monkeypatch.setenv("APP_ENV", "development")
    monkeypatch.delenv("JWT_SECRET", raising=False)
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    monkeypatch.delenv("CORS_ORIGINS", raising=False)
    return tmp_path


@pytest.fixture
def fresh_modules(monkeypatch: pytest.MonkeyPatch):
    """Drop and re-import a list of modules so they pick up fresh env vars.

    Many of our modules read env at import time (config, rate_limit, etc.).
    Tests that mutate env need a way to force a re-import.
    """
    def _reset(prefixes: tuple[str, ...] = ("src.", "fastapi_app")) -> None:
        for name in list(sys.modules):
            if name == "fastapi_app" or any(name.startswith(p) for p in prefixes):
                del sys.modules[name]

    return _reset


# ── Integration helpers (FastAPI TestClient) ─────────────────────────────────


@pytest.fixture
def fastapi_app(monkeypatch: pytest.MonkeyPatch, fresh_modules):
    """Import a fresh ``fastapi_app.app`` with model loading mocked out.

    We replace ``initialize_retrievers`` and the prompt-template seeding
    with no-ops so the test app comes up in milliseconds without GPUs.
    Tests that genuinely need the retrievers should mark themselves as
    ``slow`` and skip on CI.
    """
    fresh_modules()

    # Stub modules the lifespan touches so we don't pay the load cost.
    import types

    def _install_stub(module_name: str, attrs: dict) -> None:
        mod = types.ModuleType(module_name)
        for k, v in attrs.items():
            setattr(mod, k, v)
        monkeypatch.setitem(sys.modules, module_name, mod)

    # No-op retriever init.
    _install_stub(
        "src.models.retriever_manager",
        {"initialize_retrievers": lambda: None},
    )
    # No-op template seeding.
    _install_stub(
        "src.models.prompt_templates",
        {
            "update_system_default_template": lambda: None,
            "update_all_user_templates": lambda: None,
        },
    )
    # Memory manager shutdown hook is fine but pulls in torch/CUDA. Stub.
    _install_stub(
        "src.models.memory.memory_manager",
        {"memory_manager": types.SimpleNamespace(aggressive_cleanup=lambda: None)},
    )
    # Platform shim does nothing useful in tests.
    _install_stub(
        "src.utils.platform_shim",
        {"apply_platform_shim": lambda: None},
    )

    # fastapi_app mounts a StaticFiles route at import time and will refuse
    # to start if the directory is missing. Pre-create the dirs the config
    # module advertises, in the test's tmp_path CWD.
    for d in ("static", "sessions", "uploaded_documents", ".lancedb", "data", "logs"):
        Path(d).mkdir(parents=True, exist_ok=True)

    import fastapi_app as fa
    return fa


@pytest.fixture
def client(fastapi_app):
    """A TestClient that runs the lifespan (so DB tables exist)."""
    from fastapi.testclient import TestClient

    with TestClient(fastapi_app.app) as c:
        yield c


# ── File-lock test helper ────────────────────────────────────────────────────


@pytest.fixture
def reset_file_lock_registry():
    """Wipe the in-process file_lock registry between tests.

    file_lock keeps a module-level ``_lock_registry`` of threading.Lock
    instances per filepath. Tests that exercise locking on the same path
    in sequence would otherwise share state.
    """
    yield
    try:
        from src.utils import file_lock as fl
        with fl._registry_lock:
            fl._lock_registry.clear()
    except Exception:
        pass


# ── Threading helper ─────────────────────────────────────────────────────────


def run_in_threads(target, count: int, *args, **kwargs) -> list[Exception]:
    """Run ``target(*args, **kwargs)`` in ``count`` threads and collect exceptions."""
    errors: list[Exception] = []
    err_lock = threading.Lock()

    def _wrapper():
        try:
            target(*args, **kwargs)
        except Exception as e:
            with err_lock:
                errors.append(e)

    threads = [threading.Thread(target=_wrapper) for _ in range(count)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return errors
