"""Integration tests for the RAGRetriever ↔ reranker / HyDE wiring.

These target the tiny helper surface added by the wiring PR:
_make_llm_callable, _maybe_apply_hyde_expansion, _maybe_rerank_results.
They don't exercise the live retrieval pipeline (that needs ColPali +
LanceDB), only the integration hooks — which is the code we actually
changed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.models.rag_retriever import RAGRetriever


@pytest.fixture
def retriever(monkeypatch, tmp_path: Path) -> RAGRetriever:
    """A RAGRetriever with _make_llm_callable overridden so tests don't hit Ollama."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "sessions").mkdir()
    return RAGRetriever()


def _write_session(tmp_path: Path, uuid: str, **overrides) -> str:
    session = {"user_id": "alice", **overrides}
    (tmp_path / "sessions" / f"{uuid}.json").write_text(json.dumps(session))
    return uuid


# ── Rerank helper ────────────────────────────────────────────────────────────


def test_rerank_off_is_passthrough(retriever):
    docs = [{"text": "a", "score": 0.9}, {"text": "b", "score": 0.4}]
    out = retriever._maybe_rerank_results("q", docs, {"use_llm_rerank": False})
    assert out == docs


def test_rerank_empty_results_passthrough(retriever):
    assert retriever._maybe_rerank_results("q", [], {"use_llm_rerank": True}) == []


def test_rerank_missing_session_data_passthrough(retriever):
    docs = [{"text": "a", "score": 0.9}]
    assert retriever._maybe_rerank_results("q", docs, None) == docs
    assert retriever._maybe_rerank_results("q", docs, {}) == docs


def test_rerank_reorders_by_llm_score(retriever, monkeypatch):
    # Fake LLM gives passage 2 the top score.
    def fake(prompt):
        return "1: 2\n2: 9\n"
    monkeypatch.setattr(retriever, "_make_llm_callable", lambda model: fake)

    docs = [{"text": "low", "score": 0.9}, {"text": "high", "score": 0.5}]
    out = retriever._maybe_rerank_results(
        "q", docs, {"use_llm_rerank": True, "llm_rerank_model": "test"}
    )
    assert [d["text"] for d in out] == ["high", "low"]


def test_rerank_llm_exception_falls_back(retriever, monkeypatch):
    def exploder(prompt):
        raise RuntimeError("network down")
    monkeypatch.setattr(retriever, "_make_llm_callable", lambda model: exploder)

    docs = [{"text": "a", "score": 0.9}, {"text": "b", "score": 0.5}]
    out = retriever._maybe_rerank_results(
        "q", docs, {"use_llm_rerank": True, "llm_rerank_model": "test"}
    )
    # Fallback keeps original order.
    assert [d["text"] for d in out] == ["a", "b"]


def test_rerank_preserves_original_dict_shape(retriever, monkeypatch):
    def fake(prompt):
        return "1: 5\n2: 5\n"
    monkeypatch.setattr(retriever, "_make_llm_callable", lambda model: fake)

    docs = [
        {"text": "a", "score": 0.9, "filename": "x.pdf", "page_num": 1},
        {"text": "b", "score": 0.5, "filename": "y.pdf", "page_num": 7},
    ]
    out = retriever._maybe_rerank_results(
        "q", docs, {"use_llm_rerank": True, "llm_rerank_model": "test"}
    )
    # Each output doc is exactly the input dict (not a RerankedCandidate).
    assert all(isinstance(d, dict) for d in out)
    assert {d["filename"] for d in out} == {"x.pdf", "y.pdf"}


# ── HyDE helper ──────────────────────────────────────────────────────────────


def test_hyde_off_is_passthrough(retriever, tmp_path: Path):
    _write_session(tmp_path, "sess-a", use_hyde=False)
    assert retriever._maybe_apply_hyde_expansion("original", "sess-a") == "original"


def test_hyde_on_concatenates_original_and_passage(retriever, tmp_path: Path, monkeypatch):
    _write_session(tmp_path, "sess-b", use_hyde=True, hyde_model="test")
    monkeypatch.setattr(
        retriever, "_make_llm_callable",
        lambda model: lambda prompt: "Hypothetical answer text."
    )
    out = retriever._maybe_apply_hyde_expansion("what is the revenue?", "sess-b")
    assert "what is the revenue?" in out
    assert "Hypothetical answer text." in out


def test_hyde_missing_session_passthrough(retriever):
    assert retriever._maybe_apply_hyde_expansion("q", "no-such-session") == "q"


def test_hyde_llm_failure_passthrough(retriever, tmp_path: Path, monkeypatch):
    _write_session(tmp_path, "sess-c", use_hyde=True, hyde_model="test")
    def exploder(prompt):
        raise RuntimeError("timeout")
    monkeypatch.setattr(retriever, "_make_llm_callable", lambda model: exploder)
    assert retriever._maybe_apply_hyde_expansion("q", "sess-c") == "q"


# ── Model name handling ─────────────────────────────────────────────────────


def test_make_llm_callable_strips_ollama_prefix(retriever, monkeypatch):
    """'ollama-gemma3:4b' (UI tag) must become 'gemma3:4b' (real tag) for the client."""
    seen_models = []

    def fake_chat(*, model, messages, options):
        seen_models.append(model)
        return {"message": {"content": "ok"}}

    import ollama as _ollama
    monkeypatch.setattr(_ollama, "chat", fake_chat)

    callable_ = retriever._make_llm_callable("ollama-gemma3:4b")
    callable_("hello")
    assert seen_models == ["gemma3:4b"]


def test_make_llm_callable_leaves_bare_model_alone(retriever, monkeypatch):
    seen_models = []
    def fake_chat(*, model, messages, options):
        seen_models.append(model)
        return {"message": {"content": "ok"}}

    import ollama as _ollama
    monkeypatch.setattr(_ollama, "chat", fake_chat)

    retriever._make_llm_callable("llama3.2-vision")("x")
    assert seen_models == ["llama3.2-vision"]
