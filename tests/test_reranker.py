"""Unit tests for LLMReranker. No Ollama — the LLM is injected as a callable."""

from __future__ import annotations

import pytest

from src.models.reranker import (
    LLMReranker,
    RerankedCandidate,
    RerankerConfig,
)


def _canned_llm(answers: list[str]):
    """Returns an LLM stub that replays ``answers`` in order."""
    answers = list(answers)
    calls: list[str] = []

    def _fn(prompt: str) -> str:
        calls.append(prompt)
        return answers.pop(0)

    _fn.calls = calls  # type: ignore[attr-defined]
    return _fn


def _docs(*texts_and_scores):
    """Build a simple candidate list from (text, score) tuples."""
    return [{"text": t, "score": s, "id": f"d{i}"} for i, (t, s) in enumerate(texts_and_scores)]


# ── Basic behavior ───────────────────────────────────────────────────────────


def test_rerank_sorts_by_llm_score():
    llm = _canned_llm(["1: 3\n2: 9\n3: 5\n"])
    cands = _docs(("alpha", 0.1), ("beta", 0.1), ("gamma", 0.1))

    result = LLMReranker(llm).rerank("query", cands)

    assert [r.doc["text"] for r in result] == ["beta", "gamma", "alpha"]
    assert result[0].score == 9.0


def test_rerank_preserves_original_scores():
    llm = _canned_llm(["1: 5\n2: 5\n"])
    cands = [
        {"text": "a", "score": 0.42, "id": "d1"},
        {"text": "b", "score": 0.87, "id": "d2"},
    ]
    result = LLMReranker(llm).rerank("query", cands)
    orig = {r.doc["id"]: r.original_score for r in result}
    assert orig == {"d1": 0.42, "d2": 0.87}


def test_rerank_empty_input():
    llm = _canned_llm([])
    assert LLMReranker(llm).rerank("q", []) == []


# ── Parsing tolerance ────────────────────────────────────────────────────────


def test_parse_tolerates_markdown_bullets_and_mixed_separators():
    llm = _canned_llm(
        [
            "**1**: 8\n"
            "- 2) 3\n"
            "  3. 7.5\n"
            "Some explanation the model added anyway.\n"
        ]
    )
    cands = _docs(("first", 0), ("second", 0), ("third", 0))
    result = LLMReranker(llm).rerank("q", cands)
    # Ordering: first(8) > third(7.5) > second(3)
    assert [r.doc["text"] for r in result] == ["first", "third", "second"]


def test_out_of_bounds_scores_are_clamped():
    llm = _canned_llm(["1: 42\n2: -9\n"])
    cands = _docs(("a", 0), ("b", 0))
    result = LLMReranker(llm).rerank("q", cands)
    scores = {r.doc["text"]: r.score for r in result}
    assert scores["a"] == 10.0
    assert scores["b"] == 0.0


# ── Failure modes ────────────────────────────────────────────────────────────


def test_unparseable_response_falls_back_to_original_order():
    llm = _canned_llm(["I'm sorry, I can't help with that."])
    cands = _docs(("a", 0.9), ("b", 0.5), ("c", 0.3))
    result = LLMReranker(llm).rerank("q", cands)
    # All three get fallback scores; fallback ordering matches input.
    assert [r.doc["text"] for r in result] == ["a", "b", "c"]
    # Fallback scores are negative so any real rerank score beats them.
    assert all(r.score < 0 for r in result)


def test_llm_exception_is_caught():
    def exploder(_prompt: str) -> str:
        raise RuntimeError("network down")

    cands = _docs(("a", 0.9), ("b", 0.5))
    result = LLMReranker(exploder).rerank("q", cands)
    assert len(result) == 2
    assert [r.doc["text"] for r in result] == ["a", "b"]


def test_partial_scores_do_not_drop_unscored_candidates():
    """If the LLM only scores some, unscored ones still appear (after scored ones)."""
    llm = _canned_llm(["1: 9\n3: 2\n"])  # passage 2 silently missed
    cands = _docs(("a", 0), ("b", 0), ("c", 0))
    result = LLMReranker(llm).rerank("q", cands)
    # "a" is clearly first (score 9). "b" missed → fallback. "c" scored 2.
    assert result[0].doc["text"] == "a"
    # "c" beats "b" because c got a real score.
    ranks = {r.doc["text"]: i for i, r in enumerate(result)}
    assert ranks["c"] < ranks["b"]


# ── Batching ─────────────────────────────────────────────────────────────────


def test_batches_respect_batch_size():
    # batch_size=2 → 3 candidates → 2 LLM calls
    llm = _canned_llm(["1: 7\n2: 3\n", "1: 9\n"])
    cands = _docs(("a", 0), ("b", 0), ("c", 0))
    reranker = LLMReranker(llm, RerankerConfig(batch_size=2))
    result = reranker.rerank("q", cands)
    assert len(llm.calls) == 2  # type: ignore[attr-defined]
    # c got 9, a got 7, b got 3
    assert [r.doc["text"] for r in result] == ["c", "a", "b"]


def test_max_passage_chars_truncates_in_prompt():
    llm = _canned_llm(["1: 5\n"])
    long = "x" * 5000
    cands = [{"text": long, "score": 0, "id": "d1"}]
    reranker = LLMReranker(llm, RerankerConfig(max_passage_chars=50))
    reranker.rerank("q", cands)
    prompt = llm.calls[0]  # type: ignore[attr-defined]
    # The passage in the prompt should be truncated.
    assert "x" * 5000 not in prompt
    assert "x" * 50 in prompt


# ── Text field resolution ────────────────────────────────────────────────────


def test_falls_back_to_metadata_text():
    llm = _canned_llm(["1: 4\n"])
    cands = [{"metadata": {"text": "via metadata"}, "id": "d1"}]
    LLMReranker(llm).rerank("q", cands)
    assert "via metadata" in llm.calls[0]  # type: ignore[attr-defined]


def test_custom_text_field():
    llm = _canned_llm(["1: 4\n"])
    cands = [{"body": "alt field", "id": "d1"}]
    reranker = LLMReranker(llm, RerankerConfig(text_field="body"))
    reranker.rerank("q", cands)
    assert "alt field" in llm.calls[0]  # type: ignore[attr-defined]
