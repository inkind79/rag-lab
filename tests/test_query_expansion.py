"""Unit tests for HyDE + multi-query expanders. LLM is injected; no Ollama needed."""

from __future__ import annotations

import pytest

from src.models.query_expansion import (
    HyDEConfig,
    HyDEExpander,
    MultiQueryConfig,
    MultiQueryExpander,
)


def _stub_llm(reply: str | list[str]):
    replies = [reply] if isinstance(reply, str) else list(reply)
    calls: list[str] = []

    def _fn(prompt: str) -> str:
        calls.append(prompt)
        return replies.pop(0)

    _fn.calls = calls  # type: ignore[attr-defined]
    return _fn


# ── HyDE ─────────────────────────────────────────────────────────────────────


def test_hyde_appends_hypothetical_to_original():
    llm = _stub_llm("Q3 revenue was $42 million, up 12% year over year.")
    out = HyDEExpander(llm).expand("What was Q3 revenue?")
    assert len(out) == 1
    text = out[0]
    assert text.startswith("What was Q3 revenue?")
    assert "Q3 revenue was $42 million" in text


def test_hyde_can_replace_original():
    llm = _stub_llm("hypothetical answer")
    expander = HyDEExpander(llm, HyDEConfig(join_with_original=False))
    out = expander.expand("question")
    assert out == ["hypothetical answer"]


def test_hyde_returns_original_on_llm_error():
    def exploder(_p): raise RuntimeError("network")
    out = HyDEExpander(exploder).expand("question")
    assert out == ["question"]


def test_hyde_returns_original_on_empty_response():
    llm = _stub_llm("")
    assert HyDEExpander(llm).expand("question") == ["question"]


def test_hyde_handles_empty_query():
    llm = _stub_llm("nope")
    assert HyDEExpander(llm).expand("") == [""]
    assert HyDEExpander(llm).expand("   ") == ["   "]
    # LLM was never called for empty queries.
    assert llm.calls == []  # type: ignore[attr-defined]


def test_hyde_passes_max_words_to_prompt():
    llm = _stub_llm("answer")
    HyDEExpander(llm, HyDEConfig(max_words=120)).expand("question?")
    assert "120 words" in llm.calls[0]  # type: ignore[attr-defined]


# ── MultiQuery ───────────────────────────────────────────────────────────────


def test_multi_query_returns_original_plus_rewrites():
    llm = _stub_llm(
        "How did the company perform financially in Q3?\n"
        "What were the third quarter sales numbers?\n"
        "What revenue did the company report for Q3?\n"
    )
    out = MultiQueryExpander(llm).expand("What was Q3 revenue?")
    assert out[0] == "What was Q3 revenue?"
    assert len(out) == 4


def test_multi_query_strips_list_bullets():
    llm = _stub_llm(
        "1. First rewrite\n"
        "2) Second rewrite\n"
        "- Third rewrite\n"
        "* Fourth rewrite\n"
    )
    out = MultiQueryExpander(llm, MultiQueryConfig(n_rewrites=4, include_original=False)).expand("q")
    assert "First rewrite" in out
    assert "Second rewrite" in out
    assert "Third rewrite" in out
    assert "Fourth rewrite" in out
    # Bullets stripped — no leading "- " or "1." in the output strings.
    assert all(not r.startswith(("-", "*", "1.", "2.")) for r in out)


def test_multi_query_dedupes_rewrites():
    llm = _stub_llm(
        "Same question?\n"
        "Same question?\n"
        "Different question?\n"
    )
    out = MultiQueryExpander(llm, MultiQueryConfig(n_rewrites=3, include_original=False)).expand("q")
    assert out.count("Same question?") == 1
    assert "Different question?" in out


def test_multi_query_drops_rewrite_equal_to_original():
    llm = _stub_llm(
        "What was Q3 revenue?\n"     # same as original — should be dropped
        "How much did we earn in Q3?\n"
    )
    out = MultiQueryExpander(llm).expand("What was Q3 revenue?")
    assert out.count("What was Q3 revenue?") == 1
    assert "How much did we earn in Q3?" in out


def test_multi_query_falls_back_on_llm_error():
    def exploder(_p): raise RuntimeError("oops")
    assert MultiQueryExpander(exploder).expand("q") == ["q"]


def test_multi_query_handles_empty_query():
    llm = _stub_llm("nope")
    assert MultiQueryExpander(llm).expand("") == [""]
    assert llm.calls == []  # type: ignore[attr-defined]


def test_multi_query_caps_runaway_response():
    """A chatty model that returns 30 lines shouldn't flood the result list."""
    llm = _stub_llm("\n".join(f"Rewrite {i}?" for i in range(30)))
    out = MultiQueryExpander(llm, MultiQueryConfig(n_rewrites=3, include_original=False)).expand("q")
    # Cap is 2× expected.
    assert len(out) <= 6


def test_multi_query_passes_n_to_prompt():
    llm = _stub_llm("a?\nb?\nc?\nd?\n")
    MultiQueryExpander(llm, MultiQueryConfig(n_rewrites=4)).expand("question")
    assert "4 alternative" in llm.calls[0]  # type: ignore[attr-defined]
