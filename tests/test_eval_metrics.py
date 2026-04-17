"""Unit tests for IR metrics — pure functions, no fixtures needed."""

import math

import pytest

from src.eval.metrics import (
    average_precision,
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


# ── reciprocal_rank ──────────────────────────────────────────────────────────


def test_rr_first_position():
    assert reciprocal_rank(["a", "b", "c"], {"a"}) == 1.0


def test_rr_third_position():
    assert reciprocal_rank(["x", "y", "a"], {"a"}) == pytest.approx(1 / 3)


def test_rr_no_hit():
    assert reciprocal_rank(["x", "y", "z"], {"a"}) == 0.0


def test_rr_empty_retrieved():
    assert reciprocal_rank([], {"a"}) == 0.0


def test_rr_picks_first_of_multiple():
    """If multiple relevant docs match, only the earliest counts."""
    assert reciprocal_rank(["x", "a", "b"], {"a", "b"}) == pytest.approx(1 / 2)


# ── mean_reciprocal_rank ─────────────────────────────────────────────────────


def test_mrr_simple():
    pairs = [
        (["a", "b"], {"a"}),    # rr=1
        (["x", "a"], {"a"}),    # rr=0.5
        (["x", "y"], {"a"}),    # rr=0
    ]
    assert mean_reciprocal_rank(pairs) == pytest.approx((1 + 0.5 + 0) / 3)


def test_mrr_empty_input():
    assert mean_reciprocal_rank([]) == 0.0


# ── precision_at_k ───────────────────────────────────────────────────────────


def test_precision_all_relevant():
    assert precision_at_k(["a", "b", "c"], {"a", "b", "c"}, k=3) == 1.0


def test_precision_partial():
    assert precision_at_k(["a", "x", "b"], {"a", "b"}, k=3) == pytest.approx(2 / 3)


def test_precision_k_larger_than_results():
    """Precision is hits/k even if fewer than k docs were retrieved."""
    assert precision_at_k(["a"], {"a"}, k=5) == pytest.approx(1 / 5)


def test_precision_zero_k():
    assert precision_at_k(["a"], {"a"}, k=0) == 0.0


# ── recall_at_k ──────────────────────────────────────────────────────────────


def test_recall_full():
    assert recall_at_k(["a", "b", "c"], {"a", "b", "c"}, k=3) == 1.0


def test_recall_partial():
    assert recall_at_k(["a", "x", "y"], {"a", "b", "c"}, k=3) == pytest.approx(1 / 3)


def test_recall_no_relevant_docs():
    assert recall_at_k(["a", "b"], set(), k=3) == 0.0


def test_recall_truncates_at_k():
    """Docs beyond rank k don't count."""
    assert recall_at_k(["x", "y", "z", "a"], {"a"}, k=3) == 0.0


# ── average_precision ────────────────────────────────────────────────────────


def test_ap_perfect_ranking():
    assert average_precision(["a", "b", "c"], {"a", "b", "c"}) == 1.0


def test_ap_known_value():
    """Manual: P@1=1, P@3=2/3 → AP = (1 + 2/3) / 2 = 5/6."""
    assert average_precision(["a", "x", "b"], {"a", "b"}) == pytest.approx(5 / 6)


def test_ap_no_relevant():
    assert average_precision(["a", "b"], set()) == 0.0


# ── ndcg_at_k ────────────────────────────────────────────────────────────────


def test_ndcg_perfect_binary():
    assert ndcg_at_k(["a", "b", "c"], {"a": 1, "b": 1, "c": 1}, k=3) == 1.0


def test_ndcg_inverted_ranking_drops():
    """If we put the lowest-relevance doc first, NDCG should be < 1."""
    perfect = {"a": 3, "b": 2, "c": 1}
    score = ndcg_at_k(["c", "b", "a"], perfect, k=3)
    assert 0 < score < 1


def test_ndcg_no_relevant_docs():
    assert ndcg_at_k(["a", "b"], {}, k=3) == 0.0


def test_ndcg_zero_k():
    assert ndcg_at_k(["a"], {"a": 1}, k=0) == 0.0


def test_ndcg_handles_irrelevant_in_retrieved():
    """Docs not in the relevance map contribute zero gain."""
    score = ndcg_at_k(["a", "noise", "b"], {"a": 1, "b": 1}, k=3)
    # Ideal would put a, b at ranks 1+2; we have them at 1+3.
    # Idea DCG: 1/log2(2) + 1/log2(3) = 1 + 0.6309
    # Actual:   1/log2(2) + 1/log2(4) = 1 + 0.5
    expected = (1 + 0.5) / (1 + 1 / math.log2(3))
    assert score == pytest.approx(expected)
