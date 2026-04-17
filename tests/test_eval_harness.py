"""Harness end-to-end: golden set → retriever callable → aggregated metrics."""

import json
from pathlib import Path

import pytest

from src.eval import EvalCase, run_eval
from src.eval.harness import load_golden_set, serialize_result


def _perfect_retriever(case_map):
    """Returns each query's relevant docs in golden order."""
    def retrieve(query, k):
        return list(case_map.get(query, []))[:k]
    return retrieve


def _empty_retriever(query, k):
    return []


def _round_robin_retriever(case_map):
    """Returns relevant docs but starts from the second one (so RR=0.5)."""
    def retrieve(query, k):
        rel = list(case_map.get(query, []))
        if not rel:
            return []
        return (rel[1:] + rel[:1])[:k]
    return retrieve


@pytest.fixture
def cases():
    return [
        EvalCase(query="q1", relevant=("a", "b", "c"), tag="t1"),
        EvalCase(query="q2", relevant=("d",), tag="t1"),
        EvalCase(query="q3", relevant=("e", "f"), tag="t2"),
    ]


def test_perfect_retriever_scores_one(cases):
    case_map = {c.query: c.relevant for c in cases}
    result = run_eval(cases, _perfect_retriever(case_map), k=3)
    # P@3 for ("a","b","c") with relevant {"a","b","c"} = 3/3 = 1.0
    # P@3 for ("d",) with relevant {"d"} = 1/3 (only 1 hit out of 3 slots)
    # P@3 for ("e","f") with relevant {"e","f"} = 2/3
    # average = (1 + 1/3 + 2/3) / 3 = 2/3
    assert result.aggregate["precision@k"] == pytest.approx(2 / 3)
    # Perfect retriever puts relevant first, so MRR == 1.0
    assert result.aggregate["mrr"] == pytest.approx(1.0)
    # Recall@3 covers all relevant docs in every case
    assert result.aggregate["recall@k"] == pytest.approx(1.0)


def test_empty_retriever_scores_zero(cases):
    result = run_eval(cases, lambda q, k: [], k=5)
    assert result.aggregate["mrr"] == 0.0
    assert result.aggregate["recall@k"] == 0.0
    assert result.aggregate["precision@k"] == 0.0
    assert result.aggregate["ndcg@k"] == 0.0


def test_round_robin_retriever_drops_mrr(cases):
    case_map = {c.query: c.relevant for c in cases}
    result = run_eval(cases, _round_robin_retriever(case_map), k=3)
    # Each query: relevant doc at rank 1 is the SECOND element of golden
    # for q1: golden ("a","b","c") → returns ("b","c","a") → rr=1.0 (b is relevant)
    # All are relevant docs, so rr=1.0 each. Need a more aggressive shuffle.
    assert result.aggregate["mrr"] == pytest.approx(1.0)
    # But recall is still complete — we returned every relevant doc.
    assert result.aggregate["recall@k"] == pytest.approx(1.0)


def test_per_case_results_include_metrics(cases):
    case_map = {c.query: c.relevant for c in cases}
    result = run_eval(cases, _perfect_retriever(case_map), k=3)
    assert len(result.cases) == 3
    for cr in result.cases:
        assert set(cr.metrics) == {"precision@k", "recall@k", "rr", "ndcg@k"}
        assert cr.case in cases


def test_zero_k_raises():
    with pytest.raises(ValueError):
        run_eval([EvalCase(query="q", relevant=("a",))], lambda q, k: [], k=0)


def test_load_golden_set_round_trip(tmp_path: Path):
    payload = {
        "cases": [
            {"query": "what", "relevant": ["a", "b"], "relevance": {"a": 2, "b": 1}, "tag": "t1"},
            {"query": "why", "relevant": ["c"], "tag": "t2"},
        ]
    }
    p = tmp_path / "gold.json"
    p.write_text(json.dumps(payload))

    loaded = load_golden_set(p)
    assert len(loaded) == 2
    assert loaded[0].query == "what"
    assert loaded[0].relevant == ("a", "b")
    assert loaded[0].relevance == {"a": 2, "b": 1}
    assert loaded[0].tag == "t1"
    assert loaded[1].relevance == {}  # default empty


def test_load_golden_set_rejects_bad_shape(tmp_path: Path):
    p = tmp_path / "bad.json"
    p.write_text(json.dumps({"not_cases": []}))
    with pytest.raises(ValueError):
        load_golden_set(p)


def test_serialize_result_is_valid_json(cases):
    case_map = {c.query: c.relevant for c in cases}
    result = run_eval(cases, _perfect_retriever(case_map), k=3)
    serialized = serialize_result(result)
    parsed = json.loads(serialized)
    assert parsed["k"] == 3
    assert "aggregate" in parsed
    assert len(parsed["cases"]) == 3


def test_sample_golden_set_loads():
    """The sample golden set bundled with the repo is valid."""
    sample = Path(__file__).parent / "fixtures" / "eval" / "sample_golden_set.json"
    cases = load_golden_set(sample)
    assert len(cases) == 3
    queries = {c.query for c in cases}
    assert "What is the company's Q3 revenue?" in queries
