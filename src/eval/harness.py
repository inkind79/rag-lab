"""
Retrieval eval harness.

Drive ``retrieve(query) -> [doc_id]`` against a fixed golden set, collect
per-case + aggregate metrics, return a structured result the CLI prints
or CI uploads as an artifact.

Decoupled from the live retriever stack. For unit tests the callable is
a deterministic in-memory function; for benchmarks the callable wraps
``RAGRetriever`` against a real session.
"""

from __future__ import annotations

import json
import statistics
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Hashable

from src.eval.metrics import (
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)

DocId = Hashable
Retriever = Callable[[str, int], Sequence[DocId]]


@dataclass(frozen=True)
class EvalCase:
    """One query + the doc_ids that should be retrieved (and optional graded relevance)."""

    query: str
    relevant: tuple[DocId, ...]
    # Optional graded relevance for NDCG. Defaults to binary (every relevant
    # doc has gain 1) when empty.
    relevance: dict[DocId, float] = field(default_factory=dict)
    # Free-form tag so eval reports can group by topic / difficulty.
    tag: str = ""


@dataclass
class CaseResult:
    case: EvalCase
    retrieved: tuple[DocId, ...]
    metrics: dict[str, float]


@dataclass
class EvalResult:
    k: int
    cases: list[CaseResult]
    aggregate: dict[str, float]

    def to_dict(self) -> dict:
        return {
            "k": self.k,
            "aggregate": self.aggregate,
            "cases": [
                {
                    "query": cr.case.query,
                    "tag": cr.case.tag,
                    "relevant": list(cr.case.relevant),
                    "retrieved": list(cr.retrieved),
                    "metrics": cr.metrics,
                }
                for cr in self.cases
            ],
        }


def _case_metrics(case: EvalCase, retrieved: Sequence[DocId], k: int) -> dict[str, float]:
    relevance = case.relevance or {doc: 1.0 for doc in case.relevant}
    return {
        "precision@k": precision_at_k(retrieved, case.relevant, k),
        "recall@k": recall_at_k(retrieved, case.relevant, k),
        "rr": reciprocal_rank(retrieved, case.relevant),
        "ndcg@k": ndcg_at_k(retrieved, relevance, k),
    }


def _aggregate(case_results: list[CaseResult]) -> dict[str, float]:
    if not case_results:
        return {"precision@k": 0.0, "recall@k": 0.0, "mrr": 0.0, "ndcg@k": 0.0}
    return {
        "precision@k": statistics.fmean(cr.metrics["precision@k"] for cr in case_results),
        "recall@k": statistics.fmean(cr.metrics["recall@k"] for cr in case_results),
        "mrr": statistics.fmean(cr.metrics["rr"] for cr in case_results),
        "ndcg@k": statistics.fmean(cr.metrics["ndcg@k"] for cr in case_results),
    }


def run_eval(
    cases: Sequence[EvalCase],
    retrieve: Retriever,
    k: int = 10,
) -> EvalResult:
    """Run ``retrieve`` against every case and compute per-case + aggregate metrics.

    ``retrieve`` is called as ``retrieve(query, k)`` and must return an ordered
    sequence (best first) of doc_ids — same identifier domain as the golden set.
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")

    case_results: list[CaseResult] = []
    for case in cases:
        retrieved = tuple(retrieve(case.query, k))
        case_results.append(
            CaseResult(case=case, retrieved=retrieved, metrics=_case_metrics(case, retrieved, k))
        )

    return EvalResult(k=k, cases=case_results, aggregate=_aggregate(case_results))


def load_golden_set(path: str | Path) -> list[EvalCase]:
    """Load a golden set from JSON.

    Schema: ``{"cases": [{"query": str, "relevant": [doc_id, ...], "relevance": {doc_id: gain}, "tag": str}]}``.
    Missing fields default to empty.
    """
    raw = json.loads(Path(path).read_text())
    cases_raw = raw.get("cases") if isinstance(raw, dict) else raw
    if not isinstance(cases_raw, list):
        raise ValueError(f"golden set at {path} must contain a 'cases' list")

    cases: list[EvalCase] = []
    for entry in cases_raw:
        cases.append(
            EvalCase(
                query=entry["query"],
                relevant=tuple(entry.get("relevant", [])),
                relevance=dict(entry.get("relevance", {})),
                tag=entry.get("tag", ""),
            )
        )
    return cases


def serialize_result(result: EvalResult) -> str:
    """Render an EvalResult as pretty JSON, suitable for CI artifacts."""
    return json.dumps(result.to_dict(), indent=2, default=str)
