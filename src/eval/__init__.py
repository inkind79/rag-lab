"""
Retrieval evaluation harness.

Pure-Python module for measuring retrieval quality against a fixed
golden set. Decoupled from the live retriever stack — pass in a
``retrieve(query) -> [doc_id]`` callable and get back per-case +
aggregate metrics. Designed to be run from CI (no GPU, no Ollama)
when the callable is mocked, or end-to-end against the real stack
when wired up to RAGRetriever.

See ``docs/eval.md`` (TODO) and ``src/eval/cli.py`` for usage.
"""

from src.eval.harness import EvalCase, EvalResult, run_eval
from src.eval.metrics import (
    average_precision,
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)

__all__ = [
    "EvalCase",
    "EvalResult",
    "average_precision",
    "mean_reciprocal_rank",
    "ndcg_at_k",
    "precision_at_k",
    "recall_at_k",
    "reciprocal_rank",
    "run_eval",
]
