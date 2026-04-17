"""
Standard IR ranking metrics — pure functions, no external deps.

Conventions:
- ``retrieved`` is an ordered sequence (best first) of doc identifiers.
- ``relevant`` is a set of doc identifiers that should have appeared.
- For graded relevance (NDCG), ``relevance`` maps doc_id -> gain.
- All metrics return ``float`` in [0.0, 1.0].

These functions have no opinion about what a "doc_id" is — they're
identity-compared, so use whatever your retriever returns
(filename, page-id, hash). Keep them consistent across the golden
set and the retriever output.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Hashable

DocId = Hashable


def reciprocal_rank(retrieved: Sequence[DocId], relevant: Iterable[DocId]) -> float:
    """1 / rank of the first relevant hit, or 0 if none. Single-query MRR."""
    relevant_set = set(relevant)
    for idx, doc in enumerate(retrieved, start=1):
        if doc in relevant_set:
            return 1.0 / idx
    return 0.0


def mean_reciprocal_rank(
    rankings: Iterable[tuple[Sequence[DocId], Iterable[DocId]]],
) -> float:
    """Mean of reciprocal_rank() across an iterable of (retrieved, relevant) pairs."""
    rrs = [reciprocal_rank(r, rel) for r, rel in rankings]
    return sum(rrs) / len(rrs) if rrs else 0.0


def precision_at_k(
    retrieved: Sequence[DocId], relevant: Iterable[DocId], k: int
) -> float:
    """Fraction of the top-k results that are in ``relevant``."""
    if k <= 0:
        return 0.0
    relevant_set = set(relevant)
    top_k = list(retrieved)[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for doc in top_k if doc in relevant_set)
    return hits / k


def recall_at_k(
    retrieved: Sequence[DocId], relevant: Iterable[DocId], k: int
) -> float:
    """Fraction of ``relevant`` that appears in the top-k results."""
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    top_k = set(list(retrieved)[:k])
    hits = len(top_k & relevant_set)
    return hits / len(relevant_set)


def average_precision(
    retrieved: Sequence[DocId], relevant: Iterable[DocId]
) -> float:
    """Average of precision@k computed at each rank where a relevant doc appears."""
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    score = 0.0
    hits = 0
    for idx, doc in enumerate(retrieved, start=1):
        if doc in relevant_set:
            hits += 1
            score += hits / idx
    return score / len(relevant_set)


def ndcg_at_k(
    retrieved: Sequence[DocId],
    relevance: Mapping[DocId, float],
    k: int,
) -> float:
    """Normalized DCG with binary or graded relevance.

    ``relevance`` maps doc_id -> gain (use {doc: 1.0} for binary).
    Documents not in the map contribute 0. Uses the standard 2^gain - 1
    numerator and log2(rank + 1) discount. Returns 0 if the ideal DCG
    is 0 (i.e. there are no relevant docs at all).
    """
    if k <= 0:
        return 0.0

    def _dcg(gains: Sequence[float]) -> float:
        return sum(g / math.log2(i + 2) for i, g in enumerate(gains))

    actual_gains = [(2 ** relevance.get(doc, 0.0) - 1) for doc in list(retrieved)[:k]]
    ideal_gains = sorted(
        ((2 ** g - 1) for g in relevance.values() if g > 0),
        reverse=True,
    )[:k]

    ideal = _dcg(ideal_gains)
    if ideal == 0.0:
        return 0.0
    return _dcg(actual_gains) / ideal
