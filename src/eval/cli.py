"""
Evaluation CLI.

Usage:
    python -m src.eval.cli --golden tests/fixtures/eval/sample_golden_set.json \\
                           --session <session_uuid> [--k 10] [--out report.json]

Without --session, the CLI runs against a deterministic toy retriever that
returns the relevant docs in random-but-reproducible order — useful for
verifying the harness wiring on a clean checkout without any indexing.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

from src.eval.harness import (
    EvalResult,
    Retriever,
    load_golden_set,
    run_eval,
    serialize_result,
)


def _toy_retriever(cases) -> Retriever:
    """A perfect retriever: returns each query's relevant docs in golden order.

    Lets a developer verify the harness end-to-end without indexing anything.
    Real evaluation should use --session.
    """
    by_query = {c.query: list(c.relevant) for c in cases}

    def retrieve(query: str, k: int):
        rel = by_query.get(query, [])
        # Determinism if multiple identical queries: hash-shuffle within the relevant list.
        seed = int(hashlib.md5(query.encode()).hexdigest(), 16) % 1000
        # Rotate so adjacent queries don't trivially share rank-1.
        return (rel[seed % len(rel):] + rel[: seed % len(rel)])[:k] if rel else []

    return retrieve


def _live_retriever(session_id: str) -> Retriever:
    """Wrap RAGRetriever so the CLI can score the real stack against the golden set."""
    from src.models.retriever_manager import get_retriever
    from src.services.session_manager.manager import load_session

    retriever = get_retriever()
    session_data = load_session("sessions", session_id)
    if not session_data:
        raise SystemExit(f"Session {session_id!r} not found in sessions/")

    def retrieve(query: str, k: int):
        # Scope to whatever the session has selected; eval is "given these docs,
        # how well do we rank pages within them".
        results = retriever.retrieve(
            query=query,
            session_id=session_id,
            session_data=session_data,
            top_k=k,
        )
        # Each result is a dict with at minimum 'filename' + 'page_num' on visual
        # retrieval, or 'id' on text. Normalize to "filename#page" so the
        # golden set is human-editable.
        ids = []
        for r in results or []:
            if "filename" in r and "page_num" in r:
                ids.append(f"{r['filename']}#{r['page_num']}")
            elif "id" in r:
                ids.append(r["id"])
        return ids

    return retrieve


def _print_summary(result: EvalResult) -> None:
    agg = result.aggregate
    print(f"\n=== Aggregate (k={result.k}, n={len(result.cases)}) ===")
    print(f"  precision@k : {agg['precision@k']:.4f}")
    print(f"  recall@k    : {agg['recall@k']:.4f}")
    print(f"  MRR         : {agg['mrr']:.4f}")
    print(f"  NDCG@k      : {agg['ndcg@k']:.4f}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Run retrieval eval against a golden set.")
    p.add_argument("--golden", required=True, help="Path to golden set JSON.")
    p.add_argument("--session", help="Session UUID to run against (live retriever). Omit for toy retriever.")
    p.add_argument("--k", type=int, default=10, help="Cutoff for precision/recall/NDCG.")
    p.add_argument("--out", help="Optional path to write JSON report.")
    args = p.parse_args(argv)

    cases = load_golden_set(args.golden)
    if not cases:
        print(f"No cases in {args.golden}; nothing to do.", file=sys.stderr)
        return 1

    if args.session:
        retriever = _live_retriever(args.session)
    else:
        print("# No --session provided; running toy retriever (sanity check only).")
        retriever = _toy_retriever(cases)

    result = run_eval(cases, retriever, k=args.k)
    _print_summary(result)

    if args.out:
        Path(args.out).write_text(serialize_result(result))
        print(f"\nWrote full report to {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
