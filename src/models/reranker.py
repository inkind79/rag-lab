"""
LLM-based reranker for retrieval candidates.

Second-stage reranking is a well-known RAG quality lever: the first-stage
retriever (ColPali / BM25 / hybrid) optimizes for recall over a large
corpus, the reranker re-scores a small top-K with a richer signal. Here
we do it with a small instruction-tuned LLM that assigns a 0–10
relevance score per candidate.

Design choices:
- Pure primitive: ``LLMReranker.rerank(query, candidates)`` returns the
  candidates sorted by rerank score. No coupling to the retriever
  pipeline — that integration is a separate PR.
- Batch scoring: send up to ``batch_size`` candidates per LLM call to
  amortize latency. Parse the response line-by-line.
- Soft-fail: any parsing error / timeout returns the original ordering
  with a debug log. Reranking must never make retrieval worse by
  failing closed.
- The LLM client is injected so tests run without Ollama.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

logger = logging.getLogger(__name__)

DEFAULT_RERANK_PROMPT = (
    "You are a relevance judge for a document-retrieval system.\n"
    "Score how relevant each numbered passage is to the user's query on a 0 to 10 scale,\n"
    "where 10 means 'directly answers the query' and 0 means 'completely unrelated'.\n\n"
    "Query: {query}\n\n"
    "Passages:\n"
    "{passages}\n\n"
    "Respond with ONE line per passage in the exact format:\n"
    "  <passage_number>: <score>\n"
    "Do not add explanation. Example:\n"
    "  1: 7\n"
    "  2: 2\n"
    "  3: 9\n"
)


@dataclass
class RerankedCandidate:
    """Original candidate plus the rerank score (``score``) the LLM assigned."""

    doc: dict
    score: float
    # Keep the original retriever score alongside so callers can decide how
    # to fuse them (pure rerank, weighted blend, etc.).
    original_score: float | None = None


@dataclass
class RerankerConfig:
    batch_size: int = 10
    max_passage_chars: int = 800
    prompt_template: str = DEFAULT_RERANK_PROMPT
    # Field name in a candidate dict to use as the passage text. Falls back
    # to ``doc['metadata']['text']`` and finally to str(doc).
    text_field: str = "text"
    # Which doc field to log when recording fallback behavior.
    id_fields: tuple[str, ...] = field(default_factory=lambda: ("id", "filename", "doc_id"))


# Signature of the injected LLM: takes a prompt, returns the raw response string.
# Keeping it this narrow makes test doubles trivial.
LLMFn = Callable[[str], str]


class LLMReranker:
    """Re-score retrieval candidates with an LLM, then sort by the new score."""

    def __init__(self, llm: LLMFn, config: RerankerConfig | None = None):
        self._llm = llm
        self._cfg = config or RerankerConfig()

    # ── Public API ──────────────────────────────────────────────────────────

    def rerank(self, query: str, candidates: Sequence[dict]) -> list[RerankedCandidate]:
        """Rerank ``candidates`` by LLM-judged relevance to ``query``.

        Returns a new list sorted by score descending. On any failure
        (LLM error, parse mismatch, empty response) falls back to the
        original order with the original retriever score.
        """
        if not candidates:
            return []

        try:
            scores = self._score_all(query, candidates)
        except Exception as exc:
            logger.warning(f"Reranker failed, keeping original order: {exc}")
            return [self._fallback(i, c) for i, c in enumerate(candidates)]

        reranked: list[RerankedCandidate] = []
        for i, candidate in enumerate(candidates):
            if i in scores:
                reranked.append(
                    RerankedCandidate(
                        doc=candidate,
                        score=scores[i],
                        original_score=self._get_original_score(candidate),
                    )
                )
            else:
                # Missed passages keep original order but get the retriever score
                # clamped into a low rerank bucket so they sort after scored ones.
                reranked.append(self._fallback(i, candidate))

        reranked.sort(key=lambda r: r.score, reverse=True)
        return reranked

    # ── Internals ───────────────────────────────────────────────────────────

    def _score_all(self, query: str, candidates: Sequence[dict]) -> dict[int, float]:
        """Run rerank in batches and merge scores keyed by original index."""
        scores: dict[int, float] = {}
        cfg = self._cfg
        for batch_start in range(0, len(candidates), cfg.batch_size):
            batch = candidates[batch_start : batch_start + cfg.batch_size]
            passages = self._format_passages(batch)
            prompt = cfg.prompt_template.format(query=query, passages=passages)
            response = self._llm(prompt)
            parsed = self._parse_scores(response, len(batch))
            for local_idx, score in parsed.items():
                scores[batch_start + local_idx] = score
        return scores

    def _format_passages(self, batch: Sequence[dict]) -> str:
        lines = []
        for i, candidate in enumerate(batch, start=1):
            text = self._extract_text(candidate)
            truncated = text[: self._cfg.max_passage_chars]
            lines.append(f"{i}. {truncated}")
        return "\n\n".join(lines)

    def _extract_text(self, candidate: dict) -> str:
        field_name = self._cfg.text_field
        if field_name in candidate:
            return str(candidate[field_name])
        metadata = candidate.get("metadata")
        if isinstance(metadata, dict) and "text" in metadata:
            return str(metadata["text"])
        return str(candidate)

    def _get_original_score(self, candidate: dict) -> float | None:
        for key in ("score", "relevance", "original_score"):
            if key in candidate:
                try:
                    return float(candidate[key])
                except (TypeError, ValueError):
                    pass
        return None

    def _fallback(self, index: int, candidate: dict) -> RerankedCandidate:
        """Preserve original ordering when we can't get a rerank score."""
        # Use a tiny tiebreaker so two fallbacks don't collide at the same score.
        return RerankedCandidate(
            doc=candidate,
            score=-(index + 1) * 0.001,  # negative → sorts below any real rerank score
            original_score=self._get_original_score(candidate),
        )

    # ── Parsing ─────────────────────────────────────────────────────────────

    _SCORE_LINE = re.compile(
        r"""^[\s\-*>]*               # leading whitespace + optional list bullets
            (?:\*\*)?                # optional opening markdown bold
            \s*
            (\d+)                    # passage number
            (?:\*\*)?                # optional closing markdown bold
            \s*[:.)-]\s*             # separator: :, ., ), -
            (-?\d+(?:\.\d+)?)        # score (accept signed so we can clamp)
        """,
        re.VERBOSE | re.MULTILINE,
    )

    def _parse_scores(self, response: str, batch_size: int) -> dict[int, float]:
        """Extract "n: score" pairs from the LLM response.

        Tolerates markdown bullets, inconsistent separators, and extra
        explanation lines. Scores are clamped to [0, 10].
        """
        out: dict[int, float] = {}
        for match in self._SCORE_LINE.finditer(response):
            passage_num = int(match.group(1))
            raw_score = float(match.group(2))
            # Passage numbers are 1-indexed in the prompt; convert to 0-indexed.
            local_idx = passage_num - 1
            if 0 <= local_idx < batch_size:
                out[local_idx] = max(0.0, min(10.0, raw_score))
        return out
