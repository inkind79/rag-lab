"""
Query expansion strategies that bridge the lexical / semantic gap between
how users phrase questions and how documents phrase answers.

Two strategies live here:

- :class:`HyDEExpander` (Hypothetical Document Embeddings) generates a
  short hypothetical answer to the user query and embeds *that*. The
  retriever then matches passages by semantic proximity to a "what an
  answer would look like" vector instead of to the question itself.
  Effective when query phrasing is short or sparse and corpus passages
  are denser.

- :class:`MultiQueryExpander` rewrites the query into a small fan-out
  of paraphrases. The caller fires each one through the retriever and
  unions / fuses results. Helps when one phrasing wins recall on some
  passages and another wins on others.

Both expanders share the same shape: ``expand(query) -> list[str]`` where
the returned list is the set of strings to feed to the retriever. The
LLM is injected as a callable so unit tests don't need Ollama and the
caller can swap in any local or remote model.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Callable

logger = logging.getLogger(__name__)


# Signature of the injected LLM: prompt → raw response string.
LLMFn = Callable[[str], str]


HYDE_PROMPT = (
    "Write a short, plausible passage that would directly answer the question below.\n"
    "Do NOT include disclaimers, citations, or meta-commentary about not knowing the answer.\n"
    "Aim for {max_words} words. Use the language and vocabulary you'd expect in the source documents.\n\n"
    "Question: {query}\n\n"
    "Passage:"
)

MULTI_QUERY_PROMPT = (
    "Rewrite the following search query as {n} alternative questions that ask for the same\n"
    "information using different vocabulary. Each rewrite should be a complete question on\n"
    "its own line. Do not number them, do not add explanation.\n\n"
    "Original query: {query}\n\n"
    "Rewrites:"
)


@dataclass
class HyDEConfig:
    max_words: int = 80
    # Prepended to the original query so the retriever still sees the literal
    # user vocabulary as a tiebreaker. Empty string disables that fusion.
    join_with_original: bool = True
    prompt_template: str = HYDE_PROMPT


@dataclass
class MultiQueryConfig:
    n_rewrites: int = 3
    include_original: bool = True
    prompt_template: str = MULTI_QUERY_PROMPT


class HyDEExpander:
    """Generate one hypothetical-answer passage per query.

    Returns a list with at least one string: the hypothetical passage,
    optionally concatenated with the original query so retrievers that
    do exact-keyword matching (BM25) still benefit. On any LLM failure
    the original query is returned unchanged — never make retrieval
    worse by trying to expand it.
    """

    def __init__(self, llm: LLMFn, config: HyDEConfig | None = None):
        self._llm = llm
        self._cfg = config or HyDEConfig()

    def expand(self, query: str) -> list[str]:
        if not query or not query.strip():
            return [query]

        try:
            prompt = self._cfg.prompt_template.format(
                query=query, max_words=self._cfg.max_words
            )
            hypothetical = self._llm(prompt).strip()
        except Exception as e:
            logger.warning(f"HyDE expansion failed, using original query: {e}")
            return [query]

        if not hypothetical:
            return [query]

        if self._cfg.join_with_original:
            return [f"{query}\n\n{hypothetical}"]
        return [hypothetical]


class MultiQueryExpander:
    """Fan out a query into several paraphrases.

    Returns a list of distinct query strings. The original query is
    always included first when ``include_original`` is True (default).
    """

    def __init__(self, llm: LLMFn, config: MultiQueryConfig | None = None):
        self._llm = llm
        self._cfg = config or MultiQueryConfig()

    def expand(self, query: str) -> list[str]:
        if not query or not query.strip():
            return [query]
        cfg = self._cfg
        n = max(1, cfg.n_rewrites)

        try:
            prompt = cfg.prompt_template.format(query=query, n=n)
            response = self._llm(prompt)
        except Exception as e:
            logger.warning(f"MultiQuery expansion failed, using original query: {e}")
            return [query]

        rewrites = self._parse_rewrites(response, expected=n)

        result = [query] if cfg.include_original else []
        for r in rewrites:
            if r and r != query and r not in result:
                result.append(r)
        return result or [query]

    @staticmethod
    def _parse_rewrites(response: str, expected: int) -> list[str]:
        """Extract one rewrite per line, stripping list-bullet noise.

        Tolerates numbered lists, bullets, leading dashes, and blank
        lines. Returns at most ``expected * 2`` so a chatty model can't
        flood the result list.
        """
        rewrites: list[str] = []
        for raw_line in response.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            # Strip list-bullet noise: "1)", "1.", "- ", "* "
            line = re.sub(r'^(?:\d+[.)]\s*|[-*]\s*)', '', line).strip()
            if line:
                rewrites.append(line)
            if len(rewrites) >= expected * 2:
                break
        return rewrites
