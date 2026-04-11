"""
Hybrid Weighted Retriever

Combines ColPali visual retrieval with BM25 keyword retrieval using
normalized, weighted score fusion. Visual and text scores are independently
min-max normalized to [0, 1], then combined with configurable weights.
"""

from typing import List, Dict, Any, Optional

from src.models.retriever_base import BaseRetriever
from src.utils.logger import get_logger

logger = get_logger(__name__)


class HybridWeightedRetriever(BaseRetriever):
    """Fuses ColPali (visual) and BM25 (keyword) results with weighted scores.

    Over-retrieves from both child retrievers, normalizes scores independently,
    merges by page key (filename, page_num), and returns top-k by fused score.
    Falls back to ColPali-only if BM25 returns no results.
    """

    def __init__(
        self,
        colpali_retriever: BaseRetriever,
        bm25_retriever: BaseRetriever,
        default_visual_weight: float = 0.6,
    ):
        self._colpali = colpali_retriever
        self._bm25 = bm25_retriever
        self._default_visual_weight = default_visual_weight

    def set_embedding_adapter(self, adapter) -> None:
        self._colpali.set_embedding_adapter(adapter)

    def retrieve_documents(
        self,
        query: str,
        session_id: str,
        k: int = 3,
        selected_filenames: Optional[List[str]] = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        visual_weight = kwargs.pop('visual_weight', self._default_visual_weight)
        text_weight = 1.0 - visual_weight

        # Over-retrieve from both, disable score-slope on children
        child_kwargs = {**kwargs, 'use_score_slope': False}
        over_k = min(max(k * 3, 15), 100)

        visual_results = self._colpali.retrieve_documents(
            query, session_id, k=over_k,
            selected_filenames=selected_filenames, **child_kwargs,
        )

        bm25_results = self._bm25.retrieve_documents(
            query, session_id, k=over_k,
            selected_filenames=selected_filenames, **child_kwargs,
        )

        # Fallback: if BM25 has nothing, return visual results as-is
        if not bm25_results:
            logger.info("Hybrid: no BM25 results, falling back to ColPali-only")
            return visual_results[:k]

        # Build page-level score maps
        visual_pages = {}  # (filename, page_num) -> {score, result_dict}
        for r in visual_results:
            key = (r.get('original_filename', ''), r.get('page_num', 0))
            if key not in visual_pages or r['score'] > visual_pages[key]['score']:
                visual_pages[key] = {'score': r['score'], 'result': r}

        # Group BM25 chunks to page level (max score per page)
        bm25_pages = {}  # (filename, page_num) -> max_score
        for r in bm25_results:
            key = (r.get('original_filename', ''), r.get('page_num', 0))
            score = r.get('score', 0.0)
            if key not in bm25_pages or score > bm25_pages[key]:
                bm25_pages[key] = score

        # Normalize scores to [0, 1]
        norm_visual = _min_max_normalize({k: v['score'] for k, v in visual_pages.items()})
        norm_bm25 = _min_max_normalize(bm25_pages)

        # Merge all page keys
        all_keys = set(norm_visual.keys()) | set(norm_bm25.keys())
        fused = []
        for key in all_keys:
            v_score = norm_visual.get(key, 0.0)
            t_score = norm_bm25.get(key, 0.0)
            combined = visual_weight * v_score + text_weight * t_score

            # Use the visual result dict if available (has image path)
            if key in visual_pages:
                result = {**visual_pages[key]['result'], 'score': combined}
            else:
                # Page only found by BM25 — no image path
                result = {
                    'path': '',
                    'original_filename': key[0],
                    'page_num': key[1],
                    'score': combined,
                    'result_type': 'text',
                    'retriever_name': 'hybrid',
                }
            result['retriever_name'] = 'hybrid'
            fused.append(result)

        fused.sort(key=lambda x: x['score'], reverse=True)

        logger.info(
            f"Hybrid retriever: {len(visual_results)} visual + {len(bm25_results)} BM25 "
            f"→ {len(fused)} merged pages (returning top {k}, "
            f"weights: visual={visual_weight:.2f}, text={text_weight:.2f})"
        )
        return fused[:k]


def _min_max_normalize(scores: Dict[Any, float]) -> Dict[Any, float]:
    """Min-max normalize a dict of scores to [0, 1]."""
    if not scores:
        return {}
    vals = list(scores.values())
    lo, hi = min(vals), max(vals)
    if hi == lo:
        return {k: 1.0 for k in scores}
    rng = hi - lo
    return {k: (v - lo) / rng for k, v in scores.items()}
