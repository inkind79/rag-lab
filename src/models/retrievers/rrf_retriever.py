"""
Reciprocal Rank Fusion (RRF) Hybrid Retriever

Combines results from multiple retrievers using RRF scoring.
Score = Σ(1 / (k + rank_i)) across all retriever result lists.
"""

from typing import List, Dict, Any, Optional

from src.models.retriever_base import BaseRetriever
from src.utils.logger import get_logger

logger = get_logger(__name__)


class RRFHybridRetriever(BaseRetriever):
    """Combines multiple retrievers using Reciprocal Rank Fusion.

    Each child retriever produces a ranked list. RRF assigns each result
    a score based on its rank in each list, then merges and re-ranks.
    """

    def __init__(self, retrievers: Optional[List[BaseRetriever]] = None, k_rrf: int = 60):
        """
        Args:
            retrievers: List of child retrievers to fuse
            k_rrf: RRF constant (higher = more weight to lower ranks). Default 60.
        """
        self._retrievers = retrievers or []
        self._k_rrf = k_rrf

    def add_retriever(self, retriever: BaseRetriever):
        """Add a child retriever."""
        self._retrievers.append(retriever)

    def set_embedding_adapter(self, adapter):
        """Propagate the embedding adapter to all child retrievers."""
        for r in self._retrievers:
            r.set_embedding_adapter(adapter)

    def retrieve_documents(
        self,
        query: str,
        session_id: str,
        k: int = 3,
        selected_filenames: Optional[List[str]] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Retrieve from all child retrievers and fuse with RRF."""
        if not self._retrievers:
            logger.warning("RRFHybridRetriever has no child retrievers")
            return []

        # Collect ranked lists from each retriever
        all_ranked_lists = []
        for retriever in self._retrievers:
            try:
                results = retriever.retrieve_documents(
                    query, session_id,
                    k=k * 3,  # Over-retrieve for better fusion
                    selected_filenames=selected_filenames,
                    **kwargs
                )
                all_ranked_lists.append(results)
                logger.info(f"RRF: {type(retriever).__name__} returned {len(results)} results")
            except Exception as e:
                logger.error(f"RRF: {type(retriever).__name__} failed: {e}")
                all_ranked_lists.append([])

        # RRF scoring
        rrf_scores: Dict[str, float] = {}
        result_map: Dict[str, Dict[str, Any]] = {}

        for ranked_list in all_ranked_lists:
            for rank, result in enumerate(ranked_list):
                # Create a stable key for deduplication
                key = self._result_key(result)
                rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (self._k_rrf + rank + 1)
                if key not in result_map:
                    result_map[key] = result

        # Sort by RRF score and return top-k
        sorted_keys = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

        results = []
        for key in sorted_keys[:k]:
            result = result_map[key].copy()
            result['score'] = rrf_scores[key]
            result['retriever_name'] = 'rrf_hybrid'
            results.append(result)

        logger.info(f"RRF fusion produced {len(results)} results from {len(self._retrievers)} retrievers")
        return results

    @staticmethod
    def _result_key(result: Dict[str, Any]) -> str:
        """Create a stable deduplication key for a result."""
        # Use chunk_id if available (text results), else use path + page
        chunk_id = result.get('chunk_id')
        if chunk_id:
            return chunk_id
        path = result.get('path', '')
        page = result.get('page_num', 0)
        filename = result.get('original_filename', '')
        return f"{filename}:{page}:{path}"
