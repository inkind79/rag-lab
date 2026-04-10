"""
BM25 Text Retriever

Keyword-based retrieval using BM25 scoring over extracted text chunks.
No embedding model needed.
"""

from typing import List, Dict, Any, Optional

from src.models.retriever_base import BaseRetriever
from src.models.retrieval_result import RetrievalResult, ResultType
from src.models.vector_stores.bm25_store import get_bm25_store
from src.utils.logger import get_logger

logger = get_logger(__name__)


class BM25Retriever(BaseRetriever):
    """BM25 text retriever. Uses an inverted index, no embeddings needed."""

    def __init__(self):
        self._store = get_bm25_store()

    def retrieve_documents(
        self,
        query: str,
        session_id: str,
        k: int = 3,
        selected_filenames: Optional[List[str]] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Retrieve text chunks using BM25 keyword matching."""
        filter_dict = {'filename': selected_filenames} if selected_filenames else None
        similarity_threshold = kwargs.get('similarity_threshold', 0.0)

        metadatas, scores, ids = self._store.query(
            session_id,
            query_text=query,
            k=k,
            filter_dict=filter_dict,
            similarity_threshold=similarity_threshold,
        )

        results = []
        for meta, score, doc_id in zip(metadatas, scores, ids):
            result = RetrievalResult(
                result_type=ResultType.TEXT_CHUNK,
                score=score,
                text_content=meta.get('text', ''),
                chunk_id=doc_id,
                source_document=meta.get('filename', ''),
                page_num=meta.get('page_num', 0),
                retriever_name="bm25",
            )
            results.append(result.to_legacy_dict())

        logger.info(f"BM25 retriever returned {len(results)} results for query: '{query[:50]}...'")
        return results
