"""
Dense Bi-Encoder Retriever

Single-vector similarity search using dense embedding models (BGE-M3, E5, etc.).
"""

from typing import List, Dict, Any, Optional

from src.models.retriever_base import BaseRetriever
from src.models.retrieval_result import RetrievalResult, ResultType
from src.models.vector_stores.single_vector_lancedb_store import SingleVectorLanceDBStore
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DenseRetriever(BaseRetriever):
    """Dense single-vector retriever using LanceDB."""

    def __init__(self):
        self._embedding_adapter = None
        self._store = None

    def set_embedding_adapter(self, adapter):
        """Set the dense embedding adapter and initialize the store."""
        self._embedding_adapter = adapter
        self._store = SingleVectorLanceDBStore(
            model_name=adapter.model_name,
            dimension=adapter.dimension,
        )

    def retrieve_documents(
        self,
        query: str,
        session_id: str,
        k: int = 3,
        selected_filenames: Optional[List[str]] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Retrieve text chunks using dense vector similarity."""
        if self._embedding_adapter is None or self._store is None:
            logger.error("DenseRetriever: no embedding adapter set")
            return []

        query_embedding = self._embedding_adapter.encode_query(query)
        filter_dict = {'filename': selected_filenames} if selected_filenames else None
        similarity_threshold = kwargs.get('similarity_threshold', 0.0)

        metadatas, scores, ids = self._store.query(
            session_id,
            query_embedding=query_embedding,
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
                retriever_name="dense",
            )
            results.append(result.to_legacy_dict())

        logger.info(f"Dense retriever returned {len(results)} results for query: '{query[:50]}...'")
        return results
