"""
Multi-Vector LanceDB Store

Wraps the existing lancedb_manager.py functions in the BaseVectorStore interface.
This is the store used by ColPali-based retrievers for multi-vector (ColBERT-style)
embeddings.
"""

from typing import List, Dict, Any, Optional, Tuple

from src.models.vector_stores.base_store import BaseVectorStore
from src.utils.logger import get_logger

logger = get_logger(__name__)


class MultiVectorLanceDBStore(BaseVectorStore):
    """LanceDB store for ColPali multi-vector embeddings.

    Thin wrapper around the existing module-level functions in lancedb_manager.py.
    Each document produces multiple vectors (one per patch/token), stored together.
    """

    def __init__(self, model_name: str = "colqwen25"):
        self._model_name = model_name

    def add_documents(
        self,
        session_id: str,
        embeddings: List[Any],
        ids: List[str],
        metadatas: List[Dict[str, Any]],
        texts: Optional[List[str]] = None,
    ) -> bool:
        from src.models.vector_stores.lancedb_manager import add_embeddings_to_lancedb
        return add_embeddings_to_lancedb(
            session_id, self._model_name, embeddings, ids, metadatas
        )

    def query(
        self,
        session_id: str,
        query_embedding: Any = None,
        query_text: Optional[str] = None,
        k: int = 10,
        filter_dict: Optional[Dict] = None,
        similarity_threshold: float = 0.2,
    ) -> Tuple[List[Dict[str, Any]], List[float], List[str]]:
        from src.models.vector_stores.lancedb_manager import query_lancedb
        return query_lancedb(
            session_id, self._model_name, query_embedding, k,
            filter_dict, similarity_threshold
        )

    def delete_session(self, session_id: str) -> bool:
        from src.models.vector_stores.lancedb_manager import clear_lancedb_table
        try:
            return clear_lancedb_table(session_id, self._model_name)
        except Exception as e:
            logger.error(f"Error deleting LanceDB session {session_id}: {e}")
            return False

    def has_documents(self, session_id: str) -> bool:
        from src.models.vector_stores.lancedb_manager import has_documents
        return has_documents(session_id, self._model_name)
