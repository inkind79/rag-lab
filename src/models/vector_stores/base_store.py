"""
Base Vector Store Interface

Defines the abstract interface for all document storage backends.
Implementations include multi-vector LanceDB (ColPali), single-vector LanceDB
(dense bi-encoders), and BM25 inverted indexes.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple


class BaseVectorStore(ABC):
    """Abstract interface for document storage backends.

    All stores support adding documents and querying by embedding or text.
    The query signature returns (metadatas, scores, ids) matching the existing
    LanceDB interface for backward compatibility.
    """

    @abstractmethod
    def add_documents(
        self,
        session_id: str,
        embeddings: List[Any],
        ids: List[str],
        metadatas: List[Dict[str, Any]],
        texts: Optional[List[str]] = None,
    ) -> bool:
        """Add documents to the store.

        Args:
            session_id: Session identifier for isolation
            embeddings: List of embedding arrays (numpy/torch). None entries for BM25.
            ids: Unique document/chunk identifiers
            metadatas: Per-document metadata dicts
            texts: Raw text content (required for BM25, optional for vector stores)

        Returns:
            True if successful
        """
        ...

    @abstractmethod
    def query(
        self,
        session_id: str,
        query_embedding: Any = None,
        query_text: Optional[str] = None,
        k: int = 10,
        filter_dict: Optional[Dict] = None,
        similarity_threshold: float = 0.0,
    ) -> Tuple[List[Dict[str, Any]], List[float], List[str]]:
        """Query the store for relevant documents.

        Args:
            session_id: Session identifier
            query_embedding: Query embedding array (None for BM25)
            query_text: Raw query text (required for BM25)
            k: Maximum number of results
            filter_dict: Metadata filters (e.g., {'filename': [...]})
            similarity_threshold: Minimum score threshold

        Returns:
            Tuple of (metadatas_list, scores, ids)
        """
        ...

    @abstractmethod
    def delete_session(self, session_id: str) -> bool:
        """Delete all data for a session.

        Returns:
            True if successful
        """
        ...

    @abstractmethod
    def has_documents(self, session_id: str) -> bool:
        """Check if a session has any indexed documents."""
        ...
