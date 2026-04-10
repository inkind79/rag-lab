"""
Base Retriever Interface for RAG Lab

This module defines the base interface for all retrievers in the system.
It establishes a consistent API that all specialized retrievers must implement.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
from src.utils.logger import get_logger

logger = get_logger(__name__)

class BaseRetriever(ABC):
    """
    Abstract base class that defines the interface for all document retrievers.

    Retrievers are responsible for finding and returning relevant documents
    based on a user query, optionally filtering by specific files.
    """

    def set_embedding_adapter(self, adapter) -> None:
        """Set the embedding adapter (e.g., ColPaliAdapter, DenseEmbeddingAdapter).

        Override in subclasses that need an embedding model for retrieval.
        Default is a no-op (e.g., BM25 doesn't need embeddings).

        Args:
            adapter: An instance of BaseEmbeddingAdapter
        """
        pass

    @abstractmethod
    def retrieve_documents(
        self,
        query: str,
        session_id: str,
        k: int = 3,
        selected_filenames: Optional[List[str]] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents based on the query.

        Args:
            query: The user query to use for retrieval
            session_id: The current session ID
            k: Maximum number of documents to retrieve
            selected_filenames: Optional list of filenames to filter results
            **kwargs: Method-specific parameters (e.g., use_score_slope, use_ocr)

        Returns:
            A list of dictionaries containing document information:
            {'path': path_to_document, 'original_filename': source_filename, 'score': relevance_score}
        """
        pass

    def normalize_image_path(self, path: str, session_id: str) -> str:
        """
        Ensure image paths follow a consistent format.
        
        Args:
            path: The image path to normalize
            session_id: The current session ID
            
        Returns:
            Normalized path string
        """
        import os
        filename = os.path.basename(path)
        return os.path.join('images', session_id, filename).replace('\\', '/')

    def expand_query(self, query: str, chat_history: List[Dict[str, Any]], session_id: str) -> str:
        """
        Expand a user query by considering conversation history to resolve references.
        
        Args:
            query: The original user query
            chat_history: List of previous messages in the conversation
            session_id: The current session ID
            
        Returns:
            Expanded query with resolved references
        """
        logger.info("Using default query expansion")
        # Default implementation just returns the original query
        return query