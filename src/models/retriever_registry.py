"""
Retriever Registry

Factory pattern for creating and managing retriever instances.
Retrievers are registered by name and instantiated on demand.
"""

import threading
from typing import Dict, Type, Optional, List, Callable, Any

from src.models.retriever_base import BaseRetriever
from src.utils.logger import get_logger

logger = get_logger(__name__)


class RetrieverRegistry:
    """Factory for creating and caching retriever instances by name.

    Thread-safe via a lock guarding _instances mutations.
    """

    _registry: Dict[str, Callable[[], BaseRetriever]] = {}
    _instances: Dict[str, BaseRetriever] = {}
    _lock = threading.RLock()  # Reentrant: hybrid/rrf factories call get_retriever() recursively

    @classmethod
    def register(cls, name: str, factory: Callable[[], BaseRetriever]):
        """Register a retriever factory by name.

        Args:
            name: Retriever identifier (e.g., 'colpali', 'bm25', 'dense')
            factory: Callable that returns a new BaseRetriever instance
        """
        cls._registry[name] = factory
        logger.debug(f"Registered retriever: {name}")

    @classmethod
    def get_retriever(
        cls,
        name: str,
        embedding_adapter: Optional[Any] = None,
    ) -> BaseRetriever:
        """Get or create a retriever instance by name.

        Args:
            name: Retriever identifier
            embedding_adapter: Optional embedding adapter to set on the retriever

        Returns:
            Configured BaseRetriever instance

        Raises:
            ValueError: If the retriever name is not registered
        """
        with cls._lock:
            if name not in cls._instances:
                if name not in cls._registry:
                    available = list(cls._registry.keys())
                    raise ValueError(
                        f"Unknown retriever: '{name}'. Available: {available}"
                    )
                cls._instances[name] = cls._registry[name]()
                logger.info(f"Created retriever instance: {name}")

        instance = cls._instances[name]
        if embedding_adapter is not None:
            instance.set_embedding_adapter(embedding_adapter)
        return instance

    @classmethod
    def available_retrievers(cls) -> List[str]:
        """Return list of registered retriever names."""
        return list(cls._registry.keys())

    @classmethod
    def clear_instances(cls):
        """Clear all cached retriever instances."""
        cls._instances.clear()

    @classmethod
    def clear_all(cls):
        """Clear both registry and instances."""
        cls._registry.clear()
        cls._instances.clear()


def register_default_retrievers():
    """Register the built-in retrievers.

    Called at import time or during app initialization.
    """
    # ColPali visual retriever (existing)
    def _create_colpali():
        from src.models.rag_retriever import RAGRetriever
        return RAGRetriever()

    # BM25 text retriever
    def _create_bm25():
        from src.models.retrievers.bm25_retriever import BM25Retriever
        return BM25Retriever()

    # Dense bi-encoder retriever
    def _create_dense():
        from src.models.retrievers.dense_retriever import DenseRetriever
        return DenseRetriever()

    # RRF hybrid (ColPali + BM25) — uses registry instances for shared state
    def _create_rrf():
        from src.models.retrievers.rrf_retriever import RRFHybridRetriever
        colpali = RetrieverRegistry.get_retriever("colpali")
        bm25 = RetrieverRegistry.get_retriever("bm25")
        return RRFHybridRetriever(retrievers=[colpali, bm25])

    # Weighted hybrid (ColPali + BM25) — normalized score fusion
    def _create_hybrid():
        from src.models.retrievers.hybrid_weighted_retriever import HybridWeightedRetriever
        colpali = RetrieverRegistry.get_retriever("colpali")
        bm25 = RetrieverRegistry.get_retriever("bm25")
        return HybridWeightedRetriever(colpali_retriever=colpali, bm25_retriever=bm25)

    RetrieverRegistry.register("colpali", _create_colpali)
    RetrieverRegistry.register("bm25", _create_bm25)
    RetrieverRegistry.register("dense", _create_dense)
    RetrieverRegistry.register("hybrid_rrf", _create_rrf)
    RetrieverRegistry.register("hybrid", _create_hybrid)


# Auto-register on import
register_default_retrievers()
