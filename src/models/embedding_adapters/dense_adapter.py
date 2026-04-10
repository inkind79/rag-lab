"""
Dense Embedding Adapter

Single-vector dense embedding adapter for models like BGE-M3, E5, nomic-embed.
Uses sentence-transformers for encoding.
"""

import numpy as np
from typing import Any

from src.models.embedding_adapters.base_adapter import BaseEmbeddingAdapter, EmbeddingType
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DenseEmbeddingAdapter(BaseEmbeddingAdapter):
    """Single-vector dense embedding adapter.

    Wraps sentence-transformers models to produce normalized
    single-vector embeddings for each query/document.
    """

    def __init__(self, model_name: str = "BAAI/bge-m3"):
        self._model_name = model_name
        self._model = None
        self._dim = 1024  # Will be updated after loading

    @property
    def embedding_type(self) -> EmbeddingType:
        return EmbeddingType.SINGLE_VECTOR

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        return self._dim

    def load(self) -> None:
        """Load the sentence-transformer model."""
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )

        logger.info(f"Loading dense embedding model: {self._model_name}")
        self._model = SentenceTransformer(self._model_name)
        self._dim = self._model.get_sentence_embedding_dimension()
        logger.info(f"Loaded {self._model_name} (dimension={self._dim})")

    def encode_query(self, query: str) -> np.ndarray:
        """Encode a text query to a single normalized vector.

        Args:
            query: Text query string

        Returns:
            numpy array of shape [dim]
        """
        if self._model is None:
            self.load()
        return self._model.encode(query, normalize_embeddings=True)

    def encode_text(self, text: str) -> np.ndarray:
        """Encode text for indexing. Same as encode_query for symmetric models."""
        return self.encode_query(text)

    def encode_image(self, images: list) -> Any:
        raise NotImplementedError(
            f"DenseEmbeddingAdapter ({self._model_name}) does not support image encoding"
        )

    def unload(self) -> None:
        """Release model resources and GPU memory."""
        if self._model is not None:
            del self._model
            self._model = None
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info(f"Unloaded dense embedding model: {self._model_name}")
