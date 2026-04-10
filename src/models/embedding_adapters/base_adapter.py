"""
Base Embedding Adapter Interface

Defines the abstract interface that all embedding adapters must implement.
This allows ColPali (multi-vector), dense bi-encoders (single-vector),
and BM25 (no embeddings) to share a common interface.
"""

from abc import ABC, abstractmethod
from typing import Any, List
from enum import Enum


class EmbeddingType(Enum):
    """Type of embeddings produced by an adapter."""
    MULTI_VECTOR = "multi_vector"     # ColPali: [seq_len, dim] per document
    SINGLE_VECTOR = "single_vector"   # BGE-M3: [dim] per document
    NONE = "none"                     # BM25: no embeddings


class BaseEmbeddingAdapter(ABC):
    """Abstract base for all embedding models.

    Subclasses handle model loading, query/document encoding, and resource cleanup.
    The embedding_type property tells consumers what shape of output to expect.
    """

    @property
    @abstractmethod
    def embedding_type(self) -> EmbeddingType:
        """What kind of embeddings does this adapter produce?"""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Human-readable model identifier."""
        ...

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Embedding dimension. 0 for NONE type."""
        ...

    @abstractmethod
    def load(self) -> None:
        """Load the model into memory (GPU/CPU)."""
        ...

    @abstractmethod
    def encode_query(self, query: str) -> Any:
        """Encode a text query. Return type depends on embedding_type.

        - MULTI_VECTOR: torch.Tensor of shape [seq_len, dim]
        - SINGLE_VECTOR: numpy.ndarray of shape [dim]
        - NONE: None
        """
        ...

    def encode_text(self, text: str) -> Any:
        """Encode text for indexing. Default: same as encode_query.

        Override if query encoding differs from document encoding
        (e.g., asymmetric models).
        """
        return self.encode_query(text)

    def encode_image(self, images: List) -> Any:
        """Encode images. Only meaningful for visual models.

        Args:
            images: List of PIL.Image objects

        Raises:
            NotImplementedError: If the adapter doesn't support image encoding
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support image encoding"
        )

    def unload(self) -> None:
        """Release model resources (GPU memory, etc.). Override for cleanup."""
        pass
