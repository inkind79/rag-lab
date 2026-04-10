"""
Null Embedding Adapter

Placeholder adapter for retrieval methods that don't use embeddings (e.g., BM25).
All encoding methods return None; load/unload are no-ops.
"""

from src.models.embedding_adapters.base_adapter import BaseEmbeddingAdapter, EmbeddingType


class NullEmbeddingAdapter(BaseEmbeddingAdapter):
    """Adapter for retrievers that don't need embeddings (BM25, keyword search)."""

    @property
    def embedding_type(self) -> EmbeddingType:
        return EmbeddingType.NONE

    @property
    def model_name(self) -> str:
        return "none"

    @property
    def dimension(self) -> int:
        return 0

    def load(self) -> None:
        pass

    def encode_query(self, query: str) -> None:
        return None

    def encode_text(self, text: str) -> None:
        return None

    def unload(self) -> None:
        pass
