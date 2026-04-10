"""
Embedding Adapters for RAG Lab

Provides a unified interface for different embedding models:
- ColPaliAdapter: Multi-vector visual embeddings (ColQwen2.5, ColNomic)
- DenseEmbeddingAdapter: Single-vector text embeddings (BGE-M3, E5)
- NullEmbeddingAdapter: Placeholder for methods that don't need embeddings (BM25)
"""

from src.models.embedding_adapters.base_adapter import (
    BaseEmbeddingAdapter,
    EmbeddingType,
)

__all__ = ['BaseEmbeddingAdapter', 'EmbeddingType']
