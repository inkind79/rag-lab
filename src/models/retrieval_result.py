"""
Unified Retrieval Result Type for RAG Lab

Provides a single result type that bridges visual retrieval (page-level images from ColPali)
and text retrieval (text chunks from BM25, dense bi-encoders, etc.).
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum


class ResultType(Enum):
    """Type of retrieval result."""
    IMAGE = "image"           # Page-level image (ColPali visual retrieval)
    TEXT_CHUNK = "text"       # Text chunk (BM25, dense bi-encoder)
    HYBRID = "hybrid"         # Has both image and text content


@dataclass
class RetrievalResult:
    """Unified result from any retrieval method.

    Carries either image paths (visual retrieval) or text chunks (text retrieval),
    or both. Downstream consumers use result_type to determine rendering.
    """
    result_type: ResultType
    score: float

    # Image fields (ColPali / visual retrieval)
    image_path: Optional[str] = None
    original_filename: Optional[str] = None
    page_num: Optional[int] = None

    # Text fields (BM25 / dense retrieval)
    text_content: Optional[str] = None
    chunk_id: Optional[str] = None
    source_document: Optional[str] = None
    chunk_metadata: Dict[str, Any] = field(default_factory=dict)

    # Shared
    retriever_name: str = ""

    def to_legacy_dict(self) -> Dict[str, Any]:
        """Convert to the existing dict format for backward compatibility.

        Returns the {'path': ..., 'original_filename': ..., 'score': ...} format
        that current code expects.
        """
        return {
            'path': self.image_path or '',
            'original_filename': self.original_filename or self.source_document or '',
            'score': self.score,
            'page_num': self.page_num if self.page_num is not None else 0,
            'result_type': self.result_type.value,
            'text_content': self.text_content,
            'chunk_id': self.chunk_id,
            'retriever_name': self.retriever_name,
        }

    @staticmethod
    def from_legacy_dict(d: Dict[str, Any], retriever_name: str = "colpali") -> 'RetrievalResult':
        """Create a RetrievalResult from the existing legacy dict format.

        Args:
            d: Dict with keys like 'path', 'original_filename', 'score', 'page_num'
            retriever_name: Name of the retriever that produced this result
        """
        return RetrievalResult(
            result_type=ResultType.IMAGE,
            score=d.get('score', 0.0),
            image_path=d.get('path', ''),
            original_filename=d.get('original_filename', ''),
            page_num=d.get('page_num', 0),
            retriever_name=retriever_name,
        )
