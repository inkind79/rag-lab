"""
Document processor service for handling document uploads and indexing.

This service is responsible for processing uploaded documents,
indexing them for retrieval, and managing document metadata.
"""
from .processor import (
    process_uploaded_files,
    index_documents_for_rag,
    index_documents,
    build_document_metadata
)

__all__ = [
    'process_uploaded_files',
    'index_documents_for_rag',
    'index_documents',
    'build_document_metadata'
]
