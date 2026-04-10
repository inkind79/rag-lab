"""
Text Extraction and Chunking for RAG Lab

Provides PDF text extraction and configurable text chunking for
text-based retrieval methods (BM25, dense bi-encoder).
"""

import os
from typing import List, Tuple, Optional
from dataclasses import dataclass

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class TextChunk:
    """A chunk of extracted text with provenance metadata."""
    text: str
    chunk_id: str
    source_filename: str
    page_num: int
    chunk_index: int
    char_start: int
    char_end: int


def extract_text_from_pdf(file_path: str) -> List[Tuple[int, str]]:
    """Extract text from each page of a PDF.

    Tries PyPDF2 first (fast, no GPU). Falls back to OCR for scanned PDFs
    where PyPDF2 returns empty text.

    Args:
        file_path: Path to the PDF file

    Returns:
        List of (page_num, text) tuples. page_num is 1-indexed.
    """
    pages = []

    try:
        from PyPDF2 import PdfReader
        reader = PdfReader(file_path)
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ''
            pages.append((i + 1, text.strip()))
        logger.info(f"Extracted text from {len(pages)} pages via PyPDF2: {file_path}")
    except ImportError:
        logger.warning("PyPDF2 not installed. Trying pymupdf...")
        try:
            import fitz  # pymupdf
            doc = fitz.open(file_path)
            for i, page in enumerate(doc):
                text = page.get_text() or ''
                pages.append((i + 1, text.strip()))
            doc.close()
            logger.info(f"Extracted text from {len(pages)} pages via pymupdf: {file_path}")
        except ImportError:
            logger.error("Neither PyPDF2 nor pymupdf installed. Cannot extract PDF text.")
            return []

    # Check if extraction yielded mostly empty pages (scanned PDF)
    non_empty = sum(1 for _, t in pages if len(t) > 50)
    if pages and non_empty < len(pages) * 0.3:
        logger.info(f"Only {non_empty}/{len(pages)} pages have text — likely a scanned PDF, consider OCR")

    return pages


def extract_text_from_image(file_path: str) -> str:
    """Extract text from an image file using OCR.

    Args:
        file_path: Path to the image file

    Returns:
        Extracted text string
    """
    try:
        from src.models.ocr.ocr_processor import extract_text_from_image as ocr_extract
        result = ocr_extract(file_path)
        return result or ''
    except Exception as e:
        logger.warning(f"OCR extraction failed for {file_path}: {e}")
        return ''


def chunk_text(
    text: str,
    source_filename: str,
    page_num: int,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    doc_id: int = 0,
) -> List[TextChunk]:
    """Split text into overlapping chunks with metadata.

    Args:
        text: Source text to chunk
        source_filename: Original document filename
        page_num: Page number this text came from
        chunk_size: Maximum characters per chunk
        chunk_overlap: Characters of overlap between consecutive chunks
        doc_id: Document ID for chunk_id generation

    Returns:
        List of TextChunk objects
    """
    if not text or not text.strip():
        return []

    chunks = []
    start = 0
    chunk_index = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))

        # Try to break at a sentence boundary if possible
        if end < len(text):
            # Look back for a period, newline, or other sentence boundary
            for boundary in ['. ', '.\n', '\n\n', '\n', '? ', '! ']:
                boundary_pos = text.rfind(boundary, start + chunk_size // 2, end)
                if boundary_pos > start:
                    end = boundary_pos + len(boundary)
                    break

        chunk_text_str = text[start:end].strip()
        if chunk_text_str:
            chunks.append(TextChunk(
                text=chunk_text_str,
                chunk_id=f"{doc_id}_{page_num}_{chunk_index}",
                source_filename=source_filename,
                page_num=page_num,
                chunk_index=chunk_index,
                char_start=start,
                char_end=end,
            ))
            chunk_index += 1

        # Advance with overlap
        start = end - chunk_overlap if end < len(text) else len(text)

    logger.debug(f"Chunked page {page_num} of {source_filename}: {len(chunks)} chunks")
    return chunks


def extract_and_chunk_document(
    file_path: str,
    filename: str,
    doc_id: int = 0,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> List[TextChunk]:
    """Extract text from a document and split into chunks.

    Handles PDFs and images. For PDFs, extracts text per page then chunks.
    For images, runs OCR then chunks.

    Args:
        file_path: Full path to the document
        filename: Original filename
        doc_id: Document ID
        chunk_size: Characters per chunk
        chunk_overlap: Overlap between chunks

    Returns:
        List of all TextChunk objects from the document
    """
    all_chunks = []

    if filename.lower().endswith('.pdf'):
        pages = extract_text_from_pdf(file_path)
        for page_num, text in pages:
            chunks = chunk_text(text, filename, page_num, chunk_size, chunk_overlap, doc_id)
            all_chunks.extend(chunks)
    else:
        # Image file — try OCR
        text = extract_text_from_image(file_path)
        if text:
            chunks = chunk_text(text, filename, 1, chunk_size, chunk_overlap, doc_id)
            all_chunks.extend(chunks)

    logger.info(f"Extracted {len(all_chunks)} chunks from {filename}")
    return all_chunks
