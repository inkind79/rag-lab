"""
OCR cache utilities for token budget filtering.

This module provides functions to access OCR results for token counting.
"""
import os
import json
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

def get_ocr_cache_for_session(session_id: str) -> Dict[str, Any]:
    """
    Get the OCR cache for a session.
    
    Args:
        session_id: Session ID
        
    Returns:
        Dictionary mapping document paths to OCR results
    """
    ocr_cache = {}
    
    # Check if OCR cache directory exists
    ocr_cache_dir = os.path.join('ocr_cache', session_id)
    if not os.path.exists(ocr_cache_dir):
        return ocr_cache
    
    # Load OCR results from cache files
    try:
        for filename in os.listdir(ocr_cache_dir):
            if filename.endswith('.json'):
                cache_file_path = os.path.join(ocr_cache_dir, filename)
                try:
                    with open(cache_file_path, 'r', encoding='utf-8') as f:
                        cache_data = json.load(f)
                        
                    # Extract document path and OCR text
                    if 'path' in cache_data and 'text' in cache_data:
                        doc_path = cache_data['path']
                        ocr_cache[doc_path] = cache_data
                except Exception as e:
                    logger.warning(f"Error loading OCR cache file {cache_file_path}: {e}")
    except Exception as e:
        logger.warning(f"Error accessing OCR cache directory for session {session_id}: {e}")
    
    logger.info(f"Loaded OCR cache for session {session_id} with {len(ocr_cache)} entries")
    return ocr_cache
