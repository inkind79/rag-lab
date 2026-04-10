"""
Simple memory cleanup utilities for RAG Lab.

This module provides direct memory cleanup functions without external dependencies.
"""

import gc
import os
import torch
import psutil
import logging

# Configure logger directly
logger = logging.getLogger("memory_cleanup")

def clear_memory(session_id=None):
    """
    Simple function to clear memory after RAG processing.

    Args:
        session_id: Optional session ID for targeted LanceDB resource cleanup

    Returns:
        dict: Memory usage before and after cleanup
    """
    # Track memory usage
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / (1024 * 1024)  # MB
    gpu_before = 0
    if torch.cuda.is_available():
        gpu_before = torch.cuda.memory_allocated() / (1024 * 1024)  # MB

    logger.info(f"[MEMORY] Starting cleanup for {session_id or 'all sessions'}")
    logger.info(f"[MEMORY] Before cleanup: RAM={mem_before:.2f}MB, GPU={gpu_before:.2f}MB")

    # Clear LanceDB resources for the session if provided
    if session_id:
        try:
            # Import here to avoid circular imports
            from src.models.vector_stores.lancedb_manager import _lancedb_connections

            # Try to clear specific session LanceDB connection
            if session_id in _lancedb_connections:
                del _lancedb_connections[session_id]
                logger.info(f"[MEMORY] Removed LanceDB connection for session {session_id} from cache")

            # Also check for any other connections with this session ID
            for key in list(_lancedb_connections.keys()):
                if key.startswith(f"{session_id}_"):
                    del _lancedb_connections[key]
                    logger.info(f"[MEMORY] Removed additional LanceDB connection {key}")

        except Exception as e:
            logger.error(f"[MEMORY] Error clearing LanceDB resources: {e}")

    # Clear OCR RAG retriever caches
    try:
        from src.models.ocr_rag_retriever import ocr_rag_retriever
        if hasattr(ocr_rag_retriever, '_last_rag_results'):
            # Clear all session results (it's now a dict)
            ocr_rag_retriever._last_rag_results.clear()
            logger.info("[MEMORY] Cleared OCR retriever results cache for all sessions")
        if hasattr(ocr_rag_retriever, '_latest_ocr_results'):
            ocr_rag_retriever._latest_ocr_results = None
            logger.info("[MEMORY] Cleared OCR retriever cache")
    except Exception as e:
        logger.error(f"[MEMORY] Error clearing OCR retriever caches: {e}")

    # Run multiple garbage collection cycles
    for i in range(3):
        gc.collect()
    logger.info("[MEMORY] Ran multiple garbage collection cycles")

    # Empty CUDA cache if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("[MEMORY] CUDA cache emptied")

    # Track memory after cleanup
    mem_after = process.memory_info().rss / (1024 * 1024)  # MB
    gpu_after = 0
    if torch.cuda.is_available():
        gpu_after = torch.cuda.memory_allocated() / (1024 * 1024)  # MB

    # Calculate differences
    mem_diff = mem_before - mem_after
    gpu_diff = gpu_before - gpu_after

    logger.info(f"[MEMORY] After cleanup: RAM={mem_after:.2f}MB, GPU={gpu_after:.2f}MB")
    logger.info(f"[MEMORY] Memory freed: RAM={mem_diff:.2f}MB, GPU={gpu_diff:.2f}MB")

    return {
        'before': {'ram': mem_before, 'gpu': gpu_before},
        'after': {'ram': mem_after, 'gpu': gpu_after},
        'freed': {'ram': mem_diff, 'gpu': gpu_diff}
    }
