"""
Utility functions for cleaning up LanceDB resources.
"""

import os
import gc
import torch
import shutil
import logging
from src.utils.logger import get_logger

logger = get_logger(__name__)



def cleanup_lancedb_resources(session_id=None, model_name=None):
    """
    Clean up LanceDB resources for a specific session and model.

    This function performs aggressive memory cleanup to ensure that LanceDB resources
    are properly released and memory is returned to the system.

    Args:
        session_id: The session ID to clean up. If None, clean up all sessions.
        model_name: The model name to clean up. If None, clean up all models.

    Returns:
        dict: Information about what was cleaned up
    """
    try:
        # Track memory before cleanup
        process = __import__('psutil').Process(os.getpid())
        mem_before = process.memory_info().rss / (1024 * 1024 * 1024)  # GB

        # Import LanceDB manager to access its resources
        from src.models.vector_stores.lancedb_manager import destroy_lancedb_resources

        # Call the destroy function
        cleanup_info = destroy_lancedb_resources(session_id)

        # Force garbage collection multiple times
        for i in range(3):
            gc.collect()

        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Try to release memory back to the system using malloc_trim
        try:
            import ctypes
            libc = ctypes.CDLL('libc.so.6')
            # Call malloc_trim to release memory to the system
            libc.malloc_trim(0)
            logger.info("Called malloc_trim to release memory to the system")
        except Exception as trim_e:
            logger.warning(f"Could not call malloc_trim: {trim_e}")

        # Track memory after cleanup
        mem_after = process.memory_info().rss / (1024 * 1024 * 1024)  # GB
        mem_diff = mem_before - mem_after

        # Add memory info to cleanup_info
        cleanup_info.update({
            "memory_before_gb": round(mem_before, 2),
            "memory_after_gb": round(mem_after, 2),
            "memory_freed_gb": round(mem_diff, 2)
        })

        logger.info(f"Cleaned up LanceDB resources: {cleanup_info}")
        return cleanup_info
    except Exception as e:
        logger.error(f"Error cleaning up LanceDB resources: {e}")
        return {"error": str(e)}
