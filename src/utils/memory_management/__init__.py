"""
Memory management utilities for RAG Lab.

This module provides functions for memory cleanup and management,
including process isolation to prevent memory leaks.
"""

# Import cleanup modules
from .minimal_cleanup import minimal_cleanup
from .cleanup_lancedb_resources import cleanup_lancedb_resources
from .memory_logger import log_memory_usage, log_memory_comparison

# Try to import process isolation modules
try:
    # Import the process isolation manager
    from .process_manager import get_process_manager

    # Import the isolated RAG functionality
    from .isolated_rag import (
        get_isolated_rag_manager,
        generate_embeddings,
        add_embeddings_to_lancedb,
        query_lancedb,
        cleanup_session
    )

    # Flag indicating that process isolation is available
    PROCESS_ISOLATION_AVAILABLE = True

except ImportError as e:
    # Fallback if imports fail
    import logging
    logging.getLogger(__name__).warning(f"Process isolation not available: {e}")
    PROCESS_ISOLATION_AVAILABLE = False

# For backward compatibility, provide the cleanup_after_response function
# that matches the expected interface, but enhanced with process isolation
def cleanup_after_response(session_id=None, user_id=None, model_name=None, force_aggressive=False):
    """
    Memory cleanup function with process isolation support.

    This function performs memory cleanup after a response has been generated.
    When process isolation is available, it uses the isolated cleanup functionality.
    Otherwise, it falls back to the original cleanup implementation.

    Args:
        session_id: Optional session ID for targeted cleanup
        user_id: Optional user ID to clean up user-specific caches
        model_name: Optional model name for targeted cleanup
        force_aggressive: Whether to perform aggressive cleanup

    Returns:
        dict: Memory usage statistics
    """
    import gc
    import torch
    import logging

    logger = logging.getLogger(__name__)

    # Log memory before cleanup with dedicated memory logger
    memory_before = log_memory_usage("Before response cleanup", session_id)

    # Check if process isolation is available
    if PROCESS_ISOLATION_AVAILABLE and session_id:
        try:
            logger.info(f"[MEMORY] Using process isolation for cleanup of session {session_id}")

            # Use the isolated cleanup functionality
            from .isolated_rag import cleanup_session
            cleanup_session(session_id)

            # Additionally, run the process manager to clean up any stray processes
            from .process_manager import get_process_manager
            process_manager = get_process_manager()
            process_manager._cleanup_old_processes()

        except Exception as e:
            logger.warning(f"[MEMORY] Error using process isolation for cleanup: {e}")
            logger.warning("[MEMORY] Falling back to standard cleanup")

            # Fall back to standard cleanup
            minimal_cleanup()
            cleanup_lancedb_resources(session_id, model_name)
    else:
        # Process isolation not available, use standard cleanup

        # Perform the minimal cleanup
        minimal_cleanup()

        # Perform LanceDB-specific cleanup to prevent memory stacking
        logger.info(f"[MEMORY] Performing LanceDB cleanup for session {session_id}")

        # First try the standard cleanup
        cleanup_lancedb_resources(session_id, model_name)

        # Then use our enhanced connection pooling cleanup
        try:
            # Import the destroy function directly for this specific session
            from src.models.vector_stores.lancedb_manager import destroy_lancedb_resources

            # Force close the connection for this session
            destroy_result = destroy_lancedb_resources(session_id, preserve_active_sessions=True)
            logger.info(f"[MEMORY] Forced LanceDB connection closure: {destroy_result}")

            # Also clean up any old connections that haven't been used in a while
            from src.models.vector_stores.lancedb_manager import cleanup_old_connections
            cleaned_sessions = cleanup_old_connections(max_idle_time=120)
            if cleaned_sessions:
                logger.info(f"[MEMORY] Cleaned up {len(cleaned_sessions)} idle LanceDB connections")
        except Exception as destroy_e:
            logger.warning(f"[MEMORY] Error during forced LanceDB connection closure: {destroy_e}")

    # Force release memory to the OS
    try:
        from .periodic_cleanup import force_release_memory
        force_release_memory()
    except Exception as e:
        logger.warning(f"[MEMORY] Error during force_release_memory: {e}")

    # Force garbage collection
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Try to release memory back to the OS on Linux
    try:
        import ctypes
        libc = ctypes.CDLL('libc.so.6')
        libc.malloc_trim(0)
        logger.debug("[MEMORY] Released memory back to OS with malloc_trim")
    except Exception:
        pass

    # Log memory after cleanup with dedicated memory logger
    memory_after = log_memory_usage("After response cleanup", session_id)

    # Log comparison
    cleanup_description = f"Full response cleanup for session {session_id}"
    memory_diff = log_memory_comparison(memory_before, memory_after, cleanup_description, session_id)

    # Return stats in expected format (for backward compatibility)
    return {
        'before': {
            'ram': memory_before["ram_usage_gb"] * 1024,  # Convert to MB
            'gpu': memory_before.get("gpu_usage_gb", 0) * 1024 if memory_before.get("gpu_usage_gb") else 0
        },
        'after': {
            'ram': memory_after["ram_usage_gb"] * 1024,  # Convert to MB
            'gpu': memory_after.get("gpu_usage_gb", 0) * 1024 if memory_after.get("gpu_usage_gb") else 0
        },
        'freed': {
            'ram': memory_diff["ram_diff"] * 1024,  # Convert to MB
            'gpu': memory_diff.get("gpu_diff", 0) * 1024 if memory_diff.get("gpu_diff") else 0
        }
    }

__all__ = [
    # Core cleanup functions
    'cleanup_after_response',
    'minimal_cleanup',
    'cleanup_lancedb_resources',

    # Memory logging
    'log_memory_usage',
    'log_memory_comparison',

    # Process isolation (if available)
    'PROCESS_ISOLATION_AVAILABLE'
]

# Add process isolation functions if available
if PROCESS_ISOLATION_AVAILABLE:
    __all__.extend([
        'get_process_manager',
        'get_isolated_rag_manager',
        'generate_embeddings',
        'add_embeddings_to_lancedb',
        'query_lancedb',
        'cleanup_session'
    ])
