"""
Memory cleanup handler for RAG Lab.

This module provides targeted memory cleanup functions to prevent memory
accumulation when switching between sessions or after document processing.
"""

import gc
import torch
import psutil
import os
from typing import Optional, Dict, Any
from src.utils.logger import get_logger

logger = get_logger(__name__)


def cleanup_embedding_model_after_indexing(session_id: str, rag_models_dict: Dict[str, Any]) -> None:
    """
    Clean up embedding model after document indexing is complete.
    
    This function removes the embedding model from memory after indexing
    to prevent memory accumulation when multiple sessions upload documents.
    
    Args:
        session_id: The session ID whose model should be cleaned up
        rag_models_dict: The global RAG models dictionary
    """
    try:
        # Log initial state
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / (1024 * 1024 * 1024)  # GB
        logger.info(f"Cleanup embedding model for session {session_id}. Memory before: {mem_before:.2f}GB")
        
        # Remove the model from the dictionary
        if session_id in rag_models_dict:
            model = rag_models_dict.pop(session_id, None)
            if model is not None:
                # Try to explicitly delete the model
                try:
                    # Move to CPU first if it has a 'to' method
                    if hasattr(model, 'model') and hasattr(model.model, 'to'):
                        model.model.to('cpu')
                    # Clear any cached data
                    if hasattr(model, 'clear_cache'):
                        model.clear_cache()
                    # Delete the model
                    del model
                    logger.info(f"Successfully removed embedding model for session {session_id}")
                except Exception as e:
                    logger.warning(f"Error during model cleanup for session {session_id}: {e}")
        
        # Force garbage collection
        gc.collect()
        
        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Log memory after cleanup
        mem_after = process.memory_info().rss / (1024 * 1024 * 1024)  # GB
        mem_freed = mem_before - mem_after
        logger.info(f"Memory after cleanup: {mem_after:.2f}GB. Freed: {mem_freed:.2f}GB")
        
    except Exception as e:
        logger.error(f"Error in cleanup_embedding_model_after_indexing: {e}")


def cleanup_inactive_session_models(
    active_session_id: str, 
    rag_models_dict: Dict[str, Any],
    max_models: int = 1
) -> None:
    """
    Clean up models from inactive sessions to prevent memory accumulation.
    
    This function ensures only the active session's model (if any) remains in memory.
    
    Args:
        active_session_id: The currently active session ID
        rag_models_dict: The global RAG models dictionary
        max_models: Maximum number of models to keep (default: 1)
    """
    try:
        # Get list of sessions with models
        sessions_with_models = list(rag_models_dict.keys())
        
        if len(sessions_with_models) <= max_models:
            return  # Nothing to clean up
        
        logger.info(f"Cleaning up inactive models. Active session: {active_session_id}, "
                   f"Total models in memory: {len(sessions_with_models)}")
        
        # Remove all models except the active session
        for session_id in sessions_with_models:
            if session_id != active_session_id:
                cleanup_embedding_model_after_indexing(session_id, rag_models_dict)
        
    except Exception as e:
        logger.error(f"Error in cleanup_inactive_session_models: {e}")


def force_lancedb_cleanup(session_id: Optional[str] = None) -> None:
    """
    Force cleanup of LanceDB connections to free memory.
    
    Args:
        session_id: Specific session to clean up, or None for all idle connections
    """
    try:
        from src.models.vector_stores.lancedb_manager import cleanup_old_connections, _close_lancedb_connection
        
        if session_id:
            # Clean up specific session
            _close_lancedb_connection(session_id)
            logger.debug(f"Closed LanceDB connection for session {session_id}")
        else:
            # Clean up all idle connections (older than 2 minutes)
            closed = cleanup_old_connections(max_idle_time=120)
            logger.info(f"Closed {len(closed)} idle LanceDB connections")
            
    except Exception as e:
        logger.error(f"Error in force_lancedb_cleanup: {e}")


def comprehensive_memory_cleanup(
    active_session_id: Optional[str] = None,
    rag_models_dict: Optional[Dict[str, Any]] = None
) -> Dict[str, float]:
    """
    Perform comprehensive memory cleanup across all components.
    
    Args:
        active_session_id: The currently active session ID (if any)
        rag_models_dict: The global RAG models dictionary (if available)
        
    Returns:
        Dictionary with memory statistics
    """
    try:
        # Track initial memory
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / (1024 * 1024 * 1024)  # GB
        gpu_before = 0
        if torch.cuda.is_available():
            gpu_before = torch.cuda.memory_allocated() / (1024 * 1024 * 1024)  # GB
        
        logger.info(f"Starting comprehensive cleanup. Memory: RAM={mem_before:.2f}GB, GPU={gpu_before:.2f}GB")
        
        # 1. Clean up inactive session models
        if rag_models_dict is not None:
            cleanup_inactive_session_models(active_session_id or "", rag_models_dict)
        
        # 2. Clean up LanceDB connections
        force_lancedb_cleanup()
        
        # 3. Clear model caches
        try:
            from src.models.model_loader import clear_model_caches
            clear_model_caches(force_gpu_cleanup=True)
        except Exception as e:
            logger.warning(f"Error clearing model caches: {e}")
        
        # 4. Memory manager cleanup
        try:
            from src.models.memory.memory_manager import memory_manager
            memory_manager.aggressive_cleanup()
        except Exception as e:
            logger.warning(f"Error in memory manager cleanup: {e}")
        
        # 5. Force multiple GC passes
        for _ in range(3):
            gc.collect()
        
        # 6. Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 7. Try to release memory to OS
        try:
            import ctypes
            libc = ctypes.CDLL('libc.so.6')
            libc.malloc_trim(0)
        except:
            pass
        
        # Track final memory
        mem_after = process.memory_info().rss / (1024 * 1024 * 1024)  # GB
        gpu_after = 0
        if torch.cuda.is_available():
            gpu_after = torch.cuda.memory_allocated() / (1024 * 1024 * 1024)  # GB
        
        mem_freed = mem_before - mem_after
        gpu_freed = gpu_before - gpu_after
        
        logger.info(f"Cleanup complete. Memory: RAM={mem_after:.2f}GB (freed {mem_freed:.2f}GB), "
                   f"GPU={gpu_after:.2f}GB (freed {gpu_freed:.2f}GB)")
        
        return {
            'ram_before_gb': mem_before,
            'ram_after_gb': mem_after,
            'ram_freed_gb': mem_freed,
            'gpu_before_gb': gpu_before,
            'gpu_after_gb': gpu_after,
            'gpu_freed_gb': gpu_freed
        }
        
    except Exception as e:
        logger.error(f"Error in comprehensive_memory_cleanup: {e}")
        return {}