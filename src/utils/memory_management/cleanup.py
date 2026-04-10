"""
Memory cleanup utilities for RAG Lab.

This module provides functions to clean up memory after RAG operations.
"""

import gc
import torch
import logging
import os
import psutil
import time

from src.utils.logger import get_logger
from src.utils.memory_management.diagnostics import log_memory_report

logger = get_logger(__name__)

def cleanup_after_response(session_id=None, user_id=None, model_name=None, force_aggressive=True):
    """
    Perform memory cleanup after a response has been generated.

    This function:
    1. Clears LanceDB resources
    2. Empties GPU cache if available
    3. Forces garbage collection
    4. Reports memory usage before and after cleanup

    Args:
        session_id: Optional session ID for targeted cleanup
        user_id: Optional user ID to clean up user-specific caches
        model_name: Optional model name (e.g., "colqwen25") for targeted LanceDB cleanup

    Returns:
        dict: Memory usage statistics
    """
    # Track memory usage and run diagnostics
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / (1024 * 1024)  # MB
    gpu_mem_before = 0
    if torch.cuda.is_available():
        gpu_mem_before = torch.cuda.memory_allocated() / (1024 * 1024)  # MB

    logger.debug(f"Starting memory cleanup. Before cleanup: RAM={mem_before:.2f}MB, GPU={gpu_mem_before:.2f}MB")

    # Log detailed memory report before cleanup if debug is enabled
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("Memory report before cleanup:")
        log_memory_report()

    # Clear LanceDB resources
    try:
        from src.models.vector_stores.lancedb_manager import clear_lancedb_resources
        success = clear_lancedb_resources(session_id)
        logger.debug(f"LanceDB resources cleaned up: {success}")
    except (ImportError, ModuleNotFoundError):
        logger.debug("[MEMORY] LanceDB cleanup module not available")
    except Exception as e:
        logger.warning(f"[MEMORY] Error clearing LanceDB resources: {e}")

    # Import other caches that might need clearing
    # These are typically module-level variables that store state
    try:
        import sys
        from src.models.ocr_rag_retriever import ocr_rag_retriever
        ocr_cache_cleared = False

        # Try to reset any module-level caches in the ocr_rag_retriever
        if hasattr(ocr_rag_retriever, '_last_rag_results'):
            # Clear all session results (it's now a dict)
            ocr_rag_retriever._last_rag_results.clear()
            ocr_cache_cleared = True
            logger.debug("Cleared ocr_rag_retriever._last_rag_results cache for all sessions")

        if hasattr(ocr_rag_retriever, '_latest_ocr_results'):
            ocr_rag_retriever._latest_ocr_results = None
            ocr_cache_cleared = True
            logger.debug("Cleared ocr_rag_retriever._latest_ocr_results cache")

        logger.debug(f"OCR RAG retriever cache cleared: {ocr_cache_cleared}")
    except Exception as e:
        logger.error(f"Error clearing OCR retriever caches: {e}")

    # Clear any embeddings in the global namespace
    if force_aggressive:
        try:
            # Find and clear any large variables in the global namespace that might be holding tensors
            import sys
            import numpy as np

            # Get modules that might contain tensors to clear
            modules_to_check = []
            for module_name, module in sys.modules.items():
                if module_name.startswith('models.') or module_name.startswith('services.'):
                    modules_to_check.append((module_name, module))

            tensors_cleared = 0
            large_lists_cleared = 0
            large_dicts_cleared = 0

            for module_name, module in modules_to_check:
                for attr_name in dir(module):
                    if attr_name.startswith('_'): continue  # Skip private attributes

                    try:
                        attr = getattr(module, attr_name)

                        # Check if it's a tensor
                        if isinstance(attr, (torch.Tensor, np.ndarray)) and sys.getsizeof(attr) > 1000000:  # > 1MB
                            setattr(module, attr_name, None)
                            tensors_cleared += 1

                        # Check if it's a large list or dictionary
                        elif isinstance(attr, (list, dict)) and sys.getsizeof(attr) > 1000000:  # > 1MB
                            if isinstance(attr, list):
                                large_lists_cleared += 1
                                setattr(module, attr_name, [])
                            else:  # dict
                                large_dicts_cleared += 1
                                setattr(module, attr_name, {})
                    except:
                        # Ignore errors from accessing attributes
                        pass

            logger.debug(f"Aggressive memory cleanup - cleared {tensors_cleared} tensors, {large_lists_cleared} large lists, and {large_dicts_cleared} large dictionaries")
        except Exception as e:
            logger.error(f"Error during aggressive memory cleanup: {e}")

    # Skip clearing RAG model cache for user to preserve model caching
    # (This was previously clearing cached models, defeating our caching optimization)
    if user_id:
        logger.debug(f"Preserving RAG model cache for user {user_id} to maintain performance")

    # Add aggressive memory cleanup that was previously happening in load_embedding_model
    # This is critical for clearing retrieval-related memory between chats
    if force_aggressive:
        try:
            # Import memory manager and run aggressive cleanup (but preserve model cache)
            from src.models.memory.memory_manager import memory_manager
            logger.debug("Running aggressive cleanup to clear retrieval-related memory")
            
            # Clear global model_loader caches but preserve the actual model instances
            try:
                import importlib
                model_loader = importlib.import_module('models.model_loader')
                
                # Clear embedding and processor caches (these can accumulate retrieval data)
                if hasattr(model_loader, '_global_embedding_models'):
                    # Don't clear _global_embedding_models since we want to preserve embedding model cache
                    pass  
                
                if hasattr(model_loader, '_processor_cache'):
                    model_loader._processor_cache = {}
                    logger.debug("Cleared model_loader._processor_cache during response cleanup")
                    
            except Exception as e:
                logger.warning(f"Error clearing model_loader caches during response cleanup: {e}")
                
            # Clear LanceDB query caches without touching the model cache
            try:
                # Clear any cached query results or embeddings in LanceDB manager
                from src.models.vector_stores.lancedb_manager import clear_query_cache
                clear_query_cache()
                logger.debug("Cleared LanceDB query cache during response cleanup")
            except (ImportError, AttributeError):
                logger.debug("LanceDB query cache clearing not available")
            except Exception as e:
                logger.warning(f"Error clearing LanceDB query cache: {e}")
                
            # Use memory manager's aggressive cleanup but preserve model instances
            # Run garbage collection first to prepare for effective cleanup
            gc.collect()
            
            # Clear CUDA cache multiple times for better memory release
            if torch.cuda.is_available():
                for _ in range(5):  # Increased iterations for more thorough cleanup
                    torch.cuda.empty_cache()
                    
            # Clear search results cache that can accumulate
            try:
                from src.models.search_results_cache import get_search_cache
                cache = get_search_cache()
                cache.clear_all()
                logger.debug("Cleared search results cache during response cleanup")
            except Exception as e:
                logger.debug(f"Error clearing search results cache: {e}")
                
            # Clear any accumulated embedding vectors or retrieval results
            try:
                # Clear LanceDB manager connection pools more aggressively
                from src.models.vector_stores.lancedb_manager import cleanup_old_connections
                cleanup_old_connections(max_idle_time=0)  # Force close all connections
                logger.debug("Aggressively closed LanceDB connections during response cleanup")
            except Exception as e:
                logger.debug(f"Error during aggressive LanceDB connection cleanup: {e}")
                
            # Clear ColPali adapter internal caches if present
            try:
                # Look for any ColPali adapters in the module cache and clear their internal state
                import sys
                for module_name, module in sys.modules.items():
                    if 'colpali' in module_name.lower() and hasattr(module, '__dict__'):
                        for attr_name in list(module.__dict__.keys()):
                            if attr_name.startswith('_cache') or 'cache' in attr_name.lower():
                                try:
                                    setattr(module, attr_name, {})
                                    logger.debug(f"Cleared cache {attr_name} in {module_name}")
                                except:
                                    pass
            except Exception as e:
                logger.debug(f"Error clearing ColPali caches: {e}")
                    
            # Try malloc_trim on Linux to return memory to OS
            if os.name == 'posix':
                try:
                    import ctypes
                    libc = ctypes.CDLL('libc.so.6')
                    libc.malloc_trim(0)
                    logger.debug("System malloc_trim executed during response cleanup")
                except Exception as e:
                    logger.debug(f"System malloc_trim failed: {e}")
                    
        except Exception as e:
            logger.error(f"Error during aggressive response cleanup: {e}")

    # Force garbage collection - run multiple times to ensure cyclic references are cleaned up
    gc.collect()
    gc.collect()  # Run twice to collect objects that reference each other

    # On some systems, calling the garbage collector with different generations helps
    try:
        gc.collect(0)  # Collect youngest generation
        gc.collect(1)  # Collect middle generation
        gc.collect(2)  # Collect oldest generation
    except:
        # If collecting specific generations fails, we already did the full collection above
        pass

    # Empty CUDA cache if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.debug("CUDA cache emptied")

    # Track memory after cleanup
    time.sleep(0.1)  # Small delay to allow memory to be released
    mem_after = process.memory_info().rss / (1024 * 1024)  # MB
    gpu_mem_after = 0
    if torch.cuda.is_available():
        gpu_mem_after = torch.cuda.memory_allocated() / (1024 * 1024)  # MB

    # Calculate differences
    mem_diff = mem_before - mem_after
    gpu_mem_diff = gpu_mem_before - gpu_mem_after

    # Only log significant memory changes at INFO level
    if mem_diff > 100 or gpu_mem_diff > 100:  # Only log if more than 100MB freed
        logger.info(f"Memory cleanup freed: RAM={mem_diff:.2f}MB, GPU={gpu_mem_diff:.2f}MB")
    else:
        logger.debug(f"Memory cleanup complete. After cleanup: RAM={mem_after:.2f}MB, GPU={gpu_mem_after:.2f}MB")
        logger.debug(f"Memory freed: RAM={mem_diff:.2f}MB, GPU={gpu_mem_diff:.2f}MB")

    # Log detailed memory report after cleanup if debug is enabled
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("Memory report after cleanup:")
        log_memory_report()

    return {
        'before': {'ram': mem_before, 'gpu': gpu_mem_before},
        'after': {'ram': mem_after, 'gpu': gpu_mem_after},
        'freed': {'ram': mem_diff, 'gpu': gpu_mem_diff}
    }


def aggressive_post_response_cleanup(session_id=None, user_id=None, preserve_models=True):
    """
    Very aggressive cleanup for after response generation.
    This replicates the cleanup that was happening during model reloading.
    
    Args:
        session_id: Session ID for context
        user_id: User ID for context  
        preserve_models: If True, preserves cached models for performance
    
    Returns:
        dict: Memory usage statistics
    """
    # Track memory usage
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / (1024 * 1024)  # MB
    gpu_mem_before = 0
    if torch.cuda.is_available():
        gpu_mem_before = torch.cuda.memory_allocated() / (1024 * 1024)  # MB

    logger.info(f"Starting aggressive post-response cleanup. Before: RAM={mem_before:.2f}MB, GPU={gpu_mem_before:.2f}MB")

    try:
        # Use the memory manager's original aggressive cleanup behavior
        # but with model preservation if requested
        from src.models.memory.memory_manager import memory_manager
        
        if preserve_models:
            # Temporarily store current model state
            saved_model_cache = memory_manager.model_cache.copy() if memory_manager.model_cache else {}
            saved_current_model = memory_manager.current_model
            
        # Run the memory manager's full cleanup (this is what was working before)
        logger.info("Running memory_manager aggressive cleanup")
        memory_manager.aggressive_cleanup()
        
        if preserve_models:
            # Restore model state after cleanup
            memory_manager.model_cache = saved_model_cache
            memory_manager.current_model = saved_current_model
            logger.debug(f"Restored {len(saved_model_cache)} cached models after aggressive cleanup")
            
        # Additional cleanup that wasn't in the original but might help
        # Multiple rounds of garbage collection
        for i in range(5):
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        # System-level memory return
        if os.name == 'posix':
            try:
                import ctypes
                libc = ctypes.CDLL('libc.so.6')
                libc.malloc_trim(0)
                logger.debug("System malloc_trim executed")
            except Exception as e:
                logger.debug(f"malloc_trim failed: {e}")
                
    except Exception as e:
        logger.error(f"Error during aggressive post-response cleanup: {e}")

    # Track memory after cleanup
    time.sleep(0.2)  # Allow time for memory to be released
    mem_after = process.memory_info().rss / (1024 * 1024)  # MB
    gpu_mem_after = 0
    if torch.cuda.is_available():
        gpu_mem_after = torch.cuda.memory_allocated() / (1024 * 1024)  # MB

    # Calculate differences
    mem_diff = mem_before - mem_after
    gpu_mem_diff = gpu_mem_before - gpu_mem_after

    logger.info(f"Aggressive post-response cleanup complete. After: RAM={mem_after:.2f}MB, GPU={gpu_mem_after:.2f}MB")
    logger.info(f"Aggressive cleanup freed: RAM={mem_diff:.2f}MB, GPU={gpu_mem_diff:.2f}MB")

    return {
        'before': {'ram': mem_before, 'gpu': gpu_mem_before},
        'after': {'ram': mem_after, 'gpu': gpu_mem_after},
        'freed': {'ram': mem_diff, 'gpu': gpu_mem_diff}
    }


def get_current_memory_usage():
    """
    Get the current memory usage.

    Returns:
        dict: Memory usage statistics
    """
    process = psutil.Process(os.getpid())
    ram_usage = process.memory_info().rss / (1024 * 1024)  # MB
    ram_percent = process.memory_percent()

    gpu_usage = 0
    gpu_percent = 0
    if torch.cuda.is_available():
        gpu_usage = torch.cuda.memory_allocated() / (1024 * 1024)  # MB
        gpu_total = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
        gpu_percent = (gpu_usage / gpu_total) * 100 if gpu_total > 0 else 0

    return {
        'ram': {'usage_mb': ram_usage, 'percent': ram_percent},
        'gpu': {'usage_mb': gpu_usage, 'percent': gpu_percent},
    }



def aggressive_memory_cleanup():
    """
    Aggressive memory cleanup function for high memory usage situations.
    This is called when memory usage exceeds thresholds.
    """
    logger.info("Running aggressive memory cleanup due to high memory usage")

    try:
        # Use the aggressive post-response cleanup with model preservation
        result = aggressive_post_response_cleanup(preserve_models=True)
        logger.info(f"Aggressive cleanup freed: RAM={result['freed']['ram']:.2f}MB, GPU={result['freed']['gpu']:.2f}MB")
        return result
    except Exception as e:
        logger.error(f"Error during aggressive memory cleanup: {e}")
        # Fallback to basic cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return None


def enforce_connection_limits():
    """Enforce strict limits on all connection pools"""
    try:
        # LanceDB connections
        from src.models.vector_stores.lancedb_manager import cleanup_old_connections
        cleaned = cleanup_old_connections(max_idle_time=60)  # 1 minute timeout
        if cleaned:
            logger.info(f"Enforced LanceDB connection limit, cleaned {len(cleaned)} connections")

        # Clear any other connection pools
        import gc
        gc.collect()

    except Exception as e:
        logger.error(f"Error enforcing connection limits: {e}")
