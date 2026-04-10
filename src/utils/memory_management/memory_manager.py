"""
Memory Manager service for RAG Lab.

This module provides a centralized service for memory management,
with periodic cleanup to prevent memory leaks and excessive RAM usage.
"""

import os
import gc
import time
import threading
import psutil
import torch
from typing import Dict, List, Any, Optional, Callable

from src.utils.logger import get_logger

logger = get_logger(__name__)

class MemoryManager:
    """
    Memory manager service that handles periodic cleanup of resources.
    Designed to run in the background and prevent memory leaks.
    """
    
    def __init__(self, 
                 cleanup_interval: int = 300,  # 5 minutes default
                 lancedb_idle_time: int = 300,  # 5 minutes for LanceDB connections
                 memory_threshold_pct: float = 85.0,  # 85% RAM usage threshold
                 enable_scheduled_cleanup: bool = True):
        """
        Initialize the memory manager.
        
        Args:
            cleanup_interval: Time in seconds between periodic cleanups
            lancedb_idle_time: Time in seconds before a LanceDB connection is considered idle
            memory_threshold_pct: Memory threshold (as percentage) to trigger aggressive cleanup
            enable_scheduled_cleanup: Whether to enable scheduled cleanup
        """
        self.cleanup_interval = cleanup_interval
        self.lancedb_idle_time = lancedb_idle_time
        self.memory_threshold_pct = memory_threshold_pct
        self.enable_scheduled_cleanup = enable_scheduled_cleanup
        
        self.last_cleanup_time = 0
        self.cleanup_in_progress = False
        self.cleanup_lock = threading.RLock()
        
        # Statistics
        self.cleanup_count = 0
        self.emergency_cleanup_count = 0
        self.bytes_freed = 0
        
        # Start the background thread if scheduled cleanup is enabled
        if self.enable_scheduled_cleanup:
            self.start_background_thread()
            logger.debug(f"Memory Manager initialized with {cleanup_interval}s interval, "
                       f"{lancedb_idle_time}s LanceDB idle time, "
                       f"{memory_threshold_pct}% RAM threshold")
        else:
            logger.debug("Memory Manager initialized (scheduled cleanup disabled)")
    
    def start_background_thread(self):
        """Start the background thread for periodic cleanup."""
        thread = threading.Thread(target=self._background_cleanup_thread, daemon=True)
        thread.start()
        logger.debug("Started memory management background thread")
    
    def _background_cleanup_thread(self):
        """Background thread that performs periodic cleanup."""
        while True:
            try:
                # Sleep first to allow application to start
                time.sleep(30)  # Wait 30s before first run
                
                while True:
                    try:
                        # Check if cleanup is needed
                        self.check_and_cleanup()
                        
                        # Check if emergency cleanup is needed
                        self.check_memory_threshold()
                        
                        # Sleep until next check
                        time.sleep(self.cleanup_interval)
                    except Exception as e:
                        logger.error(f"Error in memory management thread: {e}")
                        time.sleep(60)  # Sleep for a minute on error
            except Exception as e:
                logger.error(f"Memory management thread crashed: {e}")
                time.sleep(60)  # Sleep for a minute and try again
    
    def check_and_cleanup(self) -> bool:
        """
        Check if cleanup is needed and perform it if necessary.
        
        Returns:
            True if cleanup was performed, False otherwise
        """
        with self.cleanup_lock:
            # Check if sufficient time has passed since last cleanup
            current_time = time.time()
            if current_time - self.last_cleanup_time < self.cleanup_interval:
                return False
            
            # Prevent concurrent cleanup
            if self.cleanup_in_progress:
                logger.warning("Cleanup already in progress, skipping")
                return False
            
            # Mark cleanup as in progress
            self.cleanup_in_progress = True
        
        try:
            # Measure memory before cleanup
            process = psutil.Process(os.getpid())
            mem_before = process.memory_info().rss
            
            # Perform cleanup
            self._perform_cleanup()
            
            # Measure memory after cleanup
            mem_after = process.memory_info().rss
            bytes_freed = max(0, mem_before - mem_after)  # Ensure non-negative
            self.bytes_freed += bytes_freed
            
            # Update stats
            self.cleanup_count += 1
            self.last_cleanup_time = time.time()
            
            # Only log significant memory cleanup (> 50MB)
            if bytes_freed > (50 * 1024 * 1024):
                logger.info(f"Regular cleanup freed {bytes_freed / (1024 * 1024):.2f} MB")
            else:
                logger.debug(f"Regular cleanup complete. Freed {bytes_freed / (1024 * 1024):.2f} MB. "
                           f"Total cleanups: {self.cleanup_count}")
            return True
        
        except Exception as e:
            logger.error(f"Error during memory cleanup: {e}")
            return False
        
        finally:
            # Mark cleanup as complete
            self.cleanup_in_progress = False
    
    def check_memory_threshold(self) -> bool:
        """
        Check if memory usage exceeds threshold and perform aggressive cleanup.
        
        Returns:
            True if aggressive cleanup was performed, False otherwise
        """
        try:
            # Get system memory info
            mem = psutil.virtual_memory()
            mem_percent = mem.percent
            
            # Skip if memory usage is below threshold
            if mem_percent < self.memory_threshold_pct:
                return False
            
            logger.warning(f"Memory usage ({mem_percent:.1f}%) exceeds threshold "
                          f"({self.memory_threshold_pct:.1f}%). Performing aggressive cleanup.")
            
            # Perform aggressive cleanup
            self._perform_aggressive_cleanup()
            
            # Update stats
            self.emergency_cleanup_count += 1
            
            logger.debug(f"Aggressive cleanup complete. Total emergency cleanups: {self.emergency_cleanup_count}")
            return True
        
        except Exception as e:
            logger.error(f"Error during memory threshold check: {e}")
            return False
    
    def _perform_cleanup(self):
        """Perform standard cleanup of resources."""
        # Clean up LanceDB connections
        self._cleanup_lancedb_connections()
        
        # Clean up memory stores
        self._cleanup_memory_stores()
        
        # Clean up embeddings cache
        self._cleanup_embedding_cache()
        
        # Clean up search results cache
        self._cleanup_search_results_cache()
        
        # General garbage collection
        gc.collect()
        
        # Clear CUDA cache if available
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.debug("Cleared CUDA cache")
        except (ImportError, Exception):
            pass
    
    def _perform_aggressive_cleanup(self):
        """Perform aggressive cleanup when memory usage is high."""
        # Perform standard cleanup first
        self._perform_cleanup()
        
        # More aggressive measures
        
        # Run garbage collection multiple times targeting different generations
        gc.collect()
        try:
            gc.collect(0)  # Young generation
            gc.collect(1)  # Middle generation
            gc.collect(2)  # Old generation
        except Exception:
            pass
        
        # Try to release memory back to the OS on Linux
        try:
            import ctypes
            libc = ctypes.CDLL('libc.so.6')
            libc.malloc_trim(0)
            logger.debug("Released memory back to OS with malloc_trim")
        except Exception as e:
            logger.debug(f"Could not release memory to OS: {e}")
        
        # Force cleanup of all LanceDB connections
        self._cleanup_all_lancedb_connections()
        
        # Force cleanup of all memory stores
        self._cleanup_all_memory_stores()
    
    def _cleanup_lancedb_connections(self):
        """Clean up idle LanceDB connections."""
        try:
            from src.models.vector_stores.lancedb_manager import cleanup_old_connections
            closed_sessions = cleanup_old_connections(max_idle_time=self.lancedb_idle_time)
            if closed_sessions:
                logger.debug(f"Cleaned up {len(closed_sessions)} idle LanceDB connections")
        except (ImportError, Exception) as e:
            logger.warning(f"Error cleaning up LanceDB connections: {e}")
    
    def _cleanup_all_lancedb_connections(self):
        """Force cleanup of all LanceDB connections."""
        try:
            from src.models.vector_stores.lancedb_manager import destroy_lancedb_resources
            result = destroy_lancedb_resources(preserve_active_sessions=True)
            logger.debug(f"Forced cleanup of LanceDB connections: {result}")
        except (ImportError, Exception) as e:
            logger.warning(f"Error forcing cleanup of LanceDB connections: {e}")
    
    def _cleanup_memory_stores(self):
        """Clean up idle memory stores."""
        try:
            # No built-in method to clean up idle stores only, so we simply call gc
            gc.collect()
        except Exception as e:
            logger.warning(f"Error during memory store gc: {e}")
    
    def _cleanup_all_memory_stores(self):
        """Force cleanup of all memory stores."""
        try:
            from src.models.memory.chat_memory_manager import clear_memory_store_cache
            clear_memory_store_cache()
            logger.debug("Cleared all memory stores")
        except (ImportError, Exception) as e:
            logger.warning(f"Error clearing all memory stores: {e}")
    
    def _cleanup_embedding_cache(self):
        """Clean up embedding cache."""
        try:
            # There's no direct way to clean specific items, so we rely on GC
            gc.collect()
        except Exception as e:
            logger.warning(f"Error during embedding cache cleanup: {e}")
    
    def _cleanup_search_results_cache(self):
        """Clean up search results cache if it exists."""
        try:
            from src.models.search_results_cache import get_search_cache
            cache = get_search_cache()
            # No method to clean old entries, but could be added
        except (ImportError, Exception) as e:
            logger.debug(f"Error accessing search results cache: {e}")
    
    def cleanup_after_request(self, session_id=None):
        """
        Perform cleanup after a web request completes.
        This is a lightweight version of cleanup suitable for calling after each request.
        
        Args:
            session_id: Optional session ID to target cleanup
        """
        try:
            # For simple requests, just run a garbage collection pass
            gc.collect()
            
            # Clear CUDA cache for GPU requests
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except (ImportError, Exception):
                pass
            
            # If session ID is provided, clean up session-specific resources
            if session_id:
                try:
                    from src.models.memory.chat_memory_manager import clear_memory_store_cache_for_session
                    clear_memory_store_cache_for_session(session_id)
                except (ImportError, Exception) as e:
                    logger.debug(f"Error clearing memory store for session {session_id}: {e}")
                
                try:
                    from src.models.vector_stores.lancedb_manager import _close_lancedb_connection
                    _close_lancedb_connection(session_id)
                except (ImportError, Exception) as e:
                    logger.debug(f"Error closing LanceDB connection for session {session_id}: {e}")
            
        except Exception as e:
            logger.warning(f"Error in cleanup_after_request: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get memory manager statistics.
        
        Returns:
            Dictionary of statistics
        """
        try:
            process = psutil.Process(os.getpid())
            cpu_percent = process.cpu_percent(interval=0.1)
            mem_info = process.memory_info()
            
            stats = {
                "cleanup_count": self.cleanup_count,
                "emergency_cleanup_count": self.emergency_cleanup_count,
                "bytes_freed": self.bytes_freed,
                "mb_freed": self.bytes_freed / (1024 * 1024),
                "last_cleanup_time": self.last_cleanup_time,
                "current_memory_usage": {
                    "rss": mem_info.rss,
                    "rss_mb": mem_info.rss / (1024 * 1024),
                    "vms": mem_info.vms,
                    "vms_mb": mem_info.vms / (1024 * 1024),
                    "percent": process.memory_percent(),
                    "cpu_percent": cpu_percent
                }
            }
            
            # Try to add CUDA stats if available
            try:
                import torch
                if torch.cuda.is_available():
                    stats["gpu"] = {
                        "allocated_mb": torch.cuda.memory_allocated() / (1024 * 1024),
                        "reserved_mb": torch.cuda.memory_reserved() / (1024 * 1024),
                        "device_count": torch.cuda.device_count()
                    }
            except (ImportError, Exception):
                pass
            
            return stats
        
        except Exception as e:
            logger.error(f"Error getting memory manager stats: {e}")
            return {"error": str(e)}

# Singleton instance for application-wide use
_memory_manager = None

def get_memory_manager() -> MemoryManager:
    """
    Get the singleton memory manager instance.
    
    Returns:
        The memory manager instance
    """
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = MemoryManager()
    return _memory_manager

def cleanup_after_response(session_id=None, user_id=None):
    """
    Convenience function to clean up memory after a response.
    This should be called after each RAG response to ensure memory is released.
    
    Args:
        session_id: Optional session ID to target for cleanup
        user_id: Optional user ID to target for cleanup
    """
    try:
        # Track memory before cleanup
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / (1024 * 1024)  # MB
        
        # First use the existing cleanup utility
        from src.utils.memory_management.cleanup import cleanup_after_response as existing_cleanup
        existing_cleanup(session_id=session_id, user_id=user_id)
        
        # Then use our memory manager
        mm = get_memory_manager()
        mm.cleanup_after_request(session_id=session_id)
        
        # Always force a memory check and cleanup
        mm.check_memory_threshold()  # Check for high memory usage
        mm.check_and_cleanup()       # Force periodic cleanup
        
        # LanceDB specific cleanup - use a thread safety mechanism
        try:
            # Use a class-level attribute to prevent too many concurrent cleanups
            if not hasattr(MemoryManager, '_last_lancedb_cleanup_time'):
                MemoryManager._last_lancedb_cleanup_time = 0
                
            current_time = time.time()
            # Only run LanceDB cleanup if it's been at least 10 seconds since the last one
            if current_time - MemoryManager._last_lancedb_cleanup_time > 10:
                from src.models.vector_stores.lancedb_manager import cleanup_old_connections
                
                # Close inactive connections with shorter timeout
                # But don't perform LanceDB cleanup on every request - that's too resource intensive
                try:
                    cleanup_old_connections(max_idle_time=120)  # 2 minute timeout
                    # Update the last cleanup time
                    MemoryManager._last_lancedb_cleanup_time = current_time
                except Exception as cleanup_err:
                    logger.warning(f"Error during LanceDB connection cleanup: {cleanup_err}")
                
                # No longer destroy active connections - that causes issues with RAG
        except (ImportError, Exception) as e:
            logger.warning(f"Error during LanceDB cleanup: {e}")
        
        # Force multiple garbage collection passes
        for _ in range(2):
            gc.collect()
        
        # Clear CUDA cache if available
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except (ImportError, Exception):
            pass
        
        # Try to release memory back to OS
        try:
            import ctypes
            libc = ctypes.CDLL('libc.so.6')
            libc.malloc_trim(0)
        except Exception as e:
            logger.debug(f"Could not release memory back to OS: {e}")
        
        # Track memory after cleanup
        mem_after = process.memory_info().rss / (1024 * 1024)  # MB
        mem_freed = max(0, mem_before - mem_after)
        
        # Log significant memory cleanups
        if mem_freed > 50:  # Only log if we freed more than 50MB
            logger.info(f"Memory cleanup freed {mem_freed:.2f} MB")
        
    except Exception as e:
        logger.error(f"Error in cleanup_after_response: {e}")