"""
Periodic memory cleanup for RAG Lab.

This module provides a background thread that periodically
runs memory cleanup operations to prevent memory buildup.
"""

import threading
import time
import os
import gc
import sys
import ctypes
import logging
import torch
import psutil
from typing import Optional

logger = logging.getLogger(__name__)

def force_release_memory():
    """
    Force aggressive memory release back to the operating system.
    This uses various techniques to ensure Python returns memory to the OS.
    """
    # Start with Python's garbage collection
    logger.info("[MEMORY] Running periodic memory cleanup")

    # Record starting memory
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / (1024 * 1024 * 1024)  # GB

    # Run Python's garbage collection
    for i in range(3):
        gc.collect(i)

    # Empty CUDA cache if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    # OS-specific memory release for Linux
    if os.name == 'posix':
        try:
            # Run malloc_trim to return memory to the OS
            libc = ctypes.CDLL('libc.so.6')
            libc.malloc_trim(0)

            # Additional aggressive memory release (Linux-only, no-op on other platforms)
            from src.utils.platform import flush_filesystem, drop_caches
            flush_filesystem()
            drop_caches()
        except Exception as e:
            logger.error(f"[MEMORY] OS-level cleanup error: {e}")

    # Record ending memory
    mem_after = process.memory_info().rss / (1024 * 1024 * 1024)  # GB
    mem_freed = mem_before - mem_after

    # Log results
    logger.info(f"[MEMORY] Periodic cleanup freed {mem_freed:.2f} GB RAM")

    return mem_freed

def cleanup_lancedb_globally(max_idle_time=120):
    """
    Clean up LanceDB-related memory globally, but preserve active sessions.
    This function directly accesses and cleans up LanceDB connections.

    It performs a cleanup that preserves active sessions to prevent data loss.

    Args:
        max_idle_time: Maximum idle time in seconds before a connection is closed

    Returns:
        dict: Information about the cleanup operation
    """
    try:
        logger.info("[MEMORY] Running global LanceDB cleanup")

        # Track memory before cleanup
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / (1024 * 1024 * 1024)  # GB

        # First, try to use our new connection pooling cleanup
        try:
            # Use the new cleanup_old_connections function to clean up idle connections
            from src.models.vector_stores.lancedb_manager import cleanup_old_connections
            cleaned_sessions = cleanup_old_connections(max_idle_time=max_idle_time)
            if cleaned_sessions:
                logger.info(f"[MEMORY] Cleaned up {len(cleaned_sessions)} idle LanceDB connections")
        except (ImportError, AttributeError) as e:
            logger.warning(f"[MEMORY] Could not import cleanup_old_connections: {e}")
            cleaned_sessions = []

        # If no connections were cleaned up, try the more aggressive approach
        if not cleaned_sessions:
            try:
                # Use the direct cleanup approach if available, but preserve active sessions
                from src.models.vector_stores.lancedb_manager import destroy_lancedb_resources
                cleanup_info = destroy_lancedb_resources(preserve_active_sessions=True)
                logger.info(f"[MEMORY] Cleared LanceDB resources: {cleanup_info}")
            except (ImportError, AttributeError) as e:
                logger.warning(f"[MEMORY] Could not import destroy_lancedb_resources: {e}")
                cleanup_info = {"error": str(e)}

            # Direct memory management approach - find any module containing 'lancedb' in the name
            # But be more careful about what we clear
            cleaned_modules = 0
            for name, module in list(sys.modules.items()):
                if 'lancedb' in name.lower() or 'vector' in name.lower():
                    for attr_name in dir(module):
                        # Only clear caches and temporary connections, not active ones
                        if attr_name.startswith('_lancedb_cache') or attr_name.endswith('_temp_connections'):
                            try:
                                attr_value = getattr(module, attr_name)
                                if isinstance(attr_value, dict):
                                    count = len(attr_value)
                                    attr_value.clear()
                                    cleaned_modules += 1
                                    logger.info(f"[MEMORY] Cleared {count} items from {name}.{attr_name}")
                            except Exception as e:
                                logger.warning(f"[MEMORY] Error clearing {name}.{attr_name}: {e}")

            logger.info(f"[MEMORY] Global LanceDB cleanup processed {cleaned_modules} modules")

        # Force garbage collection multiple times
        for i in range(3):
            gc.collect(i)

        # Empty CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        # Try to release memory back to the system using malloc_trim
        try:
            libc = ctypes.CDLL('libc.so.6')
            libc.malloc_trim(0)
            logger.info("[MEMORY] Called malloc_trim to release memory to the system")
        except Exception as trim_e:
            logger.warning(f"[MEMORY] Could not call malloc_trim: {trim_e}")

        # Track memory after cleanup
        mem_after = process.memory_info().rss / (1024 * 1024 * 1024)  # GB
        mem_diff = mem_before - mem_after

        logger.info(f"[MEMORY] Global LanceDB cleanup freed {mem_diff:.2f} GB RAM")

        return {
            "cleaned_sessions": len(cleaned_sessions) if cleaned_sessions else 0,
            "memory_freed_gb": mem_diff
        }
    except Exception as e:
        logger.error(f"[MEMORY] Error in global LanceDB cleanup: {e}")
        return {"error": str(e)}



class PeriodicMemoryCleanup:
    """
    Background thread that periodically cleans up memory.
    """
    def __init__(self, interval_seconds: int = 60):
        """
        Initialize the periodic cleanup thread.

        Args:
            interval_seconds: Time between cleanup operations in seconds
        """
        self.interval = interval_seconds
        self.running = False
        self.thread = None
        self.high_pressure_threshold = 0.85  # 85% RAM usage triggers extra cleanup

    def start(self):
        """Start the periodic cleanup thread."""
        if self.running:
            return

        self.running = True
        self.thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self.thread.start()
        logger.info(f"[MEMORY] Started periodic memory cleanup every {self.interval} seconds")

    def stop(self):
        """Stop the periodic cleanup thread."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
            self.thread = None
        logger.info("[MEMORY] Stopped periodic memory cleanup")

    def _cleanup_loop(self):
        """Main cleanup loop that runs in the background thread."""
        while self.running:
            try:
                # Check system memory usage
                system_mem = psutil.virtual_memory()
                percent_used = system_mem.percent / 100.0

                # Always run LanceDB cleanup first to ensure it's properly cleaned up
                # This is critical for preventing memory leaks with LanceDB
                cleanup_lancedb_globally()

                # Then run standard cleanup
                force_release_memory()

                # Under high memory pressure, perform more aggressive cleanup
                if percent_used > self.high_pressure_threshold:
                    logger.warning(f"[MEMORY] High memory pressure: {system_mem.percent}% - performing deep cleanup")

                    # Run garbage collection multiple times
                    for i in range(5):
                        gc.collect(i)

                    # Try to release memory back to the system using malloc_trim
                    try:
                        libc = ctypes.CDLL('libc.so.6')
                        libc.malloc_trim(0)
                        logger.info("[MEMORY] Called malloc_trim to release memory to the system under high pressure")
                    except Exception as trim_e:
                        logger.warning(f"[MEMORY] Could not call malloc_trim: {trim_e}")

                    # More aggressive clean after high pressure
                    force_release_memory()

            except Exception as e:
                logger.error(f"[MEMORY] Error in cleanup loop: {e}")

            # Sleep until next interval
            time.sleep(self.interval)

# Global cleanup manager instance
memory_cleanup_manager = PeriodicMemoryCleanup(interval_seconds=60)  # Every 60 seconds

def start_periodic_cleanup(interval_seconds: Optional[int] = None):
    """
    Start the periodic memory cleanup thread.

    Args:
        interval_seconds: Optional override for cleanup interval
    """
    if interval_seconds is not None:
        memory_cleanup_manager.interval = interval_seconds

    memory_cleanup_manager.start()

def stop_periodic_cleanup():
    """Stop the periodic memory cleanup thread."""
    memory_cleanup_manager.stop()