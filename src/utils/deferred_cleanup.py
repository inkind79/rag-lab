"""
Deferred Memory Cleanup Module

Provides a mechanism to defer resource cleanup until after response generation,
preventing memory pressure during critical operations.
"""

import threading
import queue
import time
import gc
from typing import Callable, Optional, Any
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DeferredCleanupManager:
    """
    Manages deferred cleanup tasks that should be executed after response generation.
    
    This helps prevent memory pressure during critical operations by deferring
    resource cleanup until after the response has been sent to the user.
    """
    
    def __init__(self, max_queue_size: int = 1000, cleanup_interval: float = 1.0):
        """
        Initialize the deferred cleanup manager.
        
        Args:
            max_queue_size: Maximum number of cleanup tasks to queue
            cleanup_interval: Interval in seconds between cleanup cycles
        """
        self.cleanup_queue = queue.Queue(maxsize=max_queue_size)
        self.cleanup_interval = cleanup_interval
        self.cleanup_thread = None
        self.running = False
        self._lock = threading.Lock()
    
    def start(self):
        """Start the cleanup thread."""
        with self._lock:
            if not self.running:
                self.running = True
                self.cleanup_thread = threading.Thread(
                    target=self._cleanup_worker,
                    daemon=True
                )
                self.cleanup_thread.start()
                logger.info("Deferred cleanup manager started")
    
    def stop(self):
        """Stop the cleanup thread."""
        with self._lock:
            if self.running:
                self.running = False
                if self.cleanup_thread:
                    self.cleanup_thread.join(timeout=5)
                logger.info("Deferred cleanup manager stopped")
    
    def defer_cleanup(self, cleanup_fn: Callable, *args, **kwargs):
        """
        Defer a cleanup task to be executed later.
        
        Args:
            cleanup_fn: Function to call for cleanup
            *args: Positional arguments for cleanup_fn
            **kwargs: Keyword arguments for cleanup_fn
        """
        try:
            if not self.running:
                # If not running, execute immediately
                cleanup_fn(*args, **kwargs)
            else:
                # Queue the cleanup task
                task = (cleanup_fn, args, kwargs)
                self.cleanup_queue.put_nowait(task)
                logger.debug(f"Deferred cleanup task: {cleanup_fn.__name__}")
        except queue.Full:
            logger.warning("Cleanup queue full, executing immediately")
            # If queue is full, execute immediately
            cleanup_fn(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error deferring cleanup: {e}")
    
    def _cleanup_worker(self):
        """Worker thread that processes cleanup tasks."""
        logger.info("Cleanup worker thread started")
        
        while self.running:
            try:
                # Wait for cleanup tasks with timeout
                try:
                    task = self.cleanup_queue.get(timeout=self.cleanup_interval)
                    cleanup_fn, args, kwargs = task
                    
                    # Execute the cleanup
                    try:
                        cleanup_fn(*args, **kwargs)
                        logger.debug(f"Executed deferred cleanup: {cleanup_fn.__name__}")
                    except Exception as e:
                        logger.error(f"Error executing cleanup {cleanup_fn.__name__}: {e}")
                    
                    self.cleanup_queue.task_done()
                except queue.Empty:
                    # No tasks available, continue waiting
                    pass
                
                # Perform periodic garbage collection
                if self.cleanup_queue.empty():
                    gc.collect()
                    
            except Exception as e:
                logger.error(f"Error in cleanup worker: {e}")
        
        # Process remaining tasks before exiting
        while not self.cleanup_queue.empty():
            try:
                task = self.cleanup_queue.get_nowait()
                cleanup_fn, args, kwargs = task
                cleanup_fn(*args, **kwargs)
                self.cleanup_queue.task_done()
            except Exception as e:
                logger.error(f"Error processing remaining cleanup tasks: {e}")
        
        logger.info("Cleanup worker thread stopped")
    
    def flush(self):
        """
        Flush all pending cleanup tasks immediately.
        
        This should be called during graceful shutdown.
        """
        logger.info("Flushing deferred cleanup tasks...")
        
        count = 0
        while not self.cleanup_queue.empty():
            try:
                task = self.cleanup_queue.get_nowait()
                cleanup_fn, args, kwargs = task
                cleanup_fn(*args, **kwargs)
                self.cleanup_queue.task_done()
                count += 1
            except Exception as e:
                logger.error(f"Error flushing cleanup task: {e}")
        
        logger.info(f"Flushed {count} cleanup tasks")
        
        # Force garbage collection after flushing
        gc.collect()


# Global instance
_cleanup_manager = None
_manager_lock = threading.Lock()


def get_cleanup_manager() -> DeferredCleanupManager:
    """Get the global deferred cleanup manager instance."""
    global _cleanup_manager
    
    with _manager_lock:
        if _cleanup_manager is None:
            _cleanup_manager = DeferredCleanupManager()
            _cleanup_manager.start()
    
    return _cleanup_manager


def defer_cleanup(cleanup_fn: Callable, *args, **kwargs):
    """
    Convenience function to defer a cleanup task.
    
    Args:
        cleanup_fn: Function to call for cleanup
        *args: Positional arguments for cleanup_fn
        **kwargs: Keyword arguments for cleanup_fn
    """
    manager = get_cleanup_manager()
    manager.defer_cleanup(cleanup_fn, *args, **kwargs)


def flush_cleanup():
    """Flush all pending cleanup tasks."""
    manager = get_cleanup_manager()
    manager.flush()


def stop_cleanup_manager():
    """Stop the cleanup manager."""
    global _cleanup_manager
    
    with _manager_lock:
        if _cleanup_manager is not None:
            _cleanup_manager.stop()
            _cleanup_manager = None