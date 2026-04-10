"""
Thread-safe model management for RAG Lab.

This module provides thread-safe wrappers for global model dictionaries
to prevent race conditions in multi-worker deployments.
"""

import threading
import time
from typing import Dict, Any, Optional, Set
from src.utils.logger import get_logger
from src.utils.resource_lifecycle_manager import get_lifecycle_manager

logger = get_logger(__name__)


class ThreadSafeModelManager:
    """
    Thread-safe manager for RAG models.
    
    This replaces the global RAG_models dictionary with proper synchronization
    and resource lifecycle management. Models are stored per session_id for
    proper session isolation.
    """

    def _check_memory_and_evict(self):
        """Check memory usage and evict oldest models if needed"""
        import psutil
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        
        # If memory usage exceeds 3GB, start evicting old models
        if memory_mb > 3072:
            logger.warning(f"High memory usage ({memory_mb:.0f}MB), evicting oldest models")
            
            # Sort by last access time and remove oldest
            sorted_models = sorted(
                self._last_access.items(),
                key=lambda x: x[1]
            )
            
            # Remove the oldest 25% of models
            to_remove = len(sorted_models) // 4
            for session_id, _ in sorted_models[:to_remove]:
                logger.info(f"Evicting model for session {session_id}")
                self.delete(session_id)
                
            # Force garbage collection
            import gc
            gc.collect()

    
    def __init__(self):
        self._models: Dict[str, Any] = {}
        self._lock = threading.RLock()
        self._access_times: Dict[str, float] = {}
        self._lifecycle_manager = get_lifecycle_manager()
        
    def get(self, session_id: str, default=None) -> Optional[Any]:
        """Get a model for a session."""
        with self._lock:
            self._access_times[session_id] = time.time()
            return self._models.get(session_id, default)
            
    def set(self, session_id: str, model: Any, cleanup_fn=None):
        """Set a model for a session."""
        with self._lock:
            # Release old model if exists
            if session_id in self._models:
                self.delete(session_id)
                
            self._models[session_id] = model
            self._access_times[session_id] = time.time()
            
            # Register with lifecycle manager
            if cleanup_fn:
                self._lifecycle_manager.acquire_resource(
                    f"rag_model_{session_id}",
                    lambda: model,
                    cleanup_fn,
                    "rag_model"
                )
            
            logger.info(f"Set model for session {session_id}")
            
    def delete(self, session_id: str) -> Optional[Any]:
        """Delete a model for a session."""
        with self._lock:
            model = self._models.pop(session_id, None)
            self._access_times.pop(session_id, None)
            
            # Release from lifecycle manager
            self._lifecycle_manager.release_resource(f"rag_model_{session_id}")
            
            if model:
                logger.info(f"Deleted model for session {session_id}")
            return model
            
    def contains(self, session_id: str) -> bool:
        """Check if a model exists for a session."""
        with self._lock:
            return session_id in self._models
            
    def clear(self):
        """Clear all models."""
        with self._lock:
            session_ids = list(self._models.keys())
            for session_id in session_ids:
                self.delete(session_id)
            logger.info("Cleared all models")
            
    def get_all_session_ids(self) -> Set[str]:
        """Get all session IDs with models."""
        with self._lock:
            return set(self._models.keys())
    
    def size(self) -> int:
        """Get the number of models currently stored."""
        with self._lock:
            return len(self._models)
            
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about model usage."""
        with self._lock:
            current_time = time.time()
            stats = {
                'total_models': len(self._models),
                'session_ids': list(self._models.keys()),
                'access_info': {}
            }
            
            for session_id, access_time in self._access_times.items():
                stats['access_info'][session_id] = {
                    'last_access': access_time,
                    'idle_seconds': current_time - access_time
                }
                
            return stats
            
    def cleanup_idle_models(self, max_idle_seconds: int = 3600):
        """Clean up models that haven't been accessed recently."""
        with self._lock:
            current_time = time.time()
            sessions_to_cleanup = []
            
            for session_id, access_time in self._access_times.items():
                if current_time - access_time > max_idle_seconds:
                    sessions_to_cleanup.append(session_id)
                    
            for session_id in sessions_to_cleanup:
                self.delete(session_id)
                logger.info(f"Cleaned up idle model for session {session_id}")
                
            return len(sessions_to_cleanup)


class ThreadSafeCache:
    """
    Generic thread-safe cache for any resource type.
    """
    
    def __init__(self, cache_name: str):
        self._cache: Dict[str, Any] = {}
        self._lock = threading.RLock()
        self._cache_name = cache_name
        self._access_times: Dict[str, float] = {}
        
    def get(self, key: str, default=None) -> Optional[Any]:
        """Get an item from cache."""
        with self._lock:
            self._access_times[key] = time.time()
            return self._cache.get(key, default)
            
    def set(self, key: str, value: Any):
        """Set an item in cache."""
        with self._lock:
            self._cache[key] = value
            self._access_times[key] = time.time()
            logger.debug(f"{self._cache_name}: Set {key}")
            
    def delete(self, key: str) -> Optional[Any]:
        """Delete an item from cache."""
        with self._lock:
            value = self._cache.pop(key, None)
            self._access_times.pop(key, None)
            if value is not None:
                logger.debug(f"{self._cache_name}: Deleted {key}")
            return value
            
    def clear(self):
        """Clear the cache."""
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            self._access_times.clear()
            logger.info(f"{self._cache_name}: Cleared {count} items")
            
    def contains(self, key: str) -> bool:
        """Check if key exists in cache."""
        with self._lock:
            return key in self._cache
            
    def size(self) -> int:
        """Get cache size."""
        with self._lock:
            return len(self._cache)
            
    def cleanup_old_entries(self, max_age_seconds: int = 3600) -> int:
        """Remove entries older than max_age_seconds."""
        with self._lock:
            current_time = time.time()
            keys_to_remove = []
            
            for key, access_time in self._access_times.items():
                if current_time - access_time > max_age_seconds:
                    keys_to_remove.append(key)
                    
            for key in keys_to_remove:
                self.delete(key)
                
            if keys_to_remove:
                logger.info(f"{self._cache_name}: Cleaned up {len(keys_to_remove)} old entries")
                
            return len(keys_to_remove)


# Global instances
_model_manager = None
_manager_lock = threading.Lock()


def get_thread_safe_model_manager() -> ThreadSafeModelManager:
    """Get or create the global thread-safe model manager."""
    global _model_manager
    
    if _model_manager is None:
        with _manager_lock:
            if _model_manager is None:
                _model_manager = ThreadSafeModelManager()
                logger.info("Initialized thread-safe model manager")
                
    return _model_manager