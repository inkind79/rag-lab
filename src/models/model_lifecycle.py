"""
Model Lifecycle Manager — single authority for GPU model loading and unloading.

Replaces the scattered _global_embedding_models cache in model_loader.py
with a thread-safe, VRAM-aware model manager. All model loading should go
through this manager to prevent duplicate loads, race conditions, and
uncontrolled GPU memory growth.

The ThreadSafeModelManager (app.RAG_models) continues to map session_ids
to models, but the actual model instances are owned by this lifecycle manager.
Multiple sessions can share the same model instance if they use the same
embedding model.
"""

import gc
import time
import threading
from typing import Dict, Optional, Any

import torch

from src.utils.logger import get_logger
from src.models.model_registry import registry

logger = get_logger(__name__)


class ModelLifecycleManager:
    """Single authority on loaded GPU embedding models.

    Key properties:
    - Thread-safe: uses RLock for all model operations
    - VRAM-aware: evicts LRU models before loading if GPU memory is tight
    - Deduplication: same model_id returns same adapter instance
    - Single cleanup path: _cleanup_gpu() is the ONE place CUDA cache is cleared
    """

    def __init__(self):
        self._loaded: Dict[str, Any] = {}  # model_id -> adapter instance
        self._access_times: Dict[str, float] = {}
        self._lock = threading.RLock()
        logger.info("ModelLifecycleManager initialized")

    def get_or_load(self, model_id: str) -> Any:
        """Get a loaded model or load it. Thread-safe with deduplication.

        If the model is already loaded, returns the cached instance.
        If GPU memory is tight, evicts the least-recently-used model first.

        Args:
            model_id: HuggingFace model ID or name that registry can detect

        Returns:
            A loaded BaseEmbeddingAdapter instance
        """
        with self._lock:
            # Normalize the model_id through registry for consistent keying
            spec = registry.detect(model_id)
            cache_key = spec.model_id if spec else model_id

            # Return cached instance if available
            if cache_key in self._loaded and self._loaded[cache_key] is not None:
                self._access_times[cache_key] = time.time()
                logger.info(f"[LIFECYCLE] Reusing loaded model: {cache_key}")
                return self._loaded[cache_key]

            # Evict if needed before loading
            self._evict_if_needed(cache_key)

            # Load the model
            logger.info(f"[LIFECYCLE] Loading model: {cache_key}")
            adapter = self._create_and_load_adapter(model_id, spec)

            # Cache it
            self._loaded[cache_key] = adapter
            self._access_times[cache_key] = time.time()

            logger.info(f"[LIFECYCLE] Model loaded successfully: {cache_key} "
                        f"(total loaded: {len(self._loaded)})")
            return adapter

    def is_loaded(self, model_id: str) -> bool:
        """Check if a model is currently loaded."""
        with self._lock:
            spec = registry.detect(model_id)
            cache_key = spec.model_id if spec else model_id
            return cache_key in self._loaded and self._loaded[cache_key] is not None

    def unload(self, model_id: str) -> None:
        """Explicitly unload a specific model and free GPU memory."""
        with self._lock:
            spec = registry.detect(model_id)
            cache_key = spec.model_id if spec else model_id

            if cache_key in self._loaded:
                adapter = self._loaded.pop(cache_key)
                self._access_times.pop(cache_key, None)

                if adapter and hasattr(adapter, 'unload'):
                    try:
                        adapter.unload()
                        logger.info(f"[LIFECYCLE] Unloaded model: {cache_key}")
                    except Exception as e:
                        logger.warning(f"[LIFECYCLE] Error unloading {cache_key}: {e}")

                del adapter
                self._cleanup_gpu()

    def unload_all(self) -> int:
        """Unload all models. Returns count of models unloaded."""
        with self._lock:
            count = len(self._loaded)
            model_ids = list(self._loaded.keys())

            for model_id in model_ids:
                adapter = self._loaded.pop(model_id, None)
                if adapter and hasattr(adapter, 'unload'):
                    try:
                        adapter.unload()
                    except Exception as e:
                        logger.warning(f"[LIFECYCLE] Error unloading {model_id}: {e}")
                del adapter

            self._access_times.clear()
            self._cleanup_gpu()

            logger.info(f"[LIFECYCLE] Unloaded all {count} models")
            return count

    def unload_idle(self, max_idle_seconds: int = 600) -> int:
        """Unload models that haven't been accessed recently.

        Args:
            max_idle_seconds: Seconds of inactivity before unloading (default 10 min)

        Returns:
            Number of models unloaded
        """
        with self._lock:
            current_time = time.time()
            to_unload = [
                model_id for model_id, access_time in self._access_times.items()
                if current_time - access_time > max_idle_seconds
            ]

            for model_id in to_unload:
                adapter = self._loaded.pop(model_id, None)
                self._access_times.pop(model_id, None)
                if adapter and hasattr(adapter, 'unload'):
                    try:
                        adapter.unload()
                        logger.info(f"[LIFECYCLE] Unloaded idle model: {model_id} "
                                    f"(idle {current_time - self._access_times.get(model_id, 0):.0f}s)")
                    except Exception as e:
                        logger.warning(f"[LIFECYCLE] Error unloading idle {model_id}: {e}")
                del adapter

            if to_unload:
                self._cleanup_gpu()
                logger.info(f"[LIFECYCLE] Unloaded {len(to_unload)} idle models")

            return len(to_unload)

    def get_stats(self) -> Dict:
        """Get statistics about loaded models."""
        with self._lock:
            current_time = time.time()
            return {
                'loaded_count': len(self._loaded),
                'models': {
                    model_id: {
                        'idle_seconds': current_time - self._access_times.get(model_id, 0),
                        'type': type(adapter).__name__,
                    }
                    for model_id, adapter in self._loaded.items()
                }
            }

    # --- Internal methods ---

    def _create_and_load_adapter(self, model_id: str, spec=None):
        """Create and load the appropriate adapter for a model."""
        # For now, all multi-vector models use ColPaliAdapter
        # Phase 6 will add adapter_factory to ModelSpec
        from src.models.colpali_adapter import ColPaliAdapter
        adapter = ColPaliAdapter(model_id)
        adapter.load()
        return adapter

    def _evict_if_needed(self, incoming_model_id: str) -> None:
        """Evict least-recently-used model if GPU memory would be exceeded."""
        if not self._loaded:
            return

        spec = registry.detect(incoming_model_id)
        needed_vram = spec.vram_gb if spec else 7.0  # Conservative default

        available_vram = self._get_available_vram_gb()
        if available_vram is not None and available_vram < needed_vram:
            # Find LRU model
            if self._access_times:
                lru_model = min(self._access_times, key=self._access_times.get)
                logger.info(f"[LIFECYCLE] Evicting LRU model {lru_model} "
                            f"(available VRAM: {available_vram:.1f}GB, need: {needed_vram:.1f}GB)")
                # Unload without the outer lock (we already hold it)
                adapter = self._loaded.pop(lru_model, None)
                self._access_times.pop(lru_model, None)
                if adapter and hasattr(adapter, 'unload'):
                    adapter.unload()
                del adapter
                self._cleanup_gpu()

    def _get_available_vram_gb(self) -> Optional[float]:
        """Get available GPU VRAM in GB, or None if no GPU."""
        if not torch.cuda.is_available():
            return None
        try:
            free, total = torch.cuda.mem_get_info()
            return free / (1024 ** 3)
        except Exception:
            return None

    def _cleanup_gpu(self) -> None:
        """The ONE strategic GPU cleanup point. Called after model unloads."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.debug("[LIFECYCLE] GPU cache cleared")


# Module-level singleton
_lifecycle_manager = None
_lifecycle_lock = threading.Lock()


def get_lifecycle_manager() -> ModelLifecycleManager:
    """Get or create the global ModelLifecycleManager singleton."""
    global _lifecycle_manager
    if _lifecycle_manager is None:
        with _lifecycle_lock:
            if _lifecycle_manager is None:
                _lifecycle_manager = ModelLifecycleManager()
    return _lifecycle_manager
