"""
Unified Memory Management Module for RAG Lab

This module consolidates all memory management functionality from across the codebase
into a single, consistent interface. It handles model loading/unloading, memory tracking,
and cleanup operations.
"""

import torch
import gc
import os
import psutil
import logging
import time
from typing import Dict, Any, Optional, Callable, Tuple, List, Union

from src.utils.logger import get_logger
logger = get_logger(__name__)

class MemoryManager:
    """
    Comprehensive memory manager that handles model loading, unloading, memory tracking,
    and cleanup operations. This class consolidates functionality previously spread across
    multiple files into a single, consistent interface.
    """

    def __init__(self):
        """Initialize the memory manager with empty state."""
        self.current_model = None
        self.model_cache = {}
        self.device = self._detect_device()
        self.last_cleanup_time = 0
        logger.info(f"Memory Manager initialized with device: {self.device}")

    def _detect_device(self) -> str:
        """
        Detects the best available device (CUDA, MPS, or CPU).

        Returns:
            str: Device name ('cuda', 'mps', or 'cpu')
        """
        if torch.cuda.is_available():
            return 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return 'mps'
        else:
            return 'cpu'

    def is_cuda_available(self) -> bool:
        """
        Check if CUDA is available.

        Returns:
            bool: True if CUDA is available, False otherwise
        """
        return self.device == 'cuda'

    def get_memory_usage(self) -> Dict[str, Any]:
        """
        Get comprehensive memory usage information in a standardized format.

        Returns:
            dict: Memory usage information including RAM and GPU metrics
        """
        # System memory (RAM)
        process = psutil.Process(os.getpid())
        ram_usage = process.memory_info().rss / (1024 * 1024 * 1024)  # in GB
        system_memory = psutil.virtual_memory()
        system_ram_percent = system_memory.percent

        # GPU memory if available
        gpu_usage = None
        gpu_total = None
        gpu_percentage = None
        gpu_reserved = None

        if self.is_cuda_available():
            try:
                gpu_usage = torch.cuda.memory_allocated() / (1024 * 1024 * 1024)  # in GB
                gpu_reserved = torch.cuda.memory_reserved() / (1024 * 1024 * 1024)  # in GB
                gpu_total = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024 * 1024)  # in GB
                gpu_percentage = (gpu_usage / gpu_total) * 100 if gpu_total > 0 else 0
            except Exception as e:
                logger.error(f"Error getting GPU memory: {e}")

        return {
            "ram_usage_gb": round(ram_usage, 2),
            "system_ram_percent": round(system_ram_percent, 2),
            "gpu_usage_gb": round(gpu_usage, 2) if gpu_usage is not None else None,
            "gpu_reserved_gb": round(gpu_reserved, 2) if gpu_reserved is not None else None,
            "gpu_total_gb": round(gpu_total, 2) if gpu_total is not None else None,
            "gpu_percentage": round(gpu_percentage, 2) if gpu_percentage is not None else None,
            "loaded_model": self.current_model,
            "device": self.device
        }

    def clear_cache(self) -> None:
        """
        Clear PyTorch cache and run garbage collection.
        """
        if self.is_cuda_available():
            torch.cuda.empty_cache()
        gc.collect()
        logger.info("Memory cache cleared")

    def get_model(self, model_name: str, loader_function: Callable) -> Any:
        """
        Get a model, either from cache or by loading it.
        This method is the primary interface for model loading.

        Args:
            model_name: Name/identifier of the model
            loader_function: Function to call to load the model if needed

        Returns:
            The loaded model or model components
        """
        # If a different model is currently loaded, unload it first
        if self.current_model is not None and self.current_model != model_name:
            logger.info(f"Unloading current model '{self.current_model}' before loading '{model_name}'")
            self.unload_model(self.current_model)

        # Check if the model is already loaded in cache
        if model_name in self.model_cache and self.model_cache[model_name] is not None:
            logger.info(f"Using cached model '{model_name}'")
            model_data = self.model_cache[model_name]
            self.current_model = model_name
            return model_data

        # Load the model and cache it
        logger.info(f"Loading model '{model_name}'")

        # Record memory before loading
        mem_before = self.get_memory_usage()

        # Load the model
        model_data = loader_function()
        self.model_cache[model_name] = model_data
        self.current_model = model_name

        # Record memory after loading
        mem_after = self.get_memory_usage()

        # Log memory difference
        if self.is_cuda_available():
            gpu_before = mem_before.get("gpu_usage_gb", 0) or 0
            gpu_after = mem_after.get("gpu_usage_gb", 0) or 0
            logger.info(f"Model loading GPU memory impact: {gpu_after - gpu_before:.2f} GB")

        ram_before = mem_before.get("ram_usage_gb", 0) or 0
        ram_after = mem_after.get("ram_usage_gb", 0) or 0
        logger.info(f"Model loading RAM impact: {ram_after - ram_before:.2f} GB")

        return model_data

    def unload_model(self, model_name: Optional[str] = None) -> None:
        """
        Unload a specific model and free memory.

        Args:
            model_name: Name/identifier of the model to unload, or None to unload current model
        """
        if model_name is None and self.current_model is not None:
            model_name = self.current_model

        if model_name is None:
            logger.info("No model to unload")
            return

        if model_name in self.model_cache:
            logger.info(f"Unloading model '{model_name}'")

            # Get the model instance
            model_data = self.model_cache[model_name]

            # Try to move model components to CPU before deletion if they're on GPU
            if model_data is not None:
                if isinstance(model_data, (tuple, list)):
                    # Handle model tuples (like model, processor, device)
                    for item in model_data:
                        if item is not None and hasattr(item, 'to') and callable(item.to):
                            try:
                                item.to('cpu')
                                logger.info(f"Moved {model_name} component to CPU")
                            except Exception as e:
                                logger.warning(f"Error moving {model_name} component to CPU: {e}")
                else:
                    # Handle single model objects
                    if hasattr(model_data, 'to') and callable(model_data.to):
                        try:
                            model_data.to('cpu')
                            logger.info(f"Moved {model_name} to CPU")
                        except Exception as e:
                            logger.warning(f"Error moving {model_name} to CPU: {e}")

            # Remove from cache
            self.model_cache.pop(model_name, None)

            # If this was the current model, clear current_model
            if self.current_model == model_name:
                self.current_model = None

            # Run garbage collection
            gc.collect()
            if self.is_cuda_available():
                torch.cuda.empty_cache()

    def unload_all_models(self, unload_storage: bool = True) -> None:
        """
        Unload all models and perform thorough memory cleanup.

        Args:
            unload_storage: Whether to also unload vector storage (LanceDB, Qdrant)
        """
        logger.info("Unloading all models with thorough memory cleanup")

        # First log current memory state
        mem_before = self.get_memory_usage()

        # Make a copy of keys to avoid modifying during iteration
        model_names = list(self.model_cache.keys())

        for model_name in model_names:
            try:
                # Get the model data
                model_data = self.model_cache[model_name]

                # Deep unload of model components
                if isinstance(model_data, (tuple, list)):
                    # If it's a tuple or list of components, process each one
                    for i, component in enumerate(model_data):
                        if component is not None:
                            # For PyTorch models, move them to CPU first, then delete
                            if hasattr(component, 'to') and callable(component.to):
                                try:
                                    logger.info(f"Moving {model_name} component to CPU")
                                    component.to('cpu')
                                except Exception as e:
                                    logger.error(f"Error moving {model_name} component to CPU: {e}")

                            # For each component, try to clear all attributes that might hold tensors
                            if hasattr(component, '__dict__'):
                                for attr_name in list(component.__dict__.keys()):
                                    try:
                                        attr_val = getattr(component, attr_name, None)
                                        # Handle tensor attributes
                                        if hasattr(attr_val, 'device') and hasattr(attr_val, 'to'):
                                            logger.debug(f"Clearing tensor attribute {attr_name}")
                                            setattr(component, attr_name, None)
                                    except Exception as e:
                                        logger.warning(f"Error clearing attribute {attr_name}: {e}")
                else:
                    # Handle single model objects
                    if model_data is not None:
                        # For PyTorch models, move them to CPU first
                        if hasattr(model_data, 'to') and callable(model_data.to):
                            try:
                                logger.info(f"Moving {model_name} to CPU")
                                model_data.to('cpu')
                            except Exception as e:
                                logger.error(f"Error moving {model_name} to CPU: {e}")

                # Remove from cache
                self.model_cache[model_name] = None

            except Exception as e:
                logger.error(f"Error during unload of model {model_name}: {e}")

        # Recreate the model_cache as a new empty dict to ensure no references remain
        self.model_cache = {}

        # Clear current model reference
        self.current_model = None

        # Also unload all models from the lifecycle manager (the canonical cache)
        try:
            from src.models.model_lifecycle import get_lifecycle_manager
            lifecycle = get_lifecycle_manager()
            lc_count = lifecycle.unload_all()
            if lc_count:
                logger.info(f"Lifecycle manager unloaded {lc_count} model(s)")
        except Exception as e:
            logger.warning(f"Error unloading from lifecycle manager: {e}")

        # Also unload LanceDB and Qdrant wrappers if requested
        if unload_storage:
            self._unload_vector_storage()

        # Run comprehensive memory cleanup
        self.aggressive_cleanup()

        # Log memory after cleanup
        mem_after = self.get_memory_usage()

        if self.is_cuda_available():
            gpu_before = mem_before.get("gpu_usage_gb", 0) or 0
            gpu_after = mem_after.get("gpu_usage_gb", 0) or 0
            freed_gpu = gpu_before - gpu_after
            logger.info(f"GPU memory freed: {freed_gpu:.2f} GB")

        ram_before = mem_before.get("ram_usage_gb", 0) or 0
        ram_after = mem_after.get("ram_usage_gb", 0) or 0
        freed_ram = ram_before - ram_after
        logger.info(f"RAM freed: {freed_ram:.2f} GB")

        logger.info("All models unloaded successfully")

    def _unload_vector_storage(self) -> None:
        """
        Unload vector storage systems (LanceDB, Qdrant).
        """
        # Clear all LanceDB resources from memory
        try:
            from src.models.vector_stores.lancedb_manager import destroy_lancedb_resources
            result = destroy_lancedb_resources()
            count = result.get("count", 0)
            logger.info(f"Cleared all {count} LanceDB resources from vector storage")
        except Exception as e:
            logger.warning(f"Error clearing LanceDB resources during vector storage unload: {e}")

    def aggressive_cleanup(self) -> None:
        """
        Perform aggressive memory cleanup operations.
        This includes multiple rounds of garbage collection and CUDA cache clearing.

        This also synchronizes with model_loader.py to ensure all global caches are cleared.
        """
        # Don't run aggressive cleanup too frequently
        current_time = time.time()
        if current_time - self.last_cleanup_time < 5:  # Only run every 5 seconds at most
            logger.debug("Skipping aggressive cleanup (ran recently)")
            return

        logger.info("Performing aggressive memory cleanup")

        # Skip clearing model cache and current_model to preserve generation model caching
        # (Previously this was clearing cached models, defeating our caching optimization)
        logger.debug("Preserving memory_manager model_cache and current_model during aggressive cleanup")

        # Note: We no longer clear model_loader._global_embedding_models here.
        # The ModelLifecycleManager owns the canonical model cache.
        # Clearing it behind the lifecycle manager's back caused stale references
        # and duplicate model loads.

        # Run multiple garbage collections
        for i in range(3):
            gc.collect(i)  # Collect each generation

        # Clear CUDA cache if available
        if self.is_cuda_available():
            for _ in range(3):
                torch.cuda.empty_cache()

        # Try to release memory back to the system on Linux
        if os.name == 'posix':
            try:
                import ctypes
                libc = ctypes.CDLL('libc.so.6')
                # Return memory to the OS
                libc.malloc_trim(0)
                logger.info("System malloc_trim executed")
            except Exception as e:
                logger.warning(f"System malloc_trim failed: {e}")

        self.last_cleanup_time = current_time
        logger.info("Aggressive memory cleanup completed")

    def unload_rag_models(self) -> None:
        """
        Special method to unload RAG models and clear related caches.
        Used by responder.py and app.py for thorough memory cleanup.
        """
        logger.info("Unloading RAG models and clearing related caches")

        # Try to clear RAG_models from app
        try:
            import sys
            if 'app' in sys.modules:
                app_module = sys.modules['app']
                if hasattr(app_module, 'RAG_models'):
                    logger.info(f"Clearing {len(app_module.RAG_models)} entries from app.RAG_models")
                    app_module.RAG_models.clear()
        except Exception as e:
            logger.warning(f"Error clearing RAG_models: {e}")

        # Try to clear model caches from model_loader
        try:
            from src.models.model_loader import clear_model_caches
            logger.info("Calling clear_model_caches from model_loader")
            clear_model_caches(force_gpu_cleanup=True)
        except Exception as e:
            logger.warning(f"Error calling clear_model_caches: {e}")

        # Run aggressive cleanup
        self.aggressive_cleanup()

# Create a global instance of the memory manager
memory_manager = MemoryManager()