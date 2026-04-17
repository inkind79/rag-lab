"""
Embedding cache for storing and retrieving embeddings.

This module provides a cache for storing and retrieving embeddings,
which is essential for consistent retrieval results.

Storage format: numpy ``.npy`` files (loaded with ``allow_pickle=False``),
with a small ``.meta.json`` sidecar recording the original tensor dtype so
torch Tensors can round-trip through numpy without type drift.
"""

import os
import hashlib
import json
import time
import torch
import numpy as np
from src.utils.logger import get_logger
from src.utils.secure_dirs import secure_makedirs

logger = get_logger(__name__)


class EmbeddingCache:
    """Cache for storing and retrieving embeddings, keyed on text + model."""

    def __init__(self, cache_dir="cache/embeddings", model_name=None, model_version=None):
        self.cache_dir = cache_dir
        self.model_name = model_name
        self.model_version = model_version

        if model_name:
            model_dir = model_name.replace("/", "_")
            self.cache_dir = os.path.join(cache_dir, model_dir)
        
        secure_makedirs(self.cache_dir)

        # Create metadata file if it doesn't exist
        self.metadata_file = os.path.join(self.cache_dir, "metadata.json")
        if not os.path.exists(self.metadata_file):
            with open(self.metadata_file, "w") as f:
                json.dump({
                    "model_name": model_name,
                    "model_version": model_version,
                    "created_at": time.time(),
                    "entries": 0,
                }, f)

        logger.info(f"Initialized embedding cache at {self.cache_dir}")

    def _get_cache_key(self, text: str) -> str:
        key_string = f"{text}_{self.model_name}_{self.model_version}"
        return hashlib.md5(key_string.encode()).hexdigest()

    def _paths(self, cache_key: str) -> tuple[str, str, str]:
        npy = os.path.join(self.cache_dir, f"{cache_key}.npy")
        meta = os.path.join(self.cache_dir, f"{cache_key}.meta.json")
        legacy_pkl = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        return npy, meta, legacy_pkl

    def get(self, text: str):
        cache_key = self._get_cache_key(text)
        npy_path, meta_path, legacy_pkl = self._paths(cache_key)

        # Drop legacy pickle entries if present — they are unsafe to load
        # and cache is regenerable.
        if os.path.exists(legacy_pkl):
            try:
                os.remove(legacy_pkl)
                logger.info(f"Removed legacy pickle cache entry: {legacy_pkl}")
            except OSError:
                pass

        if not os.path.exists(npy_path):
            return None

        try:
            array = np.load(npy_path, allow_pickle=False)
            meta = {}
            if os.path.exists(meta_path):
                with open(meta_path) as f:
                    meta = json.load(f)

            logger.info(f"Loading cached embedding for: {text[:50]}...")
            if meta.get("is_tensor"):
                tensor = torch.from_numpy(array)
                dtype = meta.get("dtype")
                if dtype and hasattr(torch, dtype):
                    tensor = tensor.to(getattr(torch, dtype))
                return tensor
            return array
        except Exception as e:
            logger.warning(f"Error loading cached embedding: {e}")
            return None

    def put(self, text: str, embedding) -> bool:
        try:
            cache_key = self._get_cache_key(text)
            npy_path, meta_path, _ = self._paths(cache_key)

            is_tensor = isinstance(embedding, torch.Tensor)
            if is_tensor:
                tensor = embedding.detach().cpu()
                dtype_name = str(tensor.dtype).split('.')[-1]  # "torch.float32" -> "float32"
                array = tensor.numpy()
            else:
                array = np.asarray(embedding)
                dtype_name = str(array.dtype)

            logger.info(f"Caching embedding for: {text[:50]}...")
            np.save(npy_path, array, allow_pickle=False)
            with open(meta_path, "w") as f:
                json.dump({"is_tensor": is_tensor, "dtype": dtype_name}, f)

            try:
                with open(self.metadata_file, "r") as f:
                    metadata = json.load(f)
                metadata["entries"] += 1
                metadata["last_updated"] = time.time()
                with open(self.metadata_file, "w") as f:
                    json.dump(metadata, f)
            except Exception as e:
                logger.warning(f"Error updating metadata: {e}")

            return True
        except Exception as e:
            logger.warning(f"Error caching embedding: {e}")
            return False

    def clear(self) -> bool:
        try:
            for filename in os.listdir(self.cache_dir):
                if filename != "metadata.json":
                    os.remove(os.path.join(self.cache_dir, filename))

            with open(self.metadata_file, "w") as f:
                json.dump({
                    "model_name": self.model_name,
                    "model_version": self.model_version,
                    "created_at": time.time(),
                    "entries": 0,
                }, f)

            logger.info(f"Cleared embedding cache at {self.cache_dir}")
            return True
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            return False
