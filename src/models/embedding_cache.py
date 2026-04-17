"""
Embedding cache for storing and retrieving embeddings.

This module provides a cache for storing and retrieving embeddings,
which is essential for consistent retrieval results.
"""

import os
import hashlib
import pickle
import json
import time
import torch
import numpy as np
from src.utils.logger import get_logger
from src.utils.secure_dirs import secure_makedirs

logger = get_logger(__name__)

class EmbeddingCache:
    """
    Cache for storing and retrieving embeddings.
    
    This cache stores embeddings on disk, keyed by a hash of the text and model information.
    It ensures that the same text and model always produce the same embedding.
    """
    
    def __init__(self, cache_dir="cache/embeddings", model_name=None, model_version=None):
        """
        Initialize the embedding cache.
        
        Args:
            cache_dir: Directory to store the cache (default: "cache/embeddings")
            model_name: Name of the model (default: None)
            model_version: Version of the model (default: None)
        """
        self.cache_dir = cache_dir
        self.model_name = model_name
        self.model_version = model_version
        
        # Create cache directory if it doesn't exist
        if model_name:
            # Replace slashes with underscores to create a valid directory name
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
                    "entries": 0
                }, f)
        
        logger.info(f"Initialized embedding cache at {self.cache_dir}")
    
    def _get_cache_key(self, text):
        """
        Generate a cache key for the given text.
        
        Args:
            text: The text to generate a key for
            
        Returns:
            A hash of the text and model information
        """
        # Create a hash of the text and model information
        key_string = f"{text}_{self.model_name}_{self.model_version}"
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def get(self, text):
        """
        Get an embedding from the cache.
        
        Args:
            text: The text to get the embedding for
            
        Returns:
            The embedding if found, None otherwise
        """
        cache_key = self._get_cache_key(text)
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        
        if os.path.exists(cache_file):
            try:
                logger.info(f"Loading cached embedding for: {text[:50]}...")
                with open(cache_file, "rb") as f:
                    embedding = pickle.load(f)
                return embedding
            except Exception as e:
                logger.warning(f"Error loading cached embedding: {e}")
                return None
        
        return None
    
    def put(self, text, embedding):
        """
        Put an embedding in the cache.
        
        Args:
            text: The text the embedding is for
            embedding: The embedding to cache
            
        Returns:
            True if successful, False otherwise
        """
        try:
            cache_key = self._get_cache_key(text)
            cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
            
            # Ensure embedding is detached from computation graph and moved to CPU
            if isinstance(embedding, torch.Tensor):
                embedding = embedding.detach().cpu()
            
            logger.info(f"Caching embedding for: {text[:50]}...")
            with open(cache_file, "wb") as f:
                pickle.dump(embedding, f)
            
            # Update metadata
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
    
    def clear(self):
        """
        Clear the cache.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Remove all cache files except metadata
            for filename in os.listdir(self.cache_dir):
                if filename != "metadata.json":
                    os.remove(os.path.join(self.cache_dir, filename))
            
            # Reset metadata
            with open(self.metadata_file, "w") as f:
                json.dump({
                    "model_name": self.model_name,
                    "model_version": self.model_version,
                    "created_at": time.time(),
                    "entries": 0
                }, f)
            
            logger.info(f"Cleared embedding cache at {self.cache_dir}")
            return True
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            return False
