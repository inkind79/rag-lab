"""
Response Cache Module

Provides caching for generated responses to avoid redundant LLM calls
and improve response times for repeated or similar queries.
"""

import hashlib
import json
import time
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime, timedelta
import threading
from collections import OrderedDict

from src.utils.logger import get_logger

logger = get_logger(__name__)


class ResponseCache:
    """
    Cache for generated responses with TTL and size limits.
    
    This cache is designed to store full responses along with their
    retrieval context to ensure consistency.
    """
    
    def __init__(self, max_size: int = 100, ttl_seconds: int = 3600):
        """
        Initialize the response cache.
        
        Args:
            max_size: Maximum number of responses to cache
            ttl_seconds: Time-to-live for cached responses in seconds
        """
        self.cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._lock = threading.RLock()
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0
        }
    
    def _generate_cache_key(
        self,
        query: str,
        selected_filenames: List[str],
        conversation_context: str,
        generation_model: str,
        retrieval_settings: Dict[str, Any]
    ) -> str:
        """
        Generate a unique cache key for the response.
        
        Args:
            query: The user's query
            selected_filenames: List of selected files
            conversation_context: Recent conversation history
            generation_model: Model used for generation
            retrieval_settings: Settings like k, similarity_threshold, etc.
        
        Returns:
            A unique cache key
        """
        # Create a dictionary with all relevant parameters
        key_data = {
            'query': query.strip().lower(),
            'files': sorted(selected_filenames) if selected_filenames else [],
            'context_hash': hashlib.md5(conversation_context.encode()).hexdigest() if conversation_context else '',
            'model': generation_model,
            'settings': {
                'k': retrieval_settings.get('k', 5),
                'similarity_threshold': retrieval_settings.get('similarity_threshold', 0.0),
                'score_slope_enabled': retrieval_settings.get('score_slope_enabled', False),
                'token_budget_enabled': retrieval_settings.get('token_budget_enabled', False)
            }
        }
        
        # Generate hash from the key data
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_str.encode()).hexdigest()
    
    def get(
        self,
        query: str,
        selected_filenames: List[str],
        conversation_context: str,
        generation_model: str,
        retrieval_settings: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Get a cached response if available and not expired.
        
        Returns:
            Cached response data or None if not found/expired
        """
        cache_key = self._generate_cache_key(
            query, selected_filenames, conversation_context,
            generation_model, retrieval_settings
        )
        
        with self._lock:
            if cache_key in self.cache:
                entry = self.cache[cache_key]
                
                # Check if entry has expired
                if time.time() - entry['timestamp'] > self.ttl_seconds:
                    # Remove expired entry
                    del self.cache[cache_key]
                    self.stats['misses'] += 1
                    logger.debug(f"Cache miss (expired) for query: '{query[:50]}...'")
                    return None
                
                # Move to end (most recently used)
                self.cache.move_to_end(cache_key)
                self.stats['hits'] += 1
                
                logger.info(f"Cache hit for query: '{query[:50]}...' (key: {cache_key[:8]}...)")
                return entry['data']
            
            self.stats['misses'] += 1
            logger.debug(f"Cache miss for query: '{query[:50]}...'")
            return None
    
    def put(
        self,
        query: str,
        selected_filenames: List[str],
        conversation_context: str,
        generation_model: str,
        retrieval_settings: Dict[str, Any],
        response_data: Dict[str, Any]
    ):
        """
        Store a response in the cache.
        
        Args:
            response_data: Dictionary containing response and metadata
        """
        cache_key = self._generate_cache_key(
            query, selected_filenames, conversation_context,
            generation_model, retrieval_settings
        )
        
        with self._lock:
            # Remove oldest entry if at capacity
            if len(self.cache) >= self.max_size and cache_key not in self.cache:
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
                self.stats['evictions'] += 1
                logger.debug(f"Evicted oldest cache entry to make room")
            
            # Store the new entry
            self.cache[cache_key] = {
                'timestamp': time.time(),
                'data': response_data,
                'query': query,
                'model': generation_model
            }
            
            # Move to end (most recently used)
            self.cache.move_to_end(cache_key)
            
            logger.info(f"Cached response for query: '{query[:50]}...' (key: {cache_key[:8]}...)")
    
    def clear(self):
        """Clear all cached responses."""
        with self._lock:
            self.cache.clear()
            logger.info("Response cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_requests = self.stats['hits'] + self.stats['misses']
            hit_rate = self.stats['hits'] / total_requests if total_requests > 0 else 0
            
            return {
                'size': len(self.cache),
                'max_size': self.max_size,
                'hits': self.stats['hits'],
                'misses': self.stats['misses'],
                'evictions': self.stats['evictions'],
                'hit_rate': hit_rate,
                'ttl_seconds': self.ttl_seconds
            }
    
    def cleanup_expired(self):
        """Remove expired entries from the cache."""
        with self._lock:
            current_time = time.time()
            expired_keys = []
            
            for key, entry in self.cache.items():
                if current_time - entry['timestamp'] > self.ttl_seconds:
                    expired_keys.append(key)
            
            for key in expired_keys:
                del self.cache[key]
            
            if expired_keys:
                logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")


# Global instance
_response_cache = None
_cache_lock = threading.Lock()


def get_response_cache() -> ResponseCache:
    """Get the global response cache instance."""
    global _response_cache
    
    with _cache_lock:
        if _response_cache is None:
            # Default: 100 responses, 1 hour TTL
            _response_cache = ResponseCache(max_size=100, ttl_seconds=3600)
    
    return _response_cache


def check_response_cache(
    query: str,
    selected_filenames: List[str],
    conversation_context: str,
    generation_model: str,
    retrieval_settings: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """
    Check if a response is cached.
    
    Returns:
        Cached response data or None
    """
    cache = get_response_cache()
    return cache.get(
        query, selected_filenames, conversation_context,
        generation_model, retrieval_settings
    )


def cache_response(
    query: str,
    selected_filenames: List[str],
    conversation_context: str,
    generation_model: str,
    retrieval_settings: Dict[str, Any],
    response_data: Dict[str, Any]
):
    """Store a response in the cache."""
    cache = get_response_cache()
    cache.put(
        query, selected_filenames, conversation_context,
        generation_model, retrieval_settings, response_data
    )