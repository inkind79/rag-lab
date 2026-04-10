"""
Query expansion cache to avoid redundant LLM calls.

Caches expanded queries to improve performance.
"""

import hashlib
import json
import time
from typing import Optional, Dict, Any
from src.utils.logger import get_logger
from src.utils.thread_safe_models import ThreadSafeCache

logger = get_logger(__name__)


class QueryExpansionCache:
    """Cache for expanded queries."""
    
    def __init__(self, max_age_seconds=3600):
        """
        Initialize query expansion cache.
        
        Args:
            max_age_seconds: Maximum age for cache entries in seconds
        """
        self.cache = ThreadSafeCache("query_expansion")
        self.max_age_seconds = max_age_seconds
        
    def _generate_key(self, query: str, context: str) -> str:
        """
        Generate a cache key from query and context.
        
        Args:
            query: The original query
            context: The conversation context
            
        Returns:
            Cache key string
        """
        # Create a hash of query + context
        content = f"{query}||{context}"
        return hashlib.md5(content.encode()).hexdigest()
        
    def get(self, query: str, context: str) -> Optional[str]:
        """
        Get expanded query from cache.
        
        Args:
            query: The original query
            context: The conversation context
            
        Returns:
            Expanded query if found and not expired, None otherwise
        """
        key = self._generate_key(query, context)
        entry = self.cache.get(key)
        
        if entry is not None:
            # Check if entry is expired
            if time.time() - entry.get('timestamp', 0) > self.max_age_seconds:
                self.cache.delete(key)
                return None
                
            logger.info(f"Found cached query expansion for: {query[:50]}...")
            return entry.get('expanded_query')
            
        return None
        
    def put(self, query: str, context: str, expanded_query: str):
        """
        Store expanded query in cache.
        
        Args:
            query: The original query
            context: The conversation context
            expanded_query: The expanded query
        """
        key = self._generate_key(query, context)
        entry = {
            'original_query': query,
            'expanded_query': expanded_query,
            'timestamp': time.time()
        }
        
        self.cache.set(key, entry)
        logger.info(f"Cached query expansion: '{query[:50]}...' -> '{expanded_query[:50]}...'")
        
    def clear(self):
        """Clear all cached expansions."""
        self.cache.clear()
        
    def cleanup_old_entries(self):
        """Remove expired entries from cache."""
        return self.cache.cleanup_old_entries(self.max_age_seconds)


# Global instance
_query_expansion_cache = None


def get_query_expansion_cache() -> QueryExpansionCache:
    """Get or create the global query expansion cache."""
    global _query_expansion_cache
    
    if _query_expansion_cache is None:
        _query_expansion_cache = QueryExpansionCache()
        logger.info("Initialized query expansion cache")
        
    return _query_expansion_cache