"""
Cache for storing and retrieving search results.

This module provides a cache for storing and retrieving search results,
which ensures consistent retrieval results for identical queries.
"""

import os
import hashlib
import pickle
import json
import time
from typing import Dict, List, Any, Optional, Tuple
from src.utils.logger import get_logger
from src.utils.secure_dirs import secure_makedirs

logger = get_logger(__name__)

# Global singleton instance
_search_cache_instance = None

class SearchResultsCache:
    """
    Cache for storing and retrieving search results.

    This cache stores search results on disk, keyed by a hash of the query text
    and document set. It ensures that identical queries always produce the same results.
    """

    def __init__(self, cache_dir="cache/search_results", cache_expiration=36000):
        """
        Initialize the search results cache.

        Args:
            cache_dir: Directory to store the cache (default: "cache/search_results")
            cache_expiration: Time in seconds after which cache entries expire (default: 10 hours)
        """
        self.cache_dir = cache_dir
        self.cache_expiration = cache_expiration

        # Create cache directory if it doesn't exist
        try:
            secure_makedirs(self.cache_dir)
            logger.info(f"Ensured cache directory exists at {self.cache_dir}")
        except Exception as e:
            logger.warning(f"Error creating cache directory {self.cache_dir}: {e}")
            # Create a fallback directory in /tmp if possible
            try:
                import tempfile
                self.cache_dir = os.path.join(tempfile.gettempdir(), 'search_results_cache')
                secure_makedirs(self.cache_dir)
                logger.warning(f"Using fallback cache directory: {self.cache_dir}")
            except Exception:
                logger.error(f"Failed to create fallback cache directory, caching may not work")

        # Create metadata file if it doesn't exist
        try:
            self.metadata_file = os.path.join(self.cache_dir, "metadata.json")
            if not os.path.exists(self.metadata_file):
                with open(self.metadata_file, "w") as f:
                    json.dump({
                        "created_at": time.time(),
                        "entries": 0,
                        "cache_expiration": self.cache_expiration
                    }, f)
                logger.info(f"Created metadata file at {self.metadata_file}")
        except Exception as e:
            logger.warning(f"Error creating metadata file: {e}")

        # Clean expired cache entries on initialization
        try:
            cleaned = self._clean_expired_entries()
            logger.info(f"Cleaned {cleaned} expired cache entries during initialization")
        except Exception as e:
            logger.warning(f"Error cleaning expired cache entries: {e}")

        logger.info(f"Initialized search results cache at {self.cache_dir} with {self.cache_expiration}s expiration")


    def _get_cache_key(self, query: str, session_id: str, selected_filenames: Optional[List[str]] = None, model_name: Optional[str] = None) -> str:
        """
        Generate a cache key for the given query and document set.

        Args:
            query: The query text
            session_id: The session ID
            selected_filenames: Optional list of selected filenames
            model_name: Optional embedding model name (prevents cross-model cache collisions)

        Returns:
            A hash of the query and document set
        """
        # Create a deterministic string representation of the inputs
        key_parts = [query, session_id]

        # Include model name to prevent cross-model cache collisions
        if model_name:
            key_parts.append(f"model:{model_name}")

        # Add selected filenames if provided
        if selected_filenames:
            # Sort to ensure deterministic order
            sorted_filenames = sorted(selected_filenames)
            key_parts.extend(sorted_filenames)

        # Add retrieval settings from session data to ensure different cache keys
        # when retrieval settings change
        try:
            from src.services.session_manager.manager import load_session
            session_data = load_session('sessions', session_id)
            if session_data:
                # Include key retrieval settings in the cache key
                retrieval_count = str(session_data.get('retrieval_count', 3))
                use_score_slope = str(session_data.get('use_score_slope', True))
                rel_drop_threshold = str(session_data.get('rel_drop_threshold', 0.65))
                abs_score_threshold = str(session_data.get('abs_score_threshold', 0.25))
                retrieval_method = str(session_data.get('retrieval_method', 'colpali'))
                hybrid_visual_weight = str(session_data.get('hybrid_visual_weight', 0.6))

                # Add these settings to the key parts
                key_parts.extend([
                    f"retrieval_count:{retrieval_count}",
                    f"use_score_slope:{use_score_slope}",
                    f"rel_drop_threshold:{rel_drop_threshold}",
                    f"abs_score_threshold:{abs_score_threshold}",
                    f"retrieval_method:{retrieval_method}",
                    f"hybrid_visual_weight:{hybrid_visual_weight}",
                ])
                logger.info(f"Including retrieval settings in cache key: count={retrieval_count}, score_slope={use_score_slope}")
        except Exception as e:
            logger.warning(f"Error loading session data for cache key: {e}")

        # Join all parts with a separator
        key_string = "|".join(key_parts)

        # Create a hash
        return hashlib.md5(key_string.encode()).hexdigest()

    def get(self, query: str, session_id: str, selected_filenames: Optional[List[str]] = None, model_name: Optional[str] = None) -> Optional[Tuple[List[Dict], Dict]]:
        """
        Get search results from the cache.

        Args:
            query: The query text
            session_id: The session ID
            selected_filenames: Optional list of selected filenames
            model_name: Optional embedding model name (prevents cross-model cache collisions)

        Returns:
            A tuple of (search results, analysis metadata) if found, None otherwise
        """
        if not query or not session_id:
            logger.warning(f"Cannot get from cache with empty query or session_id")
            return None

        try:
            # First, clean any expired entries
            try:
                self._clean_expired_entries()
            except Exception as e:
                logger.warning(f"Error cleaning expired entries: {e}, will continue checking cache anyway")

            # Generate cache key
            try:
                cache_key = self._get_cache_key(query, session_id, selected_filenames, model_name)
            except Exception as e:
                logger.warning(f"Error generating cache key: {e}")
                return None
                
            # Ensure cache directory exists
            try:
                secure_makedirs(self.cache_dir)
            except Exception as e:
                logger.warning(f"Error ensuring cache directory exists: {e}")
                return None
                
            cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")

            if os.path.exists(cache_file):
                try:
                    logger.info(f"Loading cached search results for query: '{query[:50]}...'")
                    with open(cache_file, "rb") as f:
                        cached_data = pickle.load(f)

                    # Check if the cache entry has expired
                    timestamp = cached_data.get("timestamp", 0)
                    current_time = time.time()
                    if current_time - timestamp > self.cache_expiration:
                        logger.info(f"Cache entry has expired (age: {current_time - timestamp:.0f}s), removing")
                        try:
                            os.remove(cache_file)
                        except Exception as e:
                            logger.warning(f"Error removing expired cache file: {e}")
                        return None

                    # Log some details about the cached results
                    results = cached_data.get("results", [])
                    analysis = cached_data.get("analysis", {})
                    cache_age = int(current_time - timestamp)
                    logger.info(f"Found cached results: {len(results)} documents (cache age: {cache_age}s)")

                    # Validate the cached results (make sure they match the expected format)
                    if not isinstance(results, list):
                        logger.warning(f"Invalid cached results format, expected list but got {type(results)}")
                        return None
                        
                    # If the results are valid, return them
                    return results, analysis
                except Exception as e:
                    logger.warning(f"Error loading cached search results: {e}")
                    # Remove the cache file if it's corrupted
                    try:
                        os.remove(cache_file)
                        logger.info(f"Removed corrupted cache file: {cache_file}")
                    except Exception as remove_e:
                        logger.warning(f"Error removing corrupted cache file: {remove_e}")
                    return None
            else:
                logger.debug(f"No cache file found at {cache_file}")
        except Exception as e:
            logger.warning(f"Unexpected error in cache get: {e}")
            
        return None

    def put(self, query: str, session_id: str, results: List[Dict], analysis: Dict, selected_filenames: Optional[List[str]] = None, model_name: Optional[str] = None) -> bool:
        """
        Put search results in the cache.

        Args:
            query: The query text
            session_id: The session ID
            results: The search results to cache
            analysis: The analysis metadata to cache
            selected_filenames: Optional list of selected filenames
            model_name: Optional embedding model name (prevents cross-model cache collisions)

        Returns:
            True if successful, False otherwise
        """
        if not query or not session_id or not results:
            logger.warning(f"Cannot cache with empty query, session_id or results")
            return False

        try:
            # Ensure cache directory exists
            try:
                secure_makedirs(self.cache_dir)
            except Exception as e:
                logger.warning(f"Error ensuring cache directory exists for put: {e}")
                return False

            # Generate cache key
            try:
                cache_key = self._get_cache_key(query, session_id, selected_filenames, model_name)
            except Exception as e:
                logger.warning(f"Error generating cache key for put: {e}")
                return False
                
            cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")

            # Prepare data to cache
            cache_data = {
                "query": query,
                "session_id": session_id,
                "selected_filenames": selected_filenames,
                "results": results,
                "analysis": analysis,
                "timestamp": time.time()
            }

            # Write to cache file
            try:
                logger.info(f"Caching search results for query: '{query[:50]}...'")
                with open(cache_file, "wb") as f:
                    pickle.dump(cache_data, f)
                logger.info(f"Successfully wrote {len(results)} results to cache at {cache_file}")
            except Exception as e:
                logger.warning(f"Error writing to cache file: {e}")
                return False

            # Update metadata
            try:
                if os.path.exists(self.metadata_file):
                    with open(self.metadata_file, "r") as f:
                        metadata = json.load(f)
                else:
                    metadata = {
                        "created_at": time.time(),
                        "entries": 0,
                        "cache_expiration": self.cache_expiration
                    }

                metadata["entries"] += 1
                metadata["last_updated"] = time.time()

                with open(self.metadata_file, "w") as f:
                    json.dump(metadata, f)
                logger.debug(f"Updated cache metadata, now {metadata['entries']} total entries")
            except Exception as e:
                logger.warning(f"Error updating metadata: {e}, cache will still work")
                # Don't return False here, as the cache entry was still written

            return True
        except Exception as e:
            logger.warning(f"Unexpected error caching search results: {e}")
            return False

    def clear(self) -> bool:
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
                    "created_at": time.time(),
                    "entries": 0
                }, f)

            logger.info(f"Cleared search results cache at {self.cache_dir}")
            return True
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            return False

    def _clean_expired_entries(self) -> int:
        """
        Clean expired cache entries.

        Returns:
            Number of removed entries
        """
        try:
            # Count how many files we remove
            removed_count = 0
            current_time = time.time()

            # Remove all cache files that have expired
            for filename in os.listdir(self.cache_dir):
                if filename != "metadata.json":
                    cache_file = os.path.join(self.cache_dir, filename)
                    try:
                        with open(cache_file, "rb") as f:
                            cached_data = pickle.load(f)

                        # Check if this cache entry has expired
                        timestamp = cached_data.get("timestamp", 0)
                        if current_time - timestamp > self.cache_expiration:
                            os.remove(cache_file)
                            removed_count += 1
                    except Exception:
                        # Skip files that can't be loaded
                        continue

            if removed_count > 0:
                logger.info(f"Cleaned {removed_count} expired cache entries")
            return removed_count
        except Exception as e:
            logger.warning(f"Error cleaning expired cache entries: {e}")
            return 0

    def clear_for_session(self, session_id: str) -> bool:
        """
        Clear the cache for a specific session.

        Args:
            session_id: The session ID to clear cache for

        Returns:
            True if successful, False otherwise
        """
        try:
            # Count how many files we remove
            removed_count = 0

            # Remove all cache files that contain the session ID
            for filename in os.listdir(self.cache_dir):
                if filename != "metadata.json":
                    cache_file = os.path.join(self.cache_dir, filename)
                    try:
                        with open(cache_file, "rb") as f:
                            cached_data = pickle.load(f)

                        # Check if this cache entry is for the specified session
                        if cached_data.get("session_id") == session_id:
                            os.remove(cache_file)
                            removed_count += 1
                    except Exception:
                        # Skip files that can't be loaded
                        continue

            logger.info(f"Cleared {removed_count} cache entries for session {session_id}")
            return True
        except Exception as e:
            logger.error(f"Error clearing cache for session {session_id}: {e}")
            return False
            
    def clear_for_document(self, session_id: str, document_filename: str) -> bool:
        """
        Clear the cache for a specific document within a session.

        Args:
            session_id: The session ID containing the document
            document_filename: The filename of the document to clear cache for

        Returns:
            True if successful, False otherwise
        """
        try:
            # Count how many files we remove
            removed_count = 0

            # Remove all cache files that contain both the session ID and document filename
            for filename in os.listdir(self.cache_dir):
                if filename != "metadata.json":
                    cache_file = os.path.join(self.cache_dir, filename)
                    try:
                        with open(cache_file, "rb") as f:
                            cached_data = pickle.load(f)

                        # Check if this cache entry is for the specified session
                        if cached_data.get("session_id") == session_id:
                            # Check if this cache entry includes the specified document
                            selected_filenames = cached_data.get("selected_filenames", [])
                            if isinstance(selected_filenames, list) and document_filename in selected_filenames:
                                os.remove(cache_file)
                                removed_count += 1
                    except Exception:
                        # Skip files that can't be loaded
                        continue

            logger.info(f"Cleared {removed_count} cache entries for document {document_filename} in session {session_id}")
            return True
        except Exception as e:
            logger.error(f"Error clearing cache for document {document_filename} in session {session_id}: {e}")
            return False

    def clear_for_documents(self, session_id: str, document_filenames: list) -> bool:
        """
        Clear the cache for specific documents within a session.

        Args:
            session_id: The session ID containing the documents
            document_filenames: List of document filenames to clear cache for

        Returns:
            True if successful, False otherwise
        """
        try:
            # Count how many files we remove
            removed_count = 0

            # Remove all cache files that contain both the session ID and any of the document filenames
            for filename in os.listdir(self.cache_dir):
                if filename != "metadata.json":
                    cache_file = os.path.join(self.cache_dir, filename)
                    try:
                        with open(cache_file, "rb") as f:
                            cached_data = pickle.load(f)

                        # Check if this cache entry is for the specified session
                        if cached_data.get("session_id") == session_id:
                            # Check if this cache entry includes any of the specified documents
                            selected_filenames = cached_data.get("selected_filenames", [])
                            if isinstance(selected_filenames, list) and any(doc in selected_filenames for doc in document_filenames):
                                os.remove(cache_file)
                                removed_count += 1
                    except Exception:
                        # Skip files that can't be loaded
                        continue

            logger.info(f"Cleared {removed_count} cache entries for {len(document_filenames)} documents in session {session_id}")
            return True
        except Exception as e:
            logger.error(f"Error clearing cache for documents in session {session_id}: {e}")
            return False


def get_search_cache():
    """Get the singleton SearchResultsCache instance."""
    global _search_cache_instance
    if _search_cache_instance is None:
        _search_cache_instance = SearchResultsCache()
    return _search_cache_instance