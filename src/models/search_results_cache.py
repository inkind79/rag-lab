"""
Cache for storing and retrieving search results.

Ensures consistent retrieval results for identical queries. Entries are
persisted as JSON (safely loaded, unlike pickle); numpy scalars are
coerced to native Python numbers on write so they round-trip cleanly.
"""

import hashlib
import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

from src.utils.logger import get_logger

logger = get_logger(__name__)

_search_cache_instance = None

_ENTRY_SUFFIX = ".json"
_LEGACY_SUFFIX = ".pkl"


def _to_jsonable(obj: Any) -> Any:
    """Coerce numpy scalars/arrays and other stragglers to JSON-safe values."""
    import numpy as np

    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _remove_legacy(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass


class SearchResultsCache:
    """Cache search results on disk, keyed by query + session + model + settings."""

    def __init__(self, cache_dir: str = "cache/search_results", cache_expiration: int = 36000):
        self.cache_dir = cache_dir
        self.cache_expiration = cache_expiration

        try:
            os.makedirs(self.cache_dir, exist_ok=True)
            logger.info(f"Ensured cache directory exists at {self.cache_dir}")
        except Exception as e:
            logger.warning(f"Error creating cache directory {self.cache_dir}: {e}")
            try:
                import tempfile
                self.cache_dir = os.path.join(tempfile.gettempdir(), 'search_results_cache')
                os.makedirs(self.cache_dir, exist_ok=True)
                logger.warning(f"Using fallback cache directory: {self.cache_dir}")
            except Exception:
                logger.error("Failed to create fallback cache directory, caching may not work")

        self.metadata_file = os.path.join(self.cache_dir, "metadata.json")
        try:
            if not os.path.exists(self.metadata_file):
                with open(self.metadata_file, "w") as f:
                    json.dump({
                        "created_at": time.time(),
                        "entries": 0,
                        "cache_expiration": self.cache_expiration,
                    }, f)
        except Exception as e:
            logger.warning(f"Error creating metadata file: {e}")

        try:
            cleaned = self._clean_expired_entries()
            logger.info(f"Cleaned {cleaned} expired cache entries during initialization")
        except Exception as e:
            logger.warning(f"Error cleaning expired cache entries: {e}")

        logger.info(
            f"Initialized search results cache at {self.cache_dir} "
            f"with {self.cache_expiration}s expiration"
        )

    def _entry_path(self, cache_key: str) -> str:
        return os.path.join(self.cache_dir, f"{cache_key}{_ENTRY_SUFFIX}")

    def _legacy_path(self, cache_key: str) -> str:
        return os.path.join(self.cache_dir, f"{cache_key}{_LEGACY_SUFFIX}")

    def _get_cache_key(
        self,
        query: str,
        session_id: str,
        selected_filenames: Optional[List[str]] = None,
        model_name: Optional[str] = None,
    ) -> str:
        key_parts = [query, session_id]
        if model_name:
            key_parts.append(f"model:{model_name}")
        if selected_filenames:
            key_parts.extend(sorted(selected_filenames))

        try:
            from src.services.session_manager.manager import load_session
            session_data = load_session('sessions', session_id)
            if session_data:
                for k in (
                    'retrieval_count', 'use_score_slope', 'rel_drop_threshold',
                    'abs_score_threshold', 'retrieval_method', 'hybrid_visual_weight',
                ):
                    key_parts.append(f"{k}:{session_data.get(k)}")
                logger.info(
                    f"Including retrieval settings in cache key: "
                    f"count={session_data.get('retrieval_count')}, "
                    f"score_slope={session_data.get('use_score_slope')}"
                )
        except Exception as e:
            logger.warning(f"Error loading session data for cache key: {e}")

        return hashlib.md5("|".join(str(p) for p in key_parts).encode()).hexdigest()

    def get(
        self,
        query: str,
        session_id: str,
        selected_filenames: Optional[List[str]] = None,
        model_name: Optional[str] = None,
    ) -> Optional[Tuple[List[Dict], Dict]]:
        if not query or not session_id:
            logger.warning("Cannot get from cache with empty query or session_id")
            return None

        try:
            try:
                self._clean_expired_entries()
            except Exception as e:
                logger.warning(f"Error cleaning expired entries: {e}, continuing")

            cache_key = self._get_cache_key(query, session_id, selected_filenames, model_name)
            entry_path = self._entry_path(cache_key)

            # Legacy pickle entries: drop on sight. Cache is regenerable.
            legacy = self._legacy_path(cache_key)
            if os.path.exists(legacy):
                _remove_legacy(legacy)

            if not os.path.exists(entry_path):
                logger.debug(f"No cache file found at {entry_path}")
                return None

            try:
                logger.info(f"Loading cached search results for query: '{query[:50]}...'")
                with open(entry_path, "r") as f:
                    cached_data = json.load(f)
            except Exception as e:
                logger.warning(f"Error loading cached search results: {e}")
                _remove_legacy(entry_path)
                return None

            timestamp = cached_data.get("timestamp", 0)
            current_time = time.time()
            if current_time - timestamp > self.cache_expiration:
                logger.info(f"Cache entry has expired (age: {current_time - timestamp:.0f}s), removing")
                _remove_legacy(entry_path)
                return None

            results = cached_data.get("results", [])
            analysis = cached_data.get("analysis", {})
            if not isinstance(results, list):
                logger.warning(f"Invalid cached results format, expected list but got {type(results)}")
                return None

            logger.info(
                f"Found cached results: {len(results)} documents "
                f"(cache age: {int(current_time - timestamp)}s)"
            )
            return results, analysis
        except Exception as e:
            logger.warning(f"Unexpected error in cache get: {e}")
            return None

    def put(
        self,
        query: str,
        session_id: str,
        results: List[Dict],
        analysis: Dict,
        selected_filenames: Optional[List[str]] = None,
        model_name: Optional[str] = None,
    ) -> bool:
        if not query or not session_id or not results:
            logger.warning("Cannot cache with empty query, session_id or results")
            return False

        try:
            os.makedirs(self.cache_dir, exist_ok=True)
            cache_key = self._get_cache_key(query, session_id, selected_filenames, model_name)
            entry_path = self._entry_path(cache_key)

            cache_data = _to_jsonable({
                "query": query,
                "session_id": session_id,
                "selected_filenames": selected_filenames,
                "results": results,
                "analysis": analysis,
                "timestamp": time.time(),
            })

            try:
                logger.info(f"Caching search results for query: '{query[:50]}...'")
                with open(entry_path, "w") as f:
                    json.dump(cache_data, f)
                logger.info(f"Successfully wrote {len(results)} results to cache at {entry_path}")
            except Exception as e:
                logger.warning(f"Error writing to cache file: {e}")
                return False

            try:
                if os.path.exists(self.metadata_file):
                    with open(self.metadata_file, "r") as f:
                        metadata = json.load(f)
                else:
                    metadata = {
                        "created_at": time.time(),
                        "entries": 0,
                        "cache_expiration": self.cache_expiration,
                    }
                metadata["entries"] += 1
                metadata["last_updated"] = time.time()
                with open(self.metadata_file, "w") as f:
                    json.dump(metadata, f)
            except Exception as e:
                logger.warning(f"Error updating metadata: {e}, cache still works")

            return True
        except Exception as e:
            logger.warning(f"Unexpected error caching search results: {e}")
            return False

    def clear(self) -> bool:
        try:
            for filename in os.listdir(self.cache_dir):
                if filename != "metadata.json":
                    os.remove(os.path.join(self.cache_dir, filename))
            with open(self.metadata_file, "w") as f:
                json.dump({"created_at": time.time(), "entries": 0}, f)
            logger.info(f"Cleared search results cache at {self.cache_dir}")
            return True
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            return False

    def _iter_entries(self):
        """Yield (cache_file, cached_data) for every non-legacy JSON entry.

        Silently drops legacy .pkl files and malformed JSON.
        """
        for filename in os.listdir(self.cache_dir):
            if filename == "metadata.json":
                continue
            cache_file = os.path.join(self.cache_dir, filename)
            if filename.endswith(_LEGACY_SUFFIX):
                _remove_legacy(cache_file)
                continue
            if not filename.endswith(_ENTRY_SUFFIX):
                continue
            try:
                with open(cache_file, "r") as f:
                    cached_data = json.load(f)
            except Exception:
                continue
            yield cache_file, cached_data

    def _clean_expired_entries(self) -> int:
        removed = 0
        current_time = time.time()
        for cache_file, cached_data in list(self._iter_entries()):
            if current_time - cached_data.get("timestamp", 0) > self.cache_expiration:
                try:
                    os.remove(cache_file)
                    removed += 1
                except OSError:
                    continue
        if removed > 0:
            logger.info(f"Cleaned {removed} expired cache entries")
        return removed

    def clear_for_session(self, session_id: str) -> bool:
        try:
            removed = 0
            for cache_file, cached_data in list(self._iter_entries()):
                if cached_data.get("session_id") == session_id:
                    try:
                        os.remove(cache_file)
                        removed += 1
                    except OSError:
                        continue
            logger.info(f"Cleared {removed} cache entries for session {session_id}")
            return True
        except Exception as e:
            logger.error(f"Error clearing cache for session {session_id}: {e}")
            return False

    def clear_for_document(self, session_id: str, document_filename: str) -> bool:
        return self.clear_for_documents(session_id, [document_filename])

    def clear_for_documents(self, session_id: str, document_filenames: list) -> bool:
        try:
            removed = 0
            doc_set = set(document_filenames)
            for cache_file, cached_data in list(self._iter_entries()):
                if cached_data.get("session_id") != session_id:
                    continue
                selected = cached_data.get("selected_filenames") or []
                if isinstance(selected, list) and any(d in doc_set for d in selected):
                    try:
                        os.remove(cache_file)
                        removed += 1
                    except OSError:
                        continue
            logger.info(
                f"Cleared {removed} cache entries for "
                f"{len(document_filenames)} documents in session {session_id}"
            )
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
