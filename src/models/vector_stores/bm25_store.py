"""
BM25 Inverted Index Store

Provides BM25 keyword-based retrieval without vector embeddings.
Stores tokenized text per session, persisted to disk.
"""

import os
import pickle
import threading
from typing import List, Dict, Any, Optional, Tuple

from src.models.vector_stores.base_store import BaseVectorStore
from src.utils.logger import get_logger

logger = get_logger(__name__)

BM25_DATA_PATH = os.path.join(os.getcwd(), ".bm25")


class BM25Store(BaseVectorStore):
    """BM25 inverted index store using the bm25s library.

    Each session gets its own index directory at .bm25/{session_id}/.
    The index is built from raw text chunks and persisted to disk.
    """

    def __init__(self, store_dir: str = BM25_DATA_PATH):
        self._store_dir = store_dir
        self._indices = {}  # session_id -> (bm25 index, metadata, texts, ids)
        self._lock = threading.Lock()

    def _session_dir(self, session_id: str) -> str:
        return os.path.join(self._store_dir, session_id)

    def _load_index(self, session_id: str):
        """Load a persisted BM25 index for a session."""
        if session_id in self._indices:
            return self._indices[session_id]

        index_path = os.path.join(self._session_dir(session_id), "index.pkl")
        if os.path.exists(index_path):
            try:
                with open(index_path, 'rb') as f:
                    data = pickle.load(f)
                self._indices[session_id] = data
                logger.info(f"Loaded BM25 index for session {session_id}: {len(data['ids'])} documents")
                return data
            except Exception as e:
                logger.error(f"Error loading BM25 index: {e}")
        return None

    def _save_index(self, session_id: str, data: dict):
        """Persist BM25 index to disk."""
        session_dir = self._session_dir(session_id)
        os.makedirs(session_dir, exist_ok=True)
        index_path = os.path.join(session_dir, "index.pkl")
        with open(index_path, 'wb') as f:
            pickle.dump(data, f)
        logger.info(f"Saved BM25 index for session {session_id}")

    def add_documents(
        self,
        session_id: str,
        embeddings: List[Any],
        ids: List[str],
        metadatas: List[Dict[str, Any]],
        texts: Optional[List[str]] = None,
    ) -> bool:
        """Build BM25 index from text documents. Embeddings are ignored."""
        if not texts:
            logger.error("BM25Store requires texts parameter")
            return False

        try:
            import bm25s
        except ImportError:
            logger.error("bm25s not installed. Install with: pip install bm25s")
            return False

        try:
            # Load existing or start fresh
            existing = self._load_index(session_id)
            if existing:
                # Append to existing index — rebuild with all docs
                all_texts = existing['texts'] + list(texts)
                all_ids = existing['ids'] + list(ids)
                all_metadatas = existing['metadatas'] + list(metadatas)
            else:
                all_texts = list(texts)
                all_ids = list(ids)
                all_metadatas = list(metadatas)

            # Build the BM25 index once with all documents
            all_tokens = bm25s.tokenize(all_texts, stopwords="en")
            retriever = bm25s.BM25()
            retriever.index(all_tokens)

            data = {
                'retriever': retriever,
                'texts': all_texts,
                'ids': all_ids,
                'metadatas': all_metadatas,
            }

            with self._lock:
                self._indices[session_id] = data
            self._save_index(session_id, data)

            logger.info(f"BM25 index built for session {session_id}: {len(all_texts)} documents")
            return True

        except Exception as e:
            logger.error(f"Error building BM25 index: {e}", exc_info=True)
            return False

    def query(
        self,
        session_id: str,
        query_embedding: Any = None,
        query_text: Optional[str] = None,
        k: int = 10,
        filter_dict: Optional[Dict] = None,
        similarity_threshold: float = 0.0,
    ) -> Tuple[List[Dict[str, Any]], List[float], List[str]]:
        """Query BM25 index with text query. query_embedding is ignored."""
        if not query_text:
            logger.warning("BM25Store.query requires query_text")
            return [], [], []

        try:
            import bm25s
        except ImportError:
            logger.error("bm25s not installed")
            return [], [], []

        data = self._load_index(session_id)
        if not data:
            logger.warning(f"No BM25 index found for session {session_id}")
            return [], [], []

        try:
            retriever = data['retriever']
            query_tokens = bm25s.tokenize([query_text], stopwords="en")

            # Get results
            results, scores = retriever.retrieve(query_tokens, k=min(k, len(data['texts'])))

            # results shape: [1, k] (indices into corpus)
            # scores shape: [1, k]
            result_indices = results[0]
            result_scores = scores[0]

            metadatas_out = []
            scores_out = []
            ids_out = []

            for idx, score in zip(result_indices, result_scores):
                idx = int(idx)
                if idx < 0 or idx >= len(data['texts']):
                    continue

                score = float(score)
                if score < similarity_threshold:
                    continue

                metadata = data['metadatas'][idx].copy()
                metadata['text'] = data['texts'][idx]
                doc_id = data['ids'][idx]

                # Apply filename filter if provided
                if filter_dict and 'filename' in filter_dict:
                    allowed = filter_dict['filename']
                    if isinstance(allowed, list) and metadata.get('filename') not in allowed:
                        continue

                metadatas_out.append(metadata)
                scores_out.append(score)
                ids_out.append(doc_id)

            logger.info(f"BM25 query returned {len(metadatas_out)} results for: '{query_text[:50]}{'...' if len(query_text) > 50 else ''}'")
            return metadatas_out, scores_out, ids_out

        except Exception as e:
            logger.error(f"Error querying BM25 index: {e}", exc_info=True)
            return [], [], []

    def delete_session(self, session_id: str) -> bool:
        """Delete BM25 index for a session."""
        import shutil
        session_dir = self._session_dir(session_id)
        if os.path.exists(session_dir):
            shutil.rmtree(session_dir)
        self._indices.pop(session_id, None)
        return True

    def has_documents(self, session_id: str) -> bool:
        """Check if a BM25 index exists for this session."""
        data = self._load_index(session_id)
        return data is not None and len(data.get('ids', [])) > 0


# Module-level singleton for shared access between indexing and retrieval
_default_bm25_store = None
_store_lock = threading.Lock()


def get_bm25_store(store_dir: str = BM25_DATA_PATH) -> BM25Store:
    """Get the shared BM25Store singleton.

    Ensures the same instance (and in-memory cache) is used by both
    the document indexing pipeline and the BM25Retriever.
    """
    global _default_bm25_store
    with _store_lock:
        if _default_bm25_store is None:
            _default_bm25_store = BM25Store(store_dir)
        return _default_bm25_store
