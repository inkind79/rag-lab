"""
Single-Vector LanceDB Store

Provides standard single-vector similarity search for dense bi-encoder
models (BGE-M3, E5, nomic-embed, etc.). Simpler schema than the
multi-vector store used by ColPali.
"""

import os
import numpy as np
import pyarrow as pa
from typing import List, Dict, Any, Optional, Tuple

from src.models.vector_stores.base_store import BaseVectorStore
from src.utils.logger import get_logger

logger = get_logger(__name__)

LANCEDB_DATA_PATH = os.path.join(os.getcwd(), ".lancedb")


class SingleVectorLanceDBStore(BaseVectorStore):
    """LanceDB store for single-vector embeddings.

    Uses a simple schema: id, vector (fixed-size float array), metadata, text.
    Standard cosine similarity search (no MaxSim scoring).
    """

    def __init__(self, model_name: str = "bge-m3", dimension: int = 1024):
        self._model_name = model_name.replace('/', '_')
        self._dimension = dimension
        self._table_name = f"dense_{self._model_name}"

    def _get_connection(self, session_id: str):
        """Get LanceDB connection for a session."""
        import lancedb
        db_path = os.path.join(LANCEDB_DATA_PATH, session_id)
        os.makedirs(db_path, exist_ok=True)
        return lancedb.connect(db_path)

    def _get_or_create_table(self, session_id: str, db=None):
        """Get or create the single-vector table."""
        if db is None:
            db = self._get_connection(session_id)

        try:
            table = db.open_table(self._table_name)
            return table
        except Exception:
            # Create new table with schema
            schema = pa.schema([
                pa.field("id", pa.string()),
                pa.field("vector", pa.list_(pa.float32(), self._dimension)),
                pa.field("text", pa.string()),
                pa.field("filename", pa.string()),
                pa.field("page_num", pa.int32()),
                pa.field("metadata", pa.string()),
            ])
            table = db.create_table(self._table_name, schema=schema)
            logger.info(f"Created single-vector table '{self._table_name}' for session {session_id}")
            return table

    def add_documents(
        self,
        session_id: str,
        embeddings: List[Any],
        ids: List[str],
        metadatas: List[Dict[str, Any]],
        texts: Optional[List[str]] = None,
    ) -> bool:
        """Add documents with single-vector embeddings."""
        import json

        try:
            db = self._get_connection(session_id)
            table = self._get_or_create_table(session_id, db)

            records = []
            for i, (emb, doc_id, meta) in enumerate(zip(embeddings, ids, metadatas)):
                # Convert embedding to list of floats
                if hasattr(emb, 'tolist'):
                    vec = emb.tolist()
                elif isinstance(emb, list):
                    vec = emb
                else:
                    logger.warning(f"Unexpected embedding type: {type(emb)}")
                    continue

                # Ensure correct dimension
                if len(vec) != self._dimension:
                    logger.warning(f"Embedding dim {len(vec)} != expected {self._dimension}, skipping")
                    continue

                records.append({
                    "id": doc_id,
                    "vector": vec,
                    "text": texts[i] if texts else "",
                    "filename": meta.get('filename', ''),
                    "page_num": meta.get('page_num', 0),
                    "metadata": json.dumps(meta),
                })

            if records:
                table.add(records)
                logger.info(f"Added {len(records)} single-vector documents to session {session_id}")
                return True
            else:
                logger.warning(f"No valid records to add for session {session_id} (all embeddings skipped)")
                return False

        except Exception as e:
            logger.error(f"Error adding documents to single-vector store: {e}", exc_info=True)
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
        """Query with single-vector cosine similarity."""
        if query_embedding is None:
            logger.warning("SingleVectorLanceDBStore.query requires query_embedding")
            return [], [], []

        try:
            db = self._get_connection(session_id)
            try:
                table = db.open_table(self._table_name)
            except Exception:
                logger.warning(f"Table '{self._table_name}' not found for session {session_id}")
                return [], [], []

            # Convert query embedding
            if hasattr(query_embedding, 'tolist'):
                query_vec = query_embedding.tolist()
            elif isinstance(query_embedding, list):
                query_vec = query_embedding
            else:
                query_vec = list(query_embedding)

            # Build search query
            search = table.search(query_vec).metric("cosine").limit(k)

            # Apply filename filter
            if filter_dict and 'filename' in filter_dict:
                filenames = filter_dict['filename']
                if isinstance(filenames, list) and filenames:
                    # Escape single quotes to prevent query injection
                    safe_names = [f.replace("'", "''") for f in filenames]
                    filter_expr = " OR ".join([f"filename = '{f}'" for f in safe_names])
                    search = search.where(filter_expr)

            results = search.to_pandas()

            metadatas_out = []
            scores_out = []
            ids_out = []

            for _, row in results.iterrows():
                # LanceDB returns _distance (lower is better for cosine)
                score = 1.0 - float(row.get('_distance', 0.0))

                if score < similarity_threshold:
                    continue

                metadatas_out.append({
                    'filename': row.get('filename', ''),
                    'page_num': int(row.get('page_num', 0)),
                    'text': row.get('text', ''),
                })
                scores_out.append(score)
                ids_out.append(row.get('id', ''))

            logger.info(f"Single-vector query returned {len(metadatas_out)} results")
            return metadatas_out, scores_out, ids_out

        except Exception as e:
            logger.error(f"Error querying single-vector store: {e}", exc_info=True)
            return [], [], []

    def delete_session(self, session_id: str) -> bool:
        """Delete the single-vector table for a session."""
        try:
            db = self._get_connection(session_id)
            db.drop_table(self._table_name, ignore_missing=True)
            return True
        except Exception as e:
            logger.error(f"Error deleting single-vector table: {e}")
            return False

    def has_documents(self, session_id: str) -> bool:
        """Check if documents exist in this store for the session."""
        try:
            db = self._get_connection(session_id)
            table = db.open_table(self._table_name)
            return len(table) > 0
        except Exception:
            return False
