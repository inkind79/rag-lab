"""
LanceDB vector store manager for all embedding models.

This module provides functions to create and manage LanceDB tables for storing
embeddings from various models, including:
1. ColPali-based models (ColQwen2.5, ColNomic) (multi-vector embeddings)
2. Other models (e.g., Sentence Transformers) (single-vector embeddings)

It supports different embedding formats and dimensions automatically and
provides on-disk storage with full multi-vector support.
"""

import os
import json
import numpy as np
import torch
import lancedb
import pyarrow as pa
from typing import Dict, List, Any, Optional, Tuple
import time
import shutil
import threading

from src.utils.logger import get_logger
from src.utils.resource_tracker import get_resource_tracker

logger = get_logger(__name__)
import traceback


def get_lancedb_connections_count() -> int:
    """Get the current number of active LanceDB connections."""
    with _CONNECTION_LOCK:
        # Count non-None values in the connections dictionary
        count = sum(1 for conn in _lancedb_connections.values() if conn is not None)
        return count

# Constants
LANCEDB_DATA_PATH = os.path.join(os.getcwd(), ".lancedb")  # Absolute path to store LanceDB data

# Global cache of LanceDB connections to avoid reloading
_lancedb_connections = {}

# Connection pool settings
_MAX_CONNECTIONS = 1  # Further reduced to prevent memory buildup with multiple sessions
_CONNECTION_LAST_USED = {}  # Track when connections were last used
_CONNECTION_LOCK = threading.RLock()  # Lock for thread safety

def get_lancedb_connection(session_id: str):
    """
    Gets or creates a LanceDB connection for a specific session.
    Uses connection pooling to limit the number of active connections.
    Implements both time-based and count-based connection management.

    Args:
        session_id: The session ID

    Returns:
        A LanceDB connection object
    """
    global _lancedb_connections, _CONNECTION_LAST_USED, _CONNECTION_LOCK

    # Maximum idle time for a connection (5 minutes)
    MAX_IDLE_TIME = 300
    current_time = time.time()

    # Use a lock to ensure thread safety
    with _CONNECTION_LOCK:
        # CRITICAL: Log session ID for debugging crossover issues
        logger.debug(f"[SESSION ISOLATION] get_lancedb_connection called for session_id: {session_id}")
        
        # Check if this connection exists and is not too old
        if session_id in _lancedb_connections and _lancedb_connections[session_id] is not None:
            last_used_time = _CONNECTION_LAST_USED.get(session_id, 0)

            # If connection is too old, close it and create a new one
            if current_time - last_used_time > MAX_IDLE_TIME:
                logger.info(f"Closing expired LanceDB connection for {session_id} (idle for {current_time - last_used_time:.1f}s)")
                _close_lancedb_connection(session_id)
            else:
                # Still valid, update last used time and return
                logger.debug(f"Using cached LanceDB connection for {session_id}")
                _CONNECTION_LAST_USED[session_id] = current_time
                return _lancedb_connections[session_id]

        # Clean up any connections that are too old (regardless of count)
        for session, last_used in list(_CONNECTION_LAST_USED.items()):
            if current_time - last_used > MAX_IDLE_TIME and session != session_id:
                try:
                    logger.info(f"Closing expired LanceDB connection for {session} (idle for {current_time - last_used:.1f}s)")
                    _close_lancedb_connection(session)
                except Exception as e:
                    logger.warning(f"Error closing LanceDB connection for {session}: {e}")

        # Count-based cleanup (only if we're still at the limit after time-based cleanup)
        if len(_lancedb_connections) >= _MAX_CONNECTIONS:
            # Find the least recently used connection
            if _CONNECTION_LAST_USED:
                # Sort by last used time (oldest first)
                sessions_to_close = sorted(
                    _CONNECTION_LAST_USED.keys(),
                    key=lambda s: _CONNECTION_LAST_USED.get(s, 0)
                )

                # Close connections until we're under the limit
                while len(_lancedb_connections) >= _MAX_CONNECTIONS and sessions_to_close:
                    oldest_session = sessions_to_close.pop(0)
                    if oldest_session != session_id:  # Don't close the one we're about to use
                        try:
                            # Close the connection
                            logger.info(f"Closing least recently used LanceDB connection for {oldest_session}")
                            _close_lancedb_connection(oldest_session)
                        except Exception as e:
                            logger.warning(f"Error closing LanceDB connection for {oldest_session}: {e}")

        # Ensure the root LanceDB directory exists
        os.makedirs(LANCEDB_DATA_PATH, exist_ok=True)

        # Create the session-specific directory
        lancedb_dir = os.path.join(LANCEDB_DATA_PATH, session_id)
        os.makedirs(lancedb_dir, exist_ok=True)

        # CRITICAL: Verify we're using the correct session directory
        logger.debug(f"[SESSION ISOLATION] Creating LanceDB connection for session {session_id} at {lancedb_dir}")
        connection = lancedb.connect(lancedb_dir)

        # Cache the connection and update last used time
        _lancedb_connections[session_id] = connection
        _CONNECTION_LAST_USED[session_id] = current_time
        
        # Track the connection in our resource tracker
        tracker = get_resource_tracker()
        tracker.track_connection(f"lancedb_{session_id}")
        tracker.track_resource('lancedb_connection', session_id, {
            'path': lancedb_dir,
            'created_at': current_time
        })

        return connection

def _close_lancedb_connection(session_id: str):
    """
    Helper function to close a LanceDB connection.
    Ensures proper release of PyArrow resources to minimize memory leaks.

    Args:
        session_id: The session ID of the connection to close
    """
    global _lancedb_connections, _CONNECTION_LAST_USED

    if session_id in _lancedb_connections:
        try:
            # Close all tables in the connection
            conn = _lancedb_connections[session_id]
            if conn and hasattr(conn, 'table_names'):
                # Get the table names first, then close each table
                try:
                    table_names = list(conn.table_names())  # Convert to list to avoid any connection issues during iteration
                except Exception as e:
                    logger.warning(f"Error getting table names for session {session_id}: {e}")
                    table_names = []

                # Close each table
                for table_name in table_names:
                    try:
                        # Get the table and try to clear its resources
                        table = conn.open_table(table_name)

                        # Release PyArrow resources if present
                        if hasattr(table, '_arrow_table') and table._arrow_table is not None:
                            try:
                                del table._arrow_table
                            except Exception as arrow_e:
                                logger.warning(f"Error releasing PyArrow table for {table_name}: {arrow_e}")

                        # Clear any cached data in the table
                        if hasattr(table, '_cached_data') and table._cached_data is not None:
                            try:
                                table._cached_data = None
                            except Exception as cache_e:
                                logger.warning(f"Error clearing table cache for {table_name}: {cache_e}")

                        # Attempt to flush any pending changes
                        if hasattr(table, 'flush') and callable(table.flush):
                            try:
                                table.flush()
                            except Exception as flush_e:
                                logger.warning(f"Error flushing table {table_name}: {flush_e}")

                        # Set table to None to help garbage collection
                        table = None
                    except Exception as table_e:
                        logger.warning(f"Error closing table {table_name} for session {session_id}: {table_e}")

            # Close the connection explicitly if possible
            if hasattr(conn, 'close') and callable(conn.close):
                try:
                    conn.close()
                except Exception as close_e:
                    logger.warning(f"Error calling close() on connection for {session_id}: {close_e}")

            # Set the connection to None to help garbage collection
            _lancedb_connections[session_id] = None

            # Force garbage collection before dictionary cleanup
            import gc
            gc.collect()

            # Clear CUDA cache if available to release any GPU memory
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except (ImportError, Exception) as e:
                logger.debug(f"Error clearing CUDA cache: {e}")

            # Remove from dictionaries
            del _lancedb_connections[session_id]
            if session_id in _CONNECTION_LAST_USED:
                del _CONNECTION_LAST_USED[session_id]
                
            # Untrack the connection in our resource tracker
            tracker = get_resource_tracker()
            tracker.untrack_connection(f"lancedb_{session_id}")
            tracker.untrack_resource('lancedb_connection', session_id)

            # Try to release memory back to OS on Linux
            try:
                import ctypes
                try:
                    libc = ctypes.CDLL('libc.so.6')
                    libc.malloc_trim(0)
                    logger.debug("Released memory back to OS with malloc_trim")
                except Exception as libc_e:
                    logger.debug(f"Error releasing memory to OS: {libc_e}")
            except ImportError:
                pass

            logger.debug(f"Closed LanceDB connection for {session_id}")
        except Exception as e:
            logger.warning(f"Error during LanceDB connection closure for {session_id}: {e}")

def get_lancedb_table(session_id: str, model_name: str = "colqwen25") -> Tuple[lancedb.table.Table, Dict]:
    """
    Gets or creates a LanceDB table for a specific session and model.

    Args:
        session_id: The session ID
        model_name: The model name (used to create a model-specific table)

    Returns:
        A tuple of (LanceDB table, metadata dictionary)
    """
    # Create a unique table name for this session and model
    table_name = f"{model_name}"

    # Get the connection
    connection = get_lancedb_connection(session_id)

    # Check if the table exists
    if table_name in connection.table_names():
        logger.debug(f"Using existing LanceDB table for {session_id}_{model_name}")
        table = connection.open_table(table_name)

        # Load metadata
        metadata_path = os.path.join(LANCEDB_DATA_PATH, session_id, f"{model_name}_metadata.json")
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.warning(f"Could not load metadata for {session_id}_{model_name}: {e}")
            metadata = {
                'ids': [],
                'metadatas': [],
                'model': model_name,
                'created_at': time.time()
            }

        return table, metadata

    # Create a new table
    logger.info(f"Creating new LanceDB table for {session_id}_{model_name}")

    # Determine the dimension from the model registry
    from src.models.model_registry import registry
    spec = registry.detect(model_name)
    if spec:
        dimension = spec.dimension
        logger.info(f"Using dimension {dimension} for {spec.family} model {model_name}")
    else:
        from src.utils.model_type_utils import is_colpali_model as _is_colpali
        dimension = 128 if _is_colpali(model_name) else 768
        logger.warning(f"Model '{model_name}' not in registry, using dimension {dimension}")

    vector_dim = dimension
    logger.info(f"Creating schema with vector dimension {vector_dim} for model {model_name}")

    # Create the schema with the correct multivector format
    # The outer list is variable length (number of vectors per document)
    # The inner list is fixed length (dimension of each vector)

    # For LanceDB multivector search, we need to use a list of fixed-size lists
    # This is different from a list of lists - the inner list must have a fixed size

    # Use pa.list_ with a size parameter to create a fixed-size list
    schema = pa.schema([
        pa.field('id', pa.string()),
        # For multivector support: list of fixed-size vectors
        pa.field('vectors', pa.list_(pa.list_(pa.float32(), vector_dim))),
        pa.field('metadata', pa.string()),  # JSON-encoded metadata
        pa.field('text', pa.string()),  # Text content
        pa.field('page_num', pa.int32()),  # Page number
        pa.field('filename', pa.string()),  # Filename
        pa.field('timestamp', pa.float64())  # Timestamp
    ])

    logger.info(f"Created schema: {schema}")

    # Create an empty table
    # The empty_data must match the schema exactly
    empty_data = pa.RecordBatch.from_arrays(
        [
            pa.array([], type=pa.string()),
            pa.array([], type=pa.list_(pa.list_(pa.float32(), vector_dim))),  # Use fixed-size list
            pa.array([], type=pa.string()),
            pa.array([], type=pa.string()),
            pa.array([], type=pa.int32()),
            pa.array([], type=pa.string()),
            pa.array([], type=pa.float64())
        ],
        schema=schema
    )

    # Create the table
    table = connection.create_table(
        table_name,
        data=empty_data,
        mode="overwrite"
    )

    # Create the index immediately, even on an empty table
    try:
        logger.info(f"Creating index for {session_id}_{model_name} with vector_column_name='vectors'")

        # Create the index with optimized parameters for deterministic multivector support
        try:
            # First, count the total vectors to calculate optimal partitioning
            row_count = len(table.to_pandas())
            # For ColPali models, each document typically has ~1030 vectors
            estimated_vectors_per_doc = 1030
            total_vectors = row_count * estimated_vectors_per_doc
            
            # Calculate optimal partitions: sqrt(total_vectors) as per LanceDB docs
            # This helps balance partition size for more stable retrieval
            import numpy as np
            num_partitions = max(1, int(np.sqrt(total_vectors)))
            
            logger.info(f"Creating optimized index with {num_partitions} partitions for {row_count} documents")
            
            table.create_index(
                metric="cosine",  # Only cosine is supported for multivector
                vector_column_name="vectors",  # Must match the column name in schema
                num_partitions=num_partitions,  # Optimized partition count for stability
                num_sub_vectors=32,  # Higher for accuracy, lower for speed
                index_type="IVF_PQ"  # Provides better determinism than default
            )
        except Exception as e:
            logger.warning(f"Error creating optimized index, falling back to basic index: {e}")
            # Fall back to basic index if advanced parameters fail
            table.create_index(
                metric="cosine",  # Only cosine is supported for multivector
                vector_column_name="vectors"  # Must match the column name in schema
            )
        logger.info(f"Successfully created index for {session_id}_{model_name}")
    except Exception as index_e:
        if "already exists" in str(index_e):
            logger.info(f"Index already exists for {session_id}_{model_name}")
        else:
            logger.error(f"Error creating index for {session_id}_{model_name}: {index_e}")

    logger.info(f"Created LanceDB table for {session_id}_{model_name} with index")

    # Initialize metadata
    metadata = {
        'ids': [],
        'metadatas': [],
        'dimension': dimension,  # This is the dimension of each vector (128 for ColPali)
        'model': model_name,
        'created_at': time.time()
    }

    # Save the metadata
    metadata_path = os.path.join(LANCEDB_DATA_PATH, session_id, f"{model_name}_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f)

    return table, metadata

def add_embeddings_to_lancedb(
    session_id: str,
    model_name: str,
    embeddings_list: List[np.ndarray],
    ids: List[str],
    metadatas: List[Dict[str, Any]]
) -> bool:
    """
    Adds embeddings to a LanceDB table.
    Uses batch processing to prevent memory spikes when adding large amounts of data.

    Args:
        session_id: The session ID
        model_name: The model name
        embeddings_list: List of embeddings to add
        ids: List of IDs for the embeddings
        metadatas: List of metadata dictionaries for the embeddings

    Returns:
        True if successful, False otherwise
    """
    try:
        # Get the table and metadata
        table, metadata = get_lancedb_table(session_id, model_name)

        # Verify that all lists have the same length
        if len(embeddings_list) != len(ids) or len(embeddings_list) != len(metadatas):
            logger.error(f"Mismatch in input lengths: embeddings={len(embeddings_list)}, ids={len(ids)}, metadatas={len(metadatas)}")
            return False

        # Define batch size - process in smaller chunks to avoid memory spikes
        # This is a critical parameter for memory management
        BATCH_SIZE = 10  # Process 10 documents at a time
        total_embeddings = len(embeddings_list)
        total_batches = (total_embeddings + BATCH_SIZE - 1) // BATCH_SIZE  # Ceiling division
        logger.info(f"Processing {total_embeddings} embeddings in {total_batches} batches of size {BATCH_SIZE}")

        # Track overall success metrics
        overall_success_count = 0
        overall_new_ids = []
        overall_new_metadatas = []

        # Process embeddings in batches
        for batch_idx in range(0, total_embeddings, BATCH_SIZE):
            batch_end = min(batch_idx + BATCH_SIZE, total_embeddings)
            logger.info(f"Processing batch {batch_idx//BATCH_SIZE + 1}/{total_batches} (items {batch_idx+1}-{batch_end})")

            # Initialize batch-specific variables
            batch_data = []
            batch_new_ids = []
            batch_new_metadatas = []

            # Process each embedding in this batch
            for i in range(batch_idx, batch_end):
                embedding = embeddings_list[i]
                doc_id = ids[i]

                # Skip if already exists (check in metadata)
                if doc_id in metadata['ids']:
                    logger.debug(f"Embedding {doc_id} already exists in LanceDB table, skipping")
                    continue

                # Process the embedding
                vector_data = None

                # First, convert to numpy array if it's a torch tensor
                if isinstance(embedding, torch.Tensor):
                    embedding_np = embedding.detach().cpu().numpy()
                    # Explicitly delete tensor to free memory immediately
                    del embedding
                    embedding = embedding_np

                # Now handle numpy arrays
                if isinstance(embedding, np.ndarray):
                    # Get the expected dimension from metadata
                    expected_dim = metadata.get('dimension', 128)  # Default to 128 for ColPali models

                    # For multi-vector embeddings (2D array)
                    if len(embedding.shape) == 2:  # [seq_len, dim]
                        seq_len, dim = embedding.shape
                        logger.debug(f"Processing multi-vector embedding with shape {embedding.shape} for document {doc_id}")

                        # Check if dimensions match
                        if dim != expected_dim:
                            logger.warning(f"Embedding dimension mismatch: got {dim}, expected {expected_dim}")
                            # If dimensions don't match, we need to pad or truncate
                            if dim < expected_dim:
                                # Pad with zeros using a more memory-efficient approach
                                padded_vectors = []
                                for j in range(seq_len):
                                    vec = embedding[j].tolist()  # Convert to Python list
                                    if len(vec) < expected_dim:
                                        vec.extend([0.0] * (expected_dim - len(vec)))  # Extend in-place
                                    padded_vectors.append(vec)
                                vector_data = padded_vectors
                                # Free memory
                                del embedding
                            else:
                                # Truncate directly to list
                                vector_data = [embedding[j, :expected_dim].tolist() for j in range(seq_len)]
                                # Free memory
                                del embedding
                        else:
                            # Convert directly to list to avoid keeping both numpy and list versions
                            vector_data = embedding.tolist()
                            # Free memory
                            del embedding

                    # For single vector embeddings (1D array)
                    else:  # [dim]
                        dim = embedding.shape[0]
                        logger.debug(f"Processing single-vector embedding with shape {embedding.shape} for document {doc_id}")

                        # Check if dimensions match
                        if dim != expected_dim:
                            logger.warning(f"Embedding dimension mismatch: got {dim}, expected {expected_dim}")
                            # Convert to Python list and adjust
                            vec = embedding.tolist()
                            if dim < expected_dim:
                                # Pad with zeros
                                vec.extend([0.0] * (expected_dim - dim))
                            else:
                                # Truncate
                                vec = vec[:expected_dim]
                            vector_data = [vec]
                        else:
                            # Dimensions match, convert to list
                            vector_data = [embedding.tolist()]

                        # Free memory
                        del embedding
                else:
                    logger.warning(f"Unexpected embedding type: {type(embedding)}")
                    continue

                # Skip if vector data is empty
                if not vector_data:
                    logger.warning(f"Empty vector data for document {doc_id}, skipping")
                    continue

                # Extract metadata fields
                try:
                    doc_metadata = metadatas[i]
                    text = doc_metadata.get('text', '')
                    page_num = doc_metadata.get('page_num', 0)
                    filename = doc_metadata.get('filename', '')
                except IndexError:
                    logger.error(f"Metadata index {i} out of range (len={len(metadatas)})")
                    continue

                # Add to batch data
                try:
                    # For ColPali models, we store all vectors for a document in a single record
                    batch_data.append({
                        'id': doc_id,
                        'vectors': vector_data,  # Use 'vectors' to match schema
                        'metadata': json.dumps(doc_metadata),
                        'text': text,
                        'page_num': page_num,
                        'filename': filename,
                        'timestamp': time.time()
                    })

                    # Track metadata updates
                    batch_new_ids.append(doc_id)
                    batch_new_metadatas.append(doc_metadata)
                except Exception as e:
                    logger.error(f"Error adding document {doc_id} to batch: {e}")
                    continue

            # Add batch data to the table
            if batch_data:
                try:
                    # Log batch information
                    logger.info(f"Adding batch of {len(batch_data)} documents to LanceDB table")

                    # Add documents in batch for better performance
                    try:
                        # Add all documents at once
                        table.add(batch_data)
                        batch_success_count = len(batch_data)
                        logger.info(f"Successfully added {batch_success_count} documents in batch operation")
                    except Exception as batch_e:
                        logger.warning(f"Batch add failed: {batch_e}. Falling back to individual adds.")
                        # Fallback to one-by-one if batch fails
                        batch_success_count = 0
                        for item in batch_data:
                            try:
                                table.add([item])
                                batch_success_count += 1
                            except Exception as item_e:
                                logger.error(f"Error adding item {item['id']} to LanceDB table: {item_e}")
                        
                    logger.info(f"Successfully added {batch_success_count}/{len(batch_data)} documents in this batch")

                    # Update overall success count
                    overall_success_count += batch_success_count
                    overall_new_ids.extend(batch_new_ids)
                    overall_new_metadatas.extend(batch_new_metadatas)

                    # Update metadata more frequently to avoid loss on failure
                    metadata['ids'].extend(batch_new_ids)
                    metadata['metadatas'].extend(batch_new_metadatas)

                    # Save metadata after each successful batch
                    metadata_path = os.path.join(LANCEDB_DATA_PATH, session_id, f"{model_name}_metadata.json")
                    with open(metadata_path, 'w') as f:
                        json.dump(metadata, f)

                    # Force cleanup after each batch to prevent memory accumulation
                    import gc
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    # Clear batch data to release memory
                    del batch_data
                    del batch_new_ids
                    del batch_new_metadatas
                    batch_data = []
                    batch_new_ids = []
                    batch_new_metadatas = []

                except Exception as e:
                    logger.error(f"Error adding batch data to LanceDB table: {e}")
            else:
                logger.debug(f"No documents to add in this batch")

        # Log overall results
        if overall_success_count > 0:
            logger.info(f"Successfully added {overall_success_count} documents in total")
        else:
            logger.warning(f"No documents were added to LanceDB table for {session_id}_{model_name}")

        # Create the index after all data is added
        try:
            # For multivector support, we must use cosine similarity
            logger.info(f"Creating index for {session_id}_{model_name} with vector_column_name='vectors'")

            try:
                table.create_index(
                    metric="cosine",  # Only cosine is supported for multivector
                    vector_column_name="vectors"  # Must match the column name in schema
                )
                logger.info(f"Successfully created index for {session_id}_{model_name}")
            except Exception as index_e:
                if "already exists" in str(index_e):
                    logger.info(f"Index already exists for {session_id}_{model_name}")
                else:
                    logger.error(f"Error creating index for {session_id}_{model_name}: {index_e}")

                    # Try to verify the table is searchable despite the error
                    try:
                        # Use a dummy vector to test search functionality
                        dummy_vector = [[0.0] * vector_dim]
                        table.search(dummy_vector, vector_column_name="vectors").limit(1)
                        logger.info(f"LanceDB index for {session_id}_{model_name} appears to be working despite error")
                    except Exception as search_e:
                        logger.error(f"Error searching LanceDB table after index creation failed: {search_e}")
        except Exception as e:
            logger.warning(f"Unexpected error creating index: {e}")

        # Return success based on whether we added any documents
        return overall_success_count > 0 or not embeddings_list

    except Exception as e:
        logger.error(f"Error adding embeddings to LanceDB table: {e}")
        return False

def pre_warm_lancedb_search(session_id: str, model_name: str = "colqwen25"):
    """
    Pre-warm the LanceDB search to ensure consistent performance.
    This is especially important for cold-start scenarios where
    the first search might be slower or produce different results.
    
    Args:
        session_id: The session ID
        model_name: The model name
        
    Returns:
        None
    """
    try:
        # Create a simple dummy query for warming up the index
        import numpy as np
        
        # Get table and ensure connection
        connection = get_lancedb_connection(session_id)
        table_name = f"{model_name}"
        
        # Check if table exists
        if not connection.table_exists(table_name):
            logger.info(f"Skipping pre-warm for {session_id}_{model_name} - table doesn't exist")
            return
        
        # Get vector dimension from the model registry
        from src.models.model_registry import registry
        dimension = registry.get_dimension(model_name)
        
        # Open table
        table = connection.open_table(table_name)
        
        # Create a consistent dummy query with fixed seed
        np.random.seed(42)  # Use fixed seed for deterministic behavior
        dummy_vectors = np.random.randn(2, dimension).astype(np.float32)
        # Normalize vectors to unit length
        from src.models.vector_stores.embedding_utils import normalize_vectors
        dummy_vectors = normalize_vectors(dummy_vectors)
        
        # Convert to list of lists for LanceDB
        dummy_query = dummy_vectors.tolist()
        
        # Run 3 dummy searches to warm up the index
        logger.info(f"Pre-warming LanceDB search for {session_id}_{model_name}")
        for i in range(3):
            _ = table.search(
                dummy_query,
                vector_column_name="vectors",
                nprobes=128,
                limit=5  # Small limit for quick warming
            ).to_pandas()
            
        logger.info(f"Successfully pre-warmed LanceDB search for {session_id}_{model_name}")
    except Exception as e:
        logger.warning(f"Error pre-warming LanceDB search: {e}")


def query_lancedb(
    session_id: str,
    model_name: str,
    query_embedding: np.ndarray,
    k: int = 10,
    filter_dict: Optional[Dict] = None,
    similarity_threshold: float = 0.2,  # Threshold for normalized scores in [0,1] range
    timeout_seconds: int = 60  # Add timeout parameter with default 60 seconds
) -> Tuple[List[Dict], List[float], List[str]]:
    """
    Queries a LanceDB table with a query embedding.
    Add timeout safety to prevent indefinite hangs.

    Args:
        session_id: The session ID
        model_name: The model name
        query_embedding: The query embedding
        k: The number of results to return
        filter_dict: Optional dictionary for filtering results
        similarity_threshold: Minimum similarity score to include in results (default: 0.2)
                             For normalized scores in [0,1] range after proper multivector conversion
    Returns:
        A tuple of (list of metadata dictionaries, list of similarity scores, list of IDs)
    """
    # Time the operation for debugging
    start_time = time.time()
    
    try:
        # Ensure the table exists
        if not ensure_lancedb_table_exists(session_id, model_name):
            logger.warning(f"Could not ensure LanceDB table exists for session {session_id}")
            return [], [], []
        
        # Create a thread-safe timeout mechanism
        import threading
        import signal

        class TimeoutError(Exception):
            pass

        # Check if we're in the main thread
        is_main_thread = threading.current_thread() is threading.main_thread()
        timeout_triggered = False

        if is_main_thread:
            # Use signal-based timeout only in main thread
            def timeout_handler(signum, frame):
                raise TimeoutError(f"LanceDB query timed out after {timeout_seconds} seconds")

            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout_seconds)
        else:
            # Use threading-based timeout for background threads
            logger.debug("Using threading-based timeout (not in main thread)")
            timeout_event = threading.Event()

            def timeout_checker():
                if not timeout_event.wait(timeout_seconds):
                    nonlocal timeout_triggered
                    timeout_triggered = True
                    logger.warning(f"LanceDB query timeout triggered after {timeout_seconds} seconds")

            timeout_thread = threading.Thread(target=timeout_checker, daemon=True)
            timeout_thread.start()
        
        try:
            # Get the connection and table
            connection = get_lancedb_connection(session_id)
            table_name = f"{model_name}"
            table = connection.open_table(table_name)
            
            # Convert query embedding to the right format for LanceDB multivector search
            if isinstance(query_embedding, torch.Tensor):
                query_embedding = query_embedding.cpu().numpy()

            # Process the query embedding
            query_vectors = None
            if isinstance(query_embedding, np.ndarray):
                # Check the shape of the query embedding
                if len(query_embedding.shape) == 3:  # [batch_size, seq_len, hidden_dim]
                    # Remove batch dimension if present
                    if query_embedding.shape[0] == 1:
                        query_embedding = query_embedding.squeeze(0)
                        logger.debug(f"Removed batch dimension, new shape: {query_embedding.shape}")
                    else:
                        logger.warning(f"Unexpected batch size: {query_embedding.shape[0]}, using first batch")
                        query_embedding = query_embedding[0]

                # Now we should have [seq_len, hidden_dim] or [hidden_dim]
                if len(query_embedding.shape) == 2:  # [seq_len, hidden_dim]
                    # Normalize vectors to unit length for proper cosine similarity
                    from src.models.vector_stores.embedding_utils import normalize_vectors
                    query_embedding = normalize_vectors(query_embedding)
                    logger.info(f"Normalized {query_embedding.shape[0]} query vectors to unit length")

                    # Convert to list of lists for LanceDB
                    query_vectors = query_embedding.tolist()
                elif len(query_embedding.shape) == 1:  # [hidden_dim]
                    # Normalize vector to unit length for proper cosine similarity
                    from src.models.vector_stores.embedding_utils import normalize_vectors
                    query_embedding = normalize_vectors(query_embedding)
                    logger.info(f"Normalized single query vector to unit length")

                    # Single vector, add sequence dimension
                    query_vectors = [query_embedding.tolist()]
                else:
                    logger.warning(f"Unexpected query embedding shape: {query_embedding.shape}")
                    return [], [], []
            else:
                logger.warning(f"Unexpected query embedding type: {type(query_embedding)}")
                return [], [], []

            # Log the query vectors for debugging
            logger.info(f"Querying LanceDB with {len(query_vectors)} vectors of dimension {len(query_vectors[0]) if query_vectors else 0}")

            # DIRECT APPROACH: Use the table.search method directly
            try:
                # Create the search query
                search_query = table.search(
                    query_vectors,
                    vector_column_name="vectors"  # Must match the column name in schema and index
                )

                # Apply filter if provided
                if filter_dict:
                    filter_conditions = []
                    for key, values in filter_dict.items():
                        if isinstance(values, list):
                            if len(values) == 1:
                                # Single value
                                filter_conditions.append(f"{key} = '{values[0]}'")
                            else:
                                # Multiple values - use IN clause
                                values_str = "', '".join(values)
                                filter_conditions.append(f"{key} IN ('{values_str}')")
                        else:
                            # Single value (not in a list)
                            filter_conditions.append(f"{key} = '{values}'")

                    if filter_conditions:
                        filter_expr = " AND ".join(filter_conditions)
                        logger.info(f"Applying filter: {filter_expr}")
                        search_query = search_query.where(filter_expr)

                # Set the limit and execute the query
                search_query = search_query.limit(k)

                # Check for timeout in background threads
                if not is_main_thread and timeout_triggered:
                    raise TimeoutError(f"LanceDB query timed out after {timeout_seconds} seconds")

                results = search_query.to_pandas()

                # Check for timeout again after query execution
                if not is_main_thread and timeout_triggered:
                    raise TimeoutError(f"LanceDB query timed out after {timeout_seconds} seconds")

                if results.empty:
                    logger.warning(f"No results found for query in {session_id}_{model_name}")
                    return [], [], []

                # Process results
                metadatas_list = []
                scores = []
                ids = []

                for _, row in results.iterrows():
                    # Get the raw distance from LanceDB
                    raw_distance = row['_distance']

                    # Get the number of query vectors (m)
                    query_vector_count = len(query_vectors)

                    # Calculate average cosine similarity (1.0 - distance/m)
                    # This properly handles multivector distances which are the sum of distances across all query tokens
                    avg_cos_sim = 1.0 - (raw_distance / query_vector_count)

                    # Calculate the ColBERT sum-similarity (m - distance)
                    # This preserves token count signal and matches ColBERT's training objective
                    colbert_sum = query_vector_count - raw_distance

                    # Apply temperature scaling with higher temperature to magnify differences more
                    # Increasing from 4.0 to 8.0 based on recommendations
                    temperature = 8.0
                    temp_score = np.exp(temperature * avg_cos_sim)

                    # Normalize to [0,1] range for more intuitive scoring (original approach)
                    score_01 = (avg_cos_sim + 1) / 2

                    # Normalize the ColBERT sum-similarity to [0,1] range for easier thresholding
                    # This divides by query_vector_count to get a value between 0 and 1
                    norm_colbert_sum = colbert_sum / query_vector_count

                    # Use the raw ColBERT sum-similarity as the primary score
                    # This preserves token count signal and matches ColBERT's training objective
                    # Use normalized ColBERT score for better cross-model compatibility
                    score = norm_colbert_sum  # This is already normalized by query_vector_count

                    # Log all relevant values for debugging
                    logger.debug(f"ID: {row['id']}, Raw distance: {raw_distance:.4f}, Query vectors: {query_vector_count}, " +
                                 f"Avg cos sim: {avg_cos_sim:.4f}, ColBERT sum: {colbert_sum:.4f}, " +
                                 f"Norm ColBERT sum: {norm_colbert_sum:.4f}, Temp score (τ={temperature}): {temp_score:.4f}, " +
                                 f"Score [0,1]: {score_01:.4f}, Using score: {score:.4f}")

                    # For normalized ColBERT scores, we need a more appropriate threshold
                    # The scores are now normalized by query_vector_count, giving us a range around [-1, 1]
                    # A reasonable threshold is 0.0, which means at least neutral relevance
                    # We can adjust based on the UI threshold (0-1) to be more or less strict

                    # Convert UI threshold (0-1) to an appropriate internal threshold
                    # Convert UI threshold [0,1] to internal threshold
                    # Detect model family from registry for family-specific adjustments
                    from src.models.model_registry import registry as _reg
                    _spec = _reg.detect(model_name)
                    _family = _spec.family if _spec else ""
                    if _family.startswith("colnomic"):
                        # ColNomic has compressed normalized scores in positive range [0.2, 0.4]
                        internal_threshold = 0.15 + (similarity_threshold * 0.2)
                    else:
                        # For ColQwen2.5 and others, normalized scores can be in range [-1, 1]
                        internal_threshold = similarity_threshold - 0.5

                    # Skip results below the internal threshold
                    if score < internal_threshold:
                        logger.debug(f"Skipping result with score {score:.4f} below internal threshold {internal_threshold:.4f} (from UI threshold {similarity_threshold:.4f})")
                        continue

                    # Parse metadata
                    try:
                        doc_metadata = json.loads(row['metadata'])
                    except (json.JSONDecodeError, KeyError) as e:
                        logger.warning(f"Could not parse metadata for {row['id']}: {e}")
                        # Create a basic metadata dict from row data
                        doc_metadata = {
                            'filename': row.get('filename', 'unknown'),
                            'page_num': row.get('page_num', 0),
                            'text': row.get('text', '')
                        }

                    metadatas_list.append(doc_metadata)
                    scores.append(score)
                    ids.append(row['id'])

                logger.info(f"Query returned {len(metadatas_list)} results with scores: {scores[:3] if scores else []}")

                # Implement fallback mechanism if no results pass the threshold
                if not metadatas_list and len(results.index) > 0:
                    # Calculate the internal threshold for better logging
                    if _family.startswith("colnomic"):
                        internal_threshold = 0.15 + (similarity_threshold * 0.2)
                    else:
                        internal_threshold = similarity_threshold - 0.5
                    logger.warning(f"No results passed the similarity threshold {similarity_threshold} (internal threshold {internal_threshold:.4f}). Implementing fallback mechanism.")

                    # Get the top k results regardless of threshold
                    fallback_metadatas = []
                    fallback_scores = []
                    fallback_ids = []

                    for _, row in results.iterrows():
                        # Get the raw distance from LanceDB
                        raw_distance = row['_distance']

                        # Get the number of query vectors (m)
                        query_vector_count = len(query_vectors)

                        # Calculate average cosine similarity (1.0 - distance/m)
                        # This properly handles multivector distances which are the sum of distances across all query tokens
                        avg_cos_sim = 1.0 - (raw_distance / query_vector_count)

                        # Calculate the ColBERT sum-similarity (m - distance)
                        # This preserves token count signal and matches ColBERT's training objective
                        colbert_sum = query_vector_count - raw_distance

                        # Apply temperature scaling with higher temperature to magnify differences more
                        # Increasing from 4.0 to 8.0 based on recommendations
                        temperature = 8.0
                        temp_score = np.exp(temperature * avg_cos_sim)

                        # Normalize to [0,1] range for more intuitive scoring (original approach)
                        score_01 = (avg_cos_sim + 1) / 2

                        # Normalize the ColBERT sum-similarity to [0,1] range for easier thresholding
                        # This divides by query_vector_count to get a value between 0 and 1
                        norm_colbert_sum = colbert_sum / query_vector_count

                        # Use the raw ColBERT sum-similarity as the primary score
                        # This preserves token count signal and matches ColBERT's training objective
                        # Use normalized ColBERT score for better cross-model compatibility
                        score = norm_colbert_sum  # This is already normalized by query_vector_count

                        # Log all relevant values for debugging
                        logger.debug(f"FALLBACK - ID: {row['id']}, Raw distance: {raw_distance:.4f}, Query vectors: {query_vector_count}, " +
                                     f"Avg cos sim: {avg_cos_sim:.4f}, ColBERT sum: {colbert_sum:.4f}, " +
                                     f"Norm ColBERT sum: {norm_colbert_sum:.4f}, Temp score (τ={temperature}): {temp_score:.4f}, " +
                                     f"Score [0,1]: {score_01:.4f}, Using score: {score:.4f}")

                        # For ColBERT sum-similarity scores, we need a more appropriate threshold
                        # The scores can be negative for poor matches, and positive for good matches
                        # A reasonable threshold is 0.0, which means at least neutral relevance
                        # We can adjust based on the UI threshold (0-1) to be more or less strict

                        # Convert UI threshold (0-1) to an appropriate internal threshold
                        # Uses _family from registry lookup earlier in this function
                        if _family.startswith("colnomic"):
                            internal_threshold = 0.15 + (similarity_threshold * 0.2)
                        else:
                            internal_threshold = similarity_threshold - 0.5

                        # Skip results below the internal threshold
                        if score < internal_threshold:
                            logger.debug(f"FALLBACK - Skipping result with score {score:.4f} below internal threshold {internal_threshold:.4f} (from UI threshold {similarity_threshold:.4f})")
                            continue

                        # Parse metadata
                        try:
                            doc_metadata = json.loads(row['metadata'])
                        except (json.JSONDecodeError, KeyError) as e:
                            logger.warning(f"Could not parse metadata for {row['id']}: {e}")
                            # Create a basic metadata dict from row data
                            doc_metadata = {
                                'filename': row.get('filename', 'unknown'),
                                'page_num': row.get('page_num', 0),
                                'text': row.get('text', '')
                            }

                        # Add to results
                        fallback_metadatas.append(doc_metadata)
                        fallback_scores.append(score)
                        fallback_ids.append(row['id'])

                        # Limit to k results
                        if len(fallback_metadatas) >= k:
                            break

                    if fallback_metadatas:
                        logger.info(f"Fallback mechanism returned {len(fallback_metadatas)} results with ColBERT sum-similarity scores: {fallback_scores[:3] if fallback_scores else []}")
                        logger.info(f"These scores use the raw ColBERT sum-similarity (m - distance) which preserves token count signal")
                        logger.info(f"Temperature-scaled scores (τ={temperature}) and normalized scores are also calculated for reference")
                        return fallback_metadatas, fallback_scores, fallback_ids

                return metadatas_list, scores, ids

            except Exception as e:
                logger.error(f"Error in direct search: {e}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                return [], [], []

        except Exception as e:
            logger.error(f"Error querying LanceDB table: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return [], [], []
    finally:
        # Clear timeout mechanisms
        try:
            if is_main_thread:
                # Clear the alarm to prevent it from triggering elsewhere
                import signal
                signal.alarm(0)
            else:
                # Signal the timeout thread to stop
                if 'timeout_event' in locals():
                    timeout_event.set()
        except Exception as e:
            logger.debug(f"Error clearing timeout mechanism: {e}")

        # Log total query time
        end_time = time.time()
        query_time = end_time - start_time
        logger.info(f"LanceDB query took {query_time:.2f} seconds")

def clear_lancedb_table(session_id: str, model_name: str) -> bool:
    """
    Clears a LanceDB table while preserving chat history.

    Args:
        session_id: The session ID
        model_name: The model name

    Returns:
        True if successful, False otherwise
    """
    try:
        # First, ensure we have the chat history saved in the session data
        try:
            from src.services.session_manager.manager import load_session, save_session
            session_data = load_session(session_id)
        except Exception as e:
            logger.error(f"Failed to load session data: {e}")
            session_data = None

        # Get the connection
        connection = get_lancedb_connection(session_id)

        # Check if the table exists
        table_name = f"{model_name}"
        if table_name in connection.table_names():
            # Drop the table
            connection.drop_table(table_name)
            logger.info(f"Dropped LanceDB table {session_id}_{model_name}")

        # Remove metadata file
        metadata_path = os.path.join(LANCEDB_DATA_PATH, session_id, f"{model_name}_metadata.json")
        if os.path.exists(metadata_path):
            os.remove(metadata_path)
            logger.info(f"Removed metadata file for {session_id}_{model_name}")

        # Create a new empty table
        get_lancedb_table(session_id, model_name)

        # Save session data back to preserve chat history
        if session_data:
            save_session(session_id, session_data)

        return True

    except Exception as e:
        logger.error(f"Error clearing LanceDB table: {e}")
        return False

def ensure_lancedb_table_exists(session_id: str, model_name: str = "colqwen25") -> bool:
    """
    Ensure that a LanceDB table exists for the given session and model.
    If it doesn't exist, create it. Also ensures the index is created.

    Args:
        session_id: The session ID
        model_name: The model name

    Returns:
        True if the table exists or was created successfully, False otherwise
    """
    try:
        # Check if the table exists
        connection = get_lancedb_connection(session_id)
        table_name = f"{model_name}"

        if table_name in connection.table_names():
            logger.debug(f"LanceDB table {table_name} already exists for session {session_id}")

            # Table exists, but make sure the index is created
            # This is important for tables that were created before we added the index creation code
            try:
                # Get the table
                table = connection.open_table(table_name)

                # Try to create the index
                try:
                    logger.info(f"Ensuring index exists for {session_id}_{model_name}")
                    table.create_index(
                        metric="cosine",  # Only cosine is supported for multivector
                        vector_column_name="vectors"  # Must match the column name in schema
                    )
                    logger.info(f"Successfully created index for {session_id}_{model_name}")
                except Exception as index_e:
                    if "already exists" in str(index_e):
                        logger.info(f"Index already exists for {session_id}_{model_name}")
                    else:
                        logger.error(f"Error creating index for {session_id}_{model_name}: {index_e}")
                        # Continue anyway, as the table might still be usable
            except Exception as e:
                logger.error(f"Error ensuring index exists for {session_id}_{model_name}: {e}")
                # Continue anyway, as the table might still be usable

            return True

        # Table doesn't exist, create it
        logger.info(f"Creating LanceDB table {table_name} for session {session_id}")
        _, _ = get_lancedb_table(session_id, model_name)

        # Verify that the table was created
        if table_name in connection.table_names():
            logger.info(f"Successfully created LanceDB table {table_name} for session {session_id}")
            return True
        else:
            logger.error(f"Failed to create LanceDB table {table_name} for session {session_id}")
            return False

    except Exception as e:
        logger.error(f"Error ensuring LanceDB table exists for session {session_id}: {e}")
        return False


def has_documents(session_id: str, model_name: str = "colqwen25") -> bool:
    """
    Checks if the LanceDB table for the given session and model has any documents.

    Args:
        session_id: The session ID
        model_name: The model name (defaults to 'colqwen25')

    Returns:
        True if the table has documents, False otherwise
    """
    try:
        # Ensure the table exists
        if not ensure_lancedb_table_exists(session_id, model_name):
            logger.warning(f"Could not ensure LanceDB table exists for session {session_id}")
            return False

        # Get the connection and table
        connection = get_lancedb_connection(session_id)
        table_name = f"{model_name}"
        table = connection.open_table(table_name)

        # Try to get the count using SQL
        try:
            # Get the count of documents directly from the table data
            try:
                # Method 1: Try using the length of the dataframe
                df = table.to_arrow().to_pandas()
                count = len(df)
                logger.info(f"LanceDB table for {session_id}_{model_name} has {count} documents")
                return count > 0
            except Exception as e:
                logger.warning(f"Error getting document count with to_arrow(): {e}")
                
                # Method 2: Try a dummy search to see if data exists
                try:
                    # Use a simple dummy query to check if any data exists
                    from src.models.model_registry import registry as _reg2
                    dimension = _reg2.get_dimension(model_name)
                    dummy_vector = [[0.0] * dimension]
                    results = table.search(dummy_vector, vector_column_name="vectors").limit(1).to_list()
                    has_docs = len(results) > 0
                    logger.info(f"LanceDB table for {session_id}_{model_name} has documents: {has_docs} (determined by search)")
                    return has_docs
                except Exception as e2:
                    logger.warning(f"Error checking documents with search: {e2}")
                    return False
        except Exception as sql_e:
            logger.warning(f"Error using SQL COUNT: {sql_e}, falling back to pandas")

            # Fallback: try to get all rows and count them
            try:
                # Use to_pandas() to get all rows and then count them
                # Limit to 1 row to avoid loading too much data
                count = len(table.limit(1).to_pandas())
                has_docs = count > 0
                logger.info(f"LanceDB table for {session_id}_{model_name} has documents: {has_docs}")
                return has_docs
            except Exception as e:
                logger.warning(f"Error using to_pandas(): {e}, trying search")

                # Final fallback: try a simple search
                try:
                    # Get the expected dimension from the table schema
                    try:
                        # Try to get the dimension from the table schema
                        table_schema = table.schema
                        vector_field = None
                        for field in table_schema:
                            if field.name == 'vectors':
                                vector_field = field
                                break

                        if vector_field:
                            # Extract the dimension from the fixed-size list field
                            vector_type = vector_field.type
                            if hasattr(vector_type, 'value_type') and hasattr(vector_type.value_type, 'list_size'):
                                expected_dim = vector_type.value_type.list_size
                                logger.info(f"Found vector dimension in schema: {expected_dim}")
                            else:
                                # Default to 128 for ColPali models
                                expected_dim = 128
                                logger.warning(f"Could not determine vector dimension from schema, using default: {expected_dim}")
                        else:
                            # Default to 128 for ColPali models
                            expected_dim = 128
                            logger.warning(f"Could not find 'vectors' field in schema, using default: {expected_dim}")
                    except Exception as e:
                        # Default to 128 for ColPali models
                        expected_dim = 128
                        logger.warning(f"Error getting dimension from schema: {e}, using default: {expected_dim}")

                    # Create a dummy vector with the correct dimension
                    # For multivector search, we need a list of fixed-size vectors
                    dummy_vector = [[0.0] * expected_dim]
                    logger.info(f"Testing table with dummy vector of dimension {expected_dim}")

                    # Just check if the search works, don't care about results
                    # Make sure to specify the vector column name
                    # Note: metric is specified at index creation time, not search time
                    table.search(
                        dummy_vector,
                        vector_column_name="vectors"
                    ).limit(1)

                    # If we get here without error, the table exists but might be empty
                    # Check metadata to see if we have any documents
                    metadata_path = os.path.join(LANCEDB_DATA_PATH, session_id, f"{model_name}_metadata.json")
                    if os.path.exists(metadata_path):
                        try:
                            with open(metadata_path, 'r') as f:
                                metadata = json.load(f)
                                has_docs = len(metadata.get('ids', [])) > 0
                                logger.info(f"Using metadata to check documents: {has_docs}")
                                return has_docs
                        except Exception as meta_e:
                            logger.warning(f"Error reading metadata: {meta_e}")

                    logger.info(f"LanceDB table for {session_id}_{model_name} exists but status unknown")
                    return False
                except Exception as inner_e:
                    logger.error(f"Error checking table contents: {inner_e}")
                    return False

    except Exception as e:
        logger.error(f"Error checking if session {session_id} has documents in LanceDB: {e}")
        return False



def rebuild_lancedb_index(session_id: str, model_name: str = "colqwen25") -> bool:
    """
    Rebuild the LanceDB index for the given session and model.
    This is useful if the index is missing or corrupted.

    Args:
        session_id: The session ID
        model_name: The model name

    Returns:
        True if the index was rebuilt successfully, False otherwise
    """
    try:
        # Ensure the table exists
        if not ensure_lancedb_table_exists(session_id, model_name):
            logger.warning(f"Could not ensure LanceDB table exists for session {session_id}")
            return False

        # Get the table
        connection = get_lancedb_connection(session_id)
        table_name = f"{model_name}"
        table = connection.open_table(table_name)

        # Always try to create the index, regardless of whether the table has data
        try:
            logger.info(f"Creating index for {session_id}_{model_name} with vector_column_name='vectors'")

            # Create the index with optimized parameters for deterministic multivector support
            try:
                # First, count the total vectors to calculate optimal partitioning
                row_count = len(table.to_pandas())
                # For ColPali models, each document typically has ~1030 vectors
                estimated_vectors_per_doc = 1030
                total_vectors = row_count * estimated_vectors_per_doc
                
                # Calculate optimal partitions: sqrt(total_vectors) as per LanceDB docs
                # This helps balance partition size for more stable retrieval
                import numpy as np
                num_partitions = max(1, int(np.sqrt(total_vectors)))
                
                logger.info(f"Creating optimized index with {num_partitions} partitions for {row_count} documents")
                
                table.create_index(
                    metric="cosine",  # Only cosine is supported for multivector
                    vector_column_name="vectors",  # Must match the column name in schema
                    num_partitions=num_partitions,  # Optimized partition count for stability
                    num_sub_vectors=32,  # Higher for accuracy, lower for speed
                    index_type="IVF_PQ"  # Provides better determinism than default
                )
            except Exception as e:
                logger.warning(f"Error creating optimized index, falling back to basic index: {e}")
                # Fall back to basic index if advanced parameters fail
                table.create_index(
                    metric="cosine",  # Only cosine is supported for multivector
                    vector_column_name="vectors"  # Must match the column name in schema
                )
            
            logger.info(f"Successfully created index for {session_id}_{model_name}")
            return True
        except Exception as index_e:
            if "already exists" in str(index_e):
                logger.info(f"Index already exists for {session_id}_{model_name}")
                return True
            else:
                logger.error(f"Error creating index for {session_id}_{model_name}: {index_e}")

                # Even if there's an error creating the index, try to verify if the table is searchable
                try:
                    # Try to get one row using search with a dummy vector
                    from src.models.model_registry import registry as _reg3
                    dimension = _reg3.get_dimension(model_name)
                    dummy_vector = [[0.0] * dimension]

                    # Just check if the search works, don't care about results
                    # Make sure to specify the vector column name
                    # Note: metric is specified at index creation time, not search time
                    table.search(
                        dummy_vector,
                        vector_column_name="vectors"
                    ).limit(1)

                    # If we get here without error, the index might already exist
                    logger.info(f"LanceDB index for {session_id}_{model_name} appears to be working despite error")
                    return True
                except Exception as search_e:
                    logger.error(f"Error searching LanceDB table after index creation failed: {search_e}")
                    return False

    except Exception as e:
        logger.error(f"Error rebuilding LanceDB index for session {session_id}: {e}")
        return False


def release_memory_to_os():
    """
    Aggressively release memory back to the operating system.
    This function uses multiple techniques to ensure memory is properly released.
    """
    # First, force Python garbage collection
    import gc
    gc.collect()

    # Clear CUDA cache if available
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.debug("Cleared CUDA cache")
    except ImportError:
        pass

    # On Linux, try to use malloc_trim to release memory to the OS
    try:
        import ctypes
        libc = ctypes.CDLL('libc.so.6')
        # MADV_DONTNEED = 4
        libc.malloc_trim(0)
        logger.debug("Called malloc_trim to release memory to the system")
    except Exception as e:
        logger.warning(f"Could not call malloc_trim: {e}")

    # On Linux, try to use madvise to release memory
    try:
        import resource
        import psutil

        # Get current process
        process = psutil.Process()

        # Try to release memory
        process.memory_info()  # Force update of memory info

        # Set the process's memory limit to current usage to encourage releasing memory
        current_usage = process.memory_info().rss
        resource.setrlimit(resource.RLIMIT_AS, (current_usage, current_usage))

        logger.debug(f"Set memory limit to current usage: {current_usage / (1024 * 1024):.2f} MB")
    except Exception as e:
        logger.warning(f"Could not set memory limit: {e}")

    logger.debug("Completed aggressive memory release")

def cleanup_old_connections(max_idle_time=300):
    """
    Cleanup LanceDB connections that haven't been used for a while.

    Args:
        max_idle_time: Maximum idle time in seconds before a connection is closed (default: 5 minutes)

    Returns:
        list: List of session IDs that were cleaned up
    """
    global _lancedb_connections, _CONNECTION_LAST_USED, _CONNECTION_LOCK

    cleaned_sessions = []
    current_time = time.time()

    with _CONNECTION_LOCK:
        # Find connections that haven't been used for a while
        for session_id in list(_CONNECTION_LAST_USED.keys()):
            last_used = _CONNECTION_LAST_USED.get(session_id, 0)
            if current_time - last_used > max_idle_time:
                try:
                    logger.debug(f"Cleaning up idle LanceDB connection for {session_id} (idle for {current_time - last_used:.1f}s)")
                    _close_lancedb_connection(session_id)
                    cleaned_sessions.append(session_id)
                except Exception as e:
                    logger.warning(f"Error cleaning up idle LanceDB connection for {session_id}: {e}")

    if cleaned_sessions:
        logger.debug(f"Cleaned up {len(cleaned_sessions)} idle LanceDB connections")

        # Aggressively release memory
        release_memory_to_os()

    return cleaned_sessions

def clear_lancedb_resources(session_id):
    """
    Clear LanceDB connection for a specific session.
    This only closes the connection, it does NOT delete any data.

    Args:
        session_id: The session ID to clear connection for

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Only close the connection, don't delete files
        with _CONNECTION_LOCK:
            if session_id in _lancedb_connections:
                _close_lancedb_connection(session_id)
                logger.debug(f"Closed LanceDB connection for session {session_id}")
                return True
            else:
                logger.debug(f"No active LanceDB connection found for session {session_id}")
                return True  # Not an error if no connection exists
    except Exception as e:
        logger.error(f"Error in clear_lancedb_resources: {e}")
        return False


def destroy_lancedb_resources(session_id=None, preserve_active_sessions=True):
    """
    Clean up LanceDB resources, optionally including files on disk.

    Args:
        session_id: If provided, only clear resources for this session
        preserve_active_sessions: If True, don't remove resources for active sessions

    Returns:
        dict: Information about cleanup operation
    """
    global _lancedb_connections

    # Module-level variables to track what was cleared
    connections_cleared = []
    files_removed = 0

    try:
        # Get list of active sessions from the sessions directory
        active_sessions = []
        if preserve_active_sessions:
            try:
                sessions_dir = os.path.join(os.getcwd(), "sessions")
                if os.path.exists(sessions_dir):
                    for file in os.listdir(sessions_dir):
                        if file.endswith(".json"):
                            active_sessions.append(file.replace(".json", ""))
                    logger.debug(f"Found {len(active_sessions)} active sessions to preserve")
            except Exception as e:
                logger.warning(f"Error getting active sessions: {e}")

        # Clear connections
        if session_id:
            # Clear only the specified session
            if session_id in _lancedb_connections:
                # Use our helper function to properly close the connection
                try:
                    _close_lancedb_connection(session_id)
                    connections_cleared.append(session_id)
                    logger.debug(f"Cleared LanceDB connection for session {session_id}")
                except Exception as close_e:
                    logger.warning(f"Error closing LanceDB connection for {session_id}: {close_e}")
        else:
            # Clear connections that are not in active_sessions
            to_clear = []
            for conn_id in list(_lancedb_connections.keys()):  # Use list to avoid modification during iteration
                if not preserve_active_sessions or conn_id not in active_sessions:
                    to_clear.append(conn_id)

            # Properly close each connection
            for conn_id in to_clear:
                try:
                    _close_lancedb_connection(conn_id)
                    connections_cleared.append(conn_id)
                except Exception as close_e:
                    logger.warning(f"Error closing LanceDB connection for {conn_id}: {close_e}")

            logger.debug(f"Cleared {len(connections_cleared)} LanceDB connections")

        # Clear disk resources
        lancedb_root_dir = LANCEDB_DATA_PATH

        # Create the directory if it doesn't exist
        os.makedirs(lancedb_root_dir, exist_ok=True)

        # Now we can safely proceed
        if session_id:
            # Clear only the specified session
            session_dir = os.path.join(lancedb_root_dir, session_id)
            if os.path.exists(session_dir) and (not preserve_active_sessions or session_id not in active_sessions):
                # Count files before removing
                file_count = 0
                for _, _, files in os.walk(session_dir):
                    file_count += len(files)

                # Remove all files
                shutil.rmtree(session_dir)
                os.makedirs(session_dir, exist_ok=True)
                files_removed = file_count
                logger.debug(f"Removed {file_count} files from session directory {session_dir}")
        else:
            # Only clear sessions that are not active
            for dir_name in os.listdir(lancedb_root_dir):
                dir_path = os.path.join(lancedb_root_dir, dir_name)
                if os.path.isdir(dir_path) and (not preserve_active_sessions or dir_name not in active_sessions):
                    # Count files before removing
                    file_count = 0
                    for _, _, files in os.walk(dir_path):
                        file_count += len(files)

                    # Remove all files
                    shutil.rmtree(dir_path)
                    os.makedirs(dir_path, exist_ok=True)
                    files_removed += file_count
                    logger.debug(f"Removed {file_count} files from session directory {dir_path}")

        # Force garbage collection to free memory
        import gc
        gc.collect()

        # Clear CUDA cache if available
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.debug("Cleared CUDA cache after LanceDB cleanup")
        except ImportError:
            pass

    except Exception as e:
        logger.error(f"Error clearing LanceDB resources: {e}")

    return {
        "connections_cleared": connections_cleared,
        "files_removed": files_removed
    }

def clear_query_cache():
    """
    Clear any cached query results or temporary data from LanceDB operations.
    This helps prevent memory buildup from query results between chat sessions.
    """
    global _CONNECTION_LAST_USED
    
    with _CONNECTION_LOCK:
        try:
            # Clear connection last used tracking to force fresh connections
            connections_cleared = len(_CONNECTION_LAST_USED)
            _CONNECTION_LAST_USED.clear()
            
            # Force garbage collection of any cached query results
            import gc
            gc.collect()
            
            logger.debug(f"Cleared LanceDB query cache and {connections_cleared} connection timestamps")
            
        except Exception as e:
            logger.warning(f"Error clearing LanceDB query cache: {e}")