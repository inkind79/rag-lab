"""
Core document processing functionality.

This module contains functions for processing uploaded documents,
indexing them for retrieval, and managing document metadata.
"""
import os
import logging
import torch
import gc
import tempfile
import psutil
from PIL import Image
from pdf2image import convert_from_path
from src.models.vector_stores.embedding_utils import process_embedding_for_lancedb, debug_embedding
# Import LanceDB manager
from src.models.vector_stores.lancedb_manager import add_embeddings_to_lancedb
from src.models.model_loader import load_rag_model # Need this to get the embedding model

logger = logging.getLogger(__name__)


def determine_lancedb_model_name(indexer_model):
    """Determine LanceDB model name from indexer model.

    Delegates to the centralized utility in model_type_utils.
    """
    from src.utils.model_type_utils import get_lancedb_table_name
    return get_lancedb_table_name(indexer_model)

def validate_embedding(embedding, context_name):
    """Validate that embedding is not None or empty (consolidated logic)"""
    if embedding is None or (hasattr(embedding, 'nelement') and embedding.nelement() == 0):
        logger.warning(f"Failed to generate embedding for {context_name}")
        return False
    return True

def handle_processing_result(success, context_name):
    """Handle processing result with consistent error logging (consolidated logic)"""
    if not success:
        logger.error(f"LanceDB processing failed for {context_name}, skipping")
        return False
    return True

def convert_tensor_to_numpy(embedding):
    """Convert tensor to numpy to free GPU memory (consolidated logic)"""
    if isinstance(embedding, torch.Tensor):
        embedding_np = embedding.cpu().numpy()
        del embedding
        torch.cuda.empty_cache()
        return embedding_np
    return embedding

def process_batch_if_full(lancedb_batch, session_id, lancedb_model_name, batch_size=100):
    """Process batch if it's large enough (consolidated logic)"""
    if len(lancedb_batch['embeddings']) >= batch_size:
        # Add embeddings to LanceDB
        add_embeddings_to_lancedb(
            session_id,
            lancedb_model_name,
            lancedb_batch['embeddings'],
            lancedb_batch['ids'],
            lancedb_batch['metadatas']
        )
        logger.info(f"Added batch of {len(lancedb_batch['embeddings'])} embeddings to LanceDB")
        # Clean up embeddings to free memory
        cleanup_embedding_tensors(lancedb_batch['embeddings'])
        # Reset batch
        lancedb_batch['embeddings'] = []
        lancedb_batch['metadatas'] = []
        lancedb_batch['ids'] = []

def log_memory_progress(processed_embeddings):
    """Log memory usage during indexing (consolidated logic)"""
    if processed_embeddings % 10 == 0:  # Log every 10 embeddings
        process = psutil.Process()
        ram_mb = process.memory_info().rss / 1024 / 1024
        gpu_mb = 0
        if torch.cuda.is_available():
            gpu_mb = torch.cuda.memory_allocated() / 1024 / 1024
        logger.info(f"Indexing progress: {processed_embeddings} embeddings, RAM={ram_mb:.0f}MB, GPU={gpu_mb:.0f}MB")

def process_embedding_for_storage(RAG, embedding, debug_name):
    """Process embedding for LanceDB storage (consolidated logic)"""
    # Since the system only uses ColPali models, always use standard processing
    # The standard function handles ColPali embeddings correctly
    logger.info(f"Processing embedding for LanceDB - {debug_name}")
    return process_embedding_for_lancedb(embedding, debug_name)

def process_document_embedding(RAG, embedding, current_doc_id, page_num, doc_metadata, 
                              lancedb_model_name, lancedb_batch, session_id, processed_embeddings):
    """Unified document processing function to eliminate duplication"""
    
    # Process embedding for LanceDB storage
    lancedb_embedding = process_embedding_for_storage(RAG, embedding, 
                                                     f"document_{doc_metadata['filename']}_page_{page_num}")
    
    # Add to LanceDB store
    if lancedb_embedding is not None:
        
        # Add to batch
        chroma_id = f"{current_doc_id}_{page_num}"
        page_metadata = doc_metadata.copy()
        page_metadata['page_num'] = page_num
        
        # Convert tensor to numpy to free GPU memory
        lancedb_embedding = convert_tensor_to_numpy(lancedb_embedding)
        lancedb_batch['embeddings'].append(lancedb_embedding)
        lancedb_batch['metadatas'].append(page_metadata)
        lancedb_batch['ids'].append(chroma_id)
        
        # Process batch if it's large enough
        process_batch_if_full(lancedb_batch, session_id, lancedb_model_name)
        
        # Track processed embeddings
        processed_embeddings += 1
        
        # Log memory usage during indexing
        log_memory_progress(processed_embeddings)
        
        return True, processed_embeddings
    else:
        logger.warning(f"Failed to process embedding for {doc_metadata['filename']} page {page_num} for LanceDB")
        return False, processed_embeddings

def process_uploaded_files(files, session_id, upload_folder):
    """
    Process uploaded files for a session.

    Args:
        files: List of uploaded file objects
        session_id: Current session ID
        upload_folder: Base folder for uploads

    Returns:
        List of uploaded filenames
    """
    from werkzeug.utils import secure_filename
    import shutil  # Import here for file operations

    session_folder = os.path.join(upload_folder, session_id)
    os.makedirs(session_folder, exist_ok=True)

    uploaded_files = []
    for file in files:
        if file and file.filename:
            filename = secure_filename(file.filename)
            file_path = os.path.join(session_folder, filename)
            file.save(file_path)
            uploaded_files.append(filename)
            logger.info(f"File saved: {file_path}")

    return uploaded_files

def build_document_metadata(filenames, session_folder, indexed_files_info=None):
    """
    Build metadata for documents.

    Args:
        filenames: List of filenames
        session_folder: Folder containing the files
        indexed_files_info: Optional list of already indexed file info

    Returns:
        List of metadata dictionaries for the documents
    """
    metadata_list = []

    # Create metadata with file_type information for better RAG retrieval
    for filename in filenames:
        # Look for comprehensive metadata if provided
        file_metadata = None
        if indexed_files_info:
            for info in indexed_files_info:
                if info['filename'] == filename:
                    file_metadata = info
                    break

        # Build metadata for this file
        metadata = {'filename': filename}

        # Add file_type if available
        if file_metadata and 'file_type' in file_metadata:
            metadata['file_type'] = file_metadata['file_type']
        else:
            # Try to detect file type
            file_path = os.path.join(session_folder, filename)
            try:
                import imghdr
                img_type = imghdr.what(file_path)
                if img_type:
                    metadata['file_type'] = 'image'
                elif filename.lower().endswith('.pdf'):
                    metadata['file_type'] = 'pdf'
                else:
                    metadata['file_type'] = 'unknown'
            except Exception:
                metadata['file_type'] = 'unknown'

        # Add to metadata list
        metadata_list.append(metadata)

    logger.info(f"Built enhanced metadata for indexing: {metadata_list}")
    return metadata_list

def index_documents_for_rag(session_id, session_folder, metadata_list,
                            indexer_model, app_config, rag_models):  # app_config is used by callers
    """
    Index documents for RAG retrieval using LanceDB.
    Only indexes new documents that haven't been indexed before.

    Args:
        session_id: Current session ID
        session_folder: Folder containing the files
        metadata_list: List of metadata dictionaries for the documents
        indexer_model: Model to use for indexing
        app_config: Flask app configuration
        rag_models: Dictionary of RAG models

    Returns:
        Tuple of (success, message, RAG model)
    """
    # Import needed modules at function start
    from src.models.model_loader import load_rag_model
    
    # Initialize process tracking
    process = psutil.Process(os.getpid())

    # Track initial memory state
    initial_mem = process.memory_info().rss / (1024 * 1024)  # MB
    initial_gpu = 0
    if torch.cuda.is_available():
        initial_gpu = torch.cuda.memory_allocated() / (1024 * 1024)  # MB
    logger.info(f"MEMORY DEBUG: Initial state: RAM={initial_mem:.2f}MB, GPU={initial_gpu:.2f}MB")

    # --- Session Setup ---
    # Get distance metric from session data or default to cosine
    from src.services.session_manager.manager import load_session, save_session
    session_data = load_session('sessions', session_id)
    distance_metric = session_data.get('distance_metric', 'cosine')
    logger.info(f"Using distance metric: {distance_metric} for session {session_id}")

    # Check if we need to force re-indexing due to model change
    last_indexer_model = session_data.get('last_indexer_model')
    indexed_files = session_data.get('indexed_files', [])
    model_changed = False

    if last_indexer_model is not None and last_indexer_model != indexer_model:
        # Explicit model change detected
        model_changed = True
        logger.info(f"Indexer model changed from '{last_indexer_model}' to '{indexer_model}' — forcing re-index")
    elif indexed_files and last_indexer_model is None:
        # Documents were indexed before we tracked the model — check if the target
        # LanceDB table exists and has data. If not, we need to re-index.
        from src.utils.model_type_utils import get_lancedb_table_name
        target_table = get_lancedb_table_name(indexer_model)
        try:
            from src.models.vector_stores.lancedb_manager import get_lancedb_connection
            conn = get_lancedb_connection(session_id)
            if not conn.table_exists(target_table):
                model_changed = True
                logger.info(f"Target LanceDB table '{target_table}' doesn't exist — forcing re-index for model '{indexer_model}'")
            else:
                table = conn.open_table(target_table)
                row_count = table.count_rows()
                if row_count == 0:
                    model_changed = True
                    logger.info(f"Target LanceDB table '{target_table}' is empty — forcing re-index for model '{indexer_model}'")
        except Exception as e:
            logger.warning(f"Error checking LanceDB table for model change detection: {e}")

    if model_changed:
        logger.info(f"Will re-index all documents with model '{indexer_model}'")

    # Get list of already indexed documents from session data
    indexed_files = session_data.get('indexed_files', [])
    already_indexed_filenames = set()
    if not model_changed:
        for file_info in indexed_files:
            if isinstance(file_info, dict) and 'filename' in file_info:
                already_indexed_filenames.add(file_info['filename'])
            elif isinstance(file_info, str):
                already_indexed_filenames.add(file_info)

    # Filter metadata_list to only include new documents (or all if model changed)
    new_metadata_list = []
    for doc_metadata in metadata_list:
        filename = doc_metadata.get('filename')
        if filename and filename not in already_indexed_filenames:
            new_metadata_list.append(doc_metadata)
            logger.info(f"Document {filename} {'needs re-indexing (model changed)' if model_changed else 'is new and will be indexed'}")
        elif filename:
            logger.info(f"Document {filename} is already indexed with current model, skipping")

    # If model changed, clear old indexed_files metadata so they get re-tracked
    if model_changed and new_metadata_list:
        session_data['indexed_files'] = []
        logger.info(f"Cleared indexed_files metadata for re-indexing with new model")

    # If no new documents, ensure the embedding model is still loaded in RAG_models
    if not new_metadata_list:
        logger.info(f"No new documents to index for session {session_id}")
        
        # Load the embedding model and register it in rag_models for this session
        try:
            embedding_model = load_rag_model(indexer_model)
            if embedding_model and rag_models is not None:
                # Store the embedding model in rag_models for later use
                rag_models.set(session_id, embedding_model)
                logger.info(f"Registered existing embedding model in rag_models for session {session_id}")
            else:
                logger.warning(f"Failed to load embedding model for session {session_id}")
        except Exception as e:
            logger.error(f"Error loading embedding model for existing documents: {e}")
        
        return True, "No new documents to index", None

    logger.info(f"Found {len(new_metadata_list)} new documents to index for session {session_id}")
    # --- End ChromaDB Setup ---

    # Pre-compute LanceDB model name for consistent usage
    lancedb_model_name = determine_lancedb_model_name(indexer_model)
    logger.info(f"Using LanceDB model name: {lancedb_model_name}")

    RAG = None # Initialize RAG variable

    try:
        # --- SIMPLIFIED MEMORY CLEANUP ---
        # Only perform minimal cleanup to avoid conflicts between multiple systems
        # Do NOT use aggressive_cleanup as it clears embedding model caches
        logger.info(f"Performing minimal memory cleanup before indexing for session {session_id}")
        
        # Lightweight cleanup that preserves caches
        try:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("Completed lightweight cleanup preserving model caches")
        except Exception as e:
            logger.warning(f"Error during memory cleanup: {e}")
        
        logger.info(f"Cleanup complete for session {session_id}")
        # --- END SIMPLIFIED CLEANUP ---


        # --- Load RAG Model (for Embeddings Only) ---
        # Load model using the global cache (no session cache to avoid circular imports)
        logger.info(f"Loading model '{indexer_model}' for embedding generation.")
        RAG = load_rag_model(indexer_model)

        # Validate model loading
        if RAG is None:
            return False, f"Failed to load model '{indexer_model}' for embedding.", None
        logger.info(f"Using ColPali adapter for embedding generation")
        logger.info(f"Model '{indexer_model}' loaded successfully.")
        # --- End Model Loading ---

        # --- Indexing Loop with LanceDB ---
        logger.info(f"Starting LanceDB indexing for {len(new_metadata_list)} new documents...")
        # Get the next available doc_id to avoid collisions
        doc_id_counter = len(indexed_files) # Start from the count of existing files
        processed_embeddings = 0
        
        # Initialize batch processing variables
        lancedb_batch = {
            'embeddings': [],
            'metadatas': [],
            'ids': []
        }

        for doc_metadata in new_metadata_list:
            filename = doc_metadata.get('filename')
            if not filename:
                logger.warning("Skipping metadata entry without filename.")
                continue

            file_path = os.path.join(session_folder, filename)
            if not os.path.exists(file_path):
                logger.warning(f"File not found for indexing: {file_path}, skipping.")
                continue

            # Assign a document ID (could use a hash or counter)
            # For simplicity, using a counter for now
            current_doc_id = doc_id_counter
            doc_id_counter += 1
            doc_metadata['doc_id'] = current_doc_id # Add doc_id to metadata

            # No longer need to track these for ChromaDB
            # Just track the document ID for logging

            try:
                if filename.lower().endswith('.pdf'):
                    logger.debug(f"Processing PDF: {filename}")
                    with tempfile.TemporaryDirectory() as temp_dir:
                        # Limit threads for pdf2image to avoid excessive memory use
                        num_threads = max(1, os.cpu_count() // 2)
                        logger.debug(f"Using {num_threads} threads for pdf2image")
                        images = convert_from_path(file_path, output_folder=temp_dir, paths_only=False, thread_count=num_threads)
                        logger.debug(f"Extracted {len(images)} pages from {filename}")
                        for i, page_image in enumerate(images):
                            page_num = i + 1
                            # Generate embedding for the page image
                            # Always use the adapter's encode_image method for ColPali models
                            page_embedding = RAG.encode_image([page_image])

                            if not validate_embedding(page_embedding, f"{filename} page {page_num}"):
                                continue

                            # Debug the embedding shape and properties
                            debug_embedding(page_embedding, f"document_{filename}_page_{page_num}")

                            # Process document embedding using unified function
                            success, processed_embeddings = process_document_embedding(
                                RAG, page_embedding, current_doc_id, page_num, doc_metadata,
                                lancedb_model_name, lancedb_batch, session_id, processed_embeddings
                            )
                            
                            if not handle_processing_result(success, f"{filename} page {page_num}"):
                                continue

                elif doc_metadata.get('file_type') == 'image': # Handle images
                     logger.debug(f"Processing Image: {filename}")
                     with Image.open(file_path) as img:
                         # Ensure image is in RGB format
                         img_rgb = img.convert('RGB')
                         # Always use the adapter's encode_image method for ColPali models
                         image_embedding = RAG.encode_image([img_rgb])

                         if not validate_embedding(image_embedding, f"image {filename}"):
                             continue

                         # Debug the embedding shape and properties
                         debug_embedding(image_embedding, f"document_{filename}_image")

                         # Process image embedding using unified function  
                         success, processed_embeddings = process_document_embedding(
                             RAG, image_embedding, current_doc_id, 1, doc_metadata,
                             lancedb_model_name, lancedb_batch, session_id, processed_embeddings
                         )
                         
                         if not handle_processing_result(success, f"image {filename}"):
                             continue

                else:
                    logger.warning(f"Skipping unsupported file type for indexing: {filename}")

                # Force garbage collection after processing a document
                logger.debug(f"Processed document {filename} with ID {current_doc_id}")
                gc.collect() # Collect garbage after processing a document

            except Exception as e:
                logger.error(f"Error processing document {filename} for LanceDB indexing: {e}", exc_info=True)
                # Decide whether to continue with other documents or fail

        # --- Process Final Batch ---
        # Add any remaining embeddings in the batch after the loop finishes
        if lancedb_batch and lancedb_batch['embeddings']:
            logger.info(f"Adding final batch of {len(lancedb_batch['embeddings'])} embeddings to LanceDB")
            
            # Force process the final batch (ignore size limit) using pre-computed model name
            add_embeddings_to_lancedb(
                session_id,
                lancedb_model_name,
                lancedb_batch['embeddings'],
                lancedb_batch['ids'],
                lancedb_batch['metadatas']
            )
            logger.info(f"Added final batch of {len(lancedb_batch['embeddings'])} embeddings to LanceDB")
            cleanup_embedding_tensors(lancedb_batch['embeddings'])

        logger.info(f"LanceDB indexing complete. Processed {processed_embeddings} total embeddings.")
        # --- End Indexing Loop ---

        # --- Update Session Data with Newly Indexed Files ---
        # Add the newly indexed documents to the session's indexed_files list
        for doc_metadata in new_metadata_list:
            filename = doc_metadata.get('filename')
            if filename:
                # Create a file info entry similar to what index_documents() returns
                file_info = {
                    'filename': filename,
                    'file_type': doc_metadata.get('file_type', 'unknown')
                }
                # Add page_count if available
                if doc_metadata.get('file_type') == 'pdf' and 'page_count' in doc_metadata:
                    file_info['page_count'] = doc_metadata['page_count']
                elif doc_metadata.get('file_type') == 'image':
                    file_info['page_count'] = 1

                # Add to indexed_files if not already there
                if not any(f.get('filename') == filename for f in indexed_files if isinstance(f, dict)):
                    indexed_files.append(file_info)
                    logger.info(f"Added {filename} to session's indexed_files list")

        # Save updated session data with the model used for indexing
        session_data['indexed_files'] = indexed_files
        session_data['last_indexer_model'] = indexer_model
        save_session('sessions', session_id, session_data)
        logger.info(f"Updated session data with {len(indexed_files)} indexed files (model: {indexer_model})")
        # --- End Update Session Data ---

        # Track memory after indexing
        mem_after = process.memory_info().rss / (1024 * 1024)  # MB
        gpu_mem_after = 0
        if torch.cuda.is_available():
            gpu_mem_after = torch.cuda.memory_allocated() / (1024 * 1024)  # MB

        logger.info(f"MEMORY DEBUG: After indexing: RAM={mem_after:.2f}MB, GPU={gpu_mem_after:.2f}MB")

        # --- CRITICAL: Clean up embedding model after indexing ---
        # The embedding model is no longer needed after indexing is complete.
        # Remove it from memory to prevent accumulation when multiple sessions upload documents.
        try:
            from src.models.memory_cleanup_handler import (
                cleanup_embedding_model_after_indexing,
                cleanup_inactive_session_models,
                force_lancedb_cleanup
            )
            logger.info(f"Cleaning up embedding model for session {session_id} after indexing")
            cleanup_embedding_model_after_indexing(session_id, rag_models)
            
            # Also clean up any other inactive session models
            cleanup_inactive_session_models(session_id, rag_models, max_models=0)  # Keep 0 models after indexing
            
            # Force LanceDB cleanup for this session
            force_lancedb_cleanup(session_id)
            
            # Clear the RAG reference since we removed it from the dict
            RAG = None
            
        except Exception as cleanup_error:
            logger.error(f"Error during post-indexing cleanup: {cleanup_error}")
        # --- End post-indexing cleanup ---
        
        # Return success
        return True, "Documents indexed successfully", None  # Return None instead of RAG since we cleaned it up

    except Exception as e:
        logger.error(f"Error initializing or indexing with RAG model: {e}", exc_info=True)
        # Clean up on error too
        try:
            if RAG is not None and session_id in rag_models:
                from src.models.memory_cleanup_handler import cleanup_embedding_model_after_indexing
                cleanup_embedding_model_after_indexing(session_id, rag_models)
        except:
            pass
        return False, f"Error initializing or indexing model: {str(e)}", None


def cleanup_embedding_tensors(embeddings):
    """Clean up embedding tensors to free GPU memory"""
    import torch
    import gc
    
    for embedding in embeddings:
        if isinstance(embedding, torch.Tensor):
            # Move to CPU and delete
            embedding.cpu()
            del embedding
        elif hasattr(embedding, '__iter__'):
            # If it's already a numpy array or list, just continue
            pass
    
    # Force GPU memory cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

# Save original function before defining wrapper
original_index_documents_for_rag = index_documents_for_rag

# Comprehensive error handling for document upload
def safe_index_documents_for_rag(*args, **kwargs):
    """Wrapper for index_documents_for_rag with comprehensive error handling"""
    try:
        return original_index_documents_for_rag(*args, **kwargs)
    except AttributeError as e:
        logger.error(f"Method/attribute error in document indexing: {e}")
        return False, f"Model method error: {str(e)}", None
    except ImportError as e:
        logger.error(f"Import error in document indexing: {e}")
        return False, f"Import error: {str(e)}", None
    except Exception as e:
        logger.error(f"Unexpected error in document indexing: {e}", exc_info=True)
        return False, f"Unexpected error: {str(e)}", None

# Replace the original function with the safe wrapper
index_documents_for_rag = safe_index_documents_for_rag


def index_documents_for_text_retrieval(session_id, session_folder, metadata_list,
                                        retrieval_method, embedding_adapter=None,
                                        chunk_size=512, chunk_overlap=64):
    """
    Index documents using text extraction + chunking for text-based retrieval methods.

    Parallel to index_documents_for_rag() which uses ColPali visual encoding.
    This function extracts text from PDFs/images, chunks it, and stores in the
    appropriate vector store (BM25 or single-vector LanceDB).

    Args:
        session_id: Current session ID
        session_folder: Folder containing the uploaded files
        metadata_list: List of metadata dicts with 'filename' and 'file_type' keys
        retrieval_method: 'bm25' or 'dense'
        embedding_adapter: Embedding adapter (required for 'dense', None for 'bm25')
        chunk_size: Characters per text chunk
        chunk_overlap: Overlap between consecutive chunks

    Returns:
        Tuple of (success: bool, message: str, adapter)
    """
    from src.services.document_processor.text_extractor import extract_and_chunk_document

    try:
        # Check which documents are already indexed in the text store
        already_indexed = set()
        if retrieval_method == 'bm25':
            from src.models.vector_stores.bm25_store import get_bm25_store
            store = get_bm25_store()
            existing = store._load_index(session_id)
            if existing:
                # Track filenames already in the index
                for meta in existing.get('metadatas', []):
                    already_indexed.add(meta.get('filename'))

        all_chunks = []
        for doc_idx, doc_meta in enumerate(metadata_list):
            filename = doc_meta.get('filename')
            if not filename:
                continue
            if filename in already_indexed:
                logger.info(f"Skipping already-indexed document for text retrieval: {filename}")
                continue
            file_path = os.path.join(session_folder, filename)
            if not os.path.exists(file_path):
                logger.warning(f"File not found for text indexing: {file_path}")
                continue

            chunks = extract_and_chunk_document(
                file_path, filename, doc_id=doc_idx,
                chunk_size=chunk_size, chunk_overlap=chunk_overlap
            )
            all_chunks.extend(chunks)

        if not all_chunks:
            return False, "No text extracted from documents", None

        logger.info(f"Extracted {len(all_chunks)} text chunks from {len(metadata_list)} documents")

        # Prepare data for the store
        ids = [c.chunk_id for c in all_chunks]
        texts = [c.text for c in all_chunks]
        metadatas = [{
            'filename': c.source_filename,
            'page_num': c.page_num,
            'chunk_index': c.chunk_index,
        } for c in all_chunks]

        if retrieval_method == 'bm25':
            from src.models.vector_stores.bm25_store import get_bm25_store
            store = get_bm25_store()
            embeddings = [None] * len(all_chunks)
            success = store.add_documents(session_id, embeddings, ids, metadatas, texts)

        elif retrieval_method == 'dense':
            if embedding_adapter is None:
                return False, "Dense retrieval requires an embedding adapter", None
            from src.models.vector_stores.single_vector_lancedb_store import SingleVectorLanceDBStore
            store = SingleVectorLanceDBStore(
                model_name=embedding_adapter.model_name,
                dimension=embedding_adapter.dimension
            )
            # Embed all chunks
            logger.info(f"Embedding {len(all_chunks)} chunks with {embedding_adapter.model_name}")
            embeddings = []
            for chunk in all_chunks:
                emb = embedding_adapter.encode_text(chunk.text)
                embeddings.append(emb)
            success = store.add_documents(session_id, embeddings, ids, metadatas, texts)

        else:
            return False, f"Unknown text retrieval method: {retrieval_method}", None

        if success:
            msg = f"Indexed {len(all_chunks)} text chunks via {retrieval_method}"
            logger.info(msg)
            return True, msg, embedding_adapter
        else:
            return False, f"Failed to store text chunks via {retrieval_method}", None

    except Exception as e:
        logger.error(f"Error in text indexing: {e}", exc_info=True)
        return False, f"Text indexing error: {str(e)}", None


def index_documents(session_id, session_folder, metadata_list, session_data,
                    app_config, rag_models):
    """
    Dispatch function that indexes documents using the method specified in session_data.

    For 'colpali': calls index_documents_for_rag (visual pipeline)
    For 'bm25': calls index_documents_for_text_retrieval (text pipeline)
    For 'dense': calls index_documents_for_text_retrieval with embedding adapter
    For 'hybrid_rrf': calls both visual and text pipelines

    Args:
        session_id: Current session ID
        session_folder: Upload folder path
        metadata_list: Document metadata
        session_data: Full session data dict (contains retrieval_method, indexer_model, etc.)
        app_config: Flask app configuration
        rag_models: RAG models dict

    Returns:
        Tuple of (success: bool, message: str, model_or_adapter)
    """
    retrieval_method = session_data.get('retrieval_method', 'colpali')
    indexer_model = session_data.get('indexer_model', 'athrael-soju/colqwen3.5-4.5B-v3')
    chunk_size = session_data.get('chunk_size', 512)
    chunk_overlap = session_data.get('chunk_overlap', 64)

    logger.info(f"Indexing documents with method='{retrieval_method}' for session {session_id}")

    if retrieval_method == 'colpali':
        # Existing visual pipeline
        return index_documents_for_rag(
            session_id, session_folder, metadata_list,
            indexer_model, app_config, rag_models
        )

    elif retrieval_method in ('bm25', 'dense'):
        # Text-based pipeline
        embedding_adapter = None
        if retrieval_method == 'dense':
            from src.models.model_loader import load_embedding_adapter
            text_model = session_data.get('text_embedding_model', 'BAAI/bge-m3')
            embedding_adapter = load_embedding_adapter(text_model)

        return index_documents_for_text_retrieval(
            session_id, session_folder, metadata_list,
            retrieval_method, embedding_adapter,
            chunk_size, chunk_overlap
        )

    elif retrieval_method == 'hybrid_rrf':
        # Both pipelines
        # First: visual pipeline
        vis_success, vis_msg, vis_model = index_documents_for_rag(
            session_id, session_folder, metadata_list,
            indexer_model, app_config, rag_models
        )
        # Second: BM25 text pipeline (no embedding needed)
        txt_success, txt_msg, _ = index_documents_for_text_retrieval(
            session_id, session_folder, metadata_list,
            'bm25', None, chunk_size, chunk_overlap
        )

        if vis_success and txt_success:
            return True, f"Hybrid indexing: {vis_msg}; {txt_msg}", vis_model
        elif vis_success:
            return True, f"Hybrid indexing (visual OK, text failed: {txt_msg})", vis_model
        elif txt_success:
            return True, f"Hybrid indexing (text OK, visual failed: {vis_msg})", None
        else:
            return False, f"Hybrid indexing failed: {vis_msg}; {txt_msg}", None

    else:
        logger.warning(f"Unknown retrieval method: {retrieval_method}, falling back to colpali")
        return index_documents_for_rag(
            session_id, session_folder, metadata_list,
            indexer_model, app_config, rag_models
        )
