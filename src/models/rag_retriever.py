"""
Standard RAG Lab Retriever Implementation

This module implements the standard retriever for document retrieval.
It inherits from the BaseRetriever interface.
"""

import base64
import os
import shutil
import hashlib
import re
import time
import json
import traceback
import torch
import ollama
from PIL import Image
from io import BytesIO
from typing import List, Dict, Any, Optional, Tuple
import logging  # Added import for setLevel
import numpy as np  # Added for embedding conversion

from src.models.retriever_base import BaseRetriever
from src.utils.logger import get_logger
# Using LanceDB for document retrieval
from src.models.vector_stores.embedding_utils import process_embedding_for_lancedb, debug_embedding
from src.models.vector_stores.score_analysis import analyze_score_distribution, apply_token_budget_filter
# Import LanceDB functions
from src.models.vector_stores.lancedb_manager import query_lancedb
from src.utils.ocr_cache_utils import get_ocr_cache_for_session
from src.utils.deferred_cleanup import defer_cleanup
from src.utils.memory_management.memory_logger import log_memory_usage, log_memory_comparison

# Add pdf2image and imghdr for file handling
try:
    from pdf2image import convert_from_path
except ImportError:
    logger = get_logger(__name__)
    logger.warning("pdf2image not found. Install with 'pip install pdf2image' and ensure poppler is installed.")
    convert_from_path = None
import imghdr

logger = get_logger(__name__)
logger.setLevel(logging.DEBUG) # Force debug level for this module

class RAGRetriever(BaseRetriever):
    """
    Standard retriever implementation.
    """

    def __init__(self):
        """Initialize the RAG retriever"""
        self._embedding_adapter = None
        logger.info("Initializing RAGRetriever")

    def set_embedding_adapter(self, adapter):
        """
        Set the embedding adapter to use for retrieval.

        This allows reusing an existing model instance to avoid reloading
        the embedding model multiple times, which is especially important
        for batch processing.

        Args:
            adapter: The embedding adapter instance (e.g., ColPaliAdapter)
        """
        self._embedding_adapter = adapter
        logger.info(f"RAGRetriever: Set embedding adapter with id {id(adapter)}")

    # Backward compatibility alias
    def set_rag_model(self, rag_model):
        """Legacy alias for set_embedding_adapter."""
        self.set_embedding_adapter(rag_model)

    @property
    def rag_model(self):
        """Legacy property for backward compatibility."""
        return self._embedding_adapter

    @rag_model.setter
    def rag_model(self, value):
        """Legacy setter for backward compatibility."""
        self._embedding_adapter = value

    def expand_query(self, query: str, chat_history: List[Dict[str, Any]], session_id: str) -> str:
        """
        Expand a user query by considering conversation history to resolve references.

        Args:
            query: The original user query
            chat_history: List of previous messages in the conversation
            session_id: The current session ID

        Returns:
            Expanded query with resolved references
        """
        return self._expand_query_with_context(query, chat_history, session_id)

    def _expand_query_with_context(self, query: str, chat_history: List[Dict[str, Any]], session_id: str) -> str:
        """
        Expands a user query by considering conversation history to resolve references.
        """
        logger.info("Using standard query expansion")

        contains_reference_terms = any(ref in query.lower() for ref in [
            "it", "this", "that", "they", "these", "those", "mentioned", "earlier",
            "above", "before", "previous", "you said", "tell me more", "what about",
            "can you elaborate", "more information", "more details"
        ])

        if not contains_reference_terms:
            logger.info(f"Query contains no reference terms, skipping expansion: {query[:50]}...")
            return query

        if not chat_history or len(chat_history) < 2:
            logger.info("No significant chat history to use for query expansion")
            return query

        try:
            recent_history = chat_history[-6:] if len(chat_history) > 6 else chat_history
            conversation_context = "Previous conversation:\n"
            for msg in recent_history:
                role = msg.get('role', '')
                content = msg.get('content', '')
                if isinstance(content, str):
                    content = content.replace('<p>', '\n').replace('</p>', '\n')
                    content = content.replace('<br>', '\n').replace('<br/>', '\n')
                    content = re.sub(r'<[^>]+>', '', content)
                if role == 'user': conversation_context += f"User: {content}\n"
                elif role == 'assistant': conversation_context += f"Assistant: {content}\n"

            try:
                expanded_query = self._ollama_expand_query(query, conversation_context)
                logger.info(f"Used Ollama for query expansion: '{query}' -> '{expanded_query}'")
                return expanded_query
            except Exception as e:
                logger.warning(f"Ollama query expansion failed: {e}, using original query")
                return query
        except Exception as e:
            logger.error(f"Error in query expansion: {e}")
            return query

    def _ollama_expand_query(self, query: str, conversation_context: str) -> str:
        """
        Use Ollama to expand a query based on conversation context with caching.
        """
        try:
            contains_reference = any(ref in query.lower() for ref in [
                "it", "this", "that", "these", "those", "they", "them", "their", "the document",
                "the image", "the picture", "the file", "the pdf", "the report", "the paper",
                "the chart", "the graph", "the table", "mentioned", "above", "earlier", "previous",
                "before", "last time", "you said", "you mentioned", "you talked about", "you showed",
                "tell me more", "what about", "more information", "more details", "can you elaborate",
                "explain further", "additionally", "what else", "expand on", "anything else",
                "other", "another", "additional"
            ])
            if not contains_reference: return query

            # Check cache first
            from src.utils.query_expansion_cache import get_query_expansion_cache
            cache = get_query_expansion_cache()
            
            cached_expansion = cache.get(query, conversation_context)
            if cached_expansion is not None:
                logger.info(f"Using cached query expansion: '{query}' -> '{cached_expansion}'")
                return cached_expansion

            logger.info(f"Expanding query with Ollama: {query}")
            instruction = ("Based on the conversation history below, rewrite the ambiguous user query "
                           "to include full context. Replace pronouns like 'it', 'this', 'that' with their specific referents. "
                           "Keep your response BRIEF and focused on query expansion only. DO NOT add any explanation - "
                           "just return the expanded query text directly. Make the query clear and specific.")
            prompt = f"{instruction}\n\n{conversation_context}\n\nAmbiguous query: {query}\n\nExpanded query:"

            try:
                model_to_use = 'llama3.2-vision' # Or choose another lightweight default
                response = ollama.chat(
                    model=model_to_use, messages=[{"role": "user", "content": prompt}],
                    options={"temperature": 0.1}
                )
                expanded_query = response['message']['content'].strip()
                if len(expanded_query) < len(query) or len(expanded_query) > 250: # Added length check
                    logger.warning(f"Ollama expansion unsatisfactory, using original. Expanded: {expanded_query}")
                    return query
                logger.info(f"Successfully expanded query: '{query}' -> '{expanded_query}'")
                
                # Cache the expanded query before returning
                cache.put(query, conversation_context, expanded_query)
                
                return expanded_query
            except Exception as ollama_error:
                logger.warning(f"Could not use Ollama for expansion: {ollama_error}, using original query.")
                return query
        except Exception as e:
            logger.error(f"Error in query expansion: {e}")
            return query

    def retrieve_documents(
        self,
        query: str,
        session_id: str,
        k: int = 3,
        selected_filenames: Optional[List[str]] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Retrieve documents using ColQwen2.5 adapter, handling selected filenames filtering.

        Special handling for Document Summary template:
        - When k=100 is passed (indicator from generator.py), this is a Document Summary request
        - Document Summary ALWAYS bypasses score-slope analysis to retrieve all pages
        - Document Summary uses the full k value regardless of user's retrieval count setting

        Args:
            query: The user query to use for retrieval
            session_id: The current session ID
            k: Maximum number of documents to retrieve
            selected_filenames: Optional list of filenames to filter results
            **kwargs: Additional parameters:
                - use_score_slope: Optional[bool] to override session setting
                - RAG: Legacy parameter for backward compatibility (deprecated)
        """
        use_score_slope = kwargs.get('use_score_slope', None)

        logger.info(f"RAG Retriever: Starting document retrieval for query: '{query}' for session {session_id}")
        start_time = time.time()  # Using module imported at the top

        # Use the stored embedding adapter; accept legacy RAG kwarg for backward compat
        legacy_rag = kwargs.get('RAG', None)
        RAG = self._embedding_adapter or legacy_rag
        if RAG is None:
            logger.error("No embedding adapter set. Call set_embedding_adapter() before retrieve_documents().")
            return []

        # Import necessary libraries directly in method to avoid scope issues
        import torch
        import gc
        import numpy as np

        retrieved_results_list = [] # Store dicts: {'path': ..., 'original_filename': ...}

        # Log memory at the start of the retrieval process
        memory_start = log_memory_usage("Start of retrieve_documents", session_id)

        try:
            logger.info(f"Retrieving documents for query: '{query}' with k={k}, selected_filenames={selected_filenames}")

            # Check the cache first before performing the retrieval operation
            # Import and initialize the cache outside the try block to ensure it's always available
            from src.models.search_results_cache import SearchResultsCache
            search_cache = SearchResultsCache()

            # Get model name for cache key (prevents cross-model cache collisions)
            _cache_model_name = RAG.model_name if hasattr(RAG, 'model_name') else None

            # Make sure the cache directory exists
            try:
                os.makedirs(search_cache.cache_dir, exist_ok=True)
            except Exception as e:
                logger.warning(f"Error creating cache directory: {e}, caching may not work properly")

            # Check for cached results with comprehensive error handling
            try:
                cached_results = search_cache.get(query, session_id, selected_filenames, model_name=_cache_model_name)
                if cached_results:
                    results, analysis = cached_results
                    logger.info(f"Using cached search results for query: '{query[:50]}...', found {len(results)} documents")

                    # Store token budget analysis in session data if available
                    if analysis:
                        try:
                            from src.services.session_manager.manager import load_session, save_session
                            session_data = load_session('sessions', session_id)
                            if session_data:
                                session_data['token_budget_analysis'] = analysis
                                save_session('sessions', session_id, session_data)
                                logger.info("Updated session with cached analysis data")
                        except Exception as e:
                            logger.warning(f"Failed to update session with cached analysis: {e}")

                    # Log memory at the end of cached retrieval
                    memory_end = log_memory_usage("End of retrieve_documents (from cache)", session_id)
                    log_memory_comparison(memory_start, memory_end, "Cached document retrieval process", session_id)

                    return results
            except Exception as cache_error:
                logger.warning(f"Error checking cache: {cache_error}, proceeding with retrieval")

            logger.info("No cached results found, proceeding with retrieval")

            # Get the session's distance metric and similarity threshold settings
            from src.services.session_manager.manager import load_session

            # Load session data to get settings
            try:
                session_data = load_session('sessions', session_id)
            except Exception as e:
                logger.warning(f"Could not load session data: {e}, using defaults")
                session_data = {}

            # Get distance metric with default fallback
            distance_metric = session_data.get('distance_metric', 'cosine') if session_data else 'cosine'

            # Get similarity threshold with default fallback
            similarity_threshold = session_data.get('similarity_threshold', 0.2) if session_data else 0.2

            # For ColPali models with LanceDB, we'll use the UI threshold directly
            # The LanceDB manager will convert it to the appropriate internal scale
            from src.utils.model_type_utils import is_colpali_adapter
            _is_colpali = is_colpali_adapter(RAG)
            if _is_colpali:
                # Use the UI threshold directly, LanceDB manager will convert it appropriately
                logger.info(f"Using UI threshold {similarity_threshold:.4f} for ColPali model {RAG.model_name} with LanceDB")
                logger.info(f"This will be converted to an appropriate internal threshold for ColBERT sum-similarity scores")

            logger.info(f"Using distance metric: {distance_metric}, similarity threshold: {similarity_threshold}")

            # --- LanceDB Retrieval ---
            # Determine the model name for LanceDB
            indexer_model = None
            if hasattr(RAG, 'model_name'):
                indexer_model = RAG.model_name
                logger.info(f"Using model-specific LanceDB table for {indexer_model}")

            if RAG is None:
                logger.error("No embedding model provided, cannot generate query embedding.")
                return []

            # 1. Generate Query Embedding
            # Use ColQwen2.5 adapter for all queries
            if hasattr(RAG, 'encode_text'):
                # Direct embedding model
                logger.debug("Generating query embedding with ColQwen2.5 adapter...")
                query_embedding = RAG.encode_text(query)
                # Move to GPU if available and not already there
                if isinstance(query_embedding, torch.Tensor) and torch.cuda.is_available() and query_embedding.device.type == 'cpu':
                    query_embedding = query_embedding.to('cuda')
                    logger.debug(f"Moved query embedding to GPU: {query_embedding.device}")
            else:
                logger.error("Provided model doesn't have encode_text method")
                return []

            if query_embedding is None or query_embedding.nelement() == 0:
                logger.error("Failed to generate query embedding.")
                return []

            # Debug the embedding shape and properties
            debug_embedding(query_embedding, "query")

            # Determine LanceDB table name from the model
            from src.utils.model_type_utils import get_lancedb_table_name
            use_lancedb = True  # Always use LanceDB for retrieval
            lancedb_model_name = get_lancedb_table_name(RAG.model_name if hasattr(RAG, 'model_name') else '')

            logger.info(f"Using ColPali model ({RAG.model_name}) with LanceDB table ({lancedb_model_name}) for retrieval")

            # Check if this is a Document Summary template (k>=100 is the indicator from generator.py)
            is_document_summary = (k >= 100)
            if is_document_summary:
                logger.info("Document Summary template detected - will retrieve all pages, bypassing score-slope")
            
            # Define n_results_query for LanceDB query
            # Check if score slope will be used (we already have session_data from earlier)
            use_score_slope_check = use_score_slope  # Use the parameter if provided
            if use_score_slope_check is None:
                # Use the session data we already loaded
                if session_data:
                    use_score_slope_check = session_data.get('use_score_slope', True)
                else:
                    use_score_slope_check = True  # Default to True
            
            # Document Summary always bypasses score-slope
            if is_document_summary:
                use_score_slope_check = False
                n_results_query = k  # Use the full k (now 500)
                logger.info(f"Document Summary: bypassing score-slope, retrieving {n_results_query} documents")
                # Use negative threshold to ensure we get ALL pages regardless of similarity
                similarity_threshold = -1.0  # Negative threshold to include all results
                logger.info(f"Document Summary: Using similarity threshold {similarity_threshold} to retrieve all pages")
            elif use_score_slope_check:
                # Fixed pool size for score slope analysis
                n_results_query = 10  # Standardized pool size
                logger.info(f"Score slope enabled - using fixed pool of {n_results_query} documents for analysis (requested k={k})")
            else:
                # Without score slope, retrieve exactly what was requested
                n_results_query = k
                logger.info(f"Score slope disabled - retrieving exactly {n_results_query} documents as requested")

            # Process the query embedding for LanceDB using ColQwen2.5
            if hasattr(RAG, 'process_embedding_for_lancedb'):
                # Use the adapter's custom processing method for ColQwen2.5
                logger.info("Using ColQwen2.5 custom embedding processing for LanceDB query")
                lancedb_query_embedding = RAG.process_embedding_for_lancedb(query_embedding, "query")
            else:
                # Fallback to standard processing method
                logger.info("Using standard embedding processing for LanceDB query")
                lancedb_query_embedding = process_embedding_for_lancedb(query_embedding, "query")

            # Query LanceDB - request slightly more to account for potential filtering
            # When score slope is enabled, we already have our pool size
            # When disabled, get a few extra for safety
            lancedb_k = n_results_query + 5 if use_score_slope_check else n_results_query + 2
            logger.info(f"Querying LanceDB for {lancedb_k} documents")

            # Create filter dict for selected filenames
            filter_dict = None
            if selected_filenames:
                filter_dict = {'filename': selected_filenames}

            # Query LanceDB table with robust error handling and retry
            max_retries = 2
            retry_count = 0

            while retry_count <= max_retries:
                try:
                    # Log memory before LanceDB query
                    memory_before_query = log_memory_usage("Before LanceDB query", session_id)

                    # Log detailed information about the query
                    logger.info(f"Attempting LanceDB query (try {retry_count+1}/{max_retries+1}) for session {session_id}")
                    logger.info(f"Query params: model={lancedb_model_name}, k={lancedb_k}, filter={filter_dict}, threshold={similarity_threshold}")
                    
                    # CRITICAL: Verify session isolation
                    logger.debug(f"[SESSION ISOLATION CHECK] Querying LanceDB for session: {session_id}")
                    if filter_dict and 'filename' in filter_dict:
                        logger.debug(f"[SESSION ISOLATION CHECK] Filtering for documents: {filter_dict['filename']}")

                    # Query LanceDB
                    lancedb_results, lancedb_scores, lancedb_ids = query_lancedb(
                        session_id,
                        lancedb_model_name,
                        lancedb_query_embedding,
                        k=lancedb_k,
                        filter_dict=filter_dict,
                        similarity_threshold=similarity_threshold
                    )

                    # Log memory after LanceDB query
                    memory_after_query = log_memory_usage("After LanceDB query", session_id)
                    log_memory_comparison(memory_before_query, memory_after_query, "LanceDB query operation", session_id)

                    # Check if we got any results
                    if not lancedb_results:
                        logger.warning(f"LanceDB query returned no results for {lancedb_model_name}")

                        # If no results and we haven't exhausted retries, try rebuilding the index
                        if retry_count < max_retries:
                            logger.info(f"Attempting to rebuild LanceDB index for retry {retry_count+1}")
                            from src.models.vector_stores.lancedb_manager import rebuild_lancedb_index
                            rebuild_success = rebuild_lancedb_index(session_id, lancedb_model_name)
                            if rebuild_success:
                                logger.info(f"Successfully rebuilt LanceDB index for session {session_id}, retrying query")
                                retry_count += 1
                                continue  # Retry the query

                    # If we got results or can't rebuild, break out of the retry loop
                    break

                except Exception as e:
                    logger.error(f"Error querying LanceDB (try {retry_count+1}/{max_retries+1}): {e}")
                    logger.error(f"Exception details: {traceback.format_exc()}")

                    # If we haven't exhausted retries, try again
                    if retry_count < max_retries:
                        logger.info(f"Retrying LanceDB query (attempt {retry_count+2}/{max_retries+1})...")
                        retry_count += 1

                        # Sleep briefly to allow resources to be freed
                        time.sleep(0.5)

                        # Force a garbage collection cycle
                        gc.collect()

                        continue  # Retry the query
                    else:
                        # Log the error but continue with direct retrieval if needed
                        logger.warning("LanceDB query failed after all retries, will try direct retrieval if filenames are selected")
                        use_lancedb = False
                        break

            # Only process LanceDB results if we're using LanceDB and got results
            if use_lancedb and 'lancedb_results' in locals() and lancedb_results:
                logger.info(f"LanceDB query returned {len(lancedb_results)} results")

                # Process LanceDB results
                session_images_folder = os.path.join('static', 'images', session_id)
                os.makedirs(session_images_folder, exist_ok=True)

                for i, metadata in enumerate(lancedb_results):
                    score = lancedb_scores[i]
                    original_filename = metadata.get('filename')
                    page_num = metadata.get('page_num')

                    if not original_filename or page_num is None:
                        logger.warning(f"LanceDB result missing filename or page_num in metadata: {metadata}, skipping.")
                        continue

                    # Filter by selected_filenames if provided
                    if selected_filenames and original_filename not in selected_filenames:
                        continue

                    # Reconstruct path to original uploaded file
                    upload_folder = os.path.join('uploaded_documents', session_id)
                    original_file_path = os.path.join(upload_folder, original_filename)

                    if not os.path.exists(original_file_path):
                        logger.warning(f"Original file not found at {original_file_path} for LanceDB result, skipping.")
                        continue

                    # Generate unique filename for static image
                    lancedb_id = lancedb_ids[i]
                    image_filename = f"retrieved_{lancedb_id.replace('_', 'p')}.png"
                    image_path_abs = os.path.join(session_images_folder, image_filename)
                    relative_path = self.normalize_image_path(image_path_abs, session_id)

                    # Generate and save the static image if it doesn't exist
                    if not os.path.exists(image_path_abs):
                        try:
                            if original_filename.lower().endswith('.pdf') and convert_from_path:
                                pdf_images = convert_from_path(original_file_path, first_page=page_num, last_page=page_num)
                                if pdf_images:
                                    pdf_images[0].save(image_path_abs, 'PNG')
                                else:
                                    logger.warning(f"pdf2image failed to extract page {page_num} from {original_filename}")
                                    continue
                            elif metadata.get('file_type') == 'image':
                                shutil.copy2(original_file_path, image_path_abs)
                            else:
                                logger.warning(f"Cannot generate static image for unsupported file type: {original_filename}")
                                continue
                            logger.debug(f"Saved retrieved image: {image_path_abs}")
                        except Exception as img_ex:
                            logger.error(f"Failed to save image {image_path_abs} from {original_file_path}: {img_ex}")
                            continue

                    # Add to results list
                    retrieved_results_list.append({
                        'path': relative_path,
                        'original_filename': original_filename,
                        'score': score,
                        'page_num': page_num
                    })

                # Sort by score
                retrieved_results_list.sort(key=lambda x: x['score'], reverse=True)

                # Get session data for score-slope analysis
                # Document Summary always bypasses score-slope
                if is_document_summary:
                    use_score_slope = False
                    logger.info("Document Summary template - forcing score-slope OFF")
                elif use_score_slope is not None:
                    logger.info(f"Using provided use_score_slope={use_score_slope} parameter")
                else:
                    use_score_slope = session_data.get('use_score_slope', True)  # Default to True
                    logger.info(f"Using session setting use_score_slope={use_score_slope}")

                # Use consistent threshold across the codebase
                # Standardized to 0.65 to match other defaults in the system
                rel_drop_threshold = session_data.get('rel_drop_threshold', 0.65) # Default threshold
                logger.info(f"Using rel_drop_threshold {rel_drop_threshold:.4f} for retrieval (default: 0.65)")

                abs_score_threshold = session_data.get('abs_score_threshold', 0.25)  # Default to 0.25
                generation_model = session_data.get('generation_model', 'ollama-phi4')

                # Log detailed similarity scores for testing (limit log output to first 10 for readability)
                log_limit = min(10, len(retrieved_results_list))
                score_details = [f"{item['original_filename']} (page {item.get('page_num', 'N/A')}):{item['score']:.4f}" for item in retrieved_results_list[:log_limit]]
                logger.info(f"LANCEDB RESULTS: Retrieved {len(retrieved_results_list)} documents for query: '{query}' with scores (showing first {log_limit}): {', '.join(score_details)}")

                # Defer memory cleanup after LanceDB retrieval
                # This prevents memory pressure during response generation
                def deferred_lancedb_cleanup():
                    """Perform LanceDB cleanup after response generation."""
                    try:
                        # Log memory before cleanup
                        memory_before_cleanup = log_memory_usage("Before deferred LanceDB cleanup", session_id)

                        # Run garbage collection multiple times for better cleanup
                        for _ in range(3):
                            gc.collect()

                        # Empty CUDA cache if available
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                        # Perform aggressive cleanup to prevent memory buildup
                        from src.utils.memory_management import cleanup_lancedb_resources
                        cleanup_info = cleanup_lancedb_resources(session_id, lancedb_model_name)
                        logger.info(f"[MEMORY] Performed deferred LanceDB cleanup: {cleanup_info}")

                        # Additional cleanup to ensure LanceDB connections are properly closed
                        try:
                            from src.models.vector_stores.lancedb_manager import destroy_lancedb_resources, cleanup_old_connections

                            # Force close the connection for this session
                            destroy_result = destroy_lancedb_resources(session_id, preserve_active_sessions=True)
                            logger.info(f"[MEMORY] Deferred LanceDB connection closure: {destroy_result}")

                            # Clean up old connections
                            cleaned_sessions = cleanup_old_connections(max_idle_time=120)
                            if cleaned_sessions:
                                logger.info(f"[MEMORY] Cleaned up {len(cleaned_sessions)} idle LanceDB connections")
                        except Exception as destroy_e:
                            logger.warning(f"[MEMORY] Error during deferred LanceDB connection closure: {destroy_e}")

                        # Log memory after cleanup
                        memory_after_cleanup = log_memory_usage("After deferred LanceDB cleanup", session_id)
                        memory_diff = log_memory_comparison(
                            memory_before_cleanup,
                            memory_after_cleanup,
                            "Deferred LanceDB GC",
                            session_id
                        )

                        # Force system to release memory if possible
                        try:
                            import ctypes
                            libc = ctypes.CDLL('libc.so.6')
                            libc.malloc_trim(0)
                            logger.info("[MEMORY] Called malloc_trim to release memory to the system")
                        except Exception as trim_e:
                            logger.warning(f"[MEMORY] Could not call malloc_trim: {trim_e}")

                        logger.info(f"[MEMORY] Completed deferred cleanup. Freed: RAM {memory_diff.get('ram_diff', 0):.2f} GB")

                    except Exception as e:
                        logger.error(f"[MEMORY] Error during deferred LanceDB cleanup: {e}", exc_info=True)
                
                # Schedule the cleanup to run after response generation
                defer_cleanup(deferred_lancedb_cleanup)
                logger.info("[MEMORY] Scheduled LanceDB cleanup for after response generation")

                # Get the user's actual desired retrieval count from session data
                # This is the original value before any score-slope multiplication
                user_desired_count = k  # Default to k
                if session_data and 'retrieval_count' in session_data:
                    user_desired_count = int(session_data.get('retrieval_count', k))
                    logger.info(f"Using user's desired retrieval count: {user_desired_count} (k={k})")
                
                # Always run analysis to get visualization data, but only apply filtering if enabled
                # Pass the user's desired count as minimum, not the inflated k value
                # Pass the embedding model name to allow model-specific adjustments
                # Add max_results parameter to limit the number of results even if all pass thresholds
                max_results = 20  # Reasonable default to prevent too many results
                embedding_model_name = RAG.model_name if hasattr(RAG, 'model_name') else None
                filtered_results, analysis = analyze_score_distribution(
                    retrieved_results_list,
                    rel_drop_threshold=rel_drop_threshold,
                    abs_score_threshold=abs_score_threshold,
                    min_results=1,  # Adaptive mode: let score-slope decide, minimum 1 result
                    max_results=max_results,
                    model_name=embedding_model_name
                )

                # Store analysis in session data for visualization
                if 'score_analysis' not in session_data:
                    session_data['score_analysis'] = {}

                # Store the latest analysis with timestamp
                # Using time module imported at the top level
                session_data['score_analysis'] = {
                    'timestamp': time.time(),
                    'query': query[:100],  # Store truncated query for reference
                    'analysis': analysis,
                    'scores': [item['score'] for item in retrieved_results_list[:30]],  # Store top 30 scores for visualization
                    'filenames': [f"{item['original_filename']} (p{item.get('page_num', 'N/A')})" for item in retrieved_results_list[:30]]
                }

                # Save updated session data
                from src.services.session_manager.manager import save_session
                save_session('sessions', session_id, session_data)

                # Apply score-slope analysis if enabled
                if use_score_slope:
                    logger.info(f"Applying score-slope analysis with rel_drop_threshold={rel_drop_threshold}, abs_score_threshold={abs_score_threshold}")

                    # Log analysis results
                    logger.info(f"Score-slope analysis: Original count: {analysis['original_count']}, Filtered count: {analysis['filtered_count']}, Cutoff reason: {analysis['cutoff_reason']}")

                    # Cap results by model's image capacity
                    # Each page image costs ~1500 tokens; reserve 2000 for query+response
                    from src.utils.token_utils import get_model_context_window
                    ctx = get_model_context_window(generation_model, session_data)
                    max_images = max(1, (ctx - 2000) // 1500)
                    if len(filtered_results) > max_images:
                        logger.info(f"Image cap: {len(filtered_results)} pages → {max_images} (model context {ctx} tokens, ~1500 tokens/image)")
                        filtered_results = filtered_results[:max_images]

                    # Get OCR cache for token counting
                    ocr_cache = get_ocr_cache_for_session(session_id)

                    # Apply token budget filtering to the score-slope filtered results
                    budget_filtered_results, budget_analysis = apply_token_budget_filter(
                        filtered_results,
                        generation_model,
                        session_data,
                        ocr_cache
                    )

                    # Log final results after token budget filtering
                    final_score_details = [f"{item['original_filename']} (page {item.get('page_num', 'N/A')}):{item['score']:.4f}" for item in budget_filtered_results[:log_limit]]
                    logger.info(f"FINAL RESULTS (after score-slope and token budget): {len(budget_filtered_results)} documents with scores: {', '.join(final_score_details)}")

                    # Store token budget analysis in session data
                    session_data['token_budget_analysis'] = budget_analysis

                    # Save updated session data
                    save_session('sessions', session_id, session_data)

                    # Cache the results for future use with error handling
                    try:
                        # We already imported SearchResultsCache at the top
                        # Only create a new instance if we don't already have one
                        if 'search_cache' not in locals() or search_cache is None:
                            from src.models.search_results_cache import SearchResultsCache
                            search_cache = SearchResultsCache()

                        # Ensure cache directory exists
                        os.makedirs(search_cache.cache_dir, exist_ok=True)

                        # Store results in cache
                        cache_success = search_cache.put(query, session_id, budget_filtered_results, budget_analysis, selected_filenames, model_name=_cache_model_name)
                        if cache_success:
                            logger.info(f"Successfully cached search results for future use")
                        else:
                            logger.warning(f"Failed to cache search results")
                    except Exception as cache_error:
                        logger.warning(f"Error caching search results: {cache_error}")

                    return budget_filtered_results
                else:
                    # If score-slope is disabled, limit to the user's retrieval count setting
                    logger.info(f"Score-slope analysis disabled - limiting to user's retrieval count setting")

                    # Check if this is a Document Summary template that should bypass the limit
                    # If k was specifically set to >=100 (Document Summary override), use that instead
                    if k >= 100:  # This indicates Document Summary override from generator.py
                        logger.info(f"Document Summary detected (k=100) - using full k={k} instead of user's retrieval count")
                        limited_results = retrieved_results_list[:k]
                        user_retrieval_count = k  # Set for logging purposes
                    elif kwargs.get('_skip_result_cap'):
                        # Called by a parent hybrid retriever — respect the k passed, not session settings
                        limited_results = retrieved_results_list[:k]
                        user_retrieval_count = k
                        logger.info(f"Hybrid child mode: returning up to {k} results (skipping session retrieval_count cap)")
                    else:
                        # For normal templates, use the user's retrieval count setting
                        user_retrieval_count = k  # Default to the provided k value
                        if session_data and 'retrieval_count' in session_data:
                            user_retrieval_count = int(session_data.get('retrieval_count', k))
                            logger.info(f"Using user's retrieval count setting: {user_retrieval_count}")

                        # Limit the results to the user's retrieval count setting
                        limited_results = retrieved_results_list[:user_retrieval_count]

                    logger.info(f"Limited results to {len(limited_results)} documents (from {len(retrieved_results_list)})")

                    # Get OCR cache for token counting
                    ocr_cache = get_ocr_cache_for_session(session_id)

                    # Apply token budget filtering to the limited results
                    budget_filtered_results, budget_analysis = apply_token_budget_filter(
                        limited_results,  # Use limited results based on user's retrieval count
                        generation_model,
                        session_data,
                        ocr_cache
                    )

                    # Log final results after token budget filtering
                    final_score_details = [f"{item['original_filename']} (page {item.get('page_num', 'N/A')}):{item['score']:.4f}" for item in budget_filtered_results[:log_limit]]
                    logger.info(f"FINAL RESULTS (fixed count mode, limited to {user_retrieval_count}): {len(budget_filtered_results)} documents with scores: {', '.join(final_score_details)}")

                    # Store token budget analysis in session data
                    session_data['token_budget_analysis'] = budget_analysis

                    # Save updated session data
                    save_session('sessions', session_id, session_data)

                    return budget_filtered_results

            # If LanceDB returned no results, log a warning
            if not retrieved_results_list:
                logger.warning("LanceDB query returned no results, will try direct retrieval if filenames are selected")

            # Create the session images folder for storing retrieved images
            session_images_folder = os.path.join('static', 'images', session_id)
            os.makedirs(session_images_folder, exist_ok=True)

            # If RAG search yielded no results but files were selected, try direct retrieval
            if not retrieved_results_list and selected_filenames:
                logger.info(f"RAG search yielded no results for selected files. Trying direct retrieval for: {selected_filenames}")
                logger.info(f"Direct retrieval will use k={k} to retrieve pages")

                # Set a flag to indicate that we're using direct retrieval with artificial scores
                using_direct_retrieval = True

                session_folder = os.path.join('uploaded_documents', session_id)

                for filename in selected_filenames:
                    # We no longer stop at k results - we'll use score-slope and token budget filtering instead
                    # The old code was:
                    # if len(retrieved_results_list) >= k: break # Respect k limit
                    logger.debug(f"Attempting direct retrieval for: {filename}")
                    file_path = os.path.join(session_folder, filename)
                    relative_path = None

                    if os.path.exists(file_path):
                        image_filename_base = f"direct_{os.path.basename(filename)}"
                        # Handle PDF
                        if filename.lower().endswith('.pdf') and convert_from_path:
                            try:
                                # Use the k value passed to the function (which may be overridden for Document Summary)
                                user_retrieval_count = k  # Use the k value passed to the function
                                logger.info(f"Direct retrieval: Using k={user_retrieval_count} pages")

                                # For PDFs, retrieve multiple pages based on the user's retrieval count setting
                                for page_num in range(1, user_retrieval_count + 1):
                                    image_filename = f"{image_filename_base}_p{page_num}.png"
                                    image_path_abs = os.path.join(session_images_folder, image_filename)
                                    if not os.path.exists(image_path_abs):
                                        try:
                                            pdf_images = convert_from_path(file_path, first_page=page_num, last_page=page_num)
                                            if pdf_images:
                                                pdf_images[0].save(image_path_abs, 'PNG')
                                                logger.info(f"Successfully converted page {page_num} of {filename} to {image_path_abs}")
                                            else:
                                                logger.warning(f"No images found for page {page_num} of {filename}")
                                                break  # Stop if we've reached the end of the PDF
                                        except Exception as e:
                                            logger.warning(f"Error converting page {page_num} of {filename}: {e}")
                                            break  # Stop if we encounter an error
                                    relative_path = self.normalize_image_path(image_path_abs, session_id)

                                    # Add to results if path generated
                                    if relative_path:
                                        # Check for duplicates based on relative path
                                        if not any(d['path'] == relative_path for d in retrieved_results_list):
                                            retrieved_results_list.append({
                                                'path': relative_path,
                                                'original_filename': filename,
                                                'page_num': page_num,
                                                'score': 1.0 - (page_num - 1) * 0.1  # Assign decreasing scores to pages
                                            })
                                            logger.info(f"Added directly retrieved file: {relative_path} (page {page_num})")

                                # Log the results of direct retrieval
                                direct_paths = [item['path'] for item in retrieved_results_list if item.get('original_filename') == filename]
                                logger.info(f"Direct retrieval for {filename} added {len(direct_paths)} pages: {direct_paths}")
                                logger.warning(f"IMPORTANT: Direct retrieval is using artificial scores (1.0, 0.9, 0.8, etc.) that are NOT based on embedding similarity. Score-slope settings will have limited effect.")

                                # Skip the rest of the loop since we've already added the pages
                                continue
                            except Exception as e: logger.error(f"Error converting PDF {filename}: {e}"); continue
                        # Handle Image
                        elif imghdr.what(file_path):
                            try:
                                ext = os.path.splitext(filename)[1] or '.png'
                                image_filename = f"{image_filename_base}{ext}"
                                image_path_abs = os.path.join(session_images_folder, image_filename)
                                if not os.path.exists(image_path_abs):
                                    shutil.copy2(file_path, image_path_abs)
                                relative_path = self.normalize_image_path(image_path_abs, session_id)
                            except Exception as e: logger.error(f"Error copying image {filename}: {e}"); continue
                        else: logger.warning(f"Direct retrieval: File is not PDF or image: {filename}"); continue

                        # Add to results if path generated
                        if relative_path:
                             # Check for duplicates based on relative path
                             if not any(d['path'] == relative_path for d in retrieved_results_list):
                                  retrieved_results_list.append({
                                       'path': relative_path,
                                       'original_filename': filename,
                                       'score': 0.0 # Indicate direct retrieval
                                  })
                                  logger.info(f"Added directly retrieved file: {relative_path}")
                             # else: logger.debug(f"Skipping duplicate direct path: {relative_path}") # Less verbose

                    else: logger.warning(f"Direct retrieval: File not found: {file_path}")
            else:
                # If we have results from LanceDB, set the flag to indicate we're not using direct retrieval
                using_direct_retrieval = False


            # Sort final list by score if scores are meaningful (i.e., primarily from RAG search)
            if any(item['score'] > 0.0 for item in retrieved_results_list):
                # Sort by score (higher is better)
                retrieved_results_list.sort(key=lambda x: x['score'], reverse=True)

                # Log detailed similarity scores for testing (limit log output to first 10 for readability)
                log_limit = min(10, k)
                score_details = [f"{item['original_filename']} (page {item.get('page_num', 'N/A')}):{item['score']:.4f}" for item in retrieved_results_list[:log_limit]]
                logger.info(f"SORTED RESULTS: Retrieved {len(retrieved_results_list)} documents for query: '{query}' with scores (showing first {log_limit}): {', '.join(score_details)}")

                # Check if we should use score-slope analysis
                # Document Summary always bypasses score-slope
                if is_document_summary:
                    use_score_slope = False
                    logger.info("Document Summary template - forcing score-slope OFF for direct retrieval")
                elif use_score_slope is not None:
                    logger.info(f"Using provided use_score_slope={use_score_slope} parameter")
                else:
                    use_score_slope = session_data.get('use_score_slope', True)  # Default to True
                    logger.info(f"Using session setting use_score_slope={use_score_slope}")

                # If we're using direct retrieval with artificial scores, we need to handle score-slope differently
                if 'using_direct_retrieval' in locals() and using_direct_retrieval:
                    logger.warning(f"Using direct retrieval with artificial scores. Score-slope analysis will be limited.")

                    # For direct retrieval, we'll use the user's retrieval count setting regardless of score-slope
                    # But we'll still run the analysis for visualization purposes
                    if use_score_slope:
                        logger.info(f"Score-slope is enabled, but using direct retrieval with artificial scores. Will use user's retrieval count setting instead.")
                        # Force using the user's retrieval count setting
                        use_score_slope = False

                    # Initialize rel_drop_threshold here to ensure it's defined even if we skip the model-specific code below
                    rel_drop_threshold = session_data.get('rel_drop_threshold', 0.45)  # Default threshold

                # Get score distribution data for visualization (even if not using score-slope)
                # Get thresholds from session data
                rel_drop_threshold = session_data.get('rel_drop_threshold', 0.65)  # Default to 0.65
                abs_score_threshold = session_data.get('abs_score_threshold', 0.25)  # Default to 0.25

                # Log the thresholds being used
                if _is_colpali:
                    logger.info(f"Using rel_drop_threshold {rel_drop_threshold:.4f} for ColPali model")
                    logger.info(f"Using abs_score_threshold {abs_score_threshold:.4f} for ColPali model")

                # Always run analysis to get visualization data, but only apply filtering if enabled
                # Pass k as the minimum number of results to ensure we get at least that many
                # Get the current model name from session data
                generation_model = session_data.get('generation_model', 'ollama-phi4')

                # Pass the embedding model name to allow model-specific adjustments
                # Add max_results parameter to limit the number of results even if all pass thresholds
                max_results = 20  # Reasonable default to prevent too many results
                embedding_model_name = RAG.model_name if hasattr(RAG, 'model_name') else None
                filtered_results, analysis = analyze_score_distribution(
                    retrieved_results_list,
                    rel_drop_threshold=rel_drop_threshold,
                    abs_score_threshold=abs_score_threshold,
                    min_results=1,  # Adaptive mode: let score-slope decide
                    max_results=max_results,
                    model_name=embedding_model_name
                )

                # Store analysis in session data for visualization
                if 'score_analysis' not in session_data:
                    session_data['score_analysis'] = {}

                # Store the latest analysis with timestamp
                # Using time module imported at the top level
                session_data['score_analysis'] = {
                    'timestamp': time.time(),
                    'query': query[:100],  # Store truncated query for reference
                    'analysis': analysis,
                    'scores': [item['score'] for item in retrieved_results_list[:30]],  # Store top 30 scores for visualization
                    'filenames': [f"{item['original_filename']} (p{item.get('page_num', 'N/A')})" for item in retrieved_results_list[:30]]
                }

                # Save updated session data
                from src.services.session_manager.manager import save_session
                save_session('sessions', session_id, session_data)

                # Get the current model name from session data
                generation_model = session_data.get('generation_model', 'ollama-phi4')

                # Apply score-slope analysis if enabled
                if use_score_slope:
                    logger.info(f"Applying score-slope analysis with rel_drop_threshold={rel_drop_threshold}, abs_score_threshold={abs_score_threshold}")

                    # Log analysis results
                    logger.info(f"Score-slope analysis: Original count: {analysis['original_count']}, Filtered count: {analysis['filtered_count']}, Cutoff reason: {analysis['cutoff_reason']}")

                    # Get OCR cache for token counting
                    ocr_cache = get_ocr_cache_for_session(session_id)

                    # Apply token budget filtering to the score-slope filtered results
                    # This is our primary filtering mechanism - relevance first, then token budget
                    budget_filtered_results, budget_analysis = apply_token_budget_filter(
                        filtered_results,  # Use the results already filtered by score-slope
                        generation_model,
                        session_data,
                        ocr_cache
                    )

                    # Log final results after token budget filtering
                    final_score_details = [f"{item['original_filename']} (page {item.get('page_num', 'N/A')}):{item['score']:.4f}" for item in budget_filtered_results[:log_limit]]
                    logger.info(f"FINAL RESULTS (adaptive mode, score-slope cutoff at {len(filtered_results)}): {len(budget_filtered_results)} documents with scores: {', '.join(final_score_details)}")

                    # Store token budget analysis in session data
                    session_data['token_budget_analysis'] = budget_analysis

                    # Save updated session data
                    from src.services.session_manager.manager import save_session
                    save_session('sessions', session_id, session_data)

                    # Cache the results for future use with error handling
                    try:
                        # We already imported SearchResultsCache at the top
                        # Only create a new instance if we don't already have one
                        if 'search_cache' not in locals() or search_cache is None:
                            from src.models.search_results_cache import SearchResultsCache
                            search_cache = SearchResultsCache()

                        # Ensure cache directory exists
                        os.makedirs(search_cache.cache_dir, exist_ok=True)

                        # Store results in cache
                        cache_success = search_cache.put(query, session_id, budget_filtered_results, budget_analysis, selected_filenames, model_name=_cache_model_name)
                        if cache_success:
                            logger.info(f"Successfully cached search results for future use")
                        else:
                            logger.warning(f"Failed to cache search results")
                    except Exception as cache_error:
                        logger.warning(f"Error caching search results: {cache_error}")

                    # For batch processing, return the analysis data along with the results
                    # This allows the batch processor to understand how score-slope affected the results
                    # Create a combined analysis dictionary with both score-slope and token budget analysis
                    combined_analysis = {
                        **analysis,  # Include score-slope analysis
                        **budget_analysis,  # Include token budget analysis
                        'scores': [item['score'] for item in retrieved_results_list[:30]]  # Include raw scores for debugging
                    }

                    # Check if this is being called from the batch processor
                    # We can detect this by checking the query string for the batch processing prefix
                    if query.startswith('[Processing document:'):
                        logger.info(f"Detected batch processing query, returning results with analysis data")
                        # Return a tuple of (results, ocr_results, analysis) for batch processing
                        return (budget_filtered_results, None, combined_analysis)
                    else:
                        # For regular queries, just return the results
                        return budget_filtered_results
                else:
                    # If score-slope is disabled, limit to the user's retrieval count setting
                    logger.info(f"Score-slope analysis disabled - limiting to user's retrieval count setting")

                    # Check if this is a Document Summary template that should bypass the limit
                    # If k was specifically set to >=100 (Document Summary override), use that instead
                    if k >= 100:  # This indicates Document Summary override from generator.py
                        logger.info(f"Document Summary detected (k=100) - using full k={k} instead of user's retrieval count")
                        limited_results = retrieved_results_list[:k]
                    else:
                        # For normal templates, use the user's retrieval count setting
                        user_retrieval_count = k  # Default to the provided k value
                        if session_data and 'retrieval_count' in session_data:
                            user_retrieval_count = int(session_data.get('retrieval_count', k))
                            logger.info(f"Using user's retrieval count setting: {user_retrieval_count}")
                        
                        # Limit the results to the user's retrieval count setting
                        limited_results = retrieved_results_list[:user_retrieval_count]
                    
                    logger.info(f"Limited results to {len(limited_results)} documents (from {len(retrieved_results_list)})")

                    # Get OCR cache for token counting
                    ocr_cache = get_ocr_cache_for_session(session_id)

                    # Apply token budget filtering to the limited results
                    budget_filtered_results, budget_analysis = apply_token_budget_filter(
                        limited_results,  # Use limited results based on user's retrieval count
                        generation_model,
                        session_data,
                        ocr_cache
                    )

                    # Log final results after token budget filtering
                    final_score_details = [f"{item['original_filename']} (page {item.get('page_num', 'N/A')}):{item['score']:.4f}" for item in budget_filtered_results[:log_limit]]
                    logger.info(f"FINAL RESULTS (fixed count mode, limited to {user_retrieval_count}): {len(budget_filtered_results)} documents with scores: {', '.join(final_score_details)}")

                    # Store token budget analysis in session data
                    session_data['token_budget_analysis'] = budget_analysis

                    # Save updated session data
                    from src.services.session_manager.manager import save_session
                    save_session('sessions', session_id, session_data)

                    # Cache the results for future use with error handling
                    try:
                        # We already imported SearchResultsCache at the top
                        # Only create a new instance if we don't already have one
                        if 'search_cache' not in locals() or search_cache is None:
                            from src.models.search_results_cache import SearchResultsCache
                            search_cache = SearchResultsCache()

                        # Ensure cache directory exists
                        os.makedirs(search_cache.cache_dir, exist_ok=True)

                        # Store results in cache
                        cache_success = search_cache.put(query, session_id, budget_filtered_results, budget_analysis, selected_filenames, model_name=_cache_model_name)
                        if cache_success:
                            logger.info(f"Successfully cached search results for future use")
                        else:
                            logger.warning(f"Failed to cache search results")
                    except Exception as cache_error:
                        logger.warning(f"Error caching search results: {cache_error}")

                    # For batch processing, return the analysis data along with the results
                    # Create a combined analysis dictionary with both score-slope and token budget analysis
                    combined_analysis = {
                        'original_count': len(retrieved_results_list),
                        'filtered_count': len(budget_filtered_results),
                        'cutoff_reason': 'fixed_count',
                        **budget_analysis,  # Include token budget analysis
                        'scores': [item['score'] for item in retrieved_results_list[:30]]  # Include raw scores for debugging
                    }

                    # Check if this is being called from the batch processor
                    if query.startswith('[Processing document:'):
                        logger.info(f"Detected batch processing query, returning results with analysis data")
                        # Return a tuple of (results, ocr_results, analysis) for batch processing
                        return (budget_filtered_results, None, combined_analysis)
                    else:
                        # For regular queries, just return the results
                        return budget_filtered_results
            else:
                logger.info(f"Total {len(retrieved_results_list)} documents retrieved without meaningful scores.")

                # Get the current model name from session data
                generation_model = session_data.get('generation_model', 'ollama-phi4')

                # Get OCR cache for token counting
                ocr_cache = get_ocr_cache_for_session(session_id)

                # Even for documents without meaningful scores, apply token budget filtering
                budget_filtered_results, budget_analysis = apply_token_budget_filter(
                    retrieved_results_list,
                    generation_model,
                    session_data,
                    ocr_cache
                )

                # Log final results after token budget filtering
                if 'using_direct_retrieval' in locals() and using_direct_retrieval:
                    logger.info(f"FINAL RESULTS (direct retrieval with artificial scores): {len(budget_filtered_results)} documents")
                else:
                    logger.info(f"FINAL RESULTS (token budget for non-scored docs): {len(budget_filtered_results)} documents")

                # Store token budget analysis in session data
                session_data['token_budget_analysis'] = budget_analysis

                # Save updated session data
                from src.services.session_manager.manager import save_session
                save_session('sessions', session_id, session_data)

                # Log memory at the end of successful retrieval
                memory_end = log_memory_usage("End of retrieve_documents (success)", session_id)
                log_memory_comparison(memory_start, memory_end, "Full document retrieval process", session_id)

                # Cache the results for future use with error handling
                try:
                    # We already imported SearchResultsCache at the top
                    # Only create a new instance if we don't already have one
                    if 'search_cache' not in locals() or search_cache is None:
                        from src.models.search_results_cache import SearchResultsCache
                        search_cache = SearchResultsCache()

                    # Ensure cache directory exists
                    os.makedirs(search_cache.cache_dir, exist_ok=True)

                    # Store results in cache
                    cache_success = search_cache.put(query, session_id, budget_filtered_results, budget_analysis, selected_filenames, model_name=_cache_model_name)
                    if cache_success:
                        logger.info(f"Successfully cached search results for future use")
                    else:
                        logger.warning(f"Failed to cache search results")
                except Exception as cache_error:
                    logger.warning(f"Error caching search results: {cache_error}")

                # For batch processing, return the analysis data along with the results
                # Create a combined analysis dictionary with token budget analysis
                combined_analysis = {
                    'original_count': len(retrieved_results_list),
                    'filtered_count': len(budget_filtered_results),
                    'cutoff_reason': 'token_budget',
                    **budget_analysis,  # Include token budget analysis
                    'scores': [item.get('score', 0.0) for item in retrieved_results_list[:30]]  # Include raw scores for debugging
                }

                # Check if this is being called from the batch processor
                if query.startswith('[Processing document:'):
                    logger.info(f"Detected batch processing query, returning results with analysis data")
                    # Return a tuple of (results, ocr_results, analysis) for batch processing
                    return (budget_filtered_results, None, combined_analysis)
                else:
                    # For regular queries, just return the results
                    return budget_filtered_results

        except Exception as e:
            logger.error(f"Error retrieving documents: {e}", exc_info=True)

            # Log memory even on error
            memory_error = log_memory_usage("End of retrieve_documents (error)", session_id)
            log_memory_comparison(memory_start, memory_error, "Document retrieval (failed)", session_id)

            return []

        finally:
            # Log completion timing for debugging
            end_time = time.time()
            elapsed_time = end_time - start_time
            logger.info(f"RAG Retriever: Document retrieval completed in {elapsed_time:.2f} seconds")

# Create a singleton instance for convenience
rag_retriever = RAGRetriever()