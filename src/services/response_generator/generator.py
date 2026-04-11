"""
Response generation functionality.

This module contains functions for generating streaming responses to user queries
using the appropriate model and context.
"""
import logging
import os
import time
import gc
import torch
import json

from src.models.rag_retriever import rag_retriever
from src.utils.memory_management import cleanup_after_response
from src.models.responder import generate_response as model_generate_response
from src.models.responder import generate_streaming_response as model_generate_streaming_response

logger = logging.getLogger(__name__)

def generate_streaming_response(query, session_id, generation_model, session_data, is_rag_mode, user_id, pasted_images_paths=None, rag_models=None, app=None):
    """
    Generate a streaming response with temporary reasoning display.

    Args:
        query: User query text
        session_id: Current session ID
        generation_model: Model to use for generation
        session_data: Session data dictionary
        is_rag_mode: Whether to use RAG mode or direct chat
        user_id: Current user ID
        pasted_images_paths: Optional list of pasted image paths for direct chat
        rag_models: Optional RAG models manager for accessing cached models
        app: Optional Flask app instance for application context (needed for batch processing)

    Yields chunks in the format:
    - {'type': 'reasoning', 'content': '...'} for reasoning content
    - {'type': 'response', 'content': '...'} for actual response
    - {'type': 'images', 'images': [...]} for retrieved images (RAG mode) or pasted images (direct chat)
    - {'type': 'model_info', 'model': '...', 'display_name': '...'} for model information
    - {'type': 'complete'} when done
    """
    # CRITICAL: Log session information for debugging session crossover
    logger.info(f"[SESSION TRACKING] generate_streaming_response called with session_id: {session_id}")
    logger.info(f"[SESSION TRACKING] Query: '{query[:50]}...', User: {user_id}, RAG mode: {is_rag_mode}")

    # Batch mode is driven by the UI toggle (Chat/RAG/Batch), not by template_type
    is_batch = bool(session_data and session_data.pop('_force_batch', False))
    logger.info(f"[BATCH] is_batch={is_batch}, is_rag_mode={is_rag_mode}, user={user_id}")

    # If this is a batch processing template and we have selected documents, use batch processor
    if is_batch and is_rag_mode and session_data:
        selected_docs = session_data.get('selected_docs', [])
        if selected_docs and len(selected_docs) > 0:
            logger.info(f"[BATCH PROCESSING] Batch mode for {len(selected_docs)} documents in session {session_id}")

            # Delegate to batch processor and stream the results
            try:
                # Import batch processor
                from src.services.batch_processor.processor import process_document_batch

                # Get OCR setting from session data
                use_ocr = session_data.get('use_ocr', False)

                # Send model info first
                from src.utils.model_configs import get_display_name
                model_display_name = get_display_name(generation_model)
                yield {'type': 'model_info', 'model': generation_model, 'display_name': model_display_name}

                # Ensure the RAG embedding model is loaded and cached for batch processing
                from src.models.model_loader import load_rag_model
                from src.utils.thread_safe_models import get_thread_safe_model_manager
                indexer_model = session_data.get('indexer_model', 'athrael-soju/colqwen3.5-4.5B-v3')
                batch_model_manager = get_thread_safe_model_manager()
                if not batch_model_manager.get(session_id):
                    logger.info(f"[BATCH] Loading RAG model {indexer_model} for batch processing")
                    batch_rag_model = load_rag_model(indexer_model)
                    if batch_rag_model:
                        batch_model_manager.set(session_id, batch_rag_model)
                    else:
                        yield {'type': 'error', 'error': 'Failed to load embedding model for batch processing'}
                        return

                # Stream each document as it's processed
                try:
                    for chunk in stream_batch_processing(
                        query=query,
                        session_id=session_id,
                        selected_docs=selected_docs,
                        generation_model=generation_model,
                        user_id=user_id,
                        use_ocr=use_ocr,
                    ):
                        yield chunk
                    return

                except Exception as streaming_batch_error:
                    logger.error(f"Error in streaming batch processing: {streaming_batch_error}", exc_info=True)
                    yield {'type': 'error', 'error': f"Streaming batch processing failed: {str(streaming_batch_error)}"}
                    return

                logger.info(f"Batch processing completed successfully for session {session_id}")
                return

            except Exception as batch_error:
                logger.error(f"Error in batch processing: {batch_error}", exc_info=True)
                yield {'type': 'error', 'error': f"Batch processing failed: {str(batch_error)}"}
                return

    # DEBUG: Log session data being passed
    logger.info(f"[TEMPLATE DEBUG] generate_streaming_response called with session_data:")
    if session_data:
        logger.info(f"[TEMPLATE DEBUG]   - selected_template_id: {session_data.get('selected_template_id')}")
        logger.info(f"[TEMPLATE DEBUG]   - user_id: {session_data.get('user_id')}")
    else:
        logger.error("[TEMPLATE DEBUG] session_data is None!")

    try:
        logger.info(f"Starting streaming response for session {session_id}")

        # Build conversation context: recent chat history + Mem0 memories
        max_history = session_data.get('local_history_limit', 7)
        chat_history = session_data.get('chat_history', [])

        # Sliding window of recent messages
        recent_history = chat_history[-max_history:] if len(chat_history) > max_history else chat_history
        conversation_history_for_model = [
            {"role": msg.get("role", "user"), "content": msg.get("content", "")}
            for msg in recent_history
            if msg.get("content")
        ]

        # Inject Mem0 memories as system context (non-blocking — skipped if Mem0 unavailable)
        try:
            from src.models.memory.mem0_service import get_relevant_memories
            memories = get_relevant_memories(user_id, query, limit=5)
            if memories:
                memory_context = "Relevant context from previous conversations:\n" + "\n".join(f"- {m}" for m in memories)
                conversation_history_for_model.insert(0, {"role": "system", "content": memory_context})
                logger.info(f"Injected {len(memories)} Mem0 memories into context")
        except Exception as e:
            logger.debug(f"Mem0 memory retrieval skipped: {e}")

        # Add current query if not already included
        if query and not any(msg.get('role') == 'user' and msg.get('content') == query for msg in conversation_history_for_model):
            conversation_history_for_model.append({"role": "user", "content": query})

        # Setup for RAG if needed
        retrieved_images = []
        retrieved_text_chunks = []
        rag_model = None  # Initialized here so GPU offload block doesn't hit NameError in non-RAG mode

        # Auto-downgrade RAG mode to direct chat if no documents are selected
        if is_rag_mode:
            selected_docs = session_data.get('selected_docs', [])
            if not selected_docs:
                logger.info("RAG mode requested but no documents selected — falling back to direct chat")
                is_rag_mode = False

        if is_rag_mode:
            # Send stage progression events for better UX
            yield {'type': 'stage', 'stage': 'searching'}
            selected_docs = session_data.get('selected_docs', [])
            use_ocr = session_data.get('use_ocr', False)

            logger.info(f"RAG mode: Processing {len(selected_docs)} selected documents")

            # Use the same RAG retrieval pattern as the original implementation
            from src.models.rag_retriever import rag_retriever
            from src.models.retriever_manager import select_retriever, get_current_memory_usage
            from src.models.model_loader import load_rag_model, load_embedding_adapter

            # Determine retrieval method from session data
            retrieval_method = session_data.get('retrieval_method', 'colpali')
            indexer_model = session_data.get('indexer_model', 'athrael-soju/colqwen3.5-4.5B-v3')

            # Load the appropriate embedding adapter based on retrieval method
            rag_model = None

            if retrieval_method == 'bm25':
                # BM25 doesn't need an embedding model
                from src.models.embedding_adapters.null_adapter import NullEmbeddingAdapter
                rag_model = NullEmbeddingAdapter()
                logger.info(f"Using BM25 retrieval (no embedding model needed)")

            elif retrieval_method == 'dense':
                # Load dense embedding adapter
                text_model = session_data.get('text_embedding_model', 'BAAI/bge-m3')
                rag_model = load_embedding_adapter(text_model)
                logger.info(f"Using dense retrieval with {text_model}")

            else:
                # ColPali, hybrid_rrf, or hybrid: load ColPali adapter
                # Check if the model is already cached in RAG_models first
                if rag_models:
                    try:
                        if hasattr(rag_models, 'get') and rag_models.get(session_id):
                            rag_model = rag_models.get(session_id)
                            logger.info(f"Using cached RAG model from RAG_models for session {session_id}")
                    except Exception as e:
                        logger.warning(f"Error checking RAG_models cache: {e}")

                # If not cached, load fresh model
                if not rag_model:
                    logger.info(f"Loading fresh RAG model for session {session_id}")
                    rag_model = load_rag_model(indexer_model)

                # Store the loaded model back in the session cache for future requests
                if rag_model and rag_models:
                    try:
                        if hasattr(rag_models, 'set'):
                            rag_models.set(session_id, rag_model)
                            logger.info(f"Cached RAG model for session {session_id} to avoid future reloads")
                    except Exception as e:
                        logger.warning(f"Error caching RAG model: {e}")

            if rag_model and selected_docs:
                try:
                    # Select retriever based on retrieval method
                    if retrieval_method in ('bm25', 'dense', 'hybrid_rrf', 'hybrid'):
                        # Use the registry for new retrieval methods
                        from src.models.retriever_registry import RetrieverRegistry
                        retriever_func = RetrieverRegistry.get_retriever(retrieval_method, rag_model)
                        logger.info(f"Using registry retriever: {retrieval_method}")
                    else:
                        # Use existing retriever manager for colpali (handles OCR wrapping)
                        memory_usage_pct = get_current_memory_usage()
                        retriever_func = select_retriever(
                            doc_count=len(selected_docs),
                            memory_usage_pct=memory_usage_pct,
                            session_id=session_id,
                            query=query,
                            selected_filenames=selected_docs
                        )

                    # Check if this is a Document Summary template and get template for enhanced retrieval
                    is_summary_template = False
                    template = None
                    template_id = session_data.get('selected_template_id')
                    if template_id and user_id:
                        try:
                            from src.models.prompt_templates import get_template_by_id
                            template = get_template_by_id(user_id, template_id)
                            if template and (template.get('id') == 'document-summary' or
                                           template.get('name') == 'Document Summary' or
                                           template.get('bypass_page_restriction', False)):
                                is_summary_template = True
                                logger.info(f"Document Summary template detected - will retrieve all pages")
                        except Exception as e:
                            logger.warning(f"Error checking template type: {e}")

                    # Create enhanced query for retrieval using template components
                    enhanced_query_for_retrieval = query
                    if template:
                        query_prefix = template.get('query_prefix', '')
                        query_suffix = template.get('query_suffix', '')
                        if query_prefix or query_suffix:
                            enhanced_query_for_retrieval = f"{query_prefix}{query}{query_suffix}".strip()
                            logger.info(f"Enhanced query for retrieval: '{enhanced_query_for_retrieval[:100]}...'")

                    # Perform retrieval
                    if is_summary_template:
                        # For Document Summary, retrieve many more pages (effectively all)
                        max_images = 500  # Very high number to ensure we get all pages of even large documents
                        logger.info(f"Document Summary mode: retrieving up to {max_images} pages")
                    else:
                        max_images = session_data.get('retrieval_count', 20)
                        # When adaptive is on with hybrid, use a larger pool and let
                        # score-slope analysis decide the final count
                        if (session_data.get('use_score_slope') and
                                retrieval_method == 'hybrid' and max_images < 10):
                            max_images = 10
                        logger.info(f"Performing retrieval with max_images={max_images}")

                    # Get the RAG model for retrieval from the passed rag_models parameter
                    if rag_models and hasattr(rag_models, 'get'):
                        rag_model = rag_models.get(session_id)
                    else:
                        rag_model = None
                    if not rag_model:
                        logger.error(f"No RAG model found for session {session_id}")
                        yield f"data: {json.dumps({'error': 'RAG model not loaded'})}\n\n"
                        return

                    retriever_func.set_embedding_adapter(rag_model)
                    retrieved_results = retriever_func.retrieve_documents(
                        query=enhanced_query_for_retrieval,
                        session_id=session_id,
                        k=max_images,
                        selected_filenames=selected_docs,
                        visual_weight=session_data.get('hybrid_visual_weight', 0.6),
                        use_score_slope=session_data.get('use_score_slope', False),
                        rel_drop_threshold=session_data.get('rel_drop_threshold', 0.65),
                        abs_score_threshold=session_data.get('abs_score_threshold', 0.25),
                    )

                    # Convert retriever results to streaming format
                    # Also collect text context from text-based retrievers (BM25, dense)
                    retrieved_text_chunks = []

                    if retrieved_results and hasattr(retrieved_results, '__iter__'):
                        # Handle different retriever return formats
                        actual_item_list = []
                        if isinstance(retrieved_results, (list, tuple)):
                            actual_item_list = list(retrieved_results)
                        elif hasattr(retrieved_results, 'images'):
                            actual_item_list = retrieved_results.images
                        else:
                            logger.warning(f"Unexpected retriever result type: {type(retrieved_results)}")

                        # Convert items to streaming format following original logic
                        for item in actual_item_list:
                            image_path = None
                            score = 0.0
                            text_content = None
                            result_type = None

                            if isinstance(item, dict):
                                image_path = item.get('path')
                                score = item.get('score', 0.0)
                                text_content = item.get('text_content')
                                result_type = item.get('result_type')
                            elif isinstance(item, str):
                                image_path = item

                            # Collect text content from text-based retrievers
                            if text_content and result_type == 'text':
                                source = item.get('original_filename', 'unknown')
                                page = item.get('page_num', 0)
                                page_str = f", Page {page}" if page else ""
                                retrieved_text_chunks.append(
                                    f"[Source: {source}{page_str}]\n{text_content}"
                                )

                            if image_path:
                                # Convert to relative path format for frontend display
                                if image_path.startswith('static/'):
                                    relative_path = image_path[7:]  # Remove 'static/' prefix
                                else:
                                    relative_path = image_path

                                # Store both relative path (for frontend) and full path (for model)
                                full_path = image_path if image_path.startswith('static/') else os.path.join('static', image_path).replace('\\', '/')

                                retrieved_images.append({
                                    'path': relative_path,
                                    'full_path': full_path,
                                    'score': score
                                })

                        logger.info(f"Retrieved {len(retrieved_images)} images, {len(retrieved_text_chunks)} text chunks for streaming RAG")

                        # Send stage progression and images event
                        yield {'type': 'stage', 'stage': 'extracting'}

                        # Check if we should hide retrieved pages (e.g., for Document Summary template)
                        hide_retrieved_pages = False
                        if is_summary_template:
                            # Get template details to check hide_retrieved_pages flag
                            try:
                                template_id = session_data.get('selected_template_id')
                                if template_id and user_id:
                                    from src.models.prompt_templates import get_template_by_id
                                    template = get_template_by_id(user_id, template_id)
                                    hide_retrieved_pages = template.get('hide_retrieved_pages', False)
                                    if hide_retrieved_pages:
                                        logger.info("Document Summary: Hiding retrieved pages from display")
                            except Exception as e:
                                logger.warning(f"Error checking hide_retrieved_pages flag: {e}")
                        
                        # Send images event only if we're not hiding them
                        if retrieved_images and not hide_retrieved_pages:
                            yield {'type': 'images', 'images': retrieved_images}
                        elif retrieved_images and hide_retrieved_pages:
                            # Send a special event to indicate we're processing for summary
                            logger.info(f"Sending summary_processing event with {len(retrieved_images)} pages")
                            yield {'type': 'summary_processing', 'page_count': len(retrieved_images)}

                except Exception as retrieval_error:
                    logger.error(f"Error during streaming RAG retrieval: {retrieval_error}", exc_info=True)

                # CRITICAL: Clear query cache immediately after retrieval to prevent memory buildup
                try:
                    from src.models.vector_stores.lancedb_manager import clear_query_cache
                    clear_query_cache()
                    logger.debug("Cleared LanceDB query cache after RAG retrieval")
                except Exception as cache_error:
                    logger.warning(f"Failed to clear query cache: {cache_error}")

        # Send model info and final stage
        from src.utils.model_configs import get_display_name
        model_display_name = get_display_name(generation_model)
        yield {'type': 'model_info', 'model': generation_model, 'display_name': model_display_name}

        # Send final stage before generation starts
        yield {'type': 'stage', 'stage': 'generating'}

        # Stream from the model following the same pattern as non-streaming
        # Import the streaming version of model responder
        streaming_completed = False

        # Collect OCR results if any were generated during RAG processing
        ocr_results_to_process = None

        # Handle pasted images for direct chat mode
        is_pasted_images_only = False
        final_images_for_model = []

        if not is_rag_mode and pasted_images_paths:
            # Direct chat mode with pasted images
            is_pasted_images_only = True
            # Convert relative paths to full paths for the model
            for relative_path in pasted_images_paths:
                full_path = os.path.join('static', relative_path)
                final_images_for_model.append(full_path)
            logger.info(f"Direct chat mode: Using {len(final_images_for_model)} pasted images")

            # Send pasted images as an 'images' event for frontend display
            pasted_images_for_display = []
            for relative_path in pasted_images_paths:
                pasted_images_for_display.append({
                    'path': relative_path,
                    'score': 1.0  # Pasted images have perfect relevance
                })
            yield {'type': 'images', 'images': pasted_images_for_display}

        elif is_rag_mode and retrieved_images:
            # RAG mode with retrieved images
            final_images_for_model = [img['full_path'] for img in retrieved_images]
            logger.info(f"RAG mode: Using {len(final_images_for_model)} retrieved images")

        # Get use_ocr flag from session data
        use_ocr = session_data.get('use_ocr', False) if session_data else False

        # Get use_score_slope flag from session data
        use_score_slope_flag = session_data.get('use_score_slope', False) if session_data else False

        # Get the selected template for passing to responder
        selected_template_id = session_data.get('selected_template_id') if session_data else None

        # Build text context from text-based retrieval results (BM25, dense)
        retrieved_text_context = ""
        if retrieved_text_chunks:
            retrieved_text_context = "\n\n---\n\n".join(retrieved_text_chunks)
            logger.info(f"Built retrieved_text_context: {len(retrieved_text_context)} chars from {len(retrieved_text_chunks)} chunks")

        # --- GPU Memory Optimization (best-effort) ---
        # Move ColPali embedding model to CPU before LLM generation.
        # ColPali uses 4-6GB GPU and Ollama needs 8-10GB. Both on GPU simultaneously
        # can push past the 24GB limit on RTX cards. Offloading ColPali frees that GPU
        # memory for the LLM.
        #
        # This is best-effort: if another thread is using the model concurrently
        # (unlikely in a local single-user app), the offload is skipped gracefully.
        _offloaded_model = None
        if rag_model and hasattr(rag_model, 'model') and rag_model.model is not None:
            try:
                import torch
                model_obj = rag_model.model
                if model_obj is not None and hasattr(model_obj, 'device') and str(model_obj.device) != 'cpu':
                    logger.info("[GPU OFFLOAD] Moving embedding model to CPU before LLM generation")
                    model_obj.to('cpu')
                    torch.cuda.empty_cache()
                    _offloaded_model = rag_model
                    logger.info("[GPU OFFLOAD] Freed ~4-6GB GPU memory for LLM inference")
            except Exception as offload_err:
                # Best-effort: if offload fails (e.g., model in use), continue without it
                logger.debug(f"[GPU OFFLOAD] Skipped: {offload_err}")

        try:
            for chunk in model_generate_streaming_response(
                images=final_images_for_model,
                query=query,
                session_id=session_id,
                model_choice=generation_model,
                chat_history=conversation_history_for_model,
                user_id=user_id,
                direct_ocr_results=ocr_results_to_process,
                original_query=query,
                use_ocr=use_ocr,
                is_pasted_images=is_pasted_images_only,
                use_score_slope=use_score_slope_flag,
                is_rag_mode=is_rag_mode,  # Pass RAG mode flag to responder
                selected_template_id=selected_template_id,  # Pass template ID
                retrieved_text_context=retrieved_text_context,  # Pass text context from BM25/dense
            ):
                yield chunk

            streaming_completed = True
            logger.info(f"Streaming completed successfully for session {session_id}")

        except Exception as streaming_error:
            logger.error(f"Error during streaming: {streaming_error}", exc_info=True)
            yield {'type': 'error', 'error': str(streaming_error)}
        finally:
            # --- Restore embedding model to GPU if we offloaded it ---
            if _offloaded_model is not None:
                try:
                    import torch
                    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
                    _offloaded_model.model.to(device)
                    logger.info(f"[GPU OFFLOAD] Restored embedding model to {device}")
                except Exception as restore_err:
                    logger.warning(f"[GPU OFFLOAD] Failed to restore model: {restore_err}")

        # CRITICAL: Service-level memory cleanup
        # This ensures cleanup happens after model generation completes
        try:
            logger.info(f"[SERVICE MEMORY] Performing service-level cleanup for session {session_id}")
            cleanup_stats = cleanup_after_response(
                session_id=session_id,
                user_id=user_id,
                force_aggressive=False  # Standard cleanup for services
            )

            if cleanup_stats:
                ram_freed = cleanup_stats.get('freed', {}).get('ram', 0)
                gpu_freed = cleanup_stats.get('freed', {}).get('gpu', 0)
                logger.info(f"[SERVICE MEMORY] Service cleanup freed {ram_freed:.2f} MB RAM, {gpu_freed:.2f} MB GPU")

        except Exception as service_cleanup_error:
            logger.error(f"[SERVICE MEMORY] Service-level cleanup failed: {service_cleanup_error}")

    except Exception as e:
        logger.error(f"Error in generate_streaming_response: {e}", exc_info=True)

        # CRITICAL: Ensure cleanup happens even on service-level errors
        try:
            logger.info(f"[SERVICE MEMORY] Emergency service-level cleanup for session {session_id}")
            cleanup_after_response(
                session_id=session_id,
                user_id=user_id,
                force_aggressive=True
            )
        except Exception as emergency_cleanup:
            logger.error(f"[SERVICE MEMORY] Emergency service cleanup failed: {emergency_cleanup}")

        yield {'type': 'error', 'error': str(e)}


def stream_batch_processing(query, session_id, selected_docs, generation_model, user_id, use_ocr):
    """
    Stream batch processing results in real-time as each document is processed.

    Args:
        query: User query text
        session_id: Current session ID
        selected_docs: List of selected document filenames
        generation_model: Model to use for generation
        user_id: Current user ID
        use_ocr: Whether to use OCR
        app: Flask app instance for application context

    Yields chunks in the format:
    - {'type': 'batch_start', 'total_docs': N} when batch processing starts
    - {'type': 'doc_start', 'doc_name': '...', 'doc_index': N} when starting a document
    - {'type': 'response', 'content': '...'} for document response content
    - {'type': 'doc_complete', 'doc_name': '...'} when a document is complete
    - {'type': 'complete'} when all documents are processed
    """
    logger.info(f"Starting streaming batch processing for {len(selected_docs)} documents in session {session_id}")

    try:
        # Send batch start event
        yield {'type': 'batch_start', 'total_docs': len(selected_docs)}

        # Process each document individually and stream the results
        for doc_index, doc_filename in enumerate(selected_docs):
            try:
                # Send document start event
                yield {'type': 'doc_start', 'doc_name': doc_filename, 'doc_index': doc_index}

                # Add separator between documents (except for the first one)
                if doc_index > 0:
                    yield {'type': 'response', 'content': '\n\n---\n\n'}

                # Process this single document with streaming
                try:
                    for chunk in process_single_document_streaming(
                        doc_filename=doc_filename,
                        query=query,
                        session_id=session_id,
                        generation_model=generation_model,
                        user_id=user_id,
                        use_ocr=use_ocr
                    ):
                        yield chunk

                    logger.info(f"Successfully streamed document {doc_filename}")
                except Exception as doc_error:
                    # Handle document processing error
                    error_content = f"Error processing document: {str(doc_error)}"
                    yield {'type': 'response', 'content': error_content}
                    logger.error(f"Error processing document {doc_filename}: {doc_error}", exc_info=True)

                # Send document complete event
                yield {'type': 'doc_complete', 'doc_name': doc_filename}

            except Exception as doc_error:
                logger.error(f"Error processing document {doc_filename}: {doc_error}", exc_info=True)
                error_content = f"## Document: {doc_filename}\n\nError processing document: {str(doc_error)}"
                yield {'type': 'response', 'content': error_content}
                yield {'type': 'doc_complete', 'doc_name': doc_filename}

        # Note: Completion event is sent by the route handler after session saving

        logger.info(f"Streaming batch processing completed successfully for session {session_id}")

    except Exception as e:
        logger.error(f"Error in stream_batch_processing: {e}", exc_info=True)
        yield {'type': 'error', 'error': str(e)}


def _is_generic_query(query: str) -> bool:
    """Detect generic/summarization queries that should process the full document."""
    q = query.strip().lower()
    if len(q) <= 5:
        return True
    generic_terms = ['summarize', 'summary', 'overview', 'describe', 'analyze',
                     'explain', 'review', 'extract all', 'full document']
    return any(term in q for term in generic_terms)


def _get_all_document_pages(session_id: str, doc_filename: str, session_data: dict) -> list[str]:
    """Get all page image paths for a document, sorted by page number.

    Uses the indexed_files list to find the document's index, then collects
    all retrieved_{docIndex}p{pageIndex}.png files from the session's image dir.
    """
    import os, re

    indexed_files = session_data.get('indexed_files', [])
    doc_index = None
    for i, f in enumerate(indexed_files):
        fname = f.get('filename', f) if isinstance(f, dict) else f
        if fname == doc_filename:
            doc_index = i
            break

    if doc_index is None:
        logger.warning(f"Document {doc_filename} not found in indexed_files")
        return []

    image_dir = os.path.join('static', 'images', session_id)
    if not os.path.isdir(image_dir):
        logger.warning(f"Image directory not found: {image_dir}")
        return []

    # Collect pages matching retrieved_{docIndex}p{pageIndex}.png
    pattern = re.compile(rf'^retrieved_{doc_index}p(\d+)\.png$')
    pages = []
    for fname in os.listdir(image_dir):
        m = pattern.match(fname)
        if m:
            pages.append((int(m.group(1)), os.path.join(image_dir, fname)))

    pages.sort(key=lambda x: x[0])
    return [p[1] for p in pages]


def process_single_document_streaming(doc_filename, query, session_id, generation_model, user_id, use_ocr):
    """
    Process a single document for streaming batch processing.

    Two modes:
    - **Full-document** (generic query or bypass_page_restriction template):
      Sends ALL pages in sequential chunks of PAGES_PER_CHUNK, streaming a
      section summary for each chunk. Works for any document size.
    - **Targeted** (specific query): Uses RAG retrieval for top-k relevant
      pages only (existing behavior).
    """
    PAGES_PER_CHUNK = 4

    logger.info(f"Starting streaming processing for document {doc_filename} in session {session_id}")

    try:
        import os
        from src.models.rag_retriever import rag_retriever
        from src.models.responder import generate_streaming_response as model_generate_streaming_response
        from src.services.session_manager.manager import load_session

        session_data = load_session('sessions', session_id)
        if not session_data:
            yield {'type': 'response', 'content': 'Error: Session data not found'}
            return

        retrieval_count = session_data.get('retrieval_count', 3)
        use_score_slope = session_data.get('use_score_slope', False)

        # Load template
        template = None
        template_id = session_data.get('selected_template_id')
        if template_id and user_id:
            try:
                from src.models.prompt_templates import get_template_by_id
                template = get_template_by_id(user_id, template_id)
            except Exception as e:
                logger.warning(f"Error loading template for batch processing: {e}")

        # Decide: full-document or targeted retrieval
        bypass = template.get('bypass_page_restriction', False) if template else False
        full_doc = bypass or _is_generic_query(query)

        if full_doc:
            # ── Full-document mode: retrieve ALL pages, process in chunks ──
            # Use the RAG retriever with k=page_count to get every page for this doc.
            # The retriever generates page images on-demand from the original PDF.
            indexed_files = session_data.get('indexed_files', [])
            page_count = 0
            for f in indexed_files:
                fname = f.get('filename', f) if isinstance(f, dict) else f
                if fname == doc_filename:
                    page_count = f.get('page_count', 100) if isinstance(f, dict) else 100
                    break
            if page_count == 0:
                page_count = 100  # fallback

            from src.utils.thread_safe_models import get_thread_safe_model_manager
            rag_model = get_thread_safe_model_manager().get(session_id)
            if not rag_model:
                yield {'type': 'response', 'content': 'Error: RAG model not available for session'}
                return

            rag_retriever.set_embedding_adapter(rag_model)
            retrieval_results = rag_retriever.retrieve_documents(
                query, session_id, k=page_count,
                selected_filenames=[doc_filename], use_score_slope=False
            )

            # Unpack retrieval results
            actual_item_list = []
            if isinstance(retrieval_results, tuple) and len(retrieval_results) == 2:
                potential_list = retrieval_results[0]
                if isinstance(potential_list, list):
                    actual_item_list = potential_list
            elif isinstance(retrieval_results, list):
                actual_item_list = retrieval_results

            if not actual_item_list:
                yield {'type': 'response', 'content': 'No pages found for this document.'}
                return

            # Convert to full paths, sorted by page number
            all_pages = []
            for item in actual_item_list:
                image_path = item.get('path') if isinstance(item, dict) else item
                page_num = item.get('page_num', 0) if isinstance(item, dict) else 0
                if image_path:
                    full_path = image_path if image_path.startswith('static/') else os.path.join('static', image_path).replace('\\', '/')
                    if os.path.exists(full_path):
                        all_pages.append((page_num, full_path))
            all_pages.sort(key=lambda x: x[0])
            all_page_paths = [p[1] for p in all_pages]

            total_pages = len(all_page_paths)
            num_chunks = (total_pages + PAGES_PER_CHUNK - 1) // PAGES_PER_CHUNK
            logger.info(f"[FULL-DOC] Processing {total_pages} pages in {num_chunks} chunks for {doc_filename}")

            for chunk_idx in range(num_chunks):
                start = chunk_idx * PAGES_PER_CHUNK
                end = min(start + PAGES_PER_CHUNK, total_pages)
                chunk_pages = all_page_paths[start:end]

                # Section header
                if num_chunks > 1:
                    yield {'type': 'response', 'content': f'\n\n**Pages {start + 1}–{end} of {total_pages}**\n\n'}

                # Build section-aware query
                section_query = query
                if num_chunks > 1:
                    section_query = f"Analyze pages {start + 1}-{end} of {total_pages}. {query}"

                for chunk in model_generate_streaming_response(
                    images=chunk_pages,
                    query=section_query,
                    session_id=session_id,
                    model_choice=generation_model,
                    chat_history=[],
                    user_id=user_id,
                    original_query=query,
                    use_ocr=use_ocr,
                    is_pasted_images=False,
                    use_score_slope=False,
                    is_rag_mode=True
                ):
                    chunk_type = chunk.get('type')
                    if chunk_type in ['response', 'reasoning', 'reasoning_complete', 'complete']:
                        yield chunk
                    elif chunk_type == 'error':
                        yield chunk

            # Confirmation line — no thumbnails for full-doc (would be overwhelming)
            yield {'type': 'response', 'content': f'\n\n---\n*Processed {total_pages} pages in {num_chunks} sections.*\n'}

            logger.info(f"[FULL-DOC] Completed {total_pages} pages for {doc_filename}")

        else:
            # ── Targeted mode: RAG retrieval for top-k pages ──
            enhanced_query = query
            if template:
                prefix = template.get('query_prefix', '')
                suffix = template.get('query_suffix', '')
                if prefix or suffix:
                    enhanced_query = f"{prefix}{query}{suffix}".strip()

            from src.utils.thread_safe_models import get_thread_safe_model_manager
            rag_model = get_thread_safe_model_manager().get(session_id)
            if not rag_model:
                yield {'type': 'response', 'content': 'Error: RAG model not available for session'}
                return

            rag_retriever.set_embedding_adapter(rag_model)
            retrieval_results = rag_retriever.retrieve_documents(
                enhanced_query, session_id, k=retrieval_count,
                selected_filenames=[doc_filename], use_score_slope=use_score_slope
            )

            if not retrieval_results:
                yield {'type': 'response', 'content': 'No relevant pages found in document'}
                return

            # Unpack retrieval results
            actual_item_list = []
            ocr_results_to_process = None
            if isinstance(retrieval_results, tuple) and len(retrieval_results) == 2:
                potential_list = retrieval_results[0]
                if isinstance(potential_list, list):
                    actual_item_list = potential_list
                ocr_results_to_process = retrieval_results[1]
            elif isinstance(retrieval_results, list):
                actual_item_list = retrieval_results

            if not actual_item_list:
                yield {'type': 'response', 'content': 'No relevant content found in document'}
                return

            # Convert to full paths
            doc_image_paths = []
            doc_image_paths_with_scores = []
            used_paths = set()

            for item in actual_item_list:
                image_path = item.get('path') if isinstance(item, dict) else item
                score = item.get('score', 0.0) if isinstance(item, dict) else 0.0

                if image_path:
                    full_path = image_path if image_path.startswith('static/') else os.path.join('static', image_path).replace('\\', '/')
                    if full_path not in used_paths and os.path.exists(full_path):
                        doc_image_paths_with_scores.append({'path': full_path, 'score': score})
                        doc_image_paths.append(full_path)
                        used_paths.add(full_path)

            if not doc_image_paths:
                yield {'type': 'response', 'content': 'No valid image paths found for document'}
                return

            logger.info(f"Streaming response for {doc_filename} with {len(doc_image_paths)} retrieved pages")

            for chunk in model_generate_streaming_response(
                images=doc_image_paths, query=query, session_id=session_id,
                model_choice=generation_model, chat_history=[], user_id=user_id,
                direct_ocr_results=ocr_results_to_process, original_query=query,
                use_ocr=use_ocr, is_pasted_images=False,
                use_score_slope=use_score_slope, is_rag_mode=True
            ):
                chunk_type = chunk.get('type')
                if chunk_type in ['response', 'reasoning', 'reasoning_complete', 'complete']:
                    yield chunk
                elif chunk_type == 'error':
                    yield chunk

            # Send retrieved pages as images event
            if doc_image_paths_with_scores:
                display_images = []
                for img_data in doc_image_paths_with_scores:
                    path = img_data.get('path', '')
                    score = img_data.get('score', 0.0)
                    if path:
                        rel = path[7:] if path.startswith('static/') else path
                        display_images.append({'path': rel, 'score': score, 'source': doc_filename})
                if display_images:
                    yield {'type': 'images', 'images': display_images}

            logger.info(f"Completed targeted processing for {doc_filename}")

    except Exception as e:
        logger.error(f"Error in streaming processing for document {doc_filename}: {e}", exc_info=True)
        yield {'type': 'response', 'content': f'Error processing document: {str(e)}'}


def process_single_document_for_streaming(doc_filename, query, session_id, generation_model, user_id, use_ocr):
    """
    Process a single document for streaming batch processing.

    This is a simplified version of the batch processor that handles just one document
    and returns the result immediately for streaming.

    Args:
        doc_filename: Name of the document to process
        query: User query text
        session_id: Current session ID
        generation_model: Model to use for generation
        user_id: Current user ID
        use_ocr: Whether to use OCR

    Returns:
        Dictionary with document processing result:
        - success: Boolean indicating if processing was successful
        - response: The AI response text for this document
        - image_paths_with_scores: List of retrieved images with scores
        - error: Error message if processing failed
    """
    logger.info(f"Processing single document {doc_filename} for streaming in session {session_id}")

    try:
        # Import necessary modules
        import os
        from src.models.rag_retriever import rag_retriever
        from src.models.responder import generate_response as model_generate_response
        from src.services.session_manager.manager import load_session

        # Load session data to get retrieval settings
        session_data = load_session('sessions', session_id)
        if not session_data:
            return {
                'success': False,
                'error': 'Session data not found'
            }

        # Get retrieval settings
        retrieval_count = session_data.get('retrieval_count', 3)
        use_score_slope = session_data.get('use_score_slope', False)

        logger.info(f"Using retrieval settings: count={retrieval_count}, score_slope={use_score_slope}")

        # Get template for enhanced retrieval query
        template = None
        template_id = session_data.get('selected_template_id')
        if template_id and user_id:
            try:
                from src.models.prompt_templates import get_template_by_id
                template = get_template_by_id(user_id, template_id)
            except Exception as e:
                logger.warning(f"Error loading template for batch processing: {e}")

        # Create enhanced query for retrieval using template components
        enhanced_query_for_retrieval = query
        if template:
            query_prefix = template.get('query_prefix', '')
            query_suffix = template.get('query_suffix', '')
            if query_prefix or query_suffix:
                enhanced_query_for_retrieval = f"{query_prefix}{query}{query_suffix}".strip()
                logger.info(f"Enhanced query for batch retrieval: '{enhanced_query_for_retrieval[:100]}...'")

        # Perform RAG retrieval for this specific document
        logger.info(f"Performing RAG retrieval for document {doc_filename}")

        # Get the RAG model for retrieval
        from src.utils.thread_safe_models import get_thread_safe_model_manager
        model_manager = get_thread_safe_model_manager()
        rag_model = model_manager.get(session_id)

        if not rag_model:
            return {
                'success': False,
                'error': 'RAG model not available for session'
            }

        # Use the RAG retriever to get relevant pages for this document
        rag_retriever.set_embedding_adapter(rag_model)
        retrieval_results = rag_retriever.retrieve_documents(
            enhanced_query_for_retrieval,
            session_id,
            k=retrieval_count,
            selected_filenames=[doc_filename],  # Only this document
            use_score_slope=use_score_slope
        )

        if not retrieval_results:
            logger.warning(f"No retrieval results for document {doc_filename}")
            return {
                'success': False,
                'error': 'No relevant pages found in document'
            }

        # Process retrieval results
        actual_item_list = []
        ocr_results_to_process = None

        if isinstance(retrieval_results, tuple) and len(retrieval_results) == 2:
            potential_list = retrieval_results[0]
            if isinstance(potential_list, list):
                actual_item_list = potential_list
            ocr_results_to_process = retrieval_results[1]
            logger.info(f"Processing {len(actual_item_list)} items from retriever tuple for {doc_filename}")
        elif isinstance(retrieval_results, list):
            actual_item_list = retrieval_results
            logger.info(f"Processing {len(actual_item_list)} items from retriever list for {doc_filename}")

        if not actual_item_list:
            logger.warning(f"No items in retrieval results for document {doc_filename}")
            return {
                'success': False,
                'error': 'No relevant content found in document'
            }

        # Convert items to full paths and preserve similarity scores
        doc_image_paths = []
        doc_image_paths_with_scores = []
        used_paths = set()

        for item in actual_item_list:
            image_path = None
            score = 0.0

            if isinstance(item, dict):
                image_path = item.get('path')
                score = item.get('score', 0.0)
            elif isinstance(item, str):
                image_path = item

            if image_path:
                full_path = None
                if image_path.startswith('static/'):
                    full_path = image_path
                else:
                    full_path = os.path.join('static', image_path).replace('\\', '/')

                if full_path and full_path not in used_paths:
                    if os.path.exists(full_path):
                        # Add to document-specific paths with score
                        doc_image_paths_with_scores.append({
                            'path': full_path,
                            'score': score
                        })
                        doc_image_paths.append(full_path)
                        used_paths.add(full_path)
                    else:
                        logger.warning(f"Image path does not exist: {full_path}")

        logger.info(f"Processed {len(doc_image_paths)} image paths for document {doc_filename}")

        if not doc_image_paths:
            return {
                'success': False,
                'error': 'No valid image paths found for document'
            }

        # Generate response using the model
        logger.info(f"Calling model_generate_response with {len(doc_image_paths)} image paths for {doc_filename}")

        # Use empty chat history for batch processing
        empty_chat_history = []

        response_text, _ = model_generate_response(
            doc_image_paths,
            query,
            session_id,
            generation_model,
            empty_chat_history,
            direct_ocr_results=ocr_results_to_process,
            original_query=query,
            use_ocr=use_ocr,
            is_pasted_images=False,
            user_id=user_id,
            use_score_slope=use_score_slope
        )

        logger.info(f"Successfully generated response for document {doc_filename}")

        return {
            'success': True,
            'response': response_text,
            'image_paths': doc_image_paths,
            'image_paths_with_scores': doc_image_paths_with_scores
        }

    except Exception as e:
        logger.error(f"Error processing document {doc_filename}: {e}", exc_info=True)
        return {
            'success': False,
            'error': str(e)
        }