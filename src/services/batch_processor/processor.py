"""
Core batch processing functionality.

This module contains functions for processing batches of documents,
extracting structured information, and aggregating results.
"""
import os
import gc
import json
import logging
import time
import torch
from typing import List, Dict, Any, Optional, Tuple

from src.models.model_loader import load_rag_model
from src.models.retriever_manager import select_retriever
from src.models.memory.chat_memory_manager import get_chat_history_for_llm
from src.models.responder import generate_response as model_generate_response
from src.models.rag_retriever import rag_retriever

# Configure logger
logger = logging.getLogger(__name__)

def process_document_directly(
    query: str,
    session_id: str,
    generation_model: str,
    chat_history: List[Dict[str, Any]],
    rag_model: Any,
    retriever: Any,
    selected_docs: List[str],
    user_id: str,
    use_ocr: bool = True
) -> Tuple[str, List[str]]:
    """
    Process a single document or set of documents directly without using the response generator.

    This function is used by the batch processor to avoid circular dependencies.

    Args:
        query: The query to apply to the document(s)
        session_id: Current session ID
        generation_model: Model to use for generation
        chat_history: Conversation history
        rag_model: RAG model for retrieval
        retriever: Retriever instance
        selected_docs: List of selected document filenames
        user_id: Current user ID
        use_ocr: Whether to use OCR for text extraction

    Returns:
        Tuple of (response_text, used_images)
    """
    logger.info(f"Processing document(s) directly: {selected_docs}")

    # Initialize variables
    full_image_paths = []
    ocr_results_to_process = None

    # If no retriever is provided, create one for all documents
    # This avoids reloading the embedding model for each document
    if retriever is None:
        logger.info(f"No retriever provided, creating a shared retriever for all documents")
        from src.models.retriever_manager import select_retriever
        retriever = select_retriever(
            doc_count=len(selected_docs),
            memory_usage_pct=0,
            session_id=session_id,
            query=query,
            force_ocr=use_ocr,
            selected_filenames=selected_docs
        )

        if not retriever:
            logger.error(f"Failed to create shared retriever for direct processing")
            return f"Error: Failed to create retriever for document processing", []

    try:
        # Determine retrieval count and score-slope settings from session data
        session_file = os.path.join('sessions', f"{session_id}.json")
        retrieval_count = 3  # Default
        use_score_slope = False  # Default
        if os.path.exists(session_file):
            try:
                import json
                with open(session_file, 'r') as f:
                    session_data = json.load(f)
                retrieval_count = int(session_data.get('retrieval_count', 3))
                use_score_slope = session_data.get('use_score_slope', False)

                # Store the original desired count
                desired_result_count = retrieval_count
                
                # If score-slope is enabled, use a fixed pool size for analysis
                if use_score_slope:
                    # Use a fixed pool size that gives score slope enough data to work with
                    # This should NOT scale with the user's desired count
                    score_slope_pool_size = 10  # Fixed pool for score analysis
                    logger.info(f"Score-slope enabled - retrieving {score_slope_pool_size} pages for analysis (user wants: {desired_result_count})")
                    retrieval_count = score_slope_pool_size
            except Exception as e_load:
                logger.error(f"Error loading session data for retrieval settings: {e_load}")

        # Call the retriever
        logger.info(f"Calling retriever for document(s): {selected_docs} with retrieval_count={retrieval_count}, use_score_slope={use_score_slope}")

        # Set the embedding adapter on the retriever before calling
        retriever.set_embedding_adapter(rag_model)

        # Check if this is an OCRRAGRetriever
        from src.models.ocr_rag_retriever import OCRRAGRetriever
        if isinstance(retriever, OCRRAGRetriever):
            retrieved_results = retriever.retrieve_documents(
                query, session_id,
                k=retrieval_count, selected_filenames=selected_docs,
                use_ocr=use_ocr, use_score_slope=use_score_slope
            )
        else:
            # Standard and LanceDB retrievers don't expect use_ocr
            retrieved_results = retriever.retrieve_documents(
                query, session_id,
                k=retrieval_count, selected_filenames=selected_docs,
                use_score_slope=use_score_slope
            )

        # Process retriever results
        actual_item_list = []
        score_analysis_data = None

        # Check if we got a tuple with analysis data
        if isinstance(retrieved_results, tuple):
            if len(retrieved_results) == 2:
                # Standard tuple: (results, ocr_data)
                potential_list = retrieved_results[0]
                if isinstance(potential_list, list):
                    actual_item_list = potential_list
                ocr_results_to_process = retrieved_results[1]
                logger.info(f"Processing {len(actual_item_list)} items from OCR retriever tuple.")
            elif len(retrieved_results) == 3:
                # Extended tuple with analysis: (results, ocr_data, analysis)
                potential_list = retrieved_results[0]
                if isinstance(potential_list, list):
                    actual_item_list = potential_list
                ocr_results_to_process = retrieved_results[1]
                score_analysis_data = retrieved_results[2]

                # Log detailed score-slope analysis if available
                if score_analysis_data and use_score_slope:
                    logger.info(f"Score-slope analysis: Original count: {score_analysis_data.get('original_count', 'N/A')}, " +
                               f"Filtered count: {score_analysis_data.get('filtered_count', 'N/A')}, " +
                               f"Cutoff reason: {score_analysis_data.get('cutoff_reason', 'N/A')}")

                    # Log score details if available
                    if 'scores' in score_analysis_data:
                        scores = score_analysis_data['scores']
                        score_details = [f"{i}:{score:.4f}" for i, score in enumerate(scores[:min(10, len(scores))])]
                        logger.info(f"Top scores: {', '.join(score_details)}")

                logger.info(f"Processing {len(actual_item_list)} items from OCR retriever tuple with score analysis.")
            else:
                logger.warning(f"Unexpected retriever tuple length: {len(retrieved_results)}")
                actual_item_list = []
        elif isinstance(retrieved_results, list):
            actual_item_list = retrieved_results
            logger.info(f"Processing {len(actual_item_list)} items from retriever list.")
        else:
            logger.warning(f"Unexpected retriever result type: {type(retrieved_results)}")

        # Log the effect of score-slope analysis if enabled
        if use_score_slope:
            logger.info(f"Score-slope analysis resulted in {len(actual_item_list)} items")

            # If we have score analysis data, log the cutoff details
            if score_analysis_data:
                cutoff_index = score_analysis_data.get('cutoff_index', 'N/A')
                cutoff_score = score_analysis_data.get('cutoff_score', 'N/A')
                cutoff_reason = score_analysis_data.get('cutoff_reason', 'N/A')
                logger.info(f"Score-slope cutoff at index {cutoff_index} with score {cutoff_score}, reason: {cutoff_reason}")

                # If we have score analysis data, use it to filter the results
                # This ensures we respect the score-slope analysis
                if isinstance(cutoff_index, int) and cutoff_index > 0:
                    # Only apply if we have more items than the cutoff index
                    if len(actual_item_list) > cutoff_index:
                        logger.info(f"Applying score-slope cutoff: {len(actual_item_list)} -> {cutoff_index} items")
                        actual_item_list = actual_item_list[:cutoff_index]
                    else:
                        logger.info(f"No need to apply score-slope cutoff: {len(actual_item_list)} items <= {cutoff_index}")

        # Convert items to full paths
        used_paths = set()
        for item in actual_item_list:
            image_path = None
            if isinstance(item, dict):
                image_path = item.get('path')
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
                        full_image_paths.append(full_path)
                        used_paths.add(full_path)
                    else:
                        alt_full_path = os.path.join('static', os.path.basename(image_path)).replace('\\', '/')
                        if os.path.exists(alt_full_path):
                            logger.warning(f"Used alternative path {alt_full_path} as {full_path} not found.")
                            full_image_paths.append(alt_full_path)
                            used_paths.add(alt_full_path)
                        else:
                            logger.warning(f"Constructed full image path does not exist: {full_path}")

        logger.info(f"Processed retrieval results into {len(full_image_paths)} full image paths.")

        # Get use_score_slope flag from session data
        use_score_slope = False
        try:
            from src.services.session_manager.manager import load_session
            session_data = load_session('sessions', session_id)
            if session_data:
                use_score_slope = session_data.get('use_score_slope', False)
                logger.info(f"Using score-slope flag from session: {use_score_slope}")
        except Exception as e:
            logger.error(f"Error getting use_score_slope from session: {e}")

        # Generate response using the model
        logger.info(f"Calling model_generate_response with {len(full_image_paths)} image paths.")
        response_text, images_returned = model_generate_response(
            full_image_paths,
            query,
            session_id,
            generation_model,
            chat_history,
            direct_ocr_results=ocr_results_to_process,
            original_query=query,
            use_ocr=use_ocr,
            is_pasted_images=False,
            user_id=user_id,
            use_score_slope=use_score_slope
        )

        return response_text, images_returned

    except Exception as e:
        logger.error(f"Error in process_document_directly: {e}", exc_info=True)
        return f"Error processing document(s): {str(e)}", []

def process_document_batch(
    query: str,
    session_id: str,
    selected_docs: List[str],
    generation_model: str,
    user_id: str,
    use_ocr: bool = True
) -> Dict[str, Any]:
    """
    Process a batch of documents using the same query.

    This function processes each document sequentially to minimize memory usage,
    and returns separate responses for each document instead of a combined response.

    Args:
        query: The query to apply to each document
        session_id: Current session ID
        selected_docs: List of selected document filenames
        generation_model: Model to use for generation
        user_id: Current user ID
        use_ocr: Whether to use OCR for text extraction

    Returns:
        Dictionary with batch processing results, containing separate responses for each document
    """
    logger.info(f"Starting batch processing for {len(selected_docs)} documents in session {session_id}")

    # Initialize results
    results = {
        "individual_results": [],
        "documents_processed": len(selected_docs),
        "success": True,
        "image_paths": [],  # Store image paths for display
        "separate_responses": []  # New field for separate responses
    }

    # Import memory manager for cleanup
    try:
        from src.models.memory.memory_manager import memory_manager
    except ImportError:
        logger.warning("Could not import memory_manager, memory cleanup may be limited")
        memory_manager = None

    # Get empty chat history for stateless processing
    empty_chat_history = get_chat_history_for_llm(session_id, 0)

    # Get the indexer model from session data
    try:
        from src.services.session_manager.manager import load_session
        session_data = load_session('sessions', session_id)
        if not session_data:
            logger.error(f"Failed to load session data for batch processing in session {session_id}")
            return {
                "success": False,
                "message": "Failed to load session data for batch processing",
                "image_paths": []
            }

        indexer_model = session_data.get('indexer_model', 'athrael-soju/colqwen3.5-4.5B-v3')
        logger.info(f"Using indexer model {indexer_model} for batch processing in session {session_id}")

        # Check if we already have a RAG model loaded in the app
        import sys
        rag_model = None

        # First, check if the model is already loaded - check both storage locations
        if 'app' in sys.modules:
            app_module = sys.modules['app']
            
            # Check model stored by indexer_model name (batch processing storage)
            if hasattr(app_module, 'RAG_models') and app_module.RAG_models.contains(indexer_model):
                rag_model = app_module.RAG_models.get(indexer_model)
                if rag_model is not None:
                    logger.info(f"Reusing existing RAG model for {indexer_model} from app.RAG_models (by model name)")
                    logger.info(f"Found existing RAG model of type {type(rag_model).__name__} with id {id(rag_model)}")

                    # Check if the model has an adapter (for ColPali models)
                    if hasattr(rag_model, 'adapter'):
                        logger.info(f"Model has adapter of type {type(rag_model.adapter).__name__} with id {id(rag_model.adapter)}")

                    # Check if the model has an embedding model
                    if hasattr(rag_model, 'embedding_model'):
                        logger.info(f"Model has embedding_model with id {id(rag_model.embedding_model)}")
            
            # If not found by model name, check if model is stored by user_id (standard RAG storage)
            elif hasattr(app_module, 'RAG_models') and app_module.RAG_models.contains(session_id):
                rag_model = app_module.RAG_models.get(session_id)
                if rag_model is not None:
                    logger.info(f"Reusing existing RAG model for user {user_id} from app.RAG_models (by user_id)")
                    logger.info(f"Found existing RAG model of type {type(rag_model).__name__} with id {id(rag_model)}")

                    # Check if the model has an adapter (for ColPali models)
                    if hasattr(rag_model, 'adapter'):
                        logger.info(f"Model has adapter of type {type(rag_model.adapter).__name__} with id {id(rag_model.adapter)}")

                    # Check if the model has an embedding model
                    if hasattr(rag_model, 'embedding_model'):
                        logger.info(f"Model has embedding_model with id {id(rag_model.embedding_model)}")

        # If no existing model, load a new one
        if not rag_model:
            logger.info(f"No cached RAG model found - loading fresh RAG model for {indexer_model} for batch processing")
            
            # Before loading, try a direct import approach to bypass aggressive cleanup
            try:
                from src.models.colpali_adapter import ColPaliAdapter
                logger.info(f"Loading ColPali adapter directly for batch processing: {indexer_model}")
                rag_model = ColPaliAdapter(indexer_model)
                logger.info(f"Successfully loaded RAG model directly with id {id(rag_model)}")
            except Exception as e:
                logger.warning(f"Direct ColPali loading failed: {e}, falling back to load_rag_model")
                # Fallback to original method
                rag_model = load_rag_model(indexer_model)

            # Log the type of model to help with debugging
            if rag_model:
                logger.info(f"Successfully loaded RAG model of type {type(rag_model).__name__} with id {id(rag_model)} for batch processing")

                # Log detailed information about the new model
                if hasattr(rag_model, 'adapter'):
                    logger.info(f"New model has adapter of type {type(rag_model.adapter).__name__} with id {id(rag_model.adapter)}")

                # Check if the model has an embedding model
                if hasattr(rag_model, 'embedding_model'):
                    logger.info(f"New model has embedding_model with id {id(rag_model.embedding_model)}")

            # Store in app.RAG_models for reuse across requests - store in both locations for compatibility
            if 'app' in sys.modules:
                app_module = sys.modules['app']
                if hasattr(app_module, 'RAG_models'):
                    # Store by model name (batch processing storage)
                    app_module.RAG_models.set(indexer_model, rag_model)
                    logger.info(f"Stored RAG model for {indexer_model} in app.RAG_models (by model name)")
                    
                    # Also store by user_id for compatibility with standard RAG
                    app_module.RAG_models.set(session_id, rag_model)
                    logger.info(f"Stored RAG model for user {user_id} in app.RAG_models (by user_id) for standard RAG compatibility")

        if not rag_model:
            logger.error(f"Failed to load RAG model for batch processing in session {session_id}")
            return {
                "success": False,
                "message": "Failed to load RAG model for batch processing",
                "image_paths": []
            }
    except Exception as e:
        logger.error(f"Error loading RAG model for batch processing in session {session_id}: {e}")
        return {
            "success": False,
            "message": f"Error loading RAG model: {str(e)}",
            "image_paths": []
        }

    # Process each document individually
    all_doc_results = []
    all_image_paths = []

    # Create a single retriever for all documents to avoid reloading the embedding model
    # This is more efficient than creating a new retriever for each document
    logger.info(f"Creating a shared retriever for all {len(selected_docs)} documents in batch processing")

    # Pass the already loaded RAG model to the retriever to ensure it uses the same model instance
    # This is critical to avoid reloading the embedding model for each document
    shared_retriever = select_retriever(
        doc_count=len(selected_docs),
        memory_usage_pct=0,
        session_id=session_id,
        query=query,
        force_ocr=use_ocr,
        selected_filenames=selected_docs,
        existing_rag_model=rag_model  # Pass the existing RAG model to reuse it
    )

    if not shared_retriever:
        logger.error(f"Failed to create shared retriever for batch processing")
        return {
            "success": False,
            "message": "Failed to create shared retriever for batch processing",
            "image_paths": []
        }

    # Log the type of retriever to help with debugging
    logger.info(f"Successfully created shared retriever of type {type(shared_retriever).__name__} with id {id(shared_retriever)} for batch processing")

    # Check if the retriever has a reference to the RAG model and log it
    if hasattr(shared_retriever, 'rag_model'):
        logger.info(f"Shared retriever has rag_model with id {id(shared_retriever.rag_model)}")

        # Verify it's the same model instance we loaded earlier
        if id(shared_retriever.rag_model) == id(rag_model):
            logger.info(f"Confirmed shared retriever is using the same RAG model instance")
        else:
            logger.warning(f"Shared retriever is using a different RAG model instance than expected")

    # Check if the retriever has an embedding model and log it
    if hasattr(shared_retriever, 'embedding_model'):
        logger.info(f"Shared retriever has embedding_model with id {id(shared_retriever.embedding_model)}")

    for doc_index, doc_filename in enumerate(selected_docs):
        try:
            logger.info(f"Processing document {doc_filename} ({doc_index+1}/{len(selected_docs)}) in batch for session {session_id}")

            # Use the shared retriever for all documents
            # No need to create a new retriever for each document
            logger.info(f"Using shared retriever for document {doc_filename}")

            # Create a document-specific query that includes the filename
            doc_query = f"[Processing document: {doc_filename}] {query}"

            # Get retrieval count and score-slope settings from session data
            retrieval_count = 3  # Default
            use_score_slope = False
            try:
                retrieval_count = int(session_data.get('retrieval_count', 3))
                use_score_slope = session_data.get('use_score_slope', False)

                # Store the original desired count before any modifications
                desired_result_count = retrieval_count
                
                # If score-slope is enabled, use a fixed pool size for analysis
                # to ensure we have enough documents for the score-slope analysis
                if use_score_slope:
                    # Use a fixed pool size that gives score slope enough data to work with
                    # This should NOT scale with the user's desired count
                    score_slope_pool_size = 10  # Fixed pool for score analysis
                    logger.info(f"Score-slope is enabled - using retrieval pool of {score_slope_pool_size} documents (desired: {desired_result_count})")
                    retrieval_count = score_slope_pool_size
            except Exception as e:
                logger.warning(f"Error getting retrieval settings from session: {e}, using defaults")

            # Retrieve documents directly
            logger.info(f"Retrieving content for document {doc_filename} with retrieval_count={retrieval_count}, use_score_slope={use_score_slope}")

            # Set the embedding adapter on the retriever before calling
            shared_retriever.set_embedding_adapter(rag_model)

            # Check if this is an OCRRAGRetriever
            from src.models.ocr_rag_retriever import OCRRAGRetriever
            if isinstance(shared_retriever, OCRRAGRetriever):
                retrieved_results = shared_retriever.retrieve_documents(
                    doc_query, session_id,
                    k=retrieval_count, selected_filenames=[doc_filename],
                    use_ocr=use_ocr, use_score_slope=use_score_slope
                )
            else:
                # Standard and LanceDB retrievers don't expect use_ocr
                retrieved_results = shared_retriever.retrieve_documents(
                    doc_query, session_id,
                    k=retrieval_count, selected_filenames=[doc_filename],
                    use_score_slope=use_score_slope
                )

            # Process retriever results
            doc_image_paths = []
            ocr_results_to_process = None
            score_analysis_data = None
            actual_item_list = []  # Initialize this variable

            # Check if we got a tuple with analysis data
            if isinstance(retrieved_results, tuple):
                if len(retrieved_results) == 2:
                    # Standard tuple: (results, ocr_data)
                    potential_list = retrieved_results[0]
                    if isinstance(potential_list, list):
                        actual_item_list = potential_list
                    ocr_results_to_process = retrieved_results[1]
                    logger.info(f"Processing {len(actual_item_list)} items from OCR retriever tuple.")
                elif len(retrieved_results) == 3:
                    # Extended tuple with analysis: (results, ocr_data, analysis)
                    potential_list = retrieved_results[0]
                    if isinstance(potential_list, list):
                        actual_item_list = potential_list
                    ocr_results_to_process = retrieved_results[1]
                    score_analysis_data = retrieved_results[2]

                    # Log detailed score-slope analysis if available
                    if score_analysis_data and use_score_slope:
                        logger.info(f"Score-slope analysis for {doc_filename}: Original count: {score_analysis_data.get('original_count', 'N/A')}, " +
                                   f"Filtered count: {score_analysis_data.get('filtered_count', 'N/A')}, " +
                                   f"Cutoff reason: {score_analysis_data.get('cutoff_reason', 'N/A')}")

                        # Log score details if available
                        if 'scores' in score_analysis_data:
                            scores = score_analysis_data['scores']
                            score_details = [f"{i}:{score:.4f}" for i, score in enumerate(scores[:min(10, len(scores))])]
                            logger.info(f"Top scores for {doc_filename}: {', '.join(score_details)}")

                    logger.info(f"Processing {len(actual_item_list)} items from OCR retriever tuple with score analysis.")
                else:
                    logger.warning(f"Unexpected retriever tuple length: {len(retrieved_results)}")
                    actual_item_list = []
            elif isinstance(retrieved_results, list):
                actual_item_list = retrieved_results
                logger.info(f"Processing {len(actual_item_list)} items from retriever list.")
            else:
                logger.warning(f"Unexpected retriever result type: {type(retrieved_results)}")
                actual_item_list = []

            # Log the effect of score-slope analysis if enabled
            if use_score_slope:
                logger.info(f"Score-slope analysis resulted in {len(actual_item_list)} items for document {doc_filename}")

                # If we have score analysis data, log the cutoff details
                if score_analysis_data:
                    cutoff_index = score_analysis_data.get('cutoff_index', 'N/A')
                    cutoff_score = score_analysis_data.get('cutoff_score', 'N/A')
                    cutoff_reason = score_analysis_data.get('cutoff_reason', 'N/A')
                    logger.info(f"Score-slope cutoff at index {cutoff_index} with score {cutoff_score}, reason: {cutoff_reason} for document {doc_filename}")

                    # If we have score analysis data, use it to filter the results
                    # This ensures we respect the score-slope analysis
                    if isinstance(cutoff_index, int) and cutoff_index > 0:
                        # Only apply if we have more items than the cutoff index
                        if len(actual_item_list) > cutoff_index:
                            logger.info(f"Applying score-slope cutoff: {len(actual_item_list)} -> {cutoff_index} items for document {doc_filename}")
                            actual_item_list = actual_item_list[:cutoff_index]
                        else:
                            logger.info(f"No need to apply score-slope cutoff: {len(actual_item_list)} items <= {cutoff_index} for document {doc_filename}")

            # Convert items to full paths and preserve similarity scores
            used_paths = set()
            doc_image_paths_with_scores = []  # Store paths with scores for this document

            for item in actual_item_list:
                image_path = None
                score = 0.0  # Default score

                if isinstance(item, dict):
                    image_path = item.get('path')
                    score = item.get('score', 0.0)  # Get score if available
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

                            # Also add to the global list of image paths
                            if full_path not in all_image_paths:
                                all_image_paths.append(full_path)
                        else:
                            alt_full_path = os.path.join('static', os.path.basename(image_path)).replace('\\', '/')
                            if os.path.exists(alt_full_path):
                                logger.warning(f"Used alternative path {alt_full_path} as {full_path} not found.")
                                # Add to document-specific paths with score
                                doc_image_paths_with_scores.append({
                                    'path': alt_full_path,
                                    'score': score
                                })
                                doc_image_paths.append(alt_full_path)
                                used_paths.add(alt_full_path)

                                # Also add to the global list of image paths
                                if alt_full_path not in all_image_paths:
                                    all_image_paths.append(alt_full_path)
                            else:
                                logger.warning(f"Constructed full image path does not exist: {full_path}")

            logger.info(f"Processed retrieval results into {len(doc_image_paths)} full image paths for document {doc_filename}.")

            # We already got the use_score_slope flag earlier, but log it here for clarity
            logger.info(f"Using score-slope flag for model handler: {use_score_slope} for document {doc_filename}")

            # Generate response using the model
            logger.info(f"Calling model_generate_response with {len(doc_image_paths)} image paths for document {doc_filename}.")
            response_text, _ = model_generate_response(
                doc_image_paths,
                doc_query,
                session_id,
                generation_model,
                empty_chat_history,
                direct_ocr_results=ocr_results_to_process,
                original_query=doc_query,
                use_ocr=use_ocr,
                is_pasted_images=False,
                user_id=user_id,
                use_score_slope=use_score_slope
            )

            # Store the document result with image paths and scores
            doc_result = {
                "filename": doc_filename,
                "response": response_text,
                "success": True,
                "image_paths": doc_image_paths,
                "image_paths_with_scores": doc_image_paths_with_scores  # Include paths with scores
            }

            # Add to results
            results["individual_results"].append(doc_result)
            all_doc_results.append(doc_result)

            logger.info(f"Successfully processed document {doc_filename} in batch for session {session_id}")

            # Light memory cleanup after each document (preserve models for next document)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info(f"Performed light memory cleanup after processing {doc_filename}")

            # Small delay to allow memory to be released
            time.sleep(1)

        except Exception as e:
            logger.error(f"Error processing document {doc_filename} in batch for session {session_id}: {e}")
            results["individual_results"].append({
                "filename": doc_filename,
                "success": False,
                "error": str(e)
            })

    # Store all image paths in the results
    results["image_paths"] = all_image_paths

    # Save the image paths with scores to the session data for persistence
    try:
        from src.services.session_manager.manager import load_session, save_session
        session_data = load_session('sessions', session_id)
        if session_data:
            # Initialize score_analysis if it doesn't exist
            if 'score_analysis' not in session_data:
                session_data['score_analysis'] = {}

            # Add all image paths with scores to the session data
            for doc_result in all_doc_results:
                if doc_result.get("success", False) and "image_paths_with_scores" in doc_result:
                    for img_data in doc_result["image_paths_with_scores"]:
                        path = img_data.get("path")
                        score = img_data.get("score", 0.0)
                        if path and score > 0:
                            # Store both the full path and the filename as keys
                            session_data['score_analysis'][path] = score
                            # Also store by filename for easier matching
                            filename = os.path.basename(path)
                            session_data['score_analysis'][filename] = score

            # Note: No need to store special batch processing structure since we now use standard format
            # The images will be stored in the regular chat history with the standard relative_images format
            # This allows the standard template rendering and JavaScript to handle batch processing correctly

            # Save the updated session data
            save_success = save_session('sessions', session_id, session_data)
            if save_success:
                logger.info(f"Saved {len(session_data['score_analysis'])} image scores to session data for persistence")
                logger.info(f"Batch processing will use standard format through regular chat history")
            else:
                logger.warning(f"Failed to save image scores to session data")
    except Exception as e:
        logger.error(f"Error saving image scores to session data: {e}")

    # Format separate responses for each document
    try:
        logger.info(f"Formatting separate responses for all {len(selected_docs)} documents in session {session_id}")

        # Light memory cleanup before response formatting (preserve models)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Performed light memory cleanup before response formatting")

        # Format each document response with a header
        separate_responses = []

        for i, doc_result in enumerate(all_doc_results):
            if doc_result.get("success", False):
                doc_name = doc_result['filename']
                doc_response = doc_result['response']

                # Format the response with a header indicating the document name
                formatted_response = f"## Document: {doc_name}\n\n{doc_response}"

                # Create a response object with the document info and response
                response_obj = {
                    "filename": doc_name,
                    "response": formatted_response,
                    "image_paths": doc_result.get('image_paths', []),
                    "image_paths_with_scores": doc_result.get('image_paths_with_scores', []),  # Include paths with scores
                    "success": True
                }

                separate_responses.append(response_obj)
                logger.info(f"Formatted response for document {doc_name}")

        # Store the separate responses in the results
        results["separate_responses"] = separate_responses
        logger.info(f"Successfully formatted {len(separate_responses)} separate responses for session {session_id}")

    except Exception as e:
        logger.error(f"Error formatting separate responses for session {session_id}: {e}")

        # Create simple separate responses from individual results
        separate_responses = []

        for i, doc_result in enumerate(all_doc_results):
            if doc_result.get("success", False):
                doc_name = doc_result['filename']
                doc_response = doc_result['response']

                # Format the response with a header indicating the document name
                formatted_response = f"## Document: {doc_name}\n\n{doc_response}"

                # Create a response object with the document info and response
                response_obj = {
                    "filename": doc_name,
                    "response": formatted_response,
                    "image_paths": doc_result.get('image_paths', []),
                    "image_paths_with_scores": doc_result.get('image_paths_with_scores', []),  # Include paths with scores
                    "success": True
                }

                separate_responses.append(response_obj)

        results["separate_responses"] = separate_responses
        results["success"] = False
        results["message"] = f"Error formatting separate responses: {str(e)}"

    # Final memory cleanup after all batch processing is complete
    # Now it's safe to clear models since we're done with all documents
    try:
        # Run aggressive cleanup to free memory
        if memory_manager:
            memory_manager.aggressive_cleanup()

        # Also try to clear model caches from model_loader
        try:
            from src.models.model_loader import clear_model_caches
            clear_model_caches(force_gpu_cleanup=True)
            logger.info(f"Cleared model caches after batch processing for session {session_id}")
        except Exception as e:
            logger.warning(f"Error clearing model caches: {e}")

        # Try to release memory back to the OS on Linux
        try:
            if os.name == 'posix':
                import ctypes
                libc = ctypes.CDLL('libc.so.6')
                # Return memory to the OS
                libc.malloc_trim(0)
                logger.info("System malloc_trim executed")
        except Exception as e:
            logger.warning(f"System malloc_trim failed: {e}")

        logger.info(f"Completed final memory cleanup after batch processing for session {session_id}")
    except Exception as e:
        logger.warning(f"Error during final memory cleanup: {e}")

    # Make sure we return the image paths for display
    if not results.get("image_paths") and all_image_paths:
        results["image_paths"] = all_image_paths

    return results
