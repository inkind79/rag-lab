"""
OCR-Enhanced RAG Lab Retriever

This module implements an OCR-enhanced retriever that combines visual retrieval with OCR text extraction.
It inherits from the BaseRetriever interface.
"""

import os
import shutil
import time
import threading
import imghdr
from typing import List, Dict, Any, Optional, Tuple, Union

from src.models.retriever_base import BaseRetriever
from src.models.rag_retriever import rag_retriever
from src.models.ocr.ocr_processor import extract_text_from_image, analyze_image_content
from src.models.ocr.ocr_cache import OCRCache
from src.utils.logger import get_logger

# Import centralized OCR utilities
from src.utils.ocr_utils import process_images_with_ocr, get_latest_ocr_results, format_ocr_context

try:
    from pdf2image import convert_from_path
except ImportError:
    logger = get_logger(__name__)
    logger.warning("pdf2image not found. Install with 'pip install pdf2image' and ensure poppler is installed.")
    convert_from_path = None

logger = get_logger(__name__)
# ocr_cache = OCRCache.get_instance() # Removed - Unused global instance

class OCRRAGRetriever(BaseRetriever):
    """
    OCR-enhanced retriever implementation that combines visual retrieval with OCR text extraction.
    Can be configured to use different base retrievers (Standard, LanceDB, Qdrant).
    """
    def __init__(self):
        """Initialize the OCRRAGRetriever."""
        # Default to the standard retriever, can be overridden by set_base_retriever
        self._base_retriever = rag_retriever
        # Initialize session-specific storage as a dict to store results by session
        self._last_rag_results = {}
        # Initialize embedding adapter to None
        self._embedding_adapter = None
        logger.info(f"OCRRAGRetriever initialized, default base retriever: {type(self._base_retriever).__name__}")

    def set_base_retriever(self, base_retriever: BaseRetriever):
        """Sets the base retriever instance to use for visual search."""
        if base_retriever:
            self._base_retriever = base_retriever
            logger.info(f"OCRRAGRetriever base retriever set to: {type(self._base_retriever).__name__}")
        else:
            logger.warning("Attempted to set None as base retriever, keeping default.")

    def set_embedding_adapter(self, adapter):
        """
        Set the embedding adapter to use for retrieval.

        Propagates the adapter to the base retriever as well.

        Args:
            adapter: The embedding adapter instance (e.g., ColPaliAdapter)
        """
        self._embedding_adapter = adapter
        logger.info(f"OCRRAGRetriever: Set embedding adapter with id {id(adapter)}")

        # Also set the adapter in the base retriever if it supports it
        if hasattr(self._base_retriever, 'set_embedding_adapter'):
            self._base_retriever.set_embedding_adapter(adapter)
            logger.info(f"OCRRAGRetriever: Also set embedding adapter in base retriever")

    # Backward compatibility aliases
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
        Delegates to the standard retriever's implementation.
        """
        return rag_retriever.expand_query(query, chat_history, session_id)

    def _process_ocr_background(self, img_paths: List[str], session_id: str, query: str) -> None:
        """
        Process OCR in background thread using the centralized utility.

        Args:
            img_paths: List of image paths to process
            session_id: Current session ID
            query: User query text for OCR context
        """
        # Add broad exception handling within the thread
        try:
            logger.info(f"Starting background OCR processing for {len(img_paths)} images using utility")

            # Use the centralized OCR utility
            process_images_with_ocr(
                image_paths=[os.path.join('static', p) if not p.startswith('static/') else p for p in img_paths],
                session_id=session_id,
                query=query,
                save_results=True,
                results_prefix="hybrid",
                pre_formatted=True
            )

            logger.info("Completed background OCR processing thread")
        except Exception as e:
            # Catch and log errors within the thread to prevent crashes
            logger.error(f"Error during background OCR processing thread for session {session_id}: {e}", exc_info=True)
            # Optionally, add more specific error handling if needed

    def _process_ocr_sync(self, img_paths: List[str], session_id: str, query: str) -> Optional[Dict[str, Any]]:
        """
        Process OCR synchronously using the centralized utility.

        Args:
            img_paths: List of image paths to process
            session_id: Current session ID
            query: User query text for OCR context

        Returns:
            OCR results dictionary or None if processing fails
        """
        try:
            logger.info(f"Starting synchronous OCR processing for {len(img_paths)} images using utility")

            # Use the centralized OCR utility
            results = process_images_with_ocr(
                image_paths=[os.path.join('static', p) if not p.startswith('static/') else p for p in img_paths],
                session_id=session_id,
                query=query,
                save_results=True,
                results_prefix="hybrid",
                pre_formatted=True
            )

            logger.info(f"Completed OCR processing for {len(results.get('results', []))} images synchronously")
            return results
        except Exception as e:
            logger.error(f"Error in synchronous OCR processing: {e}", exc_info=True)
            return None

    def retrieve_documents(
        self,
        query: str,
        session_id: str,
        k: int = 3,
        selected_filenames: Optional[List[str]] = None,
        **kwargs
    ) -> Union[List[Dict[str, Any]], Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]]:
        """
        Enhanced document retrieval that combines visual retrieval with OCR text extraction.

        Args:
            query: User query text
            session_id: Current session ID
            k: Number of documents to retrieve
            selected_filenames: List of selected filenames to potentially add to results
            **kwargs: Additional parameters:
                - sync_mode: bool, process OCR synchronously (default False)
                - direct_results: Optional pre-loaded OCR results
                - use_ocr: bool, whether to use OCR text extraction (default True)
                - use_score_slope: Optional[bool] to override session setting
                - RAG: Legacy parameter for backward compatibility (deprecated)

        Returns:
            If sync_mode=False: List of relative image paths
            If sync_mode=True: Tuple of (List of image paths, OCR results dictionary)
        """
        sync_mode = kwargs.get('sync_mode', False)
        direct_results = kwargs.get('direct_results', None)
        use_ocr = kwargs.get('use_ocr', True)
        use_score_slope = kwargs.get('use_score_slope', None)
        # Accept legacy RAG kwarg for backward compat
        legacy_rag = kwargs.get('RAG', None)
        RAG = self._embedding_adapter or legacy_rag
        logger.info("=" * 80)
        logger.info(f"HYBRID RETRIEVER CALLED with use_ocr={use_ocr}, use_score_slope={use_score_slope}")
        logger.info(f"sync_mode={sync_mode}, selected_docs={selected_filenames}, k={k}")
        logger.info(f"query: {query}")
        logger.info(f"direct_results provided: {direct_results is not None}")
        logger.info("=" * 80)

        # Initialize variables
        # Stores dicts: {'path': rel_path, 'original_filename': fname, 'score': score}
        rag_results_data = []
        ocr_results_to_process = None
        # Stores final relative paths for OCR/return
        final_image_paths = []

        # --- Step 1: Perform Main Retrieval (Standard RAG Search) ---
        if direct_results:
            logger.info(f"Using direct OCR results instead of standard retrieval: {len(direct_results.get('results', []))} results")
            ocr_results_to_process = direct_results
            # Extract paths and potentially original filenames if available in OCR results metadata
            temp_paths = set()
            for result in direct_results.get('results', []):
                rel_path = result.get('image_path')
                if not rel_path and 'full_path' in result:
                    full_p = result['full_path']; rel_path = full_p[7:] if full_p.startswith('static/') else full_p
                if rel_path and rel_path not in temp_paths:
                     # Attempt to find original filename if stored in OCR results (less likely)
                     original_fname = result.get('metadata', {}).get('filename', os.path.basename(rel_path)) # Fallback

                     # Try to get the original score from the result metadata
                     score = result.get('metadata', {}).get('score', 0.0)

                     # If no score in metadata, check if we have a score in the session data
                     if score == 0.0 and session_id:
                         from src.services.session_manager.manager import load_session
                         session_data = load_session('sessions', session_id)
                         if session_data and 'score_analysis' in session_data:
                             # Try to find the score by path or filename
                             score_analysis = session_data['score_analysis']
                             full_path = os.path.join('static', rel_path).replace('\\', '/')
                             if full_path in score_analysis:
                                 score = score_analysis[full_path]
                                 logger.info(f"Found score {score} for {rel_path} in session data")
                             else:
                                 # Try to find by filename
                                 filename = os.path.basename(rel_path)
                                 for key, value in score_analysis.items():
                                     if os.path.basename(key) == filename:
                                         score = value
                                         logger.info(f"Found score {score} for {filename} by matching filename in session data")
                                         break

                     # If still no score, use position-based fallback
                     if score == 0.0:
                         position = len(rag_results_data)
                         score = max(0.5, 1.0 - (position * 0.05))
                         logger.info(f"Applied fallback score {score} for {rel_path} at position {position}")

                     rag_results_data.append({'path': rel_path, 'original_filename': original_fname, 'score': score})
                     temp_paths.add(rel_path)
            logger.info(f"Extracted {len(rag_results_data)} items from direct OCR results")

        elif RAG is not None:
            try:
                start_time = time.time()
                # Use the configured base retriever implementation
                logger.info(f"Using base retriever: {type(self._base_retriever).__name__}")

                # Always use LanceDB for retrieval to eliminate in-memory storage
                # This is the whole point of our LanceDB integration
                logger.info("Using LanceDB for retrieval to eliminate in-memory storage")

                # Get the user's retrieval count setting from the session data
                from src.services.session_manager.manager import load_session
                session_data = load_session('sessions', session_id)
                user_retrieval_count = k  # Default to the provided k value

                if session_data and 'retrieval_count' in session_data:
                    user_retrieval_count = int(session_data.get('retrieval_count', k))
                    logger.info(f"Using user's retrieval count setting: {user_retrieval_count}")

                # Get the use_score_slope setting from the parameter or session data
                if use_score_slope is None:
                    # If not provided, use the session setting
                    effective_use_score_slope = session_data.get('use_score_slope', True)
                    logger.info(f"Using session setting for use_score_slope={effective_use_score_slope}")
                else:
                    # Use the provided parameter
                    effective_use_score_slope = use_score_slope
                    logger.info(f"Using provided parameter for use_score_slope={effective_use_score_slope}")

                logger.info(f"Using retrieval limit of {user_retrieval_count} documents for query: '{query}' with use_score_slope={effective_use_score_slope}")

                # Pass the use_score_slope setting to the base retriever
                # This ensures that the base retriever respects the adaptive setting

                # Ensure the base retriever has the embedding adapter set
                if RAG is not None and hasattr(self._base_retriever, 'set_embedding_adapter'):
                    self._base_retriever.set_embedding_adapter(RAG)

                rag_results_data = self._base_retriever.retrieve_documents(
                    query,
                    session_id,
                    k=user_retrieval_count,
                    selected_filenames=selected_filenames,
                    use_score_slope=effective_use_score_slope
                )

                # Store the RAG results data per session for later use
                # This is used by the response generator to preserve scores for text-only models
                self._last_rag_results[session_id] = rag_results_data

                retrieval_time = time.time() - start_time
                logger.info(f"Base RAG retrieval ({type(self._base_retriever).__name__}) returned {len(rag_results_data)} items in {retrieval_time:.2f}s")
                logger.debug(f"RAG results data: {rag_results_data}")
            except Exception as e:
                logger.error(f"Error during standard RAG retrieval: {e}", exc_info=True)
                rag_results_data = []
        else:
            logger.warning("No RAG model provided and no direct_results, cannot perform retrieval.")
            rag_results_data = []

        # Ensure results are a list of dictionaries
        if not isinstance(rag_results_data, list): rag_results_data = []
        if not all(isinstance(item, dict) for item in rag_results_data):
             logger.warning("Retriever did not return list of dicts, attempting to recover paths.")
             # Fallback: try to extract paths if it returned strings
             rag_results_data = [{'path': p, 'original_filename': os.path.basename(p), 'score': 0.0} for p in rag_results_data if isinstance(p, str)]


        # --- Step 2: Use ONLY RAG Results for OCR Context ---
        # The logic to force inclusion of all selected documents is removed
        # to prevent context contamination. OCR should only run on query-relevant results.

        # Get the session data if not already loaded
        if 'session_data' not in locals() or session_data is None:
            from src.services.session_manager.manager import load_session
            session_data = load_session('sessions', session_id)

        # Check if we should use adaptive (score-slope) or fixed retrieval count
        # Use the provided use_score_slope parameter if it's not None, otherwise use the session setting
        if use_score_slope is None:
            # If not provided, use the session setting
            effective_use_score_slope = session_data.get('use_score_slope', True)
            logger.info(f"OCR processing using session setting for use_score_slope={effective_use_score_slope}")
        else:
            # Use the provided parameter
            effective_use_score_slope = use_score_slope
            logger.info(f"OCR processing using provided parameter for use_score_slope={effective_use_score_slope}")

        logger.info(f"OCR processing with use_score_slope={effective_use_score_slope}")

        if effective_use_score_slope:
            # When adaptive is enabled, use all the results from the base retriever
            # The base retriever has already applied score-slope analysis
            final_results_data = list(rag_results_data)
            logger.info(f"Using all {len(final_results_data)} items from RAG results for OCR/return (adaptive mode enabled).")

            # Log the paths for debugging
            paths_debug = [item['path'] for item in final_results_data]
            logger.info(f"Paths for OCR/return (adaptive mode): {paths_debug}")
        else:
            # When adaptive is disabled, limit to the user's retrieval count setting
            user_retrieval_count = k  # Default to the provided k value
            if session_data and 'retrieval_count' in session_data:
                user_retrieval_count = int(session_data.get('retrieval_count', k))
                logger.info(f"Using user's retrieval count setting for OCR: {user_retrieval_count}")

            # Limit the results to the user's retrieval count setting
            final_results_data = list(rag_results_data[:user_retrieval_count])
            logger.info(f"Using {len(final_results_data)} items directly from RAG results for OCR/return (limited to user's retrieval count: {user_retrieval_count}).")

            # Log the paths for debugging
            paths_debug = [item['path'] for item in final_results_data]
            logger.info(f"Paths for OCR/return: {paths_debug}")

        # Extract final list of relative paths for OCR processing or return
        # Maintain order (RAG results first, then directly added ones)
        final_image_paths = [item['path'] for item in final_results_data]
        logger.debug(f"Final relative image paths: {final_image_paths}")


        # --- Step 3: Perform OCR (if enabled) ---
        if not use_ocr:
            logger.info("OCR is disabled by setting. Returning only paths from RAG search.")
            # Return only the paths from the final results data
            # The final_results_data already respects the adaptive setting or user's retrieval count
            # Return the full result data with scores, not just paths
            if effective_use_score_slope:
                logger.info(f"Returning {len(final_results_data)} items with OCR disabled (adaptive mode enabled)")
            else:
                logger.info(f"Returning {len(final_results_data)} items with OCR disabled (limited to user's retrieval count)")
            return final_results_data if not sync_mode else (final_results_data, None)

        if not final_image_paths:
            logger.info("No images retrieved or selected, skipping OCR.")
            return [] if not sync_mode else ([], None)

        # Perform OCR processing using the final list (RAG results + added missing selected files)
        if sync_mode:
            # In sync mode, we need to return the OCR results immediately
            logger.info(f"Processing OCR synchronously for {len(final_image_paths)} final images.")
            ocr_results = self._process_ocr_sync(final_image_paths, session_id, query)
            # Return the full results data with scores AND the OCR results
            return final_results_data, ocr_results
        else:
            # In async mode, we don't need to start a background thread
            # The OCR will be processed synchronously by the handler when needed
            # This avoids redundant OCR processing
            logger.info(f"Skipping background OCR processing for {len(final_image_paths)} final images. Will be processed by handler when needed.")
            # Return the full results data with scores, not just paths
            return final_results_data

    def get_latest_ocr_results(self, session_id: str, prefix: str = "hybrid") -> Dict[str, Any]:
        """
        Get the latest OCR results for the session.
        Delegates to the centralized OCR utility.

        Args:
            session_id: Current session ID
            prefix: Prefix used when saving the results

        Returns:
            OCR results dictionary
        """
        return get_latest_ocr_results(session_id, prefix)

    def format_ocr_context(self, ocr_results: Dict[str, Any], max_length: int = 8000) -> str:
        """
        Format OCR results as context for the LLM.
        Delegates to the centralized OCR utility.

        Args:
            ocr_results: OCR results dictionary
            max_length: Maximum length of the context

        Returns:
            Formatted OCR results as a string
        """
        return format_ocr_context(ocr_results, max_length)
    
    def get_last_rag_results(self, session_id: str) -> Optional[List[Dict[str, Any]]]:
        """Get the last RAG results for a specific session.
        
        Args:
            session_id: The session ID to retrieve results for
            
        Returns:
            The last RAG results for the session, or None if not found
        """
        return self._last_rag_results.get(session_id)
    
    def clear_session_results(self, session_id: str) -> None:
        """Clear the stored RAG results for a specific session.
        
        Args:
            session_id: The session ID to clear results for
        """
        if session_id in self._last_rag_results:
            del self._last_rag_results[session_id]
            logger.info(f"Cleared RAG results for session {session_id}")

# Create a singleton instance for convenience
ocr_rag_retriever = OCRRAGRetriever()