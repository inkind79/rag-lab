"""
OCR utilities for processing images across different application flows.

This module provides centralized functions for OCR processing that are used
in multiple places including the OCR retriever and response generator.
"""
import os
import logging
import time
import json
import threading
from typing import List, Dict, Any, Optional, Tuple, Union

from src.utils.logger import get_logger

logger = get_logger(__name__)

def _process_ocr_in_thread(
    image_paths: List[str],
    session_id: str,
    query: Optional[str] = None,
    save_results: bool = True,
    results_prefix: str = "",
    pre_formatted: bool = False
):
    """
    Process OCR in a background thread.
    This function is called by process_images_with_ocr when wait_for_results=False.
    """
    try:
        logger.info(f"Background thread: Processing OCR for {len(image_paths)} images")
        # Call the main function with wait_for_results=True to process synchronously within the thread
        process_images_with_ocr(
            image_paths=image_paths,
            session_id=session_id,
            query=query,
            save_results=save_results,
            results_prefix=results_prefix,
            pre_formatted=pre_formatted,
            wait_for_results=True  # Force synchronous processing within the thread
        )
        logger.info(f"Background thread: Completed OCR processing for {len(image_paths)} images")
    except Exception as e:
        logger.error(f"Error in background OCR processing thread: {e}", exc_info=True)

def process_images_with_ocr(
    image_paths: List[str],
    session_id: str,
    query: Optional[str] = None,
    save_results: bool = True,
    results_prefix: str = "",
    pre_formatted: bool = False,
    wait_for_results: bool = False,
    timeout: int = 1800  # Increased to 1800 seconds (30 minutes) for SmolDocling
) -> Dict[str, Any]:
    """
    Process a list of images with OCR, handling caching and formatting.

    This function centralizes OCR processing across different application flows
    (RAG retrieval, pasted images, etc.) to reduce code duplication.

    Args:
        image_paths: List of full image paths to process
        session_id: Current session ID for cache directory
        query: Optional query text for context/storage
        save_results: Whether to save results to file
        results_prefix: Optional prefix for the results filename
        pre_formatted: Whether to return pre-formatted context string
        wait_for_results: If True, process OCR synchronously; if False, use background thread
        timeout: Maximum time in seconds to wait for OCR results when wait_for_results=True

    Returns:
        Dictionary containing OCR results and metadata
    """
    # If not waiting for results, process in background thread and return empty results
    if not wait_for_results:
        # Create a background thread for OCR processing
        import threading
        thread = threading.Thread(
            target=_process_ocr_in_thread,
            args=(image_paths, session_id, query, save_results, results_prefix, pre_formatted)
        )
        thread.daemon = True
        thread.start()
        logger.info(f"Started background OCR processing for {len(image_paths)} images")
        # Return empty results immediately
        return {'results': [], 'processing': 'background', 'timestamp': time.time()}

    # If waiting for results, process synchronously with timeout
    try:
        # Import OCR components dynamically to avoid circular imports
        from src.models.ocr.ocr_processor import extract_text_from_image, analyze_image_content
        from src.models.ocr.ocr_cache import OCRCache

        # Set up cache directory
        cache_dir = os.path.join('sessions', 'ocr', session_id)
        ocr_cache = OCRCache(cache_dir)

        # Set timeout for OCR processing
        start_time = time.time()
        max_end_time = start_time + timeout

        # Process each image
        ocr_results = []

        for img_path in image_paths:
            # Check if we've exceeded the timeout
            if time.time() > max_end_time:
                logger.warning(f"OCR processing timeout reached after {timeout} seconds. Processed {len(ocr_results)}/{len(image_paths)} images.")
                break

            if not os.path.exists(img_path):
                logger.warning(f"OCR: Image file not found: {img_path}")
                continue

            # Check cache first
            cached_data = ocr_cache.get(img_path)
            ocr_text = ""
            analysis = None

            if cached_data and 'text' in cached_data:
                ocr_text = cached_data['text']
                analysis = cached_data.get('analysis')
                logger.info(f"Using cached OCR for image: {os.path.basename(img_path)}")
            else:
                try:
                    # Analyze image content (if supported)
                    try:
                        analysis = analyze_image_content(img_path)
                    except Exception as e_analysis:
                        logger.warning(f"Error analyzing image: {e_analysis}")
                        analysis = {'text_heavy': False, 'text_ratio': 0}

                    # Get OCR engine from session data
                    ocr_engine = 'easyocr'  # Default engine
                    try:
                        from src.services.session_manager.manager import load_session
                        session_data = load_session('sessions', session_id)
                        if session_data and 'ocr_engine' in session_data:
                            ocr_engine = session_data.get('ocr_engine', 'easyocr')
                            logger.info(f"Using OCR engine from session settings: {ocr_engine}")
                    except Exception as e_engine:
                        logger.warning(f"Error getting OCR settings from session data: {e_engine}")

                    # Extract text with the selected OCR engine
                    logger.info(f"Calling extract_text_from_image with engine: {ocr_engine} for image: {img_path}")
                    if ocr_engine == 'smoldocling':
                        # Use SmolDocling with default settings (full document, markdown output)
                        ocr_text = extract_text_from_image(img_path, ocr_engine=ocr_engine, output_format="markdown")
                        logger.info(f"OCR extraction complete with SmolDocling (markdown). Text length: {len(ocr_text) if ocr_text else 0}")
                    elif ocr_engine == 'smoldocling-otsl':
                        # Use SmolDocling with OTSL output format (optimized for forms and tables)
                        ocr_text = extract_text_from_image(img_path, ocr_engine=ocr_engine)
                        logger.info(f"OCR extraction complete with SmolDocling (OTSL). Text length: {len(ocr_text) if ocr_text else 0}")
                    else:
                        # Use EasyOCR
                        ocr_text = extract_text_from_image(img_path, ocr_engine=ocr_engine)
                        logger.info(f"OCR extraction complete. Text length: {len(ocr_text) if ocr_text else 0}")

                    # Cache the result if there's text or analysis
                    if ocr_text or analysis:
                        ocr_cache.set(img_path, ocr_text, analysis)
                        logger.info(f"Performed OCR on image: {os.path.basename(img_path)}")
                    else:
                        logger.warning(f"OCR returned no text for image: {os.path.basename(img_path)}")
                except Exception as ocr_err:
                    logger.error(f"Error during OCR extraction for {img_path}: {ocr_err}")

            # Only add results if we have text
            if ocr_text:
                # Convert to relative path for storage if it's in the static directory
                rel_path = img_path
                if img_path.startswith('static/'):
                    rel_path = img_path
                else:
                    try:
                        rel_path = os.path.relpath(img_path, 'static')
                    except ValueError:
                        # If relpath fails, just use the basename
                        rel_path = os.path.basename(img_path)

                # Try to get the score for this image from the session data
                score = 0.0
                try:
                    # Get the session data to check for scores
                    from src.services.session_manager.manager import load_session
                    session_data = load_session('sessions', session_id)
                    if session_data and 'score_analysis' in session_data:
                        # Try to find the score by path
                        score_analysis = session_data['score_analysis']
                        full_path_key = img_path.replace('\\', '/')
                        if full_path_key in score_analysis:
                            score = score_analysis[full_path_key]
                            logger.info(f"Found score {score} for {os.path.basename(img_path)} in session data")
                        else:
                            # Try to find by filename
                            filename = os.path.basename(img_path)
                            for key, value in score_analysis.items():
                                if os.path.basename(key) == filename:
                                    score = value
                                    logger.info(f"Found score {score} for {filename} by matching filename in session data")
                                    break
                except Exception as e:
                    logger.warning(f"Error getting score from session data: {e}")

                # Create metadata with score
                metadata = {
                    'filename': os.path.basename(img_path),
                    'score': score
                }

                ocr_results.append({
                    'image_path': rel_path,
                    'full_path': img_path,
                    'text': ocr_text,
                    'analysis': analysis,
                    'metadata': metadata
                })

        # Prepare results dictionary
        end_time = time.time()
        processing_time = end_time - start_time

        results_dict = {
            'query': query,
            'timestamp': end_time,
            'results': ocr_results,
            'processing_time': processing_time
        }

        logger.info(f"OCR processing completed in {processing_time:.2f}s for {len(ocr_results)}/{len(image_paths)} images")

        # Format context string if requested
        if pre_formatted:
            context = format_ocr_context(results_dict)
            results_dict['ocr_context'] = context

        # Save results if requested
        if save_results:
            # Use our utility function to get the session directory
            session_dir = get_ocr_session_dir(session_id)

            # Generate a unique filename
            import uuid
            results_uuid = uuid.uuid4().hex[:8]
            prefix = f"{results_prefix}_" if results_prefix else ""
            ocr_file = os.path.join(session_dir, f"{prefix}ocr_{results_uuid}.json")

            try:
                with open(ocr_file, 'w') as f:
                    json.dump(results_dict, f)
                logger.info(f"Saved OCR results to {ocr_file}: {len(ocr_results)} images processed")

                # Clean up excess files after adding a new one
                # This ensures we don't accumulate too many files per session
                _clean_excess_ocr_files(session_id)
            except Exception as e:
                logger.error(f"Error saving OCR results: {e}")

        logger.info(f"Processed OCR for {len(ocr_results)} images")

        # No explicit unload needed for EasyOCR model cache currently
        # (EasyOCR manages its own model loading/unloading implicitly)
        pass

        return results_dict

    except ImportError as e_import:
        logger.error(f"OCR modules not found: {e_import}")
        return {'results': [], 'error': str(e_import)}
    except Exception as e:
        logger.error(f"Error in OCR processing: {e}", exc_info=True)
        return {'results': [], 'error': str(e)}

def format_ocr_context(ocr_results: Dict[str, Any], max_chars: int = 5000) -> str:
    """
    Format OCR results into a context string for the model prompt.

    Args:
        ocr_results: OCR results dictionary with 'results' key containing OCR entries
        max_chars: Maximum characters to include in context

    Returns:
        Formatted context string
    """
    if not ocr_results or 'results' not in ocr_results:
        return ""

    context = "OCR TEXT EXTRACTED FROM DOCUMENTS:\n\n"
    char_count = len(context)
    processed_count = 0

    for idx, result in enumerate(ocr_results.get('results', [])):
        text = result.get('text', '').strip()
        if not text:
            continue

        # Format document header
        image_path = result.get('image_path', f"Document {idx+1}")
        doc_header = f"Document {idx+1} ({os.path.basename(image_path)}):\n"
        doc_text = f"{doc_header}{text}\n\n"

        # Check if adding this document would exceed the max_chars limit
        if char_count + len(doc_text) > max_chars:
            # Try to add partial content
            remaining_chars = max_chars - char_count - len("\n[... OCR text truncated ...]")
            if remaining_chars > len(doc_header) + 50:
                # Add header and some content
                context += doc_header
                context += text[:remaining_chars - len(doc_header)] + "\n[... OCR text truncated ...]"
                logger.warning(f"Truncated OCR text for document {idx+1} due to max_chars limit.")
            elif not context.endswith("\n[... OCR text truncated ...]"):
                # Just add truncation notice
                context += "\n[... OCR text truncated ...]"
            char_count = max_chars
            break

        # Add the full document
        context += doc_text
        char_count += len(doc_text)
        processed_count += 1

    logger.info(f"Formatted OCR context with {char_count} characters from {processed_count} documents (limit: {max_chars})")
    return context.strip()

# OCR Session Management Constants
MAX_SESSION_OCR_FILES = 10  # Maximum number of OCR files to keep per session

def get_ocr_session_dir(session_id: str, create_if_missing: bool = True) -> str:
    """
    Get the path to a session's OCR directory.

    Args:
        session_id: Session identifier
        create_if_missing: If True, create the directory if it doesn't exist

    Returns:
        Absolute path to the session's OCR directory
    """
    session_ocr_dir = os.path.join('sessions', 'ocr', session_id)
    if create_if_missing:
        os.makedirs(session_ocr_dir, exist_ok=True)
    return os.path.abspath(session_ocr_dir).replace('\\', '/')

def _clean_excess_ocr_files(session_id: str) -> int:
    """
    Clean up excess OCR files for a session to respect the MAX_SESSION_OCR_FILES limit.

    Args:
        session_id: Session ID to clean up

    Returns:
        Number of files removed
    """
    session_ocr_dir = get_ocr_session_dir(session_id, create_if_missing=False)
    if not os.path.exists(session_ocr_dir):
        return 0

    try:
        session_files = os.listdir(session_ocr_dir)
        ocr_files = [f for f in session_files if f.endswith('.json') and 'ocr' in f]

        if len(ocr_files) <= MAX_SESSION_OCR_FILES:
            return 0  # No cleanup needed

        # Process files and get their timestamps
        files_with_timestamps = []
        for filename in ocr_files:
            file_path = os.path.join(session_ocr_dir, filename)
            try:
                # Try to get timestamp from the file content first
                with open(file_path, 'r') as f:
                    try:
                        data = json.load(f)
                        timestamp = data.get('timestamp', 0)
                    except:
                        # If file isn't valid JSON or has no timestamp, use file modification time
                        timestamp = os.path.getmtime(file_path)

                files_with_timestamps.append((timestamp, filename, file_path))
            except Exception as e:
                logger.warning(f"Error reading OCR file metadata {file_path}: {e}")
                # Use file modification time as fallback
                try:
                    timestamp = os.path.getmtime(file_path)
                    files_with_timestamps.append((timestamp, filename, file_path))
                except:
                    pass  # Skip this file if we can't get any timestamp

        # Sort by timestamp (newest first)
        files_with_timestamps.sort(key=lambda x: x[0], reverse=True)

        # Keep only the most recent files up to the maximum
        files_to_keep = files_with_timestamps[:MAX_SESSION_OCR_FILES]
        files_to_remove = files_with_timestamps[MAX_SESSION_OCR_FILES:]

        # Remove excess files
        removed_count = 0
        for _, _, file_path in files_to_remove:
            try:
                os.remove(file_path)
                removed_count += 1
                logger.info(f"Removed excess OCR file: {os.path.basename(file_path)}")
            except Exception as e:
                logger.warning(f"Failed to remove excess OCR file {file_path}: {e}")

        return removed_count

    except Exception as e:
        logger.error(f"Error cleaning excess OCR files for session {session_id}: {e}")
        return 0

def get_latest_ocr_results(session_id: str, query: Optional[str] = None, clean_excess: bool = True) -> Optional[Dict[str, Any]]:
    """
    Get the latest OCR results for a session and optionally clean excess files.

    Args:
        session_id: Session ID to get results for
        query: Optional query text to match for exact results
        clean_excess: If True, remove excess OCR files beyond the maximum count

    Returns:
        OCR results dictionary or None if not found
    """
    session_ocr_dir = get_ocr_session_dir(session_id, create_if_missing=False)
    if not os.path.exists(session_ocr_dir):
        logger.info(f"No OCR directory found for session {session_id}")
        return None

    try:
        session_files = os.listdir(session_ocr_dir)
        ocr_files = [f for f in session_files if f.endswith('.json') and 'ocr' in f]

        if not ocr_files:
            logger.info(f"No OCR results found for session {session_id}")
            return None

        # Process all files to find the most relevant and collect metadata
        session_files_data = []
        for filename in ocr_files:
            file_path = os.path.join(session_ocr_dir, filename)
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)

                timestamp = data.get('timestamp', 0)

                # Check for exact query match if requested
                if query and data.get('query') == query:
                    logger.info(f"Found exact OCR match for query in file: {filename}")
                    # Don't clean if returning early with exact match
                    return data

                # Store file metadata along with data
                session_files_data.append((timestamp, filename, file_path, data))
            except Exception as e:
                logger.error(f"Error reading OCR file {file_path}: {e}")

        if not session_files_data:
            return None

        # Sort by timestamp (newest first)
        session_files_data.sort(key=lambda x: x[0], reverse=True)

        # Clean up excess files if needed
        if clean_excess and len(session_files_data) > MAX_SESSION_OCR_FILES:
            # Keep only the most recent files up to the maximum
            files_to_keep = session_files_data[:MAX_SESSION_OCR_FILES]
            files_to_remove = session_files_data[MAX_SESSION_OCR_FILES:]

            # Remove excess files
            for _, _, file_path, _ in files_to_remove:
                try:
                    os.remove(file_path)
                    logger.info(f"Removed excess OCR file: {os.path.basename(file_path)}")
                except Exception as e:
                    logger.warning(f"Failed to remove excess OCR file {file_path}: {e}")

        # Return the most recent data
        logger.info(f"No exact query match found, returning latest OCR results file for session {session_id}")
        return session_files_data[0][3]  # Return the data from the most recent file

    except Exception as e:
        logger.error(f"Error getting OCR results for session {session_id}: {e}")
        return None
