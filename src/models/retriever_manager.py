"""
Retriever Manager for selecting optimal retrieval methods.

This module provides functions to initialize and select the most appropriate
retriever based on memory usage, document count, and other runtime factors.
It works with the standardized BaseRetriever interface.
"""

import os
import logging
import torch
import psutil
from typing import Optional, List, Dict, Any, Union, Tuple

from src.utils.logger import get_logger
from src.models.memory.memory_manager import memory_manager

logger = get_logger(__name__)

# Global retriever references
_standard_retriever = None

_hybrid_retriever = None

def initialize_retrievers(app_config=None):
    """
    Initialize all available retriever methods based on what's installed.

    Args:
        app_config: Optional Flask app configuration

    Returns:
        Tuple of available retrievers
    """
    global _standard_retriever, _hybrid_retriever
    # _qdrant_retriever removed
    # Import standard RAG retriever using the existing class-based implementation
    try:
        from src.models.rag_retriever import rag_retriever
        _standard_retriever = rag_retriever
        logger.info("Standard retriever initialized")
    except ImportError:
        logger.error("Failed to import standard retriever - this is a critical error")
        raise

    # LanceDB initialization removed

    # Qdrant initialization removed

    # Load Hybrid retriever for OCR and combined retrieval strategies
    try:
        from src.models.ocr_rag_retriever import ocr_rag_retriever
        _hybrid_retriever = ocr_rag_retriever
        logger.info("OCR retriever available for text extraction and visual retrieval")
    except ImportError:
        logger.info("OCR retriever not available - documents will be processed visually only")

    return (_standard_retriever, None, _hybrid_retriever) # Return None for LanceDB slot

def select_retriever(doc_count, memory_usage_pct, session_id, query=None, force_ocr=False, selected_filenames=None, force_chroma=False, existing_rag_model=None):
    """
    Dynamically select the best retriever based on document count, memory usage, and query.
    Injects the chosen base retriever into the hybrid retriever if needed.

    Args:
        doc_count: Number of documents being retrieved
        memory_usage_pct: Current memory usage percentage
        session_id: Current session ID
        query: Optional query text to help determine retriever selection
        force_ocr: Whether to force using OCR retriever
        selected_filenames: List of selected document filenames
        force_chroma: Whether to force using ChromaDB retriever
        existing_rag_model: Optional existing RAG model to reuse (to avoid reloading)

    Returns:
        The appropriate retriever instance (BaseRetriever or configured HybridRAGRetriever)
    """
    global _standard_retriever, _hybrid_retriever

    # Ensure retrievers are initialized (or re-initialized if cleared by memory management)
    # Adjusted check to exclude LanceDB and Qdrant
    if not all([_standard_retriever, _hybrid_retriever]):
        logger.warning("Standard or Hybrid retriever instance was None. Re-initializing.")
        initialize_retrievers()

    # --- Step 1: Determine the best BASE retriever ---
    chosen_base_retriever = _standard_retriever # Default
    base_retriever_reason = "Default"

    # force_chroma parameter is no longer used - we always use LanceDB
    if force_chroma:
        logger.info("force_chroma parameter is deprecated - LanceDB is now used exclusively")

    # (Standard retriever now uses LanceDB via rag_retriever.py)
    logger.info(f"Selected base retriever: {type(chosen_base_retriever).__name__} (Reason: {base_retriever_reason})")

    # --- Step 2: Determine if the HYBRID retriever is needed ---
    use_hybrid = False
    hybrid_reason = "N/A"

    if _hybrid_retriever:
        # Check session settings for OCR
        session_file = os.path.join('sessions', f"{session_id}.json")
        use_ocr_setting = False
        generation_model = None
        try:
            if os.path.exists(session_file):
                import json
                with open(session_file, 'r') as f: session_data = json.load(f)
                use_ocr_setting = session_data.get('use_ocr', False)
                generation_model = session_data.get('generation_model', None)
        except Exception as e: logger.warning(f"Error checking OCR settings: {e}")

        # Check conditions for using hybrid
        is_text_only_model = generation_model in ['ollama-phi4', 'ollama-phi4-mini', 'ollama-phi4-mini-fp16', 'ollama-phi4-q8', 'ollama-phi4-fp16', 'ollama-olmo2', 'ollama-exaone-deep:32b', 'ollama-qwq']
        ocr_effectively_enabled = use_ocr_setting or is_text_only_model or force_ocr
        multi_doc_selected = selected_filenames and len(selected_filenames) > 1

        if force_ocr:
            use_hybrid = True
            hybrid_reason = "Forced OCR"
        elif is_text_only_model:
             use_hybrid = True
             hybrid_reason = f"Text-only model ({generation_model})"
        elif use_ocr_setting:
            use_hybrid = True
            hybrid_reason = "OCR Enabled in Settings"
        # Removed condition: elif multi_doc_selected:
        # Hybrid retriever is now only selected if OCR is explicitly needed
        # (via settings, text-only model, or force_ocr flag).

    # --- Step 3: Pass existing RAG model if provided ---
    if existing_rag_model:
        logger.info(f"Using existing RAG model with id {id(existing_rag_model)} for retriever")

        # If we have an existing model, set it in the chosen retriever
        if hasattr(chosen_base_retriever, 'set_embedding_adapter'):
            logger.info(f"Setting embedding adapter in base retriever")
            chosen_base_retriever.set_embedding_adapter(existing_rag_model)

        # If using hybrid, also set the adapter there
        if use_hybrid and _hybrid_retriever and hasattr(_hybrid_retriever, 'set_embedding_adapter'):
            logger.info(f"Setting embedding adapter in hybrid retriever")
            _hybrid_retriever.set_embedding_adapter(existing_rag_model)

    # --- Step 4: Return the final retriever (configured if hybrid) ---
    if use_hybrid and _hybrid_retriever:
        logger.info(f"Selecting HYBRID retriever (Reason: {hybrid_reason}). Configuring with base: {type(chosen_base_retriever).__name__}")
        # Inject the chosen base retriever into the hybrid instance
        _hybrid_retriever.set_base_retriever(chosen_base_retriever)

        # Log if we're using an existing model
        if existing_rag_model:
            logger.info(f"Hybrid retriever configured with existing RAG model (id: {id(existing_rag_model)})")

        return _hybrid_retriever
    else:
        # Hybrid not needed, return the chosen base retriever directly
        logger.info(f"Selecting BASE retriever: {type(chosen_base_retriever).__name__} (Reason: {base_retriever_reason})")

        # Log if we're using an existing model
        if existing_rag_model:
            logger.info(f"Base retriever configured with existing RAG model (id: {id(existing_rag_model)})")

        return chosen_base_retriever

def get_current_memory_usage():
    """
    Get the current memory usage percentage using the consolidated memory management.

    Returns:
        float: Memory usage percentage
    """
    # Get memory usage percentage using our consolidated memory manager
    memory_usage_pct = 0
    try:
        # Use the consolidated memory manager from the new location
        memory_info = memory_manager.get_memory_usage()

        if memory_info.get("gpu_percentage") is not None:
            memory_usage_pct = memory_info.get("gpu_percentage")
        else:
            # Fall back to system RAM usage if GPU info not available
            memory = psutil.virtual_memory()
            memory_usage_pct = memory.percent
    except Exception as e:
        logger.warning(f"Error getting memory usage through memory_manager: {e}")
        try:
            # Fall back to direct measurement
            if torch.cuda.is_available():
                # Get GPU memory usage if available
                total_gpu = torch.cuda.get_device_properties(0).total_memory
                used_gpu = torch.cuda.memory_allocated()
                memory_usage_pct = (used_gpu / total_gpu) * 100
            else:
                # Fall back to system RAM usage
                memory = psutil.virtual_memory()
                memory_usage_pct = memory.percent
        except Exception as e2:
            logger.warning(f"Error getting memory usage directly: {e2}")
            memory_usage_pct = 50  # Default to 50% if we can't determine

    return memory_usage_pct
