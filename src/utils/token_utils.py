"""
Token counting utilities for RAG retrieval.

This module provides functions to estimate token counts for different models,
which is used for token budget-based document filtering.
"""
import re
import math
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

# Default tokens per character ratios for different model families
# These are approximate and used when a specific tokenizer isn't available
DEFAULT_TOKENS_PER_CHAR = {
    'default': 0.25,  # ~4 characters per token for most models
    'phi': 0.25,
    'llama': 0.25,
    'mistral': 0.25,
    'gemma': 0.25,
    'granite': 0.25,
    'qwq': 0.25,
    'olmo': 0.25,
    'deepcoder': 0.25,
}

def get_model_family(model_name: str) -> str:
    """
    Extract the model family from the model name.
    
    Args:
        model_name: Full model name (e.g., 'ollama-phi4', 'ollama-llama3.2-vision')
        
    Returns:
        Model family name (e.g., 'phi', 'llama')
    """
    # Remove provider prefix if present
    if '-' in model_name:
        model_name = model_name.split('-', 1)[1]
    
    # Extract base model family
    if 'phi' in model_name:
        return 'phi'
    elif 'llama' in model_name:
        return 'llama'
    elif 'mistral' in model_name:
        return 'mistral'
    elif 'gemma' in model_name:
        return 'gemma'
    elif 'granite' in model_name:
        return 'granite'
    elif 'qwq' in model_name:
        return 'qwq'
    elif 'olmo' in model_name:
        return 'olmo'
    elif 'deepcoder' in model_name:
        return 'deepcoder'
    else:
        return 'default'

def estimate_tokens(text: str, model_name: str) -> int:
    """
    Estimate the number of tokens in a text for a specific model.
    
    Args:
        text: The text to count tokens for
        model_name: Name of the model
        
    Returns:
        Estimated token count
    """
    if not text:
        return 0
    
    # Get model family
    model_family = get_model_family(model_name)
    
    # Get tokens per character ratio for this model family
    tokens_per_char = DEFAULT_TOKENS_PER_CHAR.get(model_family, DEFAULT_TOKENS_PER_CHAR['default'])
    
    # Clean text (remove excessive whitespace)
    cleaned_text = re.sub(r'\s+', ' ', text).strip()
    
    # Estimate tokens based on character count
    char_count = len(cleaned_text)
    token_estimate = math.ceil(char_count * tokens_per_char)
    
    return token_estimate

def get_model_context_window(model_name: str, session_data: Dict[str, Any]) -> int:
    """
    Get the context window size for a specific model from session data.
    
    Args:
        model_name: Name of the model
        session_data: Session data containing model configurations
        
    Returns:
        Context window size in tokens
    """
    # Default context window size
    default_context_window = 8192
    
    # Extract provider and model name
    provider = 'ollama'  # Default provider
    base_model_name = model_name
    
    if '-' in model_name:
        parts = model_name.split('-', 1)
        if len(parts) == 2:
            provider, base_model_name = parts
    
    # Try to get context window from session data
    try:
        model_params = session_data.get('model_params', {})
        provider_params = model_params.get(provider, {})
        
        # First check if there's a num_ctx in the provider's default params
        context_window = provider_params.get('num_ctx', None)
        
        # Then check if there's a model-specific num_ctx
        models_config = provider_params.get('models', {})
        model_config = models_config.get(base_model_name, {})
        model_specific_ctx = model_config.get('num_ctx', None)
        
        if model_specific_ctx is not None:
            context_window = model_specific_ctx
        
        # If we found a context window, return it
        if context_window is not None:
            return context_window
    except Exception as e:
        logger.warning(f"Error getting context window for {model_name}: {e}")
    
    # Fallback to default context window
    return default_context_window

def get_document_text(doc: Dict[str, Any], ocr_cache: Optional[Dict[str, Any]] = None) -> str:
    """
    Get the text content of a document, either from OCR cache or metadata.
    
    Args:
        doc: Document dictionary
        ocr_cache: Optional OCR cache containing document text
        
    Returns:
        Document text content
    """
    # Try to get text from OCR cache first
    if ocr_cache and 'path' in doc:
        doc_path = doc['path']
        if doc_path in ocr_cache:
            return ocr_cache[doc_path].get('text', '')
    
    # Fallback to metadata if available
    if 'metadata' in doc and 'text' in doc['metadata']:
        return doc['metadata']['text']
    
    # If no text is available, return empty string
    return ''
