# models/responder.py

# models/responder.py
import os
import json
import re
import torch # Keep torch for memory debugging if needed
import psutil # Keep psutil for memory debugging if needed
from typing import List, Tuple, Optional, Any, Dict

from src.models.model_loader import load_model, is_single_image_model, aggressive_memory_cleanup
from src.services.session_manager.manager import load_session, _load_user_defaults # Import session loading
from src.models.prompt_templates import get_template_by_id, get_default_template
from src.api.config import SESSION_FOLDER
from src.models.memory.memory_manager import memory_manager # Keep for tracking current model
from src.utils.logger import get_logger
from src.utils.response_cache import check_response_cache, cache_response

# Import handlers
from .llm_handlers.base_handler import BaseLLMHandler
from .llm_handlers.ollama_handler import OllamaHandler
from .llm_handlers.huggingface_handler import HuggingFaceHandler

logger = get_logger(__name__)

# Mapping from model_choice prefixes/names to handler classes
HANDLER_MAP = {
    'ollama': OllamaHandler,
    'huggingface': HuggingFaceHandler,
}

# Cloud models removed — all inference is local (Ollama / HuggingFace)
CLOUD_MODELS: list[str] = []

# --- Removed Models ---
REMOVED_MODELS = {
    'qwen': "Qwen model has been removed from this version",
    'llama-vision': "The llama-vision model has been replaced with ollama-llama-vision. Please select ollama-llama-vision instead.",
    'pixtral': "Pixtral model has been removed from this version",
    'molmo': "Molmo model has been removed from this version",
    'groq-llama-vision': "Groq Llama Vision model has been removed from this version",
    'phi-4-multimodal-onnx': "Phi-4 ONNX model has been removed from this version",
    # Add any other explicitly removed models here
}

def get_handler(model_choice: str, model_obj: Any, session_model_params: Dict) -> Optional[BaseLLMHandler]:
    """Instantiates and returns the appropriate handler, passing session model parameters."""
    if model_choice in REMOVED_MODELS:
        logger.error(REMOVED_MODELS[model_choice])
        # Return None or raise an error? Returning None allows caller to handle.
        return None

    handler_class = None
    # Check for exact match first
    if model_choice in HANDLER_MAP:
        handler_class = HANDLER_MAP[model_choice]
    else:
        # Check for prefix match (e.g., 'ollama-')
        for prefix, handler in HANDLER_MAP.items():
            if model_choice.startswith(prefix + '-'):
                handler_class = handler
                break # Found the handler based on prefix

    if handler_class:
        try:
            # Pass session_model_params to the handler's constructor
            return handler_class(model_choice, model_obj, session_model_params)
        except Exception as e:
            logger.error(f"Failed to instantiate handler {handler_class.__name__} for {model_choice}: {e}", exc_info=True)
            return None
    else:
        logger.error(f"No handler found for model choice: {model_choice}")
        return None


def _prepare_conversation_context(chat_history: Optional[List[Dict[str, Any]]], session_id: str, model_choice: str, session_data: Optional[Dict] = None) -> str:
    """Formats recent chat history into a string for context."""
    conversation_context = ""
    if not chat_history:
        return conversation_context

    # Determine history limit based on model type (cloud vs local)
    max_history = 7 # Default for local
    # Use provided session_data if available, otherwise load defaults (less efficient)
    if session_data:
        is_cloud = model_choice in CLOUD_MODELS or any(model_choice.startswith(p + '-') for p in CLOUD_MODELS)
        if is_cloud:
            max_history = session_data.get('cloud_history_limit', 14) # Default 14 for cloud
        else:
            max_history = session_data.get('local_history_limit', 7) # Default 7 for local
        logger.info(f"Using history limit from session data: {max_history} messages for {model_choice}")
    else:
        # Fallback if session_data wasn't loaded successfully earlier
        logger.warning(f"Session data not available for history limit calculation (session: {session_id}). Using hardcoded defaults.")
        is_cloud = model_choice in CLOUD_MODELS or any(model_choice.startswith(p + '-') for p in CLOUD_MODELS)
        max_history = 14 if is_cloud else 7
        logger.info(f"Using default history limit: {max_history} for {model_choice}")

    # Apply the history limit

    # Apply the history limit
    recent_history = chat_history[-max_history:] if len(chat_history) > max_history else chat_history
    logger.info(f"Using {len(recent_history)} messages from chat history as context")

    # Format the history into a string
    for msg in recent_history:
        role = msg.get('role', '')
        content = msg.get('content', '')
        model_used = msg.get('model', None) # Model that generated this assistant response

        # Basic HTML tag stripping (consider a more robust library if needed)
        if isinstance(content, str):
            content = content.replace('<p>', '\n').replace('</p>', '\n')
            content = content.replace('<br>', '\n').replace('<br/>', '\n')
            content = re.sub(r'<[^>]+>', '', content).strip()

        if role == 'user':
            conversation_context += f"User: {content}\n\n"
        elif role == 'assistant':
            # Include model name if available
            if model_used:
                # Make the model name more user-friendly (optional)
                friendly_model_name = model_used.replace('ollama-', '').replace('-vision', ' Vision') # Basic cleanup
                conversation_context += f"Assistant ({friendly_model_name}): {content}\n\n"
            else:
                conversation_context += f"Assistant: {content}\n\n"

    logger.info(f"Conversation context prepared with {len(conversation_context)} characters")
    return conversation_context.strip() # Return stripped context


def _log_memory_usage(stage: str, prev_ram: float = 0, prev_gpu: float = 0) -> Tuple[float, float]:
    """Logs current RAM and GPU memory usage."""
    try:
        process = psutil.Process(os.getpid())
        current_ram = process.memory_info().rss / (1024 * 1024)  # MB
        current_gpu = 0
        if torch.cuda.is_available():
            current_gpu = torch.cuda.memory_allocated() / (1024 * 1024)  # MB

        ram_diff = current_ram - prev_ram
        gpu_diff = current_gpu - prev_gpu

        logger.info(f"MEMORY DEBUG ({stage}): RAM={current_ram:.2f}MB ({ram_diff:+.2f}MB), GPU={current_gpu:.2f}MB ({gpu_diff:+.2f}MB)")
        return current_ram, current_gpu
    except (ImportError, NameError):
        # logger.warning("psutil or torch not available for memory debugging.")
        # Don't log warning every time, just return zeros
        return 0, 0
    except Exception as e:
        logger.error(f"Error logging memory usage: {e}")
        return prev_ram, prev_gpu # Return previous values on error


# Refactored generate_response function
def generate_response(
    images: List[str],
    query: str,
    session_id: str,
    model_choice: str,
    chat_history: Optional[List[Dict[str, Any]]] = None,
    direct_ocr_results: Optional[Dict[str, Any]] = None,
    original_query: Optional[str] = None,
    use_ocr: bool = False,
    user_id: Optional[str] = None, # Keep user_id for loading defaults if session fails
    is_pasted_images: bool = False, # Flag to indicate pasted images in direct chat
    use_score_slope: bool = False # Flag to indicate if score-slope analysis is enabled
) -> Tuple[str, List[str]]:
    """
    Generates a response using the selected model handler based on the query and images.
    Handles both RAG-retrieved images and directly pasted images.
    Incorporates OCR text context when provided and appropriate.

    Args:
        images: List of FULL image paths to process (can be RAG or pasted)
        query: The text query from the user
        session_id: Current session identifier
        model_choice: Which model to use for generation
        chat_history: Optional chat history for context (list of message dicts with 'role' and 'content')
        direct_ocr_results: Optional OCR results from synchronous OCR processing (guaranteed to match images if provided)
        original_query: Optional original query if it was rewritten by generator
        use_ocr: Boolean flag indicating if OCR should be used (controlled by generator)

    Returns: (response_text, used_images_full_paths)
    """
    ram_before, gpu_before = _log_memory_usage("Responder Start")
    memory_manager.current_model = model_choice # Track model for display
    
    # Check response cache for non-direct-chat mode
    if images:
        # Extract image filenames for cache key (not full paths)
        image_filenames = [os.path.basename(img) for img in images if isinstance(img, str)]
        
        # Prepare retrieval settings for cache key
        retrieval_settings = {
            'use_ocr': use_ocr,
            'use_score_slope': use_score_slope,
            'is_pasted_images': is_pasted_images
        }
        
        # Load session data to get additional settings
        session_data = load_session(SESSION_FOLDER, session_id)
        if session_data:
            retrieval_settings.update({
                'k': session_data.get('retrieval_count', 5),
                'similarity_threshold': session_data.get('similarity_threshold', 0.0),
                'score_slope_enabled': session_data.get('score_slope_enabled', False),
                'token_budget_enabled': session_data.get('token_budget_enabled', False)
            })
        
        # Prepare conversation context for cache
        conversation_context_str = _prepare_conversation_context(chat_history, session_id, model_choice, session_data)
        
        # Check cache
        cached_response = check_response_cache(
            query=original_query or query,  # Use original query if available
            selected_filenames=image_filenames,
            conversation_context=conversation_context_str,
            generation_model=model_choice,
            retrieval_settings=retrieval_settings
        )
        
        if cached_response:
            logger.info(f"Using cached response for query: '{query[:50]}...'")
            return cached_response.get('response_text', ''), cached_response.get('used_images', [])

    # --- Handle Removed Models ---
    if model_choice in REMOVED_MODELS:
        logger.error(REMOVED_MODELS[model_choice])
        return REMOVED_MODELS[model_choice], []

    # --- Load Session Data & Extract Model Params ---
    # Check if we already loaded session_data for cache checking
    if images and 'session_data' in locals():
        # We already have session_data from cache check
        pass
    else:
        session_data = load_session(SESSION_FOLDER, session_id)
    
    session_model_params = {} # Default to empty
    selected_template = None

    if session_data:
        session_model_params = session_data.get('model_params', {})
        logger.info(f"Loaded model_params from session {session_id}")

        # Get the selected prompt template for this session
        selected_template_id = session_data.get('selected_template_id')
        logger.info(f"Session {session_id} has selected_template_id: {selected_template_id}")

        if selected_template_id and user_id:
            selected_template = get_template_by_id(user_id, selected_template_id)
            if selected_template:
                logger.info(f"Loaded prompt template '{selected_template.get('name')}' (ID: {selected_template_id}) for session {session_id}")
            else:
                logger.warning(f"Failed to load prompt template with ID {selected_template_id} for user {user_id}")
    else:
        # Fallback: Try loading user defaults if session load failed but we have user_id
        logger.warning(f"Failed to load session {session_id}. Attempting to load user defaults for model params.")
        if user_id:
            user_defaults = _load_user_defaults(user_id)
            session_model_params = user_defaults.get('model_params', {})
            logger.info(f"Loaded model_params from user defaults for user {user_id} as fallback.")
        else:
            logger.error("Cannot load session or user defaults for model params.")
            # Proceed with empty params, handler will use base defaults

    # If no template was found or loaded, use the default template
    if not selected_template and user_id:
        selected_template = get_default_template(user_id)
        logger.info(f"Using default prompt template for session {session_id}")

    # --- Prepare Conversation Context ---
    # Pass session_data to potentially use settings like history limits
    conversation_context_str = _prepare_conversation_context(chat_history, session_id, model_choice, session_data)

    # --- Direct Chat Mode Check ---
    if not images:
        logger.info(f"--- Direct Chat Mode: {model_choice} ---")
        # Clear any existing RAG models from memory before loading LLM
        try:
            # (Assuming RAG models are stored globally or accessible way)
            # Example: clear_rag_model(session_id)
            # For now, just log the intent
            logger.info("Attempting to clear RAG models before direct chat (if applicable).")
            # Aggressive cleanup before loading potentially large LLM
            aggressive_memory_cleanup()
        except Exception as e:
            logger.warning(f"Error during pre-direct-chat cleanup: {e}")

        try:
            # Load the model specifically for direct chat
            model_obj = load_model(model_choice)
            ram_after_load, gpu_after_load = _log_memory_usage("Direct Chat Load", ram_before, gpu_before)

            handler = get_handler(model_choice, model_obj, session_model_params)
            if not handler:
                return f"Model '{model_choice}' is not supported or handler failed to initialize.", []

            # In direct chat mode without images, we don't apply prompt templates
            # as they're designed for document analysis
            enhanced_query = query
            logger.info(f"Direct chat mode without images - not applying prompt template")

            # Delegate to the handler's direct chat method
            response_text, _ = handler.handle_direct_chat(enhanced_query, conversation_context_str)
            _log_memory_usage("Direct Chat End", ram_after_load, gpu_after_load)
            return response_text, [] # No images used in direct chat

        except Exception as e:
            logger.error(f"Error during direct chat processing for {model_choice}: {e}", exc_info=True)
            return f"An error occurred in direct chat mode: {str(e)}", []

    # --- RAG Mode or Pasted Images Mode ---
    mode_name = "Pasted Images" if is_pasted_images else "RAG"
    logger.info(f"--- {mode_name} Mode: {model_choice} ---")
    try:
        # Validate image paths (should be full paths already)
        valid_images = [img for img in images if isinstance(img, str) and os.path.exists(img)]
        if not valid_images:
            logger.warning("No valid image paths found for analysis.")
            return "No valid images found or provided for analysis.", []

        # Note: Single image logic (is_single_image_model) should ideally be handled
        # within the specific handler if it's model-specific (like Ollama collage logic).
        # Keeping it here for now might be redundant if handlers manage it.
        # We'll let the Ollama handler manage image count via retrieval_count for now.
        # if is_single_image_model(model_choice):
        #     valid_images = [valid_images[0]]
        #     logger.info(f"Model {model_choice} supports single image, using first: {valid_images[0]}")

        # Load the model for RAG or pasted images
        # Consider if model needs reloading or can be reused from direct chat (depends on load_model caching/state)
        model_obj = load_model(model_choice)
        ram_after_load, gpu_after_load = _log_memory_usage(f"{mode_name} Load", ram_before, gpu_before)

        handler = get_handler(model_choice, model_obj, session_model_params)
        if not handler:
             return f"Model '{model_choice}' is not supported or handler failed to initialize.", valid_images # Return images attempted

        # Apply prompt template if available
        enhanced_query = query

        if selected_template:
            # We keep template_type for future use even though we don't use it here
            template_type = selected_template.get('template_type', 'rag')
            system_prompt = selected_template.get('system_prompt', '')
            # We've removed query_prefix and query_suffix from the UI, but keep them in the backend
            # for backward compatibility
            query_prefix = selected_template.get('query_prefix', '')
            query_suffix = selected_template.get('query_suffix', '')
            # Removed ocr_prompt and non_ocr_instruction handling (now managed by model handlers)

            # Log template details for debugging
            logger.info(f"===== TEMPLATE DETAILS =====")
            logger.info(f"Template Name: {selected_template.get('name')}")
            logger.info(f"Template Type: {template_type}")
            logger.info(f"System Prompt: {system_prompt[:50]}..." if len(system_prompt) > 50 else f"System Prompt: {system_prompt}")
            logger.info(f"Query Prefix: {query_prefix[:50]}..." if len(query_prefix) > 50 else f"Query Prefix: {query_prefix}")
            logger.info(f"Query Suffix: {query_suffix[:50]}..." if len(query_suffix) > 50 else f"Query Suffix: {query_suffix}")
            logger.info(f"===== END TEMPLATE DETAILS =====")

            # Apply the template to the query if prefix/suffix exist (for backward compatibility)
            if query_prefix or query_suffix:
                enhanced_query = f"{query_prefix}{query}{query_suffix}"
                logger.info(f"Applied prompt template '{selected_template.get('name')}' to query")

            # Add system prompt to conversation context if available
            if system_prompt:
                conversation_context_str = f"System: {system_prompt}\n\n{conversation_context_str}"
                logger.info(f"Added system prompt from template to conversation context")

            # Note: OCR handling is now managed by the model handlers
            # We no longer use template-specific OCR prompts/instructions
            # Instead we use the centralized OCR processing in utils/ocr_utils.py
            logger.info(f"OCR handling will be managed by the model handlers (use_ocr={use_ocr})")
            # Log template application for pasted images
            if is_pasted_images:
                logger.info(f"===== APPLYING TEMPLATE FOR PASTED IMAGES =====")
                if not selected_template:
                    logger.info(f"No template selected - using default formatting for pasted images")
                else:
                    logger.info(f"Using template: {selected_template.get('name')}")
                    logger.info(f"OCR is {'enabled' if use_ocr else 'disabled'}")
                    logger.info(f"OCR handling will be managed by the model handler")
                logger.info(f"===== END TEMPLATE APPLICATION =====")

        # Store use_score_slope in session data if not already there
        if use_score_slope and session_data and 'use_score_slope' not in session_data:
            session_data['use_score_slope'] = use_score_slope
            from src.services.session_manager.manager import save_session
            save_session(SESSION_FOLDER, session_id, session_data)
            logger.info(f"Stored use_score_slope={use_score_slope} in session data")

        # Delegate to the handler's generate_response method
        response_text, used_images = handler.generate_response(
            images=valid_images, # Pass only validated images
            query=enhanced_query,
            session_id=session_id,
            chat_history=chat_history, # Pass raw history
            direct_ocr_results=direct_ocr_results,
            original_query=original_query,
            use_ocr=use_ocr,
            conversation_context=conversation_context_str, # Pass formatted history
            use_score_slope=use_score_slope # Pass score-slope flag to handler
            # We no longer pass template-specific OCR prompt templates
        )
        _log_memory_usage(f"{mode_name} End", ram_after_load, gpu_after_load)
        
        # Cache the response for future use
        if images and response_text:
            response_data = {
                'response_text': response_text,
                'used_images': used_images
            }
            
            cache_response(
                query=original_query or query,
                selected_filenames=image_filenames,
                conversation_context=conversation_context_str,
                generation_model=model_choice,
                retrieval_settings=retrieval_settings,
                response_data=response_data
            )
            logger.info(f"Cached response for query: '{query[:50]}...'")
        
        return response_text, used_images # Return images actually used by handler

    except Exception as e:
        logger.error(f"Error during {mode_name} processing for {model_choice}: {e}", exc_info=True)
        # Return the validated images that were attempted
        return f"An error occurred while generating the response: {str(e)}", valid_images

def generate_streaming_response(
    images: List[str],
    query: str,
    session_id: str,
    model_choice: str,
    chat_history: Optional[List[Dict[str, Any]]] = None,
    user_id: Optional[str] = None,
    direct_ocr_results: Optional[Dict[str, Any]] = None,
    original_query: Optional[str] = None,
    use_ocr: bool = False,
    is_pasted_images: bool = False,
    use_score_slope: bool = False,
    is_rag_mode: bool = False,
    selected_template_id: Optional[str] = None,
    retrieved_text_context: str = "",
):
    """
    Generates a streaming response using the specified model.
    
    Yields chunks in the format:
    - {'type': 'reasoning', 'content': '...'} for reasoning content
    - {'type': 'response', 'content': '...'} for actual response
    """
    try:
        logger.info(f"Starting streaming response generation with model: {model_choice}")
        
        # Load the model
        model_obj = load_model(model_choice)
        if not model_obj:
            yield {'type': 'error', 'error': f'Failed to load model: {model_choice}'}
            return
        
        # Load session data for model parameters
        session_data = None
        session_model_params = {}
        try:
            if True:
                session_data = load_session(SESSION_FOLDER, session_id)
                if session_data:
                    user_id = session_data.get('user_id')
                    # DEBUG: Log template selection from loaded session
                    logger.info(f"[TEMPLATE DEBUG] Loaded session data for {session_id}")
                    logger.info(f"[TEMPLATE DEBUG] selected_template_id from loaded session: {session_data.get('selected_template_id')}")
                else:
                    logger.warning(f"[TEMPLATE DEBUG] Failed to load session data for {session_id}")
        except RuntimeError:
            logger.info("Not in Flask application context, using user_id from parameter")
        
        # Load model-specific parameters and prompt template
        selected_template = None
        if user_id:
            defaults = _load_user_defaults(user_id)
            session_model_params = defaults.get('model_params', {})
        
        # Load session data and get the selected prompt template (CRITICAL FOR RAG!)
        if session_data:
            session_model_params = session_data.get('model_params', {})
            logger.info(f"Loaded model_params from session {session_id}")

            # Get the selected prompt template for this session
            # Use the passed template ID if session data doesn't have it
            session_template_id = session_data.get('selected_template_id')
            if not session_template_id and selected_template_id:
                logger.info(f"Using passed template ID {selected_template_id} as session data template was missing")
                session_template_id = selected_template_id
            logger.info(f"Session {session_id} has selected_template_id: {session_template_id}")
            # DEBUG: Additional logging
            logger.info(f"[TEMPLATE DEBUG] Before get_template_by_id: user_id={user_id}, template_id={session_template_id}")

            if session_template_id and user_id:
                selected_template = get_template_by_id(user_id, session_template_id)
                if selected_template:
                    logger.info(f"Loaded prompt template '{selected_template.get('name')}' (ID: {session_template_id}) for session {session_id}")
                    # DEBUG: Log full template details
                    logger.info(f"[TEMPLATE DEBUG] Template loaded successfully:")
                    logger.info(f"[TEMPLATE DEBUG]   - name: {selected_template.get('name')}")
                    logger.info(f"[TEMPLATE DEBUG]   - id: {selected_template.get('id')}")
                    logger.info(f"[TEMPLATE DEBUG]   - template_type: {selected_template.get('template_type')}")
                else:
                    logger.warning(f"Failed to load prompt template with ID {session_template_id} for user {user_id}")
                    logger.warning(f"[TEMPLATE DEBUG] get_template_by_id returned None")
        else:
            # Fallback: Try loading user defaults if session load failed but we have user_id
            logger.warning(f"Failed to load session {session_id}. Attempting to load user defaults for model params.")
            if user_id:
                user_defaults = _load_user_defaults(user_id)
                session_model_params = user_defaults.get('model_params', {})
                logger.info(f"Loaded model_params from user defaults for user {user_id} as fallback.")
                
                # Also try to use the passed template ID
                if selected_template_id:
                    selected_template = get_template_by_id(user_id, selected_template_id)
                    if selected_template:
                        logger.info(f"Loaded template '{selected_template.get('name')}' using passed template ID despite session load failure")

        # If no template was found or loaded, use the default template
        if not selected_template and user_id:
            selected_template = get_default_template(user_id)
            logger.info(f"Using default prompt template for session {session_id}")
        
        # Get handler
        handler = get_handler(model_choice, model_obj, session_model_params)
        if not handler:
            yield {'type': 'error', 'error': f'No handler available for model: {model_choice}'}
            return
        
        # Check if handler supports streaming
        if not hasattr(handler, 'generate_streaming_response'):
            yield {'type': 'error', 'error': f'Model {model_choice} does not support streaming'}
            return
        
        # Prepare conversation context
        conversation_context_str = _prepare_conversation_context(
            chat_history, session_id, model_choice, session_data
        )
        
        # --- Direct Chat Mode Check (using is_rag_mode flag) ---
        if not is_rag_mode and not is_pasted_images:
            logger.info(f"--- Streaming Direct Chat Mode: {model_choice} ---")
            # In direct chat mode with no images, we don't apply prompt templates
            # as they're designed for document analysis
            enhanced_query = query
            logger.info(f"Streaming direct chat mode (is_rag_mode=False, no images) - not applying prompt template")
            
            # Delegate to handler's streaming method without template or images
            # Use proper handler method signature
            for chunk in handler.generate_streaming_response(
                images=[],  # Empty images for text-only direct chat
                query=enhanced_query,
                session_id=session_id,
                chat_history=chat_history,
                conversation_context=conversation_context_str,
                direct_ocr_results=direct_ocr_results,
                original_query=original_query,
                use_ocr=use_ocr,
                use_score_slope=use_score_slope,
                retrieved_text_context=retrieved_text_context,
            ):
                yield chunk
            return
        
        # --- RAG Mode or Pasted Images Mode ---
        mode_name = "Pasted Images" if is_pasted_images else "RAG"
        logger.info(f"--- Streaming {mode_name} Mode: {model_choice} (is_rag_mode={is_rag_mode}) ---")
        
        # Apply prompt template if available (CRITICAL FOR RAG!)
        enhanced_query = query
        if selected_template:
            template_type = selected_template.get('template_type', 'rag')
            system_prompt = selected_template.get('system_prompt', '')
            query_prefix = selected_template.get('query_prefix', '')
            query_suffix = selected_template.get('query_suffix', '')

            # Log template details for debugging
            logger.info(f"===== STREAMING TEMPLATE DETAILS =====")
            logger.info(f"Template Name: {selected_template.get('name')}")
            logger.info(f"Template Type: {template_type}")
            logger.info(f"System Prompt: {system_prompt[:50]}..." if len(system_prompt) > 50 else f"System Prompt: {system_prompt}")
            logger.info(f"Query Prefix: {query_prefix[:50]}..." if len(query_prefix) > 50 else f"Query Prefix: {query_prefix}")
            logger.info(f"Query Suffix: {query_suffix[:50]}..." if len(query_suffix) > 50 else f"Query Suffix: {query_suffix}")
            logger.info(f"===== END STREAMING TEMPLATE DETAILS =====")

            # Apply the template to the query if prefix/suffix exist
            if query_prefix or query_suffix:
                enhanced_query = f"{query_prefix}{query}{query_suffix}"
                logger.info(f"Applied prompt template '{selected_template.get('name')}' to query")

            # Add system prompt to conversation context if available
            if system_prompt:
                conversation_context_str = f"System: {system_prompt}\n\n{conversation_context_str}"
                logger.info(f"Added system prompt from template to conversation context")

            logger.info(f"OCR handling will be managed by the model handlers (use_ocr={use_ocr})")
        
        # Stream from handler using enhanced query with template and all RAG parameters
        for chunk in handler.generate_streaming_response(
            images=images,
            query=enhanced_query,
            session_id=session_id,
            chat_history=chat_history,
            conversation_context=conversation_context_str,
            direct_ocr_results=direct_ocr_results,
            original_query=original_query,
            use_ocr=use_ocr,
            use_score_slope=use_score_slope,
            is_pasted_images=is_pasted_images,
            retrieved_text_context=retrieved_text_context,
        ):
            yield chunk
            
    except Exception as e:
        logger.error(f"Error in generate_streaming_response: {e}", exc_info=True)
        yield {'type': 'error', 'error': str(e)}
