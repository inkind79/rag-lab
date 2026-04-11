"""
Session management functionality.

Handles multiple sessions per user, identified by session_uuid.
Stores user_id within each session file for ownership.
Loads user-specific defaults, falling back to hardcoded defaults.
"""
import os
import json
import uuid
import logging
import copy # Needed for deep merging defaults
from typing import Dict # Import Dict for type hinting
import shutil
import time # Imported for LanceDB deletion retry
import math # For checking NaN and Inf values
from src.utils.file_lock import safe_json_read, safe_json_write

logger = logging.getLogger(__name__)

# Define the folder for user-specific defaults
USER_DEFAULTS_FOLDER = "user_defaults"
# Import base model defaults for structure/fallback
from src.utils.model_configs import DEFAULT_MODEL_CONFIGS as BASE_DEFAULT_MODEL_PARAMS

def _detect_default_model() -> str:
    """Auto-detect the first available Ollama model, or fall back to a sensible default."""
    try:
        import requests
        resp = requests.get("http://localhost:11434/api/tags", timeout=3)
        if resp.ok:
            models = resp.json().get("models", [])
            if models:
                return f"ollama-{models[0]['name']}"
    except Exception:
        pass
    return "ollama-gemma3n-vision-fp16"  # fallback if Ollama isn't running yet

# Define application defaults (General Settings)
DEFAULT_GLOBAL_SETTINGS = {
  "generation_model": _detect_default_model(),
  "retrieval_count": 1,
  "cloud_history_limit": 14,
  "local_history_limit": 7,
  "use_ocr": False,
  "ocr_engine": "easyocr",
  "ocr_language": "en",
  "ocr_vlm_model": "gemma-vision",
  "indexer_model": "athrael-soju/colqwen3.5-4.5B-v3",
  "resized_height": 448,
  "resized_width": 448,
  "indexing_backends": "lancedb",
  "distance_metric": "cosine",  # Default distance metric for vector similarity
  "similarity_threshold": 0.2,  # Default minimum similarity score for document retrieval
  "use_score_slope": False,
  "rel_drop_threshold": 0.65,   # Default threshold for relative score drop in score-slope analysis
  "abs_score_threshold": 0.25,  # Default absolute score threshold for score-slope analysis
  "show_score_viz": False,      # Whether to show score distribution visualization
  # RAG Lab retrieval method settings
  "retrieval_method": "colpali",          # colpali | bm25 | dense | hybrid_rrf | hybrid
  "text_embedding_model": "BAAI/bge-m3",  # Used when retrieval_method is dense or hybrid_rrf
  "chunk_size": 512,                       # Text chunking: characters per chunk
  "chunk_overlap": 64,                     # Text chunking: overlap between chunks
  "hybrid_visual_weight": 0.6,             # Weight for visual vs keyword in hybrid mode (0=keyword, 1=visual)
}

def _deep_merge_model_params(base: Dict, overlay: Dict) -> Dict:
    """Recursively merge model parameter dictionaries."""
    merged = copy.deepcopy(base)
    for provider, provider_config in overlay.items():
        if provider not in merged:
            merged[provider] = copy.deepcopy(provider_config)
        else:
            # Merge provider-level settings (excluding 'models')
            for key, value in provider_config.items():
                if key != 'models':
                    merged[provider][key] = value
            # Merge model-specific settings
            if 'models' in provider_config:
                if 'models' not in merged[provider]:
                    merged[provider]['models'] = {}
                for model_name, model_config in provider_config['models'].items():
                    if model_name not in merged[provider]['models']:
                        merged[provider]['models'][model_name] = copy.deepcopy(model_config)
                    else:
                        merged[provider]['models'][model_name].update(model_config)
    return merged

def _load_user_defaults(user_id):
    """Loads user-specific default settings, falling back to hardcoded defaults."""
    if not user_id:
        logger.warning("Attempted to load user defaults without user_id. Returning hardcoded defaults.")
        return DEFAULT_GLOBAL_SETTINGS.copy()

    # Ensure the user defaults directory exists (though ideally created on app start)
    os.makedirs(USER_DEFAULTS_FOLDER, exist_ok=True)
    user_defaults_path = os.path.join(USER_DEFAULTS_FOLDER, f"{user_id}.json")

    raw_user_defaults = {}
    if os.path.exists(user_defaults_path):
        try:
            with open(user_defaults_path, 'r') as f:
                raw_user_defaults = json.load(f)
            logger.debug(f"Loaded raw user defaults for {user_id} from {user_defaults_path}")
        except Exception as e:
            logger.error(f"Error loading user defaults from {user_defaults_path}: {e}. Using hardcoded defaults.")
            raw_user_defaults = {} # Fallback to empty if file is corrupt
    else:
        logger.info(f"User defaults file not found at {user_defaults_path}. Using hardcoded defaults for user {user_id}.")
        raw_user_defaults = {} # Fallback to empty if file doesn't exist

    # --- Merge General Settings ---
    final_general_defaults = DEFAULT_GLOBAL_SETTINGS.copy() # Start with hardcoded general settings
    # Override with user's general settings (keys present in DEFAULT_GLOBAL_SETTINGS)
    for key in DEFAULT_GLOBAL_SETTINGS.keys():
        if key in raw_user_defaults:
            final_general_defaults[key] = raw_user_defaults[key]

    # --- Merge Model Parameters ---
    # Start with the base default model parameters
    final_model_params = copy.deepcopy(BASE_DEFAULT_MODEL_PARAMS)
    # Load user's model params if they exist in their file
    user_model_params = raw_user_defaults.get('model_params', {})
    # Deep merge user's model params onto the base defaults
    final_model_params = _deep_merge_model_params(final_model_params, user_model_params)

    # Combine general settings and model params into the final defaults dictionary
    final_defaults = final_general_defaults
    final_defaults['model_params'] = final_model_params # Add the merged model params

    logger.debug(f"Final merged defaults for user {user_id}: {final_defaults}")
    return final_defaults


# --- Multi-Session Per User Functions ---

def create_session(session_folder, user_id, session_name=None):
    """
    Create a new session for a specific user, inheriting from user defaults.

    Args:
        session_folder: Folder to store session files.
        user_id: The ID of the user owning this session.
        session_name: Optional initial session name.

    Returns:
        Tuple of (new_session_uuid, session_data) or (None, None) on failure.
    """
    if not user_id:
        logger.error("create_session called without a user_id.")
        return None, None

    new_session_uuid = str(uuid.uuid4())

    if session_name is None:
        session_name = f"New Session {new_session_uuid[:8]}"

    # 1. Start with basic structure including owner user_id
    session_data = {
        'user_id': user_id,
        'session_name': session_name,
        'chat_history': [],
        'indexed_files': [],
        'selected_docs': [],
        'is_new_session': True,  # Flag to indicate this is a newly created session
        'memory_enabled': False,  # Disable memory by default
        'use_score_slope': False,
        'created_at': time.time(),  # Store creation timestamp for reliable sorting
    }

    # 2. Load user default settings (which includes hardcoded fallbacks)
    user_defaults = _load_user_defaults(user_id)

    # 3. Apply user default settings (general) as the base for configurable options
    # This loop copies the initial values, including the user's preferred 'use_ocr' if saved,
    # or the hardcoded 'false' if not.
    for key in DEFAULT_GLOBAL_SETTINGS.keys():
        session_data[key] = user_defaults.get(key)

    # Set default embedding model with LanceDB
    logger.info(f"Setting indexer_model to ColQwen3.5 and indexing_backends to 'lancedb' in new session {new_session_uuid}")
    session_data['indexer_model'] = "athrael-soju/colqwen3.5-4.5B-v3"
    session_data['indexing_backends'] = 'lancedb'

    # 4. Check if the default generation model requires OCR to be forced ON
    default_gen_model = session_data.get('generation_model') # Get model from session_data now
    model_params = user_defaults.get('model_params', {}) # Get full model params from defaults
    supports_vision = True # Default assumption

    if default_gen_model:
        # Determine provider and model key (simplified logic assuming structure)
        provider = 'ollama' # Default provider
        model_key = default_gen_model
        if default_gen_model.startswith('ollama-'):
             provider = 'ollama'
             model_key = default_gen_model[len('ollama-'):]
        elif default_gen_model.startswith('gemini'):
             provider = 'gemini'
             # Find the actual model key within gemini models if needed, or assume single model
             gemini_models = model_params.get('gemini', {}).get('models', {})
             model_key = list(gemini_models.keys())[0] if gemini_models else None
        elif default_gen_model.startswith('gpt-'):
             provider = 'openai'
             # OpenAI model names are usually direct keys
             model_key = default_gen_model

        # Find the specific model config
        provider_models = model_params.get(provider, {}).get('models', {})
        model_config = provider_models.get(model_key, {})

        # Check the supports_vision flag (defaulting to True if missing, except for known text-only)
        known_text_only = ['phi4', 'phi4-mini', 'olmo2', 'qwq']
        if model_key in known_text_only:
             supports_vision = model_config.get('supports_vision', False)
        else:
             supports_vision = model_config.get('supports_vision', True) # Default to True if flag missing and not known text-only

    # If the default model is text-only, force use_ocr to true for this new session.
    # Otherwise, ensure it defaults to false (overriding any user default preference for 'true').
    if not supports_vision:
        logger.info(f"Default model '{default_gen_model}' is text-only. Forcing 'use_ocr' to True for new session {new_session_uuid}.")
        session_data['use_ocr'] = True
    else:
         logger.info(f"Default model '{default_gen_model}' supports vision. Setting 'use_ocr' to False for new session {new_session_uuid}.")
         session_data['use_ocr'] = False # Explicitly set to False for VLMs


    # 5. Apply model parameters from user defaults (deep copy to avoid modifying defaults)


    # 5. Apply model parameters from user defaults (deep copy to avoid modifying defaults)
    # We copy this *after* potentially modifying use_ocr based on the default model
    session_data['model_params'] = copy.deepcopy(model_params)

    # 6. Ensure session name is set correctly (might have been defaulted above)
    session_data['session_name'] = session_name

    # Save session data using the new UUID as filename
    session_file = os.path.join(session_folder, f"{new_session_uuid}.json")
    try:
        with open(session_file, 'w') as f:
            json.dump(session_data, f, indent=2)
        logger.info(f"Created new session {new_session_uuid} for user {user_id} - Name: {session_name}")
        return new_session_uuid, session_data
    except Exception as e:
        logger.error(f"Error creating session file {session_file}: {e}")
        return None, None


def load_session(session_folder, session_uuid):
    """
    Load session data for a specific session UUID.
    Does NOT merge defaults here. Merging happens in context processor/routes.

    Args:
        session_folder: Folder containing session files.
        session_uuid: The UUID of the session to load.

    Returns:
        Session data dictionary, or None if file not found or invalid.
    """
    if not session_uuid:
        logger.error("load_session called without a session_uuid.")
        return None

    session_file = os.path.join(session_folder, f"{session_uuid}.json")
    if not os.path.exists(session_file):
        logger.warning(f"Session file not found: {session_file}")
        return None

    # Use safe JSON read with file locking
    session_data = safe_json_read(session_file)
    
    if session_data is None:
        logger.error(f"Failed to load session file: {session_file}")
        return None
        
    # Basic validation: Check if it has a user_id
    if 'user_id' not in session_data:
        logger.error(f"Session file {session_file} is missing 'user_id'. Treating as invalid.")
        return None
        
    logger.info(f"Loaded session data from: {session_file}")
    return session_data


def save_session(session_folder, session_uuid, session_data):
    """
    Save session data for a specific session UUID.

    Args:
        session_folder: Folder to store session files.
        session_uuid: The UUID of the session to save.
        session_data: Session data dictionary to save.

    Returns:
        True if successful, False otherwise.
    """
    if not session_uuid or not session_data:
        logger.error("save_session called without session_uuid or session_data.")
        return False
    # Ensure user_id is present before saving (should be added by create_session)
    if 'user_id' not in session_data:
         logger.error(f"Attempted to save session {session_uuid} without a user_id.")
         return False

    session_file = os.path.join(session_folder, f"{session_uuid}.json")

    try:
        # Ensure session folder exists
        os.makedirs(session_folder, exist_ok=True)

        # Validate session data to ensure it's serializable
        try:
            # Try to serialize the data to check for any issues
            json.dumps(session_data)
        except (TypeError, OverflowError) as e:
            logger.error(f"Session data for {session_uuid} contains non-serializable objects: {e}")
            # Try to clean up the session data
            cleaned_data = clean_session_data(session_data)
            if cleaned_data:
                session_data = cleaned_data
                logger.info(f"Successfully cleaned session data for {session_uuid}")
            else:
                logger.error(f"Failed to clean session data for {session_uuid}")
                return False

        # Log summary before saving
        log_data_summary = {
            'user_id': session_data.get('user_id'),
            'session_name': session_data.get('session_name'),
            'chat_history_len': len(session_data.get('chat_history', [])),
            'indexed_files_len': len(session_data.get('indexed_files', [])),
            'selected_docs': session_data.get('selected_docs', [])
        }
        logger.info(f"[SAVE] Session UUID: {session_uuid} - Saving Data Summary: {json.dumps(log_data_summary)}")

        # Check if the file exists and is valid JSON before overwriting
        if os.path.exists(session_file):
            try:
                with open(session_file, 'r') as f:
                    json.load(f)  # Just try to load it to check if it's valid JSON
            except json.JSONDecodeError:
                # If the file exists but is invalid JSON, make a backup before overwriting
                timestamp = int(time.time())
                backup_file = f"{session_file}.bak.{timestamp}"
                try:
                    shutil.copy2(session_file, backup_file)
                    logger.warning(f"Created backup of invalid JSON file: {backup_file}")
                except Exception as e_backup:
                    logger.error(f"Failed to create backup of invalid JSON file: {e_backup}")

        # Use safe JSON write with file locking
        success = safe_json_write(session_file, session_data)
        
        if success:
            logger.info(f"Saved session: {session_uuid}")
            return True
        else:
            logger.error(f"Failed to save session: {session_uuid}")
            return False
    except Exception as e:
        logger.error(f"Error saving session {session_uuid}: {e}")
        return False


def migrate_session_to_colqwen25(session_id):
    """
    Migrate an existing session to use ColQwen2.5 model with LanceDB.

    Args:
        session_id: The session ID to migrate

    Returns:
        Tuple of (success, message)
    """
    try:
        # Load the session data
        session_data = load_session('sessions', session_id)
        if not session_data:
            return False, f"Session {session_id} not found"

        # Check if migration is needed
        current_model = session_data.get('indexer_model')
        current_backend = session_data.get('indexing_backends')

        if current_model == "athrael-soju/colqwen3.5-4.5B-v3" and current_backend == "lancedb":
            return True, f"Session {session_id} already using ColQwen3.5 with LanceDB"

        # Update the session data
        session_data['indexer_model'] = "athrael-soju/colqwen3.5-4.5B-v3"
        session_data['indexing_backends'] = "lancedb"

        # Save the updated session data
        success = save_session('sessions', session_id, session_data)
        if not success:
            return False, f"Failed to save updated session data for {session_id}"

        # Reindex documents if needed
        indexed_files = session_data.get('indexed_files', [])
        if indexed_files:
            logger.info(f"Session {session_id} has {len(indexed_files)} indexed files that need reindexing with ColQwen2.5")
            # Note: Actual reindexing would be handled by the document processor
            # This function just updates the session settings

        return True, f"Successfully migrated session {session_id} to use ColQwen2.5 with LanceDB"
    except Exception as e:
        logger.error(f"Error migrating session {session_id} to ColQwen2.5: {e}")
        return False, f"Error migrating session: {str(e)}"

def clean_session_data(session_data):
    """
    Clean session data to ensure it's serializable.

    Args:
        session_data: The session data to clean.

    Returns:
        Cleaned session data dictionary or None if cleaning failed.
    """
    try:
        # Import numpy here to check for numpy types
        try:
            import numpy as np
            has_numpy = True
        except ImportError:
            has_numpy = False

        # Helper function to convert numpy types to Python native types
        def convert_numpy_types(obj):
            if has_numpy:
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
            return obj

        # Helper function to recursively clean a dictionary
        def clean_dict(d):
            result = {}
            for k, v in d.items():
                if isinstance(v, dict):
                    result[k] = clean_dict(v)
                elif isinstance(v, list):
                    result[k] = clean_list(v)
                else:
                    result[k] = convert_numpy_types(v)
            return result

        # Helper function to recursively clean a list
        def clean_list(lst):
            result = []
            for item in lst:
                if isinstance(item, dict):
                    result.append(clean_dict(item))
                elif isinstance(item, list):
                    result.append(clean_list(item))
                else:
                    result.append(convert_numpy_types(item))
            return result

        # Create a deep copy to avoid modifying the original
        cleaned_data = copy.deepcopy(session_data)

        # Apply recursive cleaning to the entire data structure
        cleaned_data = clean_dict(cleaned_data)

        # Clean score_analysis
        if 'score_analysis' in cleaned_data:
            if not isinstance(cleaned_data['score_analysis'], dict):
                cleaned_data['score_analysis'] = {}
            else:
                # Ensure all scores are valid numbers
                valid_scores = {}
                for key, value in cleaned_data['score_analysis'].items():
                    try:
                        # Convert to float and check if it's valid
                        float_value = float(value)
                        if not math.isnan(float_value) and not math.isinf(float_value):
                            valid_scores[key] = float_value
                    except (ValueError, TypeError, OverflowError):
                        continue
                cleaned_data['score_analysis'] = valid_scores

        # Clean chat_history
        if 'chat_history' in cleaned_data and isinstance(cleaned_data['chat_history'], list):
            cleaned_history = []
            for msg in cleaned_data['chat_history']:
                if not isinstance(msg, dict):
                    continue  # Skip non-dict messages

                # Create a cleaned message with only serializable fields
                cleaned_msg = {}

                # Handle essential fields
                if 'id' in msg:
                    cleaned_msg['id'] = str(msg['id'])
                if 'role' in msg:
                    cleaned_msg['role'] = str(msg['role'])
                if 'timestamp' in msg:
                    # Ensure timestamp is a string or number
                    try:
                        if isinstance(msg['timestamp'], (int, float)):
                            cleaned_msg['timestamp'] = msg['timestamp']
                        else:
                            cleaned_msg['timestamp'] = str(msg['timestamp'])
                    except:
                        pass

                # Handle content field (convert to string to avoid Markup objects)
                if 'content' in msg:
                    try:
                        cleaned_msg['content'] = str(msg['content'])
                    except:
                        cleaned_msg['content'] = ""

                # Handle model field if present
                if 'model' in msg:
                    try:
                        cleaned_msg['model'] = str(msg['model'])
                    except:
                        pass

                # Handle template_name field if present
                if 'template_name' in msg:
                    try:
                        cleaned_msg['template_name'] = str(msg['template_name'])
                    except:
                        pass

                # Handle images field if present
                if 'images' in msg and msg.get('images'):
                    try:
                        # Images should already be serializable (list of strings or dicts)
                        cleaned_msg['images'] = msg['images']
                    except:
                        pass

                # Add the cleaned message to the history
                cleaned_history.append(cleaned_msg)

            # Replace the original history with the cleaned one
            cleaned_data['chat_history'] = cleaned_history

        # Ensure the cleaned data is serializable
        json.dumps(cleaned_data)
        return cleaned_data
    except Exception as e:
        logger.error(f"Failed to clean session data: {e}")
        return None


def get_all_sessions(session_folder, user_id):
    """
    Get all sessions belonging to a specific user.

    Args:
        session_folder: Folder containing session files.
        user_id: The ID of the user whose sessions to retrieve.

    Returns:
        List of session dictionaries [{'id': session_uuid, 'name': session_name}]
        sorted by creation time (newest first).
    """
    if not user_id:
        logger.error("get_all_sessions called without user_id.")
        return []

    all_files = []
    try:
        # Ensure session folder exists before listing
        os.makedirs(session_folder, exist_ok=True)
        all_files = [f for f in os.listdir(session_folder) if f.endswith('.json')]
    except FileNotFoundError:
        logger.warning(f"Session folder not found when listing: {session_folder}")
        return []
    except Exception as e:
        logger.error(f"Error listing session folder {session_folder}: {e}")
        return []

    user_sessions = []
    for filename in all_files:
        session_uuid = filename[:-5]
        file_path = os.path.join(session_folder, filename)
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            # Check if the session belongs to the requested user
            if data.get('user_id') == user_id:
                name = data.get('session_name', f'Session {session_uuid[:8]}')
                # Use stored creation timestamp if available, otherwise fall back to file creation time
                # This ensures newly created sessions appear first
                created_at = data.get('created_at')
                if created_at is None:
                    # For older sessions without created_at, use file creation time
                    created_at = os.path.getctime(file_path)
                user_sessions.append({'id': session_uuid, 'name': name, 'created_at': created_at})
        except json.JSONDecodeError:
             logger.error(f"Skipping invalid JSON file during session list: {filename}")
        except Exception as e:
            logger.error(f"Error loading or checking ownership for session file {filename}: {e}")

    # Sort sessions by creation time (newest first)
    # This ensures newly created sessions appear at the top
    # For sessions without created_at, use file creation time as fallback
    def get_session_creation_time(session_data):
        """Get creation time for a session, with fallback to file creation time"""
        # First try the stored created_at timestamp
        if 'created_at' in session_data and session_data['created_at']:
            return session_data['created_at']
        
        # Fallback to file creation time
        session_file = os.path.join(session_folder, f"{session_data['id']}.json")
        if os.path.exists(session_file):
            try:
                return os.path.getctime(session_file)
            except:
                pass
        
        # Last resort: return 0 so these go to the end
        return 0
    
    user_sessions.sort(key=get_session_creation_time, reverse=True)

    # Return only id and name after sorting
    return [{'id': s['id'], 'name': s['name']} for s in user_sessions]


def get_active_session_id():
    """
    Get the active session ID from Flask's session object.

    Returns:
        The active session UUID or None if not found.
    """
    try:
        from flask import session
        return session.get('active_session_uuid')
    except (ImportError, RuntimeError):
        return None
    except Exception as e:
        logger.error(f"Error getting active session ID: {e}")
        return None


def delete_session_data(session_uuid, app_config, rag_models=None):
    """
    Delete all data associated with a specific session UUID.

    Args:
        session_uuid: The UUID of the session to delete.
        app_config: Flask app configuration.
        rag_models: Optional dictionary of RAG models (keyed by user_id).

    Returns:
        True if successful, False otherwise.
    """
    if not session_uuid:
        logger.error("delete_session_data called without session_uuid.")
        return False

    session_file = os.path.join(app_config['SESSION_FOLDER'], f"{session_uuid}.json")
    user_id = None # We need the user_id for RAG model cleanup

    try:
        # Load session data first to get user_id (if file exists)
        if os.path.exists(session_file):
            try:
                # Use load_session to handle potential errors loading the file
                session_data = load_session(app_config['SESSION_FOLDER'], session_uuid)
                if session_data:
                    user_id = session_data.get('user_id')
                    logger.info(f"Loaded user_id {user_id} from session file {session_uuid} before deletion.")
                else:
                     logger.warning(f"Could not load session file {session_uuid} to get user_id before deletion (file might be invalid).")
            except Exception as e: # Catch any other unexpected error during load
                logger.error(f"Unexpected error loading session file {session_uuid} to get user_id before deletion: {e}")

            # Delete session file regardless of whether we got user_id
            try:
                os.remove(session_file)
                logger.info(f"Deleted session file: {session_file}")
            except OSError as e_os:
                 logger.error(f"Error removing session file {session_file}: {e_os}")
                 # Continue with other cleanup even if file removal fails

        # Delete associated folders using session_uuid
        session_upload_folder = os.path.join(app_config['UPLOAD_FOLDER'], session_uuid)
        if os.path.exists(session_upload_folder):
            shutil.rmtree(session_upload_folder, ignore_errors=True)
            logger.info(f"Deleted uploaded documents folder: {session_upload_folder}")

        session_images_folder = os.path.join(app_config['STATIC_FOLDER'], 'images', session_uuid)
        if os.path.exists(session_images_folder):
            shutil.rmtree(session_images_folder, ignore_errors=True)
            logger.info(f"Deleted session images folder: {session_images_folder}")

        # Clear OCR cache entries for this session's images
        try:
            from src.utils.ocr_cache_cleanup import clear_ocr_cache_for_session
            ocr_cleared = clear_ocr_cache_for_session(session_uuid, app_config)
            logger.info(f"Cleared {ocr_cleared} OCR cache entries for session {session_uuid}")
        except ImportError:
            logger.warning("OCR cache cleanup utility not available")
        except Exception as e:
            logger.warning(f"Error clearing OCR cache for session {session_uuid}: {e}")

        # Byaldi index folder cleanup removed - no longer needed

        # Clear LanceDB resources for the session
        try:
            from src.models.vector_stores.lancedb_manager import destroy_lancedb_resources
            # Clear LanceDB resources for the session
            cleanup_info = destroy_lancedb_resources(session_uuid)
            logger.info(f"Cleared LanceDB resources for session {session_uuid}: {cleanup_info}")
        except ImportError:
            logger.debug("LanceDB manager not available, skipping LanceDB resource clearing.")
        except Exception as e:
            logger.warning(f"Error clearing LanceDB resources for session {session_uuid}: {e}")

        # Delete LanceDB session folder
        lancedb_folder = os.path.join(app_config.get('LANCEDB_FOLDER', '.lancedb'), session_uuid)
        if os.path.exists(lancedb_folder):
            try:
                # Try multiple times with delay if the first attempt fails
                max_retries = 3
                retry_count = 0
                deleted = False

                while not deleted and retry_count <= max_retries:
                    try:
                        shutil.rmtree(lancedb_folder, ignore_errors=True) # Use ignore_errors
                        if not os.path.exists(lancedb_folder):
                            deleted = True
                            logger.info(f"Deleted LanceDB folder: {lancedb_folder}" +
                                       (f" on retry {retry_count}" if retry_count > 0 else ""))
                            break
                    except Exception as e_retry:
                        logger.warning(f"Error in LanceDB folder deletion on attempt {retry_count+1}: {e_retry}")

                    retry_count += 1
                    if retry_count <= max_retries:
                        time.sleep(1)  # Wait 1 second before retrying

                if not deleted:
                    logger.error(f"Failed to delete LanceDB folder after {max_retries+1} attempts: {lancedb_folder}")
            except Exception as e:
                logger.error(f"Error in LanceDB folder deletion process: {e}")



        # Vector store resource management now handled by LanceDB manager


        # Remove RAG model from memory for this session
        # Note: RAG model cache is now per-session_uuid, not per-user
        if rag_models and rag_models.contains(session_uuid):
            logger.info(f"Removing RAG model for session {session_uuid} from cache.")
            try:
                rag_models.delete(session_uuid)
                logger.info(f"Successfully removed RAG model for session {session_uuid}")
            except KeyError:
                logger.warning(f"Tried to remove RAG model for session {session_uuid}, but key was already gone.")
            except Exception as e_del:
                logger.error(f"Error removing RAG model for session {session_uuid}: {e_del}")
        else:
            logger.debug(f"No RAG model found in cache for session {session_uuid}")

        # --- Delete Memory Collections ---
        try:
            # Clear memory store cache
            try:
                from src.models.memory.chat_memory_manager import clear_memory_store_cache_for_session
                clear_memory_store_cache_for_session(session_uuid)
                logger.info(f"Cleared memory store cache for session {session_uuid}")
            except ImportError:
                logger.warning("Memory manager not found, skipping memory store cache clearing.")
            except Exception as e_cache:
                logger.error(f"Error clearing memory store cache for session {session_uuid}: {e_cache}")
            
            # Delete the ChromaDB collection itself
            try:
                from src.models.memory.memory_chroma_manager import delete_memory_collection
                collection_deleted = delete_memory_collection(session_uuid)
                if collection_deleted:
                    logger.info(f"Deleted ChromaDB memory collection for session {session_uuid}")
                else:
                    logger.debug(f"No ChromaDB memory collection found for session {session_uuid}")
            except ImportError:
                logger.warning("Memory ChromaDB manager not found, skipping collection deletion.")
            except Exception as e_collection:
                logger.error(f"Error deleting memory collection for session {session_uuid}: {e_collection}")
                
        except Exception as e_memory:
            logger.error(f"Error handling memory cleanup for session {session_uuid}: {e_memory}")
        # --- End Memory Deletion ---

        return True
    except Exception as e:
        logger.error(f"Error deleting session {session_uuid}: {e}")
        return False
