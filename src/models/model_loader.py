# models/model_loader.py

import os
import torch
import time
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
from transformers import MllamaForConditionalGeneration
from transformers import AutoModelForCausalLM
from dotenv import load_dotenv
from src.utils.logger import get_logger
from src.models.memory.memory_manager import memory_manager, MemoryManager
from src.utils.resource_tracker import get_resource_tracker, track_operation
import gc  # Import gc for explicit calls

# Load environment variables from .env file
load_dotenv()

logger = get_logger(__name__)

# Models that only support single image processing
SINGLE_IMAGE_MODELS = {
    # 'phi-4-multimodal': True  # Removed - we'll handle this with collage logic
    # 'ollama-llama-vision': True  # Removed - we handle this with special collage logic
}

def is_single_image_model(model_choice):
    """Returns True if the model only supports processing a single image."""
    return model_choice in SINGLE_IMAGE_MODELS

def detect_device():
    """
    Detects the best available device (CUDA, MPS, or CPU).
    Delegates to memory_manager to ensure consistency.
    """
    return memory_manager.device

# Global caches for embedding models and processors
# Note: Main model caching is handled by memory_manager
_processor_cache = {}  # Cache for tokenizers and processors specifically
_global_embedding_models = {}  # Cache for embedding components (needed by load_rag_model)

# Track when each session's model was last accessed for idle unloading
_session_model_access_time = {}  # session_id -> timestamp


def unload_idle_session_models(rag_models, max_idle_seconds=600):
    """Unload embedding models for sessions idle longer than max_idle_seconds.

    This prevents GPU memory accumulation when multiple sessions have been used.
    Each idle ColPali model takes 4-6GB GPU — unloading them frees that memory.

    Args:
        rag_models: The RAG_models dict (ThreadSafeModelManager) from app config
        max_idle_seconds: Seconds of inactivity before unloading (default 10 min)

    Returns:
        Number of models unloaded
    """
    import time
    current_time = time.time()
    unloaded = 0

    idle_sessions = [
        sid for sid, access_time in list(_session_model_access_time.items())
        if current_time - access_time > max_idle_seconds
    ]

    for session_id in idle_sessions:
        try:
            if rag_models and hasattr(rag_models, 'get') and rag_models.get(session_id):
                model = rag_models.get(session_id)
                if model and hasattr(model, 'unload'):
                    model.unload()
                if hasattr(rag_models, 'remove'):
                    rag_models.remove(session_id)
                elif hasattr(rag_models, 'pop'):
                    rag_models.pop(session_id, None)

                _session_model_access_time.pop(session_id, None)
                unloaded += 1
                logger.info(f"[IDLE UNLOAD] Unloaded embedding model for idle session {session_id}")
        except Exception as e:
            logger.warning(f"[IDLE UNLOAD] Error unloading session {session_id}: {e}")

    if unloaded > 0:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info(f"[IDLE UNLOAD] Unloaded {unloaded} idle session models, freed GPU memory")

    return unloaded


def record_session_access(session_id):
    """Record that a session's model was accessed (called during retrieval)."""
    import time
    _session_model_access_time[session_id] = time.time()

def clear_model_caches(force_gpu_cleanup=True):
    """
    Clear all global model caches to ensure we don't have stale model instances.
    This function completely delegates to the memory manager for consistent memory
    management throughout the application, but first clears the local module caches.

    This function is maintained for backward compatibility - direct use of
    memory_manager.unload_all_models() is preferred for new code.

    Args:
        force_gpu_cleanup: If True, will attempt to explicitly free GPU memory
    """
    global _processor_cache, _global_embedding_models

    # Record memory usage before cleaning
    memory_before = memory_manager.get_memory_usage()

    # Count objects being cleared for logging
    proc_count = len(_processor_cache)
    embed_count = len(_global_embedding_models)

    logger.info(f"Clearing global model caches: {embed_count} embedding models, {proc_count} processors")

    # Clear processor cache (tokenizers are lightweight and safe to clear)
    _processor_cache.clear()
    _processor_cache = {}

    # Unload embedding models through the lifecycle manager (the canonical owner)
    # Don't clear _global_embedding_models directly — let lifecycle manager do it
    try:
        from src.models.model_lifecycle import get_lifecycle_manager
        lifecycle = get_lifecycle_manager()
        lc_count = lifecycle.unload_all()
        logger.info(f"Lifecycle manager unloaded {lc_count} model(s)")
        # Now safe to clear the shadow cache
        _global_embedding_models.clear()
        _global_embedding_models = {}
    except Exception as e:
        logger.warning(f"Error unloading via lifecycle manager: {e}")
        # Fallback: clear the shadow cache directly
        _global_embedding_models.clear()
        _global_embedding_models = {}

    # Clear LanceDB resources if available
    try:
        from src.models.vector_stores.lancedb_manager import destroy_lancedb_resources
        cleanup_info = destroy_lancedb_resources()
        logger.info(f"Cleared LanceDB resources: {cleanup_info}")
    except (ImportError, AttributeError):
        logger.debug("LanceDB manager not available for cache clearing")
    except Exception as e:
        logger.warning(f"Error clearing LanceDB resources: {e}")

    # Let the memory manager handle all system-level memory cleanup
    memory_manager.unload_all_models()

    # Log memory change
    memory_after = memory_manager.get_memory_usage()

    if memory_manager.is_cuda_available():
        gpu_before = memory_before.get('gpu_usage_gb', 0) or 0
        gpu_after = memory_after.get('gpu_usage_gb', 0) or 0
        freed_gpu = gpu_before - gpu_after
        logger.info(f"GPU memory freed: {freed_gpu:.2f}GB")

    ram_before = memory_before.get('ram_usage_gb', 0) or 0
    ram_after = memory_after.get('ram_usage_gb', 0) or 0
    freed_ram = ram_before - ram_after
    logger.info(f"RAM freed: {freed_ram:.2f}GB")

def aggressive_memory_cleanup():
    """
    Comprehensive memory cleanup function that can be called from anywhere.
    This function is maintained for backward compatibility - direct use of
    memory_manager.unload_all_models() followed by memory_manager.aggressive_cleanup()
    is preferred for new code.
    """
    logger.info("Performing aggressive memory cleanup through unified memory manager")
    memory_manager.unload_all_models()
    memory_manager.aggressive_cleanup()

    # Log memory status after cleanup
    memory = memory_manager.get_memory_usage()
    logger.info(f"Memory after aggressive cleanup: RAM={memory.get('ram_usage_gb')}GB, GPU={memory.get('gpu_usage_gb')}GB")

def load_embedding_model(indexer_model):
    """
    Loader for embedding models only. Returns a cached instance when possible.
    # Clear global caches to prevent memory accumulation
    global _processor_cache, _global_embedding_models
    _processor_cache.clear()
    _global_embedding_models.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    This function is optimized for ColQwen2.5 models.

    Args:
        indexer_model: The name of the embedding model to load

    Returns:
        The embedding model (ColQwen2.5 adapter).
    """
    global _global_embedding_models
    tracker = get_resource_tracker()

    # Check if we already have embedding components cached we could reuse
    embedding_key = f"embedding_{indexer_model}"

    # Ensure _global_embedding_models is initialized
    if not isinstance(_global_embedding_models, dict):
        _global_embedding_models = {}

    if embedding_key in _global_embedding_models:
        logger.info(f"Using cached embedding model for {indexer_model}")
        tracker.increment_reference(f"embedding_model_{embedding_key}")
        return _global_embedding_models[embedding_key]

    # We need to load a fresh model
    logger.info(f"Loading fresh embedding model for {indexer_model}")

    # Use our shared memory cleanup function
    logger.info("Running aggressive cleanup before model loading")
    memory_manager.aggressive_cleanup()

    # Load the model with resource tracking
    with track_operation(f"load_embedding_model_{indexer_model}"):
        try:
            # Check which adapter to use based on the model name
            if "colnomic" in indexer_model.lower():
                # Use the ColQwen2.5 adapter (handles ColNomic too)
                from src.models.colpali_adapter import ColPaliAdapter
                logger.info(f"Loading ColPali adapter for embedding: {indexer_model}")
                adapter = ColPaliAdapter(indexer_model)
                adapter.load()
            else:
                # Use the ColQwen2.5 adapter
                from src.models.colpali_adapter import ColPaliAdapter
                logger.info(f"Loading ColPali adapter for embedding: {indexer_model}")
                adapter = ColPaliAdapter(indexer_model)
                adapter.load()

            # Store the model in the cache
            _global_embedding_models[embedding_key] = adapter  # Re-enabled to fix double loading

            # Track the model in our resource tracker
            tracker.track_resource('embedding_model', embedding_key, {
                'model_name': indexer_model,
                'adapter_type': 'ColPaliAdapter'
            })

            return adapter
        except Exception as e:
            logger.error(f"Error loading embedding model {indexer_model}: {e}", exc_info=True)
            return None

def load_embedding_adapter(model_name):
    """Factory function to load the appropriate embedding adapter.

    Detects the model type from the name and returns the correct adapter:
    - ColPali models (colqwen, colnomic, colpali) -> ColPaliAdapter
    - Dense models (bge, e5, gte, nomic-embed) -> DenseEmbeddingAdapter
    - None/bm25 -> NullEmbeddingAdapter

    Args:
        model_name: Model identifier string

    Returns:
        A BaseEmbeddingAdapter instance (loaded and ready to use)
    """
    from src.models.model_registry import registry
    from src.models.embedding_adapters.base_adapter import EmbeddingType

    if not model_name or (model_name.lower() in ('none', 'bm25')):
        from src.models.embedding_adapters.null_adapter import NullEmbeddingAdapter
        return NullEmbeddingAdapter()

    spec = registry.detect(model_name)

    if spec and spec.embedding_type == EmbeddingType.MULTI_VECTOR:
        # ColPali-family visual embedding model
        return load_rag_model(model_name)

    if spec and spec.embedding_type == EmbeddingType.SINGLE_VECTOR:
        from src.models.embedding_adapters.dense_adapter import DenseEmbeddingAdapter
        adapter = DenseEmbeddingAdapter(model_name)
        adapter.load()
        return adapter

    # Fallback: keyword detection for models not in the registry
    model_lower = model_name.lower()
    if any(kw in model_lower for kw in ('colqwen', 'colnomic', 'colpali', 'colsmol')):
        return load_rag_model(model_name)

    if any(kw in model_lower for kw in ('bge', 'e5-', 'gte', 'nomic-embed', 'sentence-transformers')):
        from src.models.embedding_adapters.dense_adapter import DenseEmbeddingAdapter
        adapter = DenseEmbeddingAdapter(model_name)
        adapter.load()
        return adapter

    # Default: try ColPali for backward compatibility
    logger.warning(f"Unknown model type '{model_name}', defaulting to ColPali adapter")
    return load_rag_model(model_name)


def load_rag_model(indexer_model):
    """Load an embedding model via the lifecycle manager.

    The lifecycle manager handles caching, deduplication, VRAM eviction, and
    thread safety. This function is a thin entry point that preserves the
    existing call signature used throughout the codebase.

    Args:
        indexer_model: The name of the indexer model to load

    Returns:
        A loaded BaseEmbeddingAdapter instance, or None on error.
    """
    import psutil, os

    # Log memory before loading
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / (1024 * 1024)
    gpu_mem_before = torch.cuda.memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0
    logger.info(f"MEMORY DEBUG: Before RAG model loading: RAM={mem_before:.2f}MB, GPU={gpu_mem_before:.2f}MB")

    try:
        from src.models.model_lifecycle import get_lifecycle_manager
        lifecycle = get_lifecycle_manager()
        adapter = lifecycle.get_or_load(indexer_model)

        # Also keep in _global_embedding_models for backward compat with code that reads it
        global _global_embedding_models
        if isinstance(_global_embedding_models, dict):
            _global_embedding_models[f"embedding_{indexer_model}"] = adapter

        # Log memory after loading
        mem_after = process.memory_info().rss / (1024 * 1024)
        gpu_mem_after = torch.cuda.memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0
        logger.info(f"MEMORY DEBUG: After RAG model loading: RAM={mem_after:.2f}MB, GPU={gpu_mem_after:.2f}MB")
        logger.info(f"MEMORY DEBUG: Difference: RAM={mem_after-mem_before:.2f}MB, GPU={gpu_mem_after-gpu_mem_before:.2f}MB")

        return adapter
    except Exception as e:
        logger.error(f"Error loading RAG model {indexer_model}: {e}", exc_info=True)
        memory_manager.aggressive_cleanup()
        return None


def get_current_model():
    """
    Returns the currently loaded model from the memory manager.

    Returns:
        The currently loaded model, or None if no model is loaded.
    """
    # Check if there's a model in the memory manager's model_cache
    if hasattr(memory_manager, 'model_cache') and memory_manager.model_cache:
        # Get the first model in the cache (there should only be one active model)
        for model_name, model_data in memory_manager.model_cache.items():
            if model_data is not None:
                logger.info(f"Returning current model: {model_name}")
                return model_data

    logger.warning("No model currently loaded")
    return None

def load_model(model_choice):
    """
    Loads and caches the specified model using the model_manager.
    This improves memory management by unloading inactive models.

    When loading a new model, we automatically unload any previously loaded models
    to prevent CUDA memory buildup.

    NOTE: If the model is already loaded (current_model == model_choice), we will
    return the cached model instead of reloading it.
    """
    # MEMORY DEBUG: Log memory before model load function
    import psutil, os
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / (1024 * 1024)  # MB
    gpu_mem_before = 0
    if torch.cuda.is_available():
        gpu_mem_before = torch.cuda.memory_allocated() / (1024 * 1024)  # MB

    logger.info(f"MEMORY DEBUG: Start of load_model: RAM={mem_before:.2f}MB, GPU={gpu_mem_before:.2f}MB")

    # Defensive check - ensure model_cache exists in memory_manager
    if not hasattr(memory_manager, 'model_cache') or memory_manager.model_cache is None:
        memory_manager.model_cache = {}
        logger.warning("model_cache was not initialized in memory_manager, created empty cache")

    # Check if the requested model is already loaded and active
    has_model_in_cache = (model_choice in memory_manager.model_cache and
                          memory_manager.model_cache[model_choice] is not None)

    # First check if model is in cache regardless of current_model state
    if has_model_in_cache:
        logger.info(f"Model '{model_choice}' found in cache, using cached version")
        cached_model = memory_manager.model_cache[model_choice]

        # Update current model reference to match reality
        memory_manager.current_model = model_choice

        # MEMORY DEBUG: Log memory after finding cached model
        mem_after = process.memory_info().rss / (1024 * 1024)  # MB
        gpu_mem_after = 0
        if torch.cuda.is_available():
            gpu_mem_after = torch.cuda.memory_allocated() / (1024 * 1024)  # MB

        logger.info(f"MEMORY DEBUG: After using cached model: RAM={mem_after:.2f}MB, GPU={gpu_mem_after:.2f}MB")
        logger.info(f"MEMORY DEBUG: Difference: RAM={mem_after-mem_before:.2f}MB, GPU={gpu_mem_after-gpu_mem_before:.2f}MB")

        return cached_model

    # If current_model is set but model not in cache, clear the stale reference
    elif model_choice == memory_manager.current_model:
        logger.info(f"Model '{model_choice}' was marked as current but not in cache - clearing stale reference")
        memory_manager.current_model = None

    # If we need to load a different model, clear current models to prevent memory buildup
    logger.info(f"Loading model '{model_choice}' (current model was '{memory_manager.current_model}')")

    # Only clear app.RAG_models when actually switching to a different model
    # If we're just reloading the same model due to cache issues, preserve RAG_models
    previous_model = memory_manager.current_model
    if previous_model is not None and previous_model != model_choice:
        try:
            import sys
            if 'app' in sys.modules:
                app_module = sys.modules['app']
                if hasattr(app_module, 'RAG_models'):
                    logger.info(f"Switching from '{previous_model}' to '{model_choice}' - clearing {len(app_module.RAG_models)} entries from app.RAG_models")
                    app_module.RAG_models.clear()
        except Exception as e:
            logger.warning(f"Error clearing RAG_models: {e}")
    else:
        logger.info(f"Reloading same model '{model_choice}' - preserving RAG_models cache")

    # Delegate all memory management to the memory manager
    # This ensures consistent memory cleanup across the application
    memory_manager.unload_all_models()

    # Aggressive cleanup is now handled by memory_manager automatically
    # Define model loading functions for each model type




    def load_ollama_llama_vision():
        logger.info("Loading Ollama Llama 3.2 Vision model (Q4)")
        return (None, None, None, "llama3.2-vision")

    def load_ollama_llama_vision_q8():
        logger.info("Loading Ollama Llama 3.2 Vision model (Q8)")
        return (None, None, None, "llama3.2-vision:11b-instruct-q8_0")

    def load_ollama_llama_vision_fp16():
        logger.info("Loading Ollama Llama 3.2 Vision model (FP16)")
        return (None, None, None, "llama3.2-vision:11b-instruct-fp16")

    def load_ollama_minicpm_vision():
        logger.info("Ollama MiniCPM Vision model ready to use.")
        return (None, None, None, "minicpm-v")


    # Phi-3.5-Vision model loaders have been removed

    # Add text-only LLM loaders using Ollama

    def load_ollama_phi4():
        logger.info("Loading Ollama Phi-4 Plus 14B (Q4) model")
        return (None, None, None, "phi4-reasoning:plus")

    def load_ollama_phi4_q8():
        logger.info("Loading Ollama Phi-4 Plus 14B (Q8) model")
        return (None, None, None, "phi4-reasoning:14b-plus-q8_0")

    def load_ollama_phi4_fp16():
        logger.info("Loading Ollama Phi-4 Plus 14B (FP16) model")
        return (None, None, None, "phi4-reasoning:14b-plus-fp16")

    def load_ollama_phi4_mini():
        logger.info("Loading Ollama Phi-4 Mini Reasoning 3.8B (Q4) model")
        return (None, None, None, "phi4-mini-reasoning:latest")

    def load_ollama_phi4_mini_fp16():
        logger.info("Loading Ollama Phi-4 Mini Reasoning 3.8B (FP16) model")
        return (None, None, None, "phi4-mini-reasoning:3.8b-fp16")

    def load_ollama_olmo2():
        logger.info("Loading Ollama OLMo 2:13B text-only model")
        return (None, None, None, "olmo2:13b")

    def load_ollama_granite_vision():
        logger.info("Loading Ollama Granite-3.2 Vision model")
        return (None, None, None, "granite3.2-vision")

    def load_ollama_gemma_vision():
        logger.info("Loading Ollama Gemma 3 Vision model (12B Q4)")
        return (None, None, None, "gemma3:12b")

    def load_ollama_gemma_vision_12b_q8():
        logger.info("Loading Ollama Gemma 3 Vision model (12B Q8)")
        return (None, None, None, "gemma3:12b-it-q8_0")

    def load_ollama_gemma_vision_12b_fp16():
        logger.info("Loading Ollama Gemma 3 Vision model (12B FP16)")
        return (None, None, None, "gemma3:12b-it-fp16")

    def load_ollama_gemma_vision_27b():
        logger.info("Loading Ollama Gemma 3 Vision model (27B Q4)")
        return (None, None, None, "gemma3:27b")

    def load_ollama_gemma_vision_27b_q8():
        logger.info("Loading Ollama Gemma 3 Vision model (27B Q8)")
        return (None, None, None, "gemma3:27b-it-q8_0")

    def load_ollama_gemma_vision_27b_fp16():
        logger.info("Loading Ollama Gemma 3 Vision model (27B FP16)")
        return (None, None, None, "gemma3:27b-it-fp16")

    def load_ollama_gemma3n_vision_fp16():
        logger.info("Loading Ollama Gemma 3N Vision model (FP16)")
        return (None, None, None, "gemma3n:e4b-it-fp16")

    def load_ollama_exaone_deep_32b():
        logger.info("Loading Ollama Exaone Deep 32B text-only model")
        return (None, None, None, "exaone-deep:32b")

    def load_ollama_qwq():
        logger.info("Loading Ollama QWQ text-only model")
        return (None, None, None, "qwq")

    def load_ollama_qwen3():
        logger.info("Loading Ollama Qwen3 30B (Q4) text-only model")
        return (None, None, None, "qwen3:30b-a3b")

    def load_ollama_qwen3_q8():
        logger.info("Loading Ollama Qwen3 30B (Q8) text-only model")
        return (None, None, None, "qwen3:30b-a3b-q8_0")

    def load_ollama_qwen3_fp16():
        logger.info("Loading Ollama Qwen3 30B (FP16) text-only model")
        return (None, None, None, "qwen3:30b-a3b-fp16")

    def load_ollama_mistral_small32():
        logger.info("Loading Ollama Mistral Small 3.2 vision model")
        return (None, None, None, "mistral-small3.2:latest")

    def load_ollama_mistral_small32_24b():
        logger.info("Loading Ollama Mistral Small 3.2 24B (Q8) model")
        return (None, None, None, "mistral-small3.2:24b-instruct-2506-q8_0")

    def load_ollama_mistral_small32_24b_fp16():
        logger.info("Loading Ollama Mistral Small 3.2 24B (FP16) model")
        return (None, None, None, "mistral-small3.2:24b-instruct-2506-fp16")

    # --- Add new loader functions for Agentica DeepCoder ---
    def load_ollama_deepcoder_14b_q8():
        logger.info("Loading Ollama Agentica DeepCoder 14B (Q8) text-only model")
        return (None, None, None, "deepcoder:14b-preview-q8_0")

    def load_ollama_deepcoder_14b_fp16():
        logger.info("Loading Ollama Agentica DeepCoder 14B (FP16) text-only model")
        return (None, None, None, "deepcoder:14b-preview-fp16")


    def load_ollama_granite3_3():
        logger.info("Loading Ollama Granite 3.3 (Q4) model")
        return (None, None, None, "granite3.3")

    def load_ollama_granite3_3_2b():
        logger.info("Loading Ollama Granite 3.3 2B (Q4) model")
        return (None, None, None, "granite3.3:2b")

    def load_ollama_llama3_2_vision():
        logger.info("Loading Ollama Llama 3.2 Vision 90B (Q4) model")
        return (None, None, None, "llama3.2-vision")

    def load_ollama_llama4_scout():
        logger.info("Loading Ollama Llama 4 Scout 17B (Q4) model")
        return (None, None, None, "llama4:scout")

    def load_ollama_llama4_scout_q8():
        logger.info("Loading Ollama Llama 4 Scout 17B (Q8) model")
        return (None, None, None, "llama4:17b-scout-16e-instruct-q8_0")

    def load_ollama_llama4_scout_fp16():
        logger.info("Loading Ollama Llama 4 Scout 17B (FP16) model")
        return (None, None, None, "llama4:17b-scout-16e-instruct-fp16")

    def load_ollama_llama4_maverick():
        logger.info("Loading Ollama Llama 4 Maverick 17B (Q4) model")
        return (None, None, None, "llama4:maverick")

    def load_ollama_llama4_maverick_q8():
        logger.info("Loading Ollama Llama 4 Maverick 17B (Q8) model")
        return (None, None, None, "llama4:17b-maverick-128e-instruct-q8_0")

    def load_ollama_llama4_maverick_fp16():
        logger.info("Loading Ollama Llama 4 Maverick 17B (FP16) model")
        return (None, None, None, "llama4:17b-maverick-128e-instruct-fp16")

    def load_ollama_qwen2_5vl():
        logger.info("Loading Ollama Qwen 2.5 Vision 7B (Q4) model")
        return (None, None, None, "qwen2.5vl:7b")

    def load_ollama_qwen2_5vl_q8():
        logger.info("Loading Ollama Qwen 2.5 Vision 7B (Q8) model")
        return (None, None, None, "qwen2.5vl:7b-q8_0")

    def load_ollama_qwen2_5vl_fp16():
        logger.info("Loading Ollama Qwen 2.5 Vision 7B (FP16) model")
        return (None, None, None, "qwen2.5vl:7b-fp16")

    # 32B model loaders
    def load_ollama_qwen2_5vl_32b():
        logger.info("Loading Ollama Qwen 2.5 Vision 32B (Q4) model")
        return (None, None, None, "qwen2.5vl:32b")

    def load_ollama_qwen2_5vl_32b_q8():
        logger.info("Loading Ollama Qwen 2.5 Vision 32B (Q8) model")
        return (None, None, None, "qwen2.5vl:32b-q8_0")

    def load_ollama_qwen2_5vl_32b_fp16():
        logger.info("Loading Ollama Qwen 2.5 Vision 32B (FP16) model")
        return (None, None, None, "qwen2.5vl:32b-fp16")

    # 72B model loaders
    def load_ollama_qwen2_5vl_72b():
        logger.info("Loading Ollama Qwen 2.5 Vision 72B (Q4) model")
        return (None, None, None, "qwen2.5vl:72b")

    def load_ollama_qwen2_5vl_72b_q8():
        logger.info("Loading Ollama Qwen 2.5 Vision 72B (Q8) model")
        return (None, None, None, "qwen2.5vl:72b-q8_0")

    def load_ollama_qwen2_5vl_72b_fp16():
        logger.info("Loading Ollama Qwen 2.5 Vision 72B (FP16) model")
        return (None, None, None, "qwen2.5vl:72b-fp16")

    # Mistral Codestral models
    def load_ollama_devstral():
        logger.info("Loading Ollama Mistral Codestral 24B (Q4) model")
        return (None, None, None, "devstral:24b")

    def load_ollama_devstral_q8():
        logger.info("Loading Ollama Mistral Codestral 24B (Q8) model")
        return (None, None, None, "devstral:24b-small-2505-q8_0")

    def load_ollama_devstral_fp16():
        logger.info("Loading Ollama Mistral Codestral 24B (FP16) model")
        return (None, None, None, "devstral:24b-small-2505-fp16")

    # --- Add new loader functions for DeepSeek R1 ---
    def load_ollama_deepseek_r1_q4():
        logger.info("Loading Ollama DeepSeek R1 8B (Q4) reasoning model")
        return (None, None, None, "deepseek-r1:8b-0528-qwen3-q4_K_M")

    def load_ollama_deepseek_r1_q8():
        logger.info("Loading Ollama DeepSeek R1 8B (Q8) reasoning model")
        return (None, None, None, "deepseek-r1:8b-0528-qwen3-q8_0")

    def load_ollama_deepseek_r1_fp16():
        logger.info("Loading Ollama DeepSeek R1 8B (FP16) reasoning model")
        return (None, None, None, "deepseek-r1:8b-0528-qwen3-fp16")

    # --- Add Mistral Magistral loader functions ---
    def load_ollama_magistral():
        logger.info("Loading Ollama Mistral Magistral 24B (Q4) text-only model")
        return (None, None, None, "magistral:24b")

    def load_ollama_magistral_q8():
        logger.info("Loading Ollama Mistral Magistral 24B (Q8) text-only model")
        return (None, None, None, "magistral:24b-small-2506-q8_0")

    def load_ollama_magistral_fp16():
        logger.info("Loading Ollama Mistral Magistral 24B (FP16) text-only model")
        return (None, None, None, "magistral:24b-small-2506-fp16")

    # --- Ollama cloud-proxied and new models ---

    def load_ollama_glm5_1():
        logger.info("Loading Ollama GLM 5.1 (cloud-proxied)")
        return (None, None, None, "glm-5.1:cloud")

    def load_ollama_minimax_m2_7():
        logger.info("Loading Ollama MiniMax M2.7 (cloud-proxied)")
        return (None, None, None, "minimax-m2.7:cloud")

    def load_ollama_gpt_oss():
        logger.info("Loading Ollama GPT-OSS 20B")
        return (None, None, None, "gpt-oss:20b")

    # --- HuggingFace Models ---

    def load_huggingface_points_reader():
        """Load POINTS-Reader model from HuggingFace."""
        logger.info("Loading POINTS-Reader document extraction model from HuggingFace...")

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from transformers import Qwen2VLImageProcessor

            model_name = "tencent/POINTS-Reader"
            device = detect_device()

            logger.info(f"Loading model {model_name} on device {device}...")

            # Load model
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            logger.info(f"Model loaded successfully on {device}")

            # Load tokenizer and image processor
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            image_processor = Qwen2VLImageProcessor.from_pretrained(model_name)

            # Put model in eval mode
            model.eval()

            return (model, (tokenizer, image_processor), model_name)

        except Exception as e:
            logger.error(f"Error loading POINTS-Reader: {e}", exc_info=True)
            raise

    def load_huggingface_kosmos25():
        """Load Kosmos-2.5 model from HuggingFace."""
        logger.info("Loading Kosmos-2.5 OCR model from HuggingFace...")

        try:
            from transformers import AutoProcessor, Kosmos2_5ForConditionalGeneration

            model_name = "microsoft/kosmos-2.5"
            device = detect_device()

            logger.info(f"Loading model {model_name} on device {device}...")

            # Load model
            model = Kosmos2_5ForConditionalGeneration.from_pretrained(
                model_name,
                device_map="auto",
                torch_dtype=torch.bfloat16
            )
            logger.info(f"Model loaded successfully on {device}")

            # Load processor
            processor = AutoProcessor.from_pretrained(model_name)

            # Put model in eval mode
            model.eval()

            return (model, processor, model_name)

        except Exception as e:
            logger.error(f"Error loading Kosmos-2.5: {e}", exc_info=True)
            raise

    def load_huggingface_qwen3vl():
        """Load Qwen3-VL-8B model from HuggingFace."""
        logger.info("Loading Qwen3-VL-8B-Instruct vision-language model from HuggingFace...")

        try:
            from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

            model_name = "Qwen/Qwen3-VL-8B-Instruct"
            device = detect_device()

            logger.info(f"Loading model {model_name} on device {device}...")

            # Load model
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype="auto",
                device_map="auto"
            )
            logger.info(f"Model loaded successfully on {device}")

            # Load processor
            processor = AutoProcessor.from_pretrained(model_name)

            # Put model in eval mode
            model.eval()

            return (model, processor, model_name)

        except Exception as e:
            logger.error(f"Error loading Qwen3-VL-8B: {e}", exc_info=True)
            raise

    def load_huggingface_qwen3vl_4b():
        """Load Qwen3-VL-4B model from HuggingFace."""
        logger.info("Loading Qwen3-VL-4B-Instruct vision-language model from HuggingFace...")

        try:
            from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

            model_name = "Qwen/Qwen3-VL-4B-Instruct"
            device = detect_device()

            logger.info(f"Loading model {model_name} on device {device}...")

            # Load model
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype="auto",
                device_map="auto"
            )
            logger.info(f"Model loaded successfully on {device}")

            # Load processor
            processor = AutoProcessor.from_pretrained(model_name)

            # Put model in eval mode
            model.eval()

            return (model, processor, model_name)

        except Exception as e:
            logger.error(f"Error loading Qwen3-VL-4B: {e}", exc_info=True)
            raise

    def load_huggingface_qwen3vl_fp8():
        """Load Qwen3-VL-8B FP8 quantized model from HuggingFace."""
        logger.info("Loading Qwen3-VL-8B-Instruct-FP8 vision-language model from HuggingFace...")

        try:
            from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

            model_name = "Qwen/Qwen3-VL-8B-Instruct-FP8"
            device = detect_device()

            logger.info(f"Loading FP8 quantized model {model_name} on device {device}...")

            # Load model (FP8 quantization is built into the model weights)
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype="auto",
                device_map="auto"
            )
            logger.info(f"FP8 model loaded successfully on {device}")

            # Load processor
            processor = AutoProcessor.from_pretrained(model_name)

            # Put model in eval mode
            model.eval()

            return (model, processor, model_name)

        except Exception as e:
            logger.error(f"Error loading Qwen3-VL-8B-FP8: {e}", exc_info=True)
            raise

    def load_huggingface_qwen3vl_4b_fp8():
        """Load Qwen3-VL-4B FP8 quantized model from HuggingFace."""
        logger.info("Loading Qwen3-VL-4B-Instruct-FP8 vision-language model from HuggingFace...")

        try:
            from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

            model_name = "Qwen/Qwen3-VL-4B-Instruct-FP8"
            device = detect_device()

            logger.info(f"Loading FP8 quantized model {model_name} on device {device}...")

            # Load model (FP8 quantization is built into the model weights)
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype="auto",
                device_map="auto"
            )
            logger.info(f"FP8 model loaded successfully on {device}")

            # Load processor
            processor = AutoProcessor.from_pretrained(model_name)

            # Put model in eval mode
            model.eval()

            return (model, processor, model_name)

        except Exception as e:
            logger.error(f"Error loading Qwen3-VL-4B-FP8: {e}", exc_info=True)
            raise

    def load_huggingface_nanonets_ocr2():
        """Load Nanonets-OCR2-3B model from HuggingFace."""
        logger.info("Loading Nanonets-OCR2-3B OCR model from HuggingFace...")

        try:
            from transformers import AutoProcessor, AutoModelForImageTextToText

            model_name = "nanonets/Nanonets-OCR2-3B"
            device = detect_device()

            logger.info(f"Loading model {model_name} on device {device}...")

            # Load model with flash attention if available
            try:
                model = AutoModelForImageTextToText.from_pretrained(
                    model_name,
                    torch_dtype="auto",
                    device_map="auto",
                    attn_implementation="flash_attention_2"
                )
                logger.info(f"Model loaded with flash_attention_2 on {device}")
            except Exception as e:
                logger.warning(f"Flash attention not available, using default: {e}")
                model = AutoModelForImageTextToText.from_pretrained(
                    model_name,
                    torch_dtype="auto",
                    device_map="auto"
                )
                logger.info(f"Model loaded successfully on {device}")

            # Load processor
            processor = AutoProcessor.from_pretrained(model_name)

            # Put model in eval mode
            model.eval()

            return (model, processor, model_name)

        except Exception as e:
            logger.error(f"Error loading Nanonets-OCR2-3B: {e}", exc_info=True)
            raise

    def load_huggingface_paddleocr_vl():
        """Load PaddleOCR-VL model using PaddleOCR library."""
        logger.info("Loading PaddleOCR-VL document parser...")

        try:
            from paddleocr import PaddleOCR

            model_name = "PaddlePaddle/PaddleOCR-VL"

            logger.info(f"Initializing PaddleOCR pipeline...")

            # Initialize PaddleOCR with document-focused settings
            pipeline = PaddleOCR(
                use_angle_cls=True,  # Enable text orientation classification
                lang='en',  # Default to English, supports 80+ languages
                use_gpu=True,  # Use GPU if available
                show_log=False  # Reduce console output
            )
            logger.info(f"PaddleOCR pipeline initialized successfully")

            # Return pipeline as both model and processor
            # (to maintain compatibility with handler expectations)
            return (pipeline, None, model_name)

        except ImportError as e:
            logger.error(f"PaddleOCR not installed. Install with: pip install paddleocr")
            logger.error(f"Error: {e}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"Error loading PaddleOCR: {e}", exc_info=True)
            raise

    def load_huggingface_olmocr():
        """Load olmOCR-2-7B-1025 model from HuggingFace."""
        logger.info("Loading olmOCR-2-7B-1025 OCR model from HuggingFace...")

        try:
            from transformers import AutoModelForVision2Seq, AutoProcessor

            model_name = "allenai/olmOCR-2-7B-1025"
            device = detect_device()

            logger.info(f"Loading model {model_name} on device {device}...")

            # Load model with automatic architecture detection
            # olmOCR is based on Qwen2.5-VL, so we use AutoModelForVision2Seq
            model = AutoModelForVision2Seq.from_pretrained(
                model_name,
                dtype="auto",
                device_map="auto"
            )
            logger.info(f"Model loaded successfully on {device}")

            # Load processor
            processor = AutoProcessor.from_pretrained(model_name)

            # Put model in eval mode
            model.eval()

            return (model, processor, model_name)

        except Exception as e:
            logger.error(f"Error loading olmOCR-2-7B-1025: {e}", exc_info=True)
            raise

    # Map model choices to their loading functions
    loaders = {
        'ollama-llama-vision': load_ollama_llama_vision,
        'ollama-phi4:14b-fp16': load_ollama_phi4_fp16,

        'ollama-phi4:14b-q8': load_ollama_phi4_q8,

        'ollama-phi4': load_ollama_phi4,

        'ollama-llama3.2-vision': load_ollama_llama3_2_vision,

        'ollama-granite3.3:2b': load_ollama_granite3_3_2b,

        'ollama-granite3.3': load_ollama_granite3_3,

        'ollama-llama-vision-q8': load_ollama_llama_vision_q8,
        'ollama-llama-vision-fp16': load_ollama_llama_vision_fp16,
        'ollama-minicpm-vision': load_ollama_minicpm_vision,
        'ollama-granite-vision': load_ollama_granite_vision,
        'ollama-gemma-vision': load_ollama_gemma_vision,
        'ollama-gemma-vision-12b-q8': load_ollama_gemma_vision_12b_q8,
        'ollama-gemma-vision-12b-fp16': load_ollama_gemma_vision_12b_fp16,
        'ollama-gemma-vision-27b': load_ollama_gemma_vision_27b,
        'ollama-gemma-vision-27b-q8': load_ollama_gemma_vision_27b_q8,
        'ollama-gemma-vision-27b-fp16': load_ollama_gemma_vision_27b_fp16,
        'ollama-gemma3n-vision-fp16': load_ollama_gemma3n_vision_fp16,
        'ollama-phi4': load_ollama_phi4,
        'ollama-phi4-q8': load_ollama_phi4_q8,
        'ollama-phi4-fp16': load_ollama_phi4_fp16,
        'ollama-phi4-mini': load_ollama_phi4_mini,
        'ollama-phi4-mini-fp16': load_ollama_phi4_mini_fp16,
        'ollama-phi4-mini-reasoning:latest': load_ollama_phi4_mini,
        'ollama-phi4-reasoning:plus': load_ollama_phi4,
        'ollama-phi4-reasoning:14b-plus-q8_0': load_ollama_phi4_q8,
        'ollama-phi4-reasoning:14b-plus-fp16': load_ollama_phi4_fp16,
        'ollama-olmo2': load_ollama_olmo2,
        'ollama-exaone-deep:32b': load_ollama_exaone_deep_32b,
        'ollama-qwq': load_ollama_qwq,
        'ollama-mistral-small32': load_ollama_mistral_small32,
        'ollama-mistral-small32-24b': load_ollama_mistral_small32_24b,
        'ollama-mistral-small32-24b-fp16': load_ollama_mistral_small32_24b_fp16,
        # --- Add Agentica DeepCoder mappings ---
        'ollama-deepcoder:14b-preview-q8_0': load_ollama_deepcoder_14b_q8,
        'ollama-deepcoder:14b-preview-fp16': load_ollama_deepcoder_14b_fp16,
        'ollama-granite3.3': load_ollama_granite3_3,
        # --- Add Qwen3 models ---
        'ollama-qwen3': load_ollama_qwen3,
        'ollama-qwen3-q8': load_ollama_qwen3_q8,
        'ollama-qwen3-fp16': load_ollama_qwen3_fp16,
        # --- Add Llama4 Scout models ---
        'ollama-llama4-scout': load_ollama_llama4_scout,
        'ollama-llama4-scout-q8': load_ollama_llama4_scout_q8,
        'ollama-llama4-scout-fp16': load_ollama_llama4_scout_fp16,
        # --- Add Llama4 Maverick models ---
        'ollama-llama4-maverick': load_ollama_llama4_maverick,
        'ollama-llama4-maverick-q8': load_ollama_llama4_maverick_q8,
        'ollama-llama4-maverick-fp16': load_ollama_llama4_maverick_fp16,
        # --- Add Qwen 2.5 Vision models ---
        'ollama-qwen2.5vl': load_ollama_qwen2_5vl,
        'ollama-qwen2.5vl-q8': load_ollama_qwen2_5vl_q8,
        'ollama-qwen2.5vl-fp16': load_ollama_qwen2_5vl_fp16,
        'ollama-qwen2.5vl-32b': load_ollama_qwen2_5vl_32b,
        'ollama-qwen2.5vl-32b-q8': load_ollama_qwen2_5vl_32b_q8,
        'ollama-qwen2.5vl-32b-fp16': load_ollama_qwen2_5vl_32b_fp16,
        'ollama-qwen2.5vl-72b': load_ollama_qwen2_5vl_72b,
        'ollama-qwen2.5vl-72b-q8': load_ollama_qwen2_5vl_72b_q8,
        'ollama-qwen2.5vl-72b-fp16': load_ollama_qwen2_5vl_72b_fp16,
        # Mistral Codestral models
        'ollama-devstral': load_ollama_devstral,
        'ollama-devstral-q8': load_ollama_devstral_q8,
        'ollama-devstral-fp16': load_ollama_devstral_fp16,
        # --- Add DeepSeek R1 mappings ---
        'ollama-deepseek-r1-q4': load_ollama_deepseek_r1_q4,
        'ollama-deepseek-r1-q8': load_ollama_deepseek_r1_q8,
        'ollama-deepseek-r1-fp16': load_ollama_deepseek_r1_fp16,
        # --- Add Mistral Magistral mappings ---
        'ollama-magistral': load_ollama_magistral,
        'ollama-magistral-q8': load_ollama_magistral_q8,
        'ollama-magistral-fp16': load_ollama_magistral_fp16,
        # --- Ollama cloud-proxied and new models ---
        'ollama-glm5.1': load_ollama_glm5_1,
        'ollama-minimax-m2.7': load_ollama_minimax_m2_7,
        'ollama-gpt-oss': load_ollama_gpt_oss,
        # --- HuggingFace models ---
        'huggingface-points-reader': load_huggingface_points_reader,
        'huggingface-kosmos25': load_huggingface_kosmos25,
        'huggingface-qwen3vl': load_huggingface_qwen3vl,
        'huggingface-qwen3vl-4b': load_huggingface_qwen3vl_4b,
        'huggingface-qwen3vl-fp8': load_huggingface_qwen3vl_fp8,
        'huggingface-qwen3vl-4b-fp8': load_huggingface_qwen3vl_4b_fp8,
        'huggingface-nanonets-ocr2': load_huggingface_nanonets_ocr2,
        'huggingface-paddleocr-vl': load_huggingface_paddleocr_vl,
        'huggingface-olmocr': load_huggingface_olmocr,
    }


    if model_choice not in loaders:
        # Fallback: handle dynamic Ollama models (e.g. "ollama-gemma4:latest")
        if model_choice.startswith('ollama-'):
            ollama_tag = model_choice[len('ollama-'):]
            logger.info(f"Using dynamic Ollama model: {ollama_tag}")
            return (None, None, None, ollama_tag)
        logger.error(f"Invalid model choice: {model_choice}")
        raise ValueError("Invalid model choice.")

    # Use the memory manager to get the model, passing the loader function
    return memory_manager.get_model(model_choice, loaders[model_choice])
