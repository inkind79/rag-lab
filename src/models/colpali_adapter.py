"""
Custom adapter for ColPali-based models (ColQwen2.5, ColNomic).

This module provides a custom adapter for models using the ColPali engine
(like tsystems/colqwen2.5-3b-multilingual-v1.0, nomic-ai/colnomic-embed-multimodal-*)
that allows them to be used with the RAG Lab application.
"""

import torch
import numpy as np
from src.utils.logger import get_logger
from src.models.embedding_adapters.base_adapter import BaseEmbeddingAdapter, EmbeddingType
from transformers.utils.import_utils import is_flash_attn_2_available

logger = get_logger(__name__)

class ColPaliAdapter(BaseEmbeddingAdapter):
    """
    Adapter for ColPali-based models (ColQwen2.5, ColNomic).

    This adapter provides methods to load and use models based on the ColPali engine
    (like tsystems/colqwen2.5-3b-multilingual-v1.0, nomic-ai/colnomic-embed-multimodal-*)
    with the RAG Lab application.

    These models use a ColBERT-style multi-vector representation, which requires
    special handling for both encoding and scoring via the colpali_engine library.
    """

    # --- BaseEmbeddingAdapter interface ---

    @property
    def embedding_type(self) -> EmbeddingType:
        return EmbeddingType.MULTI_VECTOR

    @property
    def dimension(self) -> int:
        from src.models.model_registry import registry
        return registry.get_dimension(self._model_name)

    def unload(self) -> None:
        """Release GPU resources."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info(f"Unloaded ColPali model: {self._model_name}")

    # --- End BaseEmbeddingAdapter interface ---

    def __init__(self, model_name="athrael-soju/colqwen3.5-4.5B-v3"):
        """
        Initialize the adapter.

        Args:
            model_name: The name of the model to load. Supports ColQwen2.5, ColQwen3,
                       ColQwen3.5, and ColSmol models via the model registry.
        """
        self._model_name = model_name
        self.model = None
        self.processor = None

        # Enable deterministic mode
        from src.utils.deterministic_config import set_deterministic_mode
        set_deterministic_mode(seed=42)

    @property
    def model_name(self) -> str:
        return self._model_name

    @model_name.setter
    def model_name(self, value: str):
        self._model_name = value

    def load(self):
        """
        Load the model and processor using the model registry to determine
        the correct colpali_engine class.

        Returns:
            The loaded model.
        """
        try:
            import colpali_engine.models as ce_models
            from src.models.model_registry import registry

            # Look up model class from registry
            spec = registry.detect(self.model_name)
            if spec:
                model_cls_name = spec.model_class_name
                proc_cls_name = spec.processor_class_name
            else:
                # Fallback for unregistered models
                model_cls_name = "ColQwen2_5"
                proc_cls_name = "ColQwen2_5_Processor"
                logger.warning(f"Model '{self.model_name}' not in registry, defaulting to {model_cls_name}")

            model_cls = getattr(ce_models, model_cls_name)
            proc_cls = getattr(ce_models, proc_cls_name)
            logger.info(f"Using {model_cls_name}/{proc_cls_name} for {self.model_name}")

            # Set up flash attention if available
            attn_implementation = "flash_attention_2" if is_flash_attn_2_available() else None

            # Load the model on GPU
            logger.info(f"Loading ColPali model: {self.model_name} on {'GPU' if torch.cuda.is_available() else 'CPU'}")
            dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
            logger.info(f"Using {dtype} precision for {'GPU' if torch.cuda.is_available() else 'CPU'} operation")

            self.model = model_cls.from_pretrained(
                self.model_name,
                torch_dtype=dtype,
                device_map="cuda:0" if torch.cuda.is_available() else "cpu",
                attn_implementation=attn_implementation if torch.cuda.is_available() else None,
            ).eval()

            # Load the processor
            logger.info(f"Loading ColPali processor: {self.model_name}")
            self.processor = proc_cls.from_pretrained(self.model_name)

            logger.info(f"Successfully loaded ColPali model and processor for {self.model_name}")
            return self.model

        except Exception as e:
            logger.error(f"Error loading ColPali model ({self.model_name}): {e}", exc_info=True)
            raise

    def encode_query(self, query):
        """
        Encode a query using the model with caching support.

        Args:
            query: The query to encode.

        Returns:
            The encoded query.
        """
        # Check cache first
        from src.models.embedding_cache import EmbeddingCache
        cache = EmbeddingCache(model_name=self.model_name)
        
        cached_embedding = cache.get(query)
        if cached_embedding is not None:
            logger.info(f"Using cached embedding for query: '{query[:50]}...'")
            return cached_embedding
            
        if self.model is None or self.processor is None:
            self.load()

        try:
            # Process the query
            logger.info(f"Processing query with ColPali ({self.model_name}): '{query[:50]}...'")

            # Get user settings for query processing
            # We'll use a reasonable default but not truncate too aggressively
            max_query_length = 512  # More generous limit for query length

            # Try to get user settings from session
            try:
                from src.services.session_manager.manager import load_session
                try:
                    from flask import session as flask_session
                    session_id = flask_session.get('active_session_uuid')
                except (ImportError, RuntimeError):
                    session_id = None

                if session_id:
                    # Load session data
                    session_data = load_session('sessions', session_id)

                    # Get model settings directly from session_data
                    if session_data and 'model_params' in session_data:
                        model_params = session_data.get('model_params', {})
                        # Use num_ctx if available as a guide for query length
                        if 'ollama' in model_params:
                            ollama_params = model_params['ollama']
                            if 'num_ctx' in ollama_params:
                                # Use a fraction of the context length
                                max_query_length = min(512, int(ollama_params['num_ctx'] / 16))
                                logger.debug(f"Using context-based query length limit: {max_query_length}")
            except Exception as e:
                logger.warning(f"Error getting user query settings, using defaults: {e}")

            # Truncate long queries to improve speed
            if len(query) > max_query_length:
                logger.debug(f"Truncating query from {len(query)} to {max_query_length} characters")
                query = query[:max_query_length]

            # Process the query
            # The ColPali processor doesn't support custom parameters
            # So we'll just use the default processing
            batch_query = self.processor.process_queries([query]).to(self.model.device)

            # Forward pass with memory optimization and deterministic settings
            with torch.no_grad():
                # Ensure deterministic behavior
                from src.utils.deterministic_config import is_deterministic_mode_enabled
                if not is_deterministic_mode_enabled():
                    logger.warning("Deterministic mode not enabled, embeddings may vary between runs")
                    from src.utils.deterministic_config import set_deterministic_mode
                    set_deterministic_mode(seed=42)

                # Use torch.amp.autocast for mixed precision to reduce memory usage and improve speed
                # This is needed for FlashAttention compatibility - keep as in original code
                if torch.cuda.is_available():
                    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                        query_embedding = self.model(**batch_query)
                else:
                    query_embedding = self.model(**batch_query)

            # Clear CUDA cache to free up memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Cache the result before returning
            cache.put(query, query_embedding)
            logger.info(f"Cached embedding for query: '{query[:50]}...'")
            
            return query_embedding

        except Exception as e:
            logger.error(f"Error encoding query: {e}", exc_info=True)
            raise

    def encode_text(self, text):
        """
        Encode text using the model. Delegates to encode_query which handles
        caching internally.

        Args:
            text: The text to encode.

        Returns:
            The encoded text.
        """
        # encode_query already handles caching, deterministic mode, and GPU optimization.
        # No need to duplicate that logic here.
        return self.encode_query(text)

    def encode_image(self, images):
        """
        Encode images using the model.

        Args:
            images: The images to encode.

        Returns:
            The encoded images.
        """
        if self.model is None or self.processor is None:
            self.load()

        try:
            # Process the images
            logger.info(f"Processing {len(images)} images with ColPali ({self.model_name})")

            # Let each model's processor handle resolution natively.
            # Forced 448x448 resize was a legacy hack that destroyed detail
            # and produced poor retrieval quality for all model families.
            batch_images = self.processor.process_images(images).to(self.model.device)

            # Forward pass with memory optimization and deterministic settings
            with torch.no_grad():
                # Ensure deterministic behavior
                from src.utils.deterministic_config import is_deterministic_mode_enabled
                if not is_deterministic_mode_enabled():
                    logger.warning("Deterministic mode not enabled, embeddings may vary between runs")
                    from src.utils.deterministic_config import set_deterministic_mode
                    set_deterministic_mode(seed=42)

                # Use torch.amp.autocast for mixed precision to reduce memory usage and improve speed
                # This is needed for FlashAttention compatibility - keep as in original code
                if torch.cuda.is_available():
                    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                        image_embeddings = self.model(**batch_images)
                else:
                    image_embeddings = self.model(**batch_images)

            # Clear CUDA cache to free up memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return image_embeddings

        except Exception as e:
            logger.error(f"Error encoding images: {e}", exc_info=True)
            raise

    def process_embedding_for_lancedb(self, embedding, source="unknown"):
        """
        Process a multi-vector embedding for LanceDB storage.
        This implementation preserves the full multi-vector representation.

        Args:
            embedding: The embedding tensor from ColQwen2.5
            source: Source identifier for logging

        Returns:
            A numpy array preserving the multi-vector representation
        """
        try:
            logger.info(f"Processing ColPali ({self.model_name}) {source} embedding for LanceDB")

            # Convert to CPU and numpy
            if isinstance(embedding, torch.Tensor):
                # Get the shape for logging
                shape = embedding.shape
                logger.info(f"Original embedding shape: {shape}")

                # Move to CPU first if on GPU
                if embedding.device.type == 'cuda':
                    embedding = embedding.cpu()

                # Convert to float32 for consistent dtype
                if embedding.dtype != torch.float32:
                    embedding = embedding.to(torch.float32)

                # For LanceDB, we preserve the multi-vector representation
                if len(shape) == 3:  # [batch_size, seq_len, hidden_dim]
                    # Remove batch dimension if present
                    if shape[0] == 1:
                        embedding = embedding.squeeze(0)
                        logger.info(f"Removed batch dimension, new shape: {embedding.shape}")

                    # Convert to numpy, preserving the multi-vector structure
                    processed = embedding.numpy()

                    # Log detailed information about the embedding
                    logger.info(f"Processed embedding shape: {processed.shape}")
                    logger.info(f"Embedding stats: min={processed.min():.4f}, max={processed.max():.4f}, mean={processed.mean():.4f}")

                    # Filter out zero or near-zero vectors
                    norms = np.linalg.norm(processed, axis=1)
                    zero_vectors = np.sum(norms < 1e-6)
                    if zero_vectors > 0:
                        logger.warning(f"Found {zero_vectors} vectors with near-zero norm out of {processed.shape[0]}")
                        # Filter out zero vectors if there are too many
                        if zero_vectors < processed.shape[0]:  # Don't filter if all vectors are zero
                            processed = processed[norms >= 1e-6]
                            logger.info(f"Filtered out zero vectors, new shape: {processed.shape}")

                    # Log the first few vectors for debugging
                    if processed.shape[0] > 0:
                        first_vector = processed[0]
                        logger.debug(f"First vector (first 5 values): {first_vector[:5]}")

                        if processed.shape[0] > 1:
                            second_vector = processed[1]
                            # Calculate cosine similarity between first two vectors to check diversity
                            similarity = np.dot(first_vector, second_vector) / (np.linalg.norm(first_vector) * np.linalg.norm(second_vector))
                            logger.debug(f"Similarity between first two vectors: {similarity:.4f}")
                    # Normalize vectors to unit length for cosine similarity
                    norms = np.linalg.norm(processed, axis=1, keepdims=True)
                    # Avoid division by zero
                    norms = np.maximum(norms, 1e-10)
                    processed = processed / norms
                    logger.info(f"Normalized embedding shape: {processed.shape}")
                    logger.info(f"Normalized embedding stats: min={processed.min():.4f}, max={processed.max():.4f}, mean={processed.mean():.4f}")

                    return processed
                elif len(shape) == 2:  # [seq_len, hidden_dim]
                    # Already in the right format, just convert to numpy
                    processed = embedding.numpy()
                    logger.info(f"Embedding already in correct shape: {processed.shape}")

                    # Filter out zero or near-zero vectors
                    norms = np.linalg.norm(processed, axis=1)
                    zero_vectors = np.sum(norms < 1e-6)
                    if zero_vectors > 0:
                        logger.warning(f"Found {zero_vectors} vectors with near-zero norm out of {processed.shape[0]}")
                        # Filter out zero vectors if there are too many
                        if zero_vectors < processed.shape[0]:  # Don't filter if all vectors are zero
                            processed = processed[norms >= 1e-6]
                            logger.info(f"Filtered out zero vectors, new shape: {processed.shape}")
                    # Normalize vectors to unit length for cosine similarity
                    norms = np.linalg.norm(processed, axis=1, keepdims=True)
                    # Avoid division by zero
                    norms = np.maximum(norms, 1e-10)
                    processed = processed / norms
                    logger.info(f"Normalized embedding shape: {processed.shape}")
                    logger.info(f"Normalized embedding stats: min={processed.min():.4f}, max={processed.max():.4f}, mean={processed.mean():.4f}")

                    return processed
                elif len(shape) == 1:  # [hidden_dim]
                    # Single vector, add sequence dimension
                    embedding = embedding.unsqueeze(0)  # [1, hidden_dim]
                    processed = embedding.numpy()
                    logger.info(f"Added sequence dimension to single vector: {processed.shape}")
                    return processed
                else:
                    logger.warning(f"Unexpected embedding shape: {shape}, attempting to reshape")
                    # Try to reshape to [seq_len, hidden_dim] if possible
                    if embedding.numel() % 128 == 0:
                        # Reshape to have 128-dimensional vectors
                        embedding = embedding.reshape(-1, 128)
                        processed = embedding.numpy()
                        logger.info(f"Reshaped to {processed.shape[0]} vectors of dimension 128")
                        return processed
                    else:
                        # Fallback: just convert to numpy
                        processed = embedding.numpy()
                        logger.warning(f"Could not reshape to standard format, using shape: {processed.shape}")
                        return processed
            elif isinstance(embedding, np.ndarray):
                # Already a numpy array, ensure it has the right shape
                shape = embedding.shape
                logger.info(f"Processing numpy array with shape: {shape}")

                if len(shape) == 3:  # [batch_size, seq_len, hidden_dim]
                    # Remove batch dimension if present
                    if shape[0] == 1:
                        embedding = embedding.squeeze(0)
                        logger.info(f"Removed batch dimension, new shape: {embedding.shape}")

                    # Filter out zero or near-zero vectors
                    norms = np.linalg.norm(embedding, axis=1)
                    zero_vectors = np.sum(norms < 1e-6)
                    if zero_vectors > 0:
                        logger.warning(f"Found {zero_vectors} vectors with near-zero norm out of {embedding.shape[0]}")
                        # Filter out zero vectors if there are too many
                        if zero_vectors < embedding.shape[0]:  # Don't filter if all vectors are zero
                            embedding = embedding[norms >= 1e-6]
                            logger.info(f"Filtered out zero vectors, new shape: {embedding.shape}")

                    return embedding
                elif len(shape) == 2:  # [seq_len, hidden_dim]
                    # Already in the right format
                    # Filter out zero or near-zero vectors
                    norms = np.linalg.norm(embedding, axis=1)
                    zero_vectors = np.sum(norms < 1e-6)
                    if zero_vectors > 0:
                        logger.warning(f"Found {zero_vectors} vectors with near-zero norm out of {embedding.shape[0]}")
                        # Filter out zero vectors if there are too many
                        if zero_vectors < embedding.shape[0]:  # Don't filter if all vectors are zero
                            embedding = embedding[norms >= 1e-6]
                            logger.info(f"Filtered out zero vectors, new shape: {embedding.shape}")

                    return embedding
                elif len(shape) == 1:  # [hidden_dim]
                    # Single vector, add sequence dimension
                    embedding = embedding.reshape(1, -1)  # [1, hidden_dim]
                    logger.info(f"Added sequence dimension to single vector: {embedding.shape}")
                    return embedding
                else:
                    logger.warning(f"Unexpected numpy array shape: {shape}")
                    return embedding
            else:
                logger.error(f"Expected torch.Tensor or numpy.ndarray, got {type(embedding)}")
                return None

        except Exception as e:
            logger.error(f"Error processing {source} embedding for LanceDB: {e}")
            return None

