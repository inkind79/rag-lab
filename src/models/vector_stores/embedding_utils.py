"""
Utilities for processing embeddings for vector stores.
Ensures consistent embedding formats between indexing and querying.
Supports ChromaDB and LanceDB vector stores.
"""
import torch
import numpy as np  # Used for numpy.ndarray conversion
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Global variable to store the expected embedding dimension
_EMBEDDING_DIM = None

def normalize_vectors(vectors):
    """
    Normalize vectors to unit length for proper cosine similarity.

    Args:
        vectors: numpy array of vectors to normalize
                Can be 1D [dim] or 2D [num_vectors, dim]

    Returns:
        Normalized vectors with unit L2 norm
    """
    if isinstance(vectors, np.ndarray):
        # For 2D array [num_vectors, dim]
        if len(vectors.shape) == 2:
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            # Avoid division by zero
            norms[norms == 0] = 1.0
            normalized = vectors / norms
            logger.debug(f"Normalized {vectors.shape[0]} vectors to unit length")
            return normalized
        # For 1D array [dim]
        elif len(vectors.shape) == 1:
            norm = np.linalg.norm(vectors)
            if norm == 0:
                return vectors
            normalized = vectors / norm
            logger.debug(f"Normalized single vector to unit length")
            return normalized
    elif isinstance(vectors, torch.Tensor):
        # Convert to numpy, normalize, and convert back to tensor
        device = vectors.device
        dtype = vectors.dtype
        vectors_np = vectors.cpu().numpy()
        normalized_np = normalize_vectors(vectors_np)
        return torch.tensor(normalized_np, dtype=dtype, device=device)

    # If not numpy array or tensor, return as is
    logger.warning(f"Cannot normalize vectors of type {type(vectors)}")
    return vectors

def process_embedding_for_chroma(embedding_tensor, source="unknown"):
    """
    Process a PyTorch embedding tensor for ChromaDB storage.
    Ensures consistent shape and dtype across document and query embeddings.
    Uses GPU for processing when available.

    Args:
        embedding_tensor: The PyTorch tensor containing the embedding
        source: Source identifier for logging (e.g., "document", "query")

    Returns:
        list: The processed embedding as a list of floats
    """
    global _EMBEDDING_DIM

    try:
        logger.debug(f"Processing {source} embedding for ChromaDB")
        logger.debug(f"Original shape: {embedding_tensor.shape}, dtype: {embedding_tensor.dtype}, device: {embedding_tensor.device}")

        # Note: GPU operations have been removed as Byaldi is designed for CPU usage
        # We keep the tensor on CPU for processing
        logger.debug(f"Processing tensor on CPU: {embedding_tensor.device}")

        # Extract the hidden dimension (last dimension)
        hidden_dim = embedding_tensor.shape[-1]

        # For document embeddings, store the dimension for future reference
        if source == "document" and _EMBEDDING_DIM is None:
            # Use the hidden dimension as the basis for our embedding dimension
            _EMBEDDING_DIM = hidden_dim
            logger.info(f"Set reference embedding dimension to {_EMBEDDING_DIM}")

        # Handle different tensor shapes
        if embedding_tensor.dim() > 2:
            # For tensors with shape [batch, seq_len, hidden_dim, ...]
            # Take the first item if it's a batch
            if embedding_tensor.shape[0] == 1:
                embedding_tensor = embedding_tensor[0]
                logger.debug(f"Removed batch dimension, new shape: {embedding_tensor.shape}")

        # For consistent embeddings, we'll use the mean of the sequence dimension
        # This ensures document and query embeddings have the same dimension
        if embedding_tensor.dim() > 1:
            # Take mean across sequence dimension (dim 0 after removing batch)
            original_shape = embedding_tensor.shape
            embedding_tensor = embedding_tensor.mean(dim=0)
            logger.debug(f"Took mean across sequence dimension, shape: {original_shape} -> {embedding_tensor.shape}")

        # Convert to float32 for consistent dtype
        if embedding_tensor.dtype != torch.float32:
            embedding_tensor = embedding_tensor.to(torch.float32)
            logger.debug(f"Converted dtype to {embedding_tensor.dtype}")

        # Convert to list for ChromaDB (must move to CPU for numpy conversion)
        embedding_list = embedding_tensor.cpu().numpy().tolist()
        logger.debug(f"Final {source} embedding length: {len(embedding_list)}")

        return embedding_list
    except Exception as e:
        logger.error(f"Error processing {source} embedding: {e}")
        return None





def process_embedding_for_lancedb(embedding_tensor, source="unknown"):
    """
    Process a PyTorch embedding tensor for LanceDB storage.
    Ensures consistent shape and dtype across document and query embeddings.
    Preserves the full multi-vector embedding structure for better retrieval quality.
    Specifically designed to maintain the full multi-vector representation for ColPali models (128-dimensional vectors).
    Normalizes all vectors to unit length for proper cosine similarity.

    Args:
        embedding_tensor: The PyTorch tensor containing the embedding
        source: Source identifier for logging (e.g., "document", "query")

    Returns:
        numpy.ndarray: The processed embedding as a numpy array with shape [seq_len, hidden_dim]
                      or [hidden_dim] for single vectors, with all vectors normalized to unit length
    """
    global _EMBEDDING_DIM

    try:
        logger.debug(f"Processing {source} embedding for LanceDB")
        logger.debug(f"Original shape: {embedding_tensor.shape}, dtype: {embedding_tensor.dtype}, device: {embedding_tensor.device}")

        # Keep the tensor on CPU for processing
        if embedding_tensor.device.type == 'cuda':
            embedding_tensor = embedding_tensor.cpu()
            logger.debug(f"Moved tensor to CPU")

        # Convert to float32 for consistent dtype
        if embedding_tensor.dtype != torch.float32:
            embedding_tensor = embedding_tensor.to(torch.float32)
            logger.debug(f"Converted dtype to {embedding_tensor.dtype}")

        # Handle different tensor shapes
        if embedding_tensor.dim() > 2:
            # For tensors with shape [batch, seq_len, hidden_dim, ...]
            # Take the first item if it's a batch
            if embedding_tensor.shape[0] == 1:
                embedding_tensor = embedding_tensor[0]
                logger.debug(f"Removed batch dimension, new shape: {embedding_tensor.shape}")

        # For LanceDB, we preserve the full embedding structure
        # Convert to numpy array for LanceDB
        embedding_np = embedding_tensor.numpy()

        # For LanceDB multivector support, ensure we have a 2D array [seq_len, hidden_dim]
        # If we have a 1D array [hidden_dim], reshape to [1, hidden_dim]
        if len(embedding_np.shape) == 1:
            embedding_np = embedding_np.reshape(1, -1)
            logger.debug(f"Reshaped 1D embedding to 2D: {embedding_np.shape}")

        # For ColPali models, we expect a 2D array with shape [seq_len, 128]
        # This preserves the multi-vector representation (one vector per token/patch)
        # LanceDB will store this as a list of vectors for each document
        if len(embedding_np.shape) == 2:
            seq_len, dim = embedding_np.shape
            logger.debug(f"Multi-vector embedding with {seq_len} vectors of dimension {dim}")

            # Verify that each vector has a reasonable norm (not all zeros)
            norms = np.linalg.norm(embedding_np, axis=1)
            zero_vectors = np.sum(norms < 1e-6)
            if zero_vectors > 0:
                logger.warning(f"Found {zero_vectors} vectors with near-zero norm out of {seq_len}")
                # Filter out zero vectors if there are too many
                if zero_vectors < seq_len:  # Don't filter if all vectors are zero
                    embedding_np = embedding_np[norms >= 1e-6]
                    logger.debug(f"Filtered out zero vectors, new shape: {embedding_np.shape}")

            # Normalize all vectors to unit length for proper cosine similarity
            embedding_np = normalize_vectors(embedding_np)

            # Verify normalization
            norms_after = np.linalg.norm(embedding_np, axis=1)
            logger.debug(f"Vector norms after normalization: min={norms_after.min():.4f}, max={norms_after.max():.4f}, mean={norms_after.mean():.4f}")

        logger.debug(f"Final {source} embedding shape for LanceDB: {embedding_np.shape}")

        # Log some stats about the normalized embedding
        if len(embedding_np.shape) == 2:
            logger.debug(f"Embedding stats: min={embedding_np.min():.4f}, max={embedding_np.max():.4f}, mean={embedding_np.mean():.4f}")

            # Log the first vector for debugging
            if embedding_np.shape[0] > 0:
                logger.debug(f"First vector (first 5 values): {embedding_np[0, :5]}")

            # Check similarity between first two vectors if available
            if embedding_np.shape[0] > 1:
                sim = np.dot(embedding_np[0], embedding_np[1])
                logger.debug(f"Similarity between first two vectors: {sim:.4f}")

        return embedding_np
    except Exception as e:
        logger.error(f"Error processing {source} embedding for LanceDB: {e}")
        return None

def debug_embedding(embedding_tensor, source="unknown"):
    """
    Debug embedding dimensions and properties.

    Args:
        embedding_tensor: The PyTorch tensor to debug
        source: Source identifier for logging (e.g., "document", "query")
    """
    logger.debug(f"--- {source.upper()} EMBEDDING DEBUG ---")

    if not isinstance(embedding_tensor, torch.Tensor):
        logger.debug(f"Not a tensor: {embedding_tensor}")
        return

    logger.debug(f"Shape: {embedding_tensor.shape}, Dtype: {embedding_tensor.dtype}, Device: {embedding_tensor.device}")

    if embedding_tensor.dim() > 1:
        logger.debug(f"Dimensions: {embedding_tensor.shape}, Elements: {embedding_tensor.numel()}")
        if embedding_tensor.shape[0] == 1:
            logger.debug(f"Embedding shape without batch: {embedding_tensor[0].shape}")
