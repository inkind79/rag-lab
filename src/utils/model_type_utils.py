"""
Model Type Utilities

Thin wrappers around the model registry. These functions preserve the existing
API used throughout the codebase while delegating to the central registry.
"""

from src.utils.logger import get_logger

logger = get_logger(__name__)


def is_colpali_model(model_name: str) -> bool:
    """Check if a model name refers to a ColPali-family (multi-vector) model.

    Args:
        model_name: Model identifier string (e.g., 'tsystems/colqwen2.5-3b-multilingual-v1.0')

    Returns:
        True if this is a multi-vector visual embedding model
    """
    if not model_name:
        return False
    from src.models.model_registry import registry
    if registry.is_multi_vector(model_name):
        return True
    # Fallback: keyword check for models not yet in the registry
    name_lower = model_name.lower()
    return 'colqwen' in name_lower or 'colnomic' in name_lower or 'colpali' in name_lower or 'colsmol' in name_lower


def is_colpali_adapter(adapter) -> bool:
    """Check if an embedding adapter is a ColPali-type (multi-vector) adapter.

    Preferred over string-based model name checks when you have the adapter instance.

    Args:
        adapter: An embedding adapter instance (or any object with embedding_type)

    Returns:
        True if the adapter produces multi-vector embeddings
    """
    try:
        from src.models.embedding_adapters.base_adapter import EmbeddingType
        return hasattr(adapter, 'embedding_type') and adapter.embedding_type == EmbeddingType.MULTI_VECTOR
    except ImportError:
        # Fallback to string check if base_adapter not available
        return hasattr(adapter, 'model_name') and is_colpali_model(adapter.model_name)


def get_lancedb_table_name(model_name: str) -> str:
    """Determine the LanceDB table name for a given model.

    Each model gets its own table to prevent cross-model embedding collisions.

    Args:
        model_name: Model identifier string

    Returns:
        LanceDB table name string
    """
    if not model_name:
        return "default_lancedb"

    from src.models.model_registry import registry
    table = registry.get_lancedb_table(model_name)
    if table != "default_lancedb":
        return table

    # Fallback for unregistered models — preserve legacy behavior
    name_lower = model_name.lower()
    if 'colqwen' in name_lower:
        return "colqwen25"
    elif 'colnomic' in name_lower:
        return "colnomic_other"
    elif 'colpali' in name_lower:
        return "colpali_other"
    else:
        logger.debug(f"Non-ColPali model '{model_name}', using default table name")
        return "default_lancedb"
