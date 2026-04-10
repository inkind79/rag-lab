"""
RAG Lab Models Package

This package contains all model-related functionality for the RAG Lab system.
"""
# Import memory management components directly from their source
from src.models.memory.memory_manager import memory_manager

# Import the model_manager alias (which points to memory_manager) for backward compatibility
from src.models.memory import model_manager

# Define which symbols are exported from this package
__all__ = ['memory_manager', 'model_manager']