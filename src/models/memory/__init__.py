"""
Memory management module for RAG Lab

This module provides unified memory management functionality through memory_manager.
The model_manager has been consolidated into memory_manager and is maintained here
only as an alias for backward compatibility.
"""

from src.models.memory.memory_manager import memory_manager

# Create model_manager as an alias to memory_manager for backward compatibility
# This ensures existing code importing model_manager continues to work
model_manager = memory_manager

# Export both names for backward compatibility
__all__ = ['model_manager', 'memory_manager']

# Define the preferred import to guide developers
recommended_import = memory_manager