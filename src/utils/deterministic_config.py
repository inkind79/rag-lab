"""
Utility functions for setting up deterministic behavior in PyTorch.

This module provides functions to configure PyTorch for deterministic behavior,
which is essential for consistent embeddings and retrieval results.
"""

import os
import random
import numpy as np
import torch
from src.utils.logger import get_logger

logger = get_logger(__name__)

def set_deterministic_mode(seed=42):
    """
    Configure PyTorch and related libraries for deterministic behavior.
    
    This function sets random seeds for Python, NumPy, and PyTorch,
    and configures PyTorch to use deterministic algorithms.
    
    Args:
        seed: The random seed to use (default: 42)
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Set random seeds
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Set CUDA seeds if available
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        
        # Configure PyTorch for deterministic behavior
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # Set environment variable for CUBLAS workspace
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        
        # FlashAttention is not compatible with fully deterministic mode
        # So we avoid enabling it for better compatibility
        # The above settings still give reasonable determinism
        
        logger.info(f"Deterministic mode enabled with seed {seed}")
        return True
    
    except Exception as e:
        logger.error(f"Error setting deterministic mode: {e}")
        return False

def is_deterministic_mode_enabled():
    """
    Check if deterministic mode is enabled.
    
    Returns:
        True if deterministic mode is enabled, False otherwise
    """
    try:
        return torch.backends.cudnn.deterministic
    except:
        return False
