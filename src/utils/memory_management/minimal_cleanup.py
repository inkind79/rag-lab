"""
Minimal memory cleanup for RAG Lab.

This module provides the absolute minimum memory cleanup 
functions to avoid interfering with application logic.
"""

import gc
import torch
import logging

# Get the standard logger
logger = logging.getLogger(__name__)

def minimal_cleanup():
    """
    Perform minimal memory cleanup operations that are guaranteed not to interfere with application logic.
    
    This function:
    1. Runs garbage collection
    2. Empties CUDA cache if available
    
    No exceptions will be raised from this function.
    """
    try:
        # Run basic garbage collection
        gc.collect()
        
        # Empty CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("[MEMORY] CUDA cache emptied")
        
        logger.info("[MEMORY] Basic memory cleanup completed")
        return True
    except Exception as e:
        # Catch any possible error to avoid affecting application
        logger.error(f"[MEMORY] Error during minimal cleanup: {e}")
        return False
