"""
Process-Isolated RAG Implementation.

This module wraps RAG operations with process isolation to prevent memory leaks
and ensure complete memory release after each operation.
"""

import os
import time
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union

from src.utils.logger import get_logger
from src.utils.memory_management.process_isolation import LanceDBProcessService, get_lancedb_process_service, add_embeddings_to_lancedb_isolated, query_lancedb_isolated
from src.utils.memory_management.embedding_process import EmbeddingProcessService, get_embedding_process_service, generate_embeddings_isolated
from src.utils.memory_management.process_manager import ProcessManager, get_process_manager
from src.utils.memory_management.ipc import to_shared_memory, from_shared_memory, cleanup_shared_memory

logger = get_logger(__name__)

class IsolatedRAGManager:
    """
    Manager for process-isolated RAG operations.
    
    This class wraps the RAG operations to ensure they run in separate processes,
    which guarantees memory is fully released after each operation.
    """
    
    def __init__(self):
        """Initialize the isolated RAG manager."""
        # Get service instances
        self.lancedb_service = get_lancedb_process_service()
        self.embedding_service = get_embedding_process_service()
        self.process_manager = get_process_manager()
    
    def generate_embeddings(self, texts: List[str], model_name: str) -> np.ndarray:
        """
        Generate embeddings using an isolated process.
        
        Args:
            texts: List of texts to generate embeddings for
            model_name: Name of the embedding model to use
            
        Returns:
            Numpy array of embeddings
        """
        logger.info(f"Generating embeddings for {len(texts)} texts using model {model_name} in isolated process")
        start_time = time.time()
        
        # Generate embeddings in isolated process
        embeddings = generate_embeddings_isolated(texts, model_name)
        
        # Log performance
        duration = time.time() - start_time
        logger.info(f"Generated embeddings in {duration:.2f}s using isolated process")
        
        return embeddings
    
    def add_embeddings(self, session_id: str, model_name: str, 
                      embeddings_list: List[np.ndarray], 
                      ids: List[str], 
                      metadatas: List[Dict[str, Any]]) -> bool:
        """
        Add embeddings to LanceDB using an isolated process.
        
        Args:
            session_id: The session ID
            model_name: The model name
            embeddings_list: List of embeddings to add
            ids: List of IDs for the embeddings
            metadatas: List of metadata dictionaries for the embeddings
            
        Returns:
            True if successful, False otherwise
        """
        logger.info(f"Adding {len(embeddings_list)} embeddings to LanceDB for session {session_id} using isolated process")
        start_time = time.time()
        
        # Add embeddings in isolated process
        success = add_embeddings_to_lancedb_isolated(
            session_id=session_id,
            model_name=model_name,
            embeddings_list=embeddings_list,
            ids=ids,
            metadatas=metadatas
        )
        
        # Log performance
        duration = time.time() - start_time
        logger.info(f"Added embeddings in {duration:.2f}s using isolated process: success={success}")
        
        return success
    
    def query(self, session_id: str, model_name: str, query_embedding: np.ndarray,
             k: int = 10, filter_dict: Optional[Dict] = None,
             similarity_threshold: float = 0.2) -> Tuple[List[Dict], List[float], List[str]]:
        """
        Query LanceDB using an isolated process.
        
        Args:
            session_id: The session ID
            model_name: The model name
            query_embedding: The query embedding
            k: Number of results to return
            filter_dict: Optional filter dictionary
            similarity_threshold: Minimum similarity threshold
            
        Returns:
            Tuple of (metadatas, scores, ids)
        """
        logger.info(f"Querying LanceDB for session {session_id} using isolated process")
        start_time = time.time()
        
        # Query in isolated process
        metadatas, scores, ids = query_lancedb_isolated(
            session_id=session_id,
            model_name=model_name,
            query_embedding=query_embedding,
            k=k,
            filter_dict=filter_dict,
            similarity_threshold=similarity_threshold
        )
        
        # Log performance
        duration = time.time() - start_time
        logger.info(f"Queried LanceDB in {duration:.2f}s using isolated process: found {len(metadatas)} results")
        
        return metadatas, scores, ids
    
    def cleanup_session(self, session_id: str) -> bool:
        """
        Clean up resources for a session.
        
        Args:
            session_id: The session ID
            
        Returns:
            True if successful, False otherwise
        """
        logger.info(f"Cleaning up session {session_id} resources using isolated process")
        
        try:
            # Force cleanup of LanceDB resources
            self.lancedb_service.run_operation(
                'cleanup',
                session_id=session_id
            )
            
            # Force a full memory cleanup
            import gc
            gc.collect()
            
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except (ImportError, Exception):
                pass
            
            return True
        except Exception as e:
            logger.error(f"Error cleaning up session {session_id}: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the isolated RAG manager.
        
        Returns:
            Dictionary of statistics
        """
        return {
            'process_manager': self.process_manager.get_stats()
        }

# Module-level singleton instance
_isolated_rag_manager = None

def get_isolated_rag_manager() -> IsolatedRAGManager:
    """
    Get the isolated RAG manager singleton.
    
    Returns:
        The isolated RAG manager
    """
    global _isolated_rag_manager
    if _isolated_rag_manager is None:
        _isolated_rag_manager = IsolatedRAGManager()
    return _isolated_rag_manager

# Enhanced drop-in replacements for existing functions

def generate_embeddings(texts: List[str], model_name: str) -> np.ndarray:
    """
    Generate embeddings using an isolated process.
    
    Args:
        texts: List of texts to generate embeddings for
        model_name: Name of the embedding model to use
        
    Returns:
        Numpy array of embeddings
    """
    manager = get_isolated_rag_manager()
    return manager.generate_embeddings(texts, model_name)

def add_embeddings_to_lancedb(session_id: str, model_name: str,
                             embeddings_list: List[np.ndarray],
                             ids: List[str],
                             metadatas: List[Dict[str, Any]]) -> bool:
    """
    Add embeddings to LanceDB using an isolated process.
    
    Args:
        session_id: The session ID
        model_name: The model name
        embeddings_list: List of embeddings to add
        ids: List of IDs for the embeddings
        metadatas: List of metadata dictionaries for the embeddings
        
    Returns:
        True if successful, False otherwise
    """
    manager = get_isolated_rag_manager()
    return manager.add_embeddings(
        session_id=session_id,
        model_name=model_name,
        embeddings_list=embeddings_list,
        ids=ids,
        metadatas=metadatas
    )

def query_lancedb(session_id: str, model_name: str, query_embedding: np.ndarray,
                 k: int = 10, filter_dict: Optional[Dict] = None,
                 similarity_threshold: float = 0.2) -> Tuple[List[Dict], List[float], List[str]]:
    """
    Query LanceDB using an isolated process.
    
    Args:
        session_id: The session ID
        model_name: The model name
        query_embedding: The query embedding
        k: Number of results to return
        filter_dict: Optional filter dictionary
        similarity_threshold: Minimum similarity threshold
        
    Returns:
        Tuple of (metadatas, scores, ids)
    """
    manager = get_isolated_rag_manager()
    return manager.query(
        session_id=session_id,
        model_name=model_name,
        query_embedding=query_embedding,
        k=k,
        filter_dict=filter_dict,
        similarity_threshold=similarity_threshold
    )

def cleanup_session(session_id: str) -> bool:
    """
    Clean up resources for a session.
    
    Args:
        session_id: The session ID
        
    Returns:
        True if successful, False otherwise
    """
    manager = get_isolated_rag_manager()
    return manager.cleanup_session(session_id)