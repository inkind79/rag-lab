"""
Process Isolation for Embedding Generation.

This module provides functionality to run embedding generation in separate
processes to ensure memory is completely released after each operation.
"""

import os
import time
import pickle
import signal
import tempfile
import multiprocessing
from multiprocessing import Process, Queue
from typing import Dict, List, Any, Optional, Tuple, Union, Callable

import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)

class EmbeddingProcessService:
    """
    Service that runs embedding generation in separate processes.
    
    This allows memory to be fully released when the process exits,
    preventing memory leaks, especially for GPU operations.
    """
    
    def __init__(self, timeout: int = 120):
        """
        Initialize the embedding process service.
        
        Args:
            timeout: Maximum time in seconds to wait for embedding generation
        """
        self.timeout = timeout
        self.temp_files = []
    
    def _embedding_worker(self, model_name: str, texts: List[str], 
                         result_queue: Queue, error_queue: Queue):
        """
        Worker function that runs in a separate process.
        
        Args:
            model_name: The name of the embedding model to use
            texts: List of texts to generate embeddings for
            result_queue: Queue to store the embeddings
            error_queue: Queue to store errors
        """
        try:
            pid = os.getpid()
            logger.info(f"Process {pid} starting embedding generation with model {model_name}")
            
            start_time = time.time()
            
            # Import here to ensure modules are only loaded in the child process
            from src.models.model_loader import load_embedding_model
            
            # Load the embedding model
            embedding_model = load_embedding_model(model_name)
            
            # Generate embeddings
            embeddings = embedding_model.encode(texts)
            
            # Convert to numpy array if it's a torch tensor
            if str(type(embeddings)).startswith("<class 'torch."):
                embeddings = embeddings.detach().cpu().numpy()
            
            # Create a temporary file for the embeddings
            fd, path = tempfile.mkstemp(suffix='.npy')
            os.close(fd)
            
            # Save the embeddings to the temporary file
            np.save(path, embeddings)
            
            # Put the path in the result queue
            result_queue.put(path)
            
            duration = time.time() - start_time
            logger.info(f"Process {pid} completed embedding generation in {duration:.2f}s")
            
        except Exception as e:
            logger.error(f"Error in embedding process {os.getpid()}: {e}")
            error_queue.put(str(e))
            
        finally:
            # Force cleanup
            import gc
            gc.collect()
            
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except (ImportError, Exception):
                pass
            
            try:
                import ctypes
                libc = ctypes.CDLL('libc.so.6')
                libc.malloc_trim(0)
            except Exception:
                pass
    
    def _ollama_embedding_worker(self, model_name: str, texts: List[str], 
                               result_queue: Queue, error_queue: Queue):
        """
        Worker function for Ollama embeddings that runs in a separate process.
        
        Args:
            model_name: The name of the Ollama embedding model
            texts: List of texts to generate embeddings for
            result_queue: Queue to store the embeddings
            error_queue: Queue to store errors
        """
        try:
            pid = os.getpid()
            logger.info(f"Process {pid} starting Ollama embedding generation with model {model_name}")
            
            start_time = time.time()
            
            # Import here to ensure modules are only loaded in the child process
            from src.models.memory.ollama_embeddings import get_ollama_embeddings
            
            # Get the Ollama embedding model
            embedding_model = get_ollama_embeddings(model_name)
            
            # Generate embeddings
            embeddings = []
            for text in texts:
                embedding = embedding_model.embed_query(text)
                embeddings.append(embedding)
            
            # Convert to numpy array
            embeddings_array = np.array(embeddings)
            
            # Create a temporary file for the embeddings
            fd, path = tempfile.mkstemp(suffix='.npy')
            os.close(fd)
            
            # Save the embeddings to the temporary file
            np.save(path, embeddings_array)
            
            # Put the path in the result queue
            result_queue.put(path)
            
            duration = time.time() - start_time
            logger.info(f"Process {pid} completed Ollama embedding generation in {duration:.2f}s")
            
        except Exception as e:
            logger.error(f"Error in Ollama embedding process {os.getpid()}: {e}")
            error_queue.put(str(e))
            
        finally:
            # Force cleanup
            import gc
            gc.collect()
            
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except (ImportError, Exception):
                pass
            
            try:
                import ctypes
                libc = ctypes.CDLL('libc.so.6')
                libc.malloc_trim(0)
            except Exception:
                pass
    
    def generate_embeddings(self, model_name: str, texts: List[str], use_ollama: bool = False) -> np.ndarray:
        """
        Generate embeddings in a separate process.
        
        Args:
            model_name: The name of the embedding model to use
            texts: List of texts to generate embeddings for
            use_ollama: Whether to use Ollama for embedding generation
            
        Returns:
            Numpy array of embeddings
        """
        # Create queues for results and errors
        result_queue = Queue()
        error_queue = Queue()
        
        # Choose the appropriate worker
        worker_func = self._ollama_embedding_worker if use_ollama else self._embedding_worker
        
        # Start a new process for embedding generation
        process = Process(
            target=worker_func,
            args=(model_name, texts, result_queue, error_queue)
        )
        
        # Start the process
        process.start()
        logger.info(f"Started process {process.pid} for embedding generation with model {model_name}")
        
        try:
            # Wait for the process to complete (with timeout)
            process.join(timeout=self.timeout)
            
            # Check if the process is still running
            if process.is_alive():
                logger.warning(f"Process {process.pid} timed out, terminating")
                process.terminate()
                process.join(timeout=5)
                
                # If still alive, kill with SIGKILL
                if process.is_alive():
                    logger.warning(f"Process {process.pid} did not terminate, sending SIGKILL")
                    try:
                        os.kill(process.pid, signal.SIGKILL)
                    except Exception as e:
                        logger.error(f"Error killing process {process.pid}: {e}")
                
                raise TimeoutError(f"Embedding generation timed out after {self.timeout} seconds")
            
            # Check for errors
            if not error_queue.empty():
                error_message = error_queue.get()
                raise RuntimeError(f"Error in embedding generation: {error_message}")
            
            # Get the result
            if not result_queue.empty():
                # Get the path to the temporary file
                path = result_queue.get()
                self.temp_files.append(path)
                
                # Load the embeddings
                embeddings = np.load(path)
                logger.info(f"Loaded embeddings from temporary file: {path}")
                
                # Clean up the temporary file
                try:
                    os.remove(path)
                    self.temp_files.remove(path)
                    logger.debug(f"Removed temporary file: {path}")
                except Exception as e:
                    logger.warning(f"Error removing temporary file {path}: {e}")
                
                return embeddings
            else:
                raise RuntimeError("No result returned from embedding generation")
                
        finally:
            # Close queues
            result_queue.close()
            error_queue.close()
            
            # Clean up any remaining temporary files
            self._cleanup_temp_files()
    
    def _cleanup_temp_files(self):
        """Clean up any temporary files created during operations."""
        for path in list(self.temp_files):
            try:
                if os.path.exists(path):
                    os.remove(path)
                    self.temp_files.remove(path)
                    logger.debug(f"Removed temporary file: {path}")
            except Exception as e:
                logger.error(f"Error removing temp file {path}: {e}")

# Singleton instance
_embedding_process_service = None

def get_embedding_process_service() -> EmbeddingProcessService:
    """
    Get the embedding process service singleton.
    
    Returns:
        The embedding process service
    """
    global _embedding_process_service
    if _embedding_process_service is None:
        _embedding_process_service = EmbeddingProcessService()
    return _embedding_process_service

# Convenience functions

def generate_embeddings_isolated(texts: List[str], model_name: str) -> np.ndarray:
    """
    Generate embeddings in a separate process.
    
    Args:
        texts: List of texts to generate embeddings for
        model_name: The name of the embedding model to use
        
    Returns:
        Numpy array of embeddings
    """
    service = get_embedding_process_service()
    
    # Check if this is an Ollama model
    use_ollama = model_name.startswith('ollama:') or model_name in [
        'nomic-embed-text', 'nomic-embed-text-v1.5', 'mxbai-embed-large'
    ]
    
    if use_ollama:
        # Extract the actual model name if prefixed with 'ollama:'
        if model_name.startswith('ollama:'):
            model_name = model_name[7:]
        
        return service.generate_embeddings(model_name, texts, use_ollama=True)
    else:
        return service.generate_embeddings(model_name, texts, use_ollama=False)