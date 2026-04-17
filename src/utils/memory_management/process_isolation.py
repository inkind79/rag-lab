"""
Process Isolation Service for LanceDB Operations.

This module provides a way to run LanceDB operations in separate processes to
ensure memory is completely released after each operation.

Security note: argument marshalling to the child process uses numpy.save /
np.savez / torch.save exclusively. Pickle is never used — torch.load is
called with weights_only=True so tensor payloads can't execute arbitrary
code during deserialization. Legacy "._type: list" temp files from older
builds are rejected on load.
"""

import os
import json
import time
import signal
import tempfile
import multiprocessing
from multiprocessing import Process, Queue, Manager
from typing import Dict, List, Any, Optional, Tuple, Callable

from src.utils.logger import get_logger

logger = get_logger(__name__)

class LanceDBProcessService:
    """
    Service that runs LanceDB operations in separate processes.
    
    This allows memory to be fully released when the process exits,
    preventing memory leaks and excessive RAM usage.
    """
    
    def __init__(self, max_process_lifetime: int = 5):
        """
        Initialize the LanceDB process service.
        
        Args:
            max_process_lifetime: Maximum lifetime of a process in minutes
        """
        self.max_process_lifetime = max_process_lifetime * 60  # Convert to seconds
        self.manager = Manager()
        self.active_processes = self.manager.dict()
    
    def _process_worker(self, operation: str, args: Dict[str, Any], 
                        result_queue: Queue, error_queue: Queue):
        """
        Worker function that runs in a separate process.
        
        Args:
            operation: The operation to perform (e.g., 'add_embeddings', 'query')
            args: Arguments for the operation
            result_queue: Queue to store the operation result
            error_queue: Queue to store errors
        """
        try:
            logger.info(f"Process {os.getpid()} starting LanceDB operation: {operation}")
            
            # Import LanceDB modules here to ensure they're only loaded in the child process
            if operation == 'add_embeddings':
                from src.models.vector_stores.lancedb_manager import add_embeddings_to_lancedb
                
                # Extract arguments
                session_id = args.get('session_id')
                model_name = args.get('model_name')
                embeddings_list = args.get('embeddings_list')
                ids = args.get('ids')
                metadatas = args.get('metadatas')
                
                # Perform operation
                success = add_embeddings_to_lancedb(
                    session_id=session_id,
                    model_name=model_name,
                    embeddings_list=embeddings_list,
                    ids=ids,
                    metadatas=metadatas
                )
                
                # Put result in queue
                result_queue.put(success)
                
            elif operation == 'query':
                from src.models.vector_stores.lancedb_manager import query_lancedb
                
                # Extract arguments
                session_id = args.get('session_id')
                model_name = args.get('model_name')
                query_embedding = args.get('query_embedding')
                k = args.get('k', 10)
                filter_dict = args.get('filter_dict')
                similarity_threshold = args.get('similarity_threshold', 0.2)
                
                # Perform operation
                metadatas, scores, ids = query_lancedb(
                    session_id=session_id,
                    model_name=model_name,
                    query_embedding=query_embedding,
                    k=k,
                    filter_dict=filter_dict,
                    similarity_threshold=similarity_threshold
                )
                
                # Put result in queue
                result_queue.put((metadatas, scores, ids))
                
            elif operation == 'cleanup':
                from src.models.vector_stores.lancedb_manager import _close_lancedb_connection
                
                # Extract arguments
                session_id = args.get('session_id')
                
                # Perform operation
                _close_lancedb_connection(session_id)
                
                # Put success in queue
                result_queue.put(True)
                
            else:
                raise ValueError(f"Unknown operation: {operation}")
                
            logger.info(f"Process {os.getpid()} completed LanceDB operation: {operation}")
            
        except Exception as e:
            logger.error(f"Error in process {os.getpid()} for operation {operation}: {e}")
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
    
    def run_operation(self, operation: str, **kwargs) -> Any:
        """
        Run a LanceDB operation in a separate process.
        
        Args:
            operation: The operation to perform (e.g., 'add_embeddings', 'query')
            **kwargs: Arguments for the operation
            
        Returns:
            The result of the operation
        """
        # Create queues for results and errors
        result_queue = Queue()
        error_queue = Queue()
        
        # Handle large numpy arrays or tensors by serializing them to temp files
        # This avoids issues with pickle and multiprocessing
        args = self._prepare_args_for_transfer(kwargs)
        
        # Start a new process for the operation
        process = Process(
            target=self._process_worker,
            args=(operation, args, result_queue, error_queue)
        )
        
        # Store process information
        process_id = process.pid if process._parent_pid else os.getpid()
        self.active_processes[process_id] = {
            'operation': operation,
            'start_time': time.time(),
            'process': process
        }
        
        # Start the process
        process.start()
        logger.info(f"Started process {process_id} for LanceDB operation: {operation}")
        
        try:
            # Wait for the process to complete (with timeout)
            process.join(timeout=60)  # 60-second timeout
            
            # Check if the process is still running
            if process.is_alive():
                logger.warning(f"Process {process_id} timed out, terminating")
                process.terminate()
                process.join(timeout=5)
                
                # If still alive, kill with SIGKILL
                if process.is_alive():
                    logger.warning(f"Process {process_id} did not terminate, sending SIGKILL")
                    try:
                        os.kill(process_id, signal.SIGKILL)
                    except Exception as e:
                        logger.error(f"Error killing process {process_id}: {e}")
                
                raise TimeoutError(f"Operation {operation} timed out")
            
            # Check for errors
            if not error_queue.empty():
                error_message = error_queue.get()
                raise RuntimeError(f"Error in LanceDB operation: {error_message}")
            
            # Get the result
            if not result_queue.empty():
                result = result_queue.get()
                logger.info(f"Completed LanceDB operation {operation} in separate process")
                return self._retrieve_result_from_transfer(result)
            else:
                raise RuntimeError(f"No result returned from operation {operation}")
                
        finally:
            # Clean up
            if process_id in self.active_processes:
                del self.active_processes[process_id]
                
            # Close queues
            result_queue.close()
            error_queue.close()
            
            # Clean up any temporary files
            self._cleanup_temp_files()
    
    def _prepare_args_for_transfer(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare arguments for transfer to another process.
        Large numpy arrays and tensors are saved to temp files.
        
        Args:
            args: The arguments to prepare
            
        Returns:
            The prepared arguments
        """
        import numpy as np
        
        prepared_args = {}
        temp_files = []
        
        for key, value in args.items():
            # Handle numpy arrays
            if isinstance(value, np.ndarray):
                # Create a temporary file
                fd, path = tempfile.mkstemp(suffix='.npy')
                os.close(fd)
                
                # Save the array
                np.save(path, value)
                
                # Store the path
                prepared_args[key] = {'_type': 'numpy', 'path': path}
                temp_files.append(path)
                
            # Handle torch tensors
            elif str(type(value)).startswith("<class 'torch."):
                try:
                    import torch
                    
                    # Create a temporary file
                    fd, path = tempfile.mkstemp(suffix='.pt')
                    os.close(fd)
                    
                    # Save the tensor
                    torch.save(value, path)
                    
                    # Store the path
                    prepared_args[key] = {'_type': 'torch', 'path': path}
                    temp_files.append(path)
                except Exception as e:
                    logger.error(f"Error saving tensor to temp file: {e}")
                    prepared_args[key] = value
            
            # Handle lists of numpy arrays or tensors. Use torch.save (which is
            # safely loadable via weights_only=True) or numpy's .npz; never pickle.
            elif isinstance(value, list) and value and (
                    isinstance(value[0], np.ndarray) or
                    str(type(value[0])).startswith("<class 'torch.")):

                if isinstance(value[0], np.ndarray):
                    fd, path = tempfile.mkstemp(suffix='.npz')
                    os.close(fd)
                    np.savez(path, *value)
                    prepared_args[key] = {'_type': 'list_numpy', 'path': path}
                else:
                    import torch
                    fd, path = tempfile.mkstemp(suffix='.pt')
                    os.close(fd)
                    torch.save(value, path)
                    prepared_args[key] = {'_type': 'list_torch', 'path': path}

                temp_files.append(path)
            
            # Handle other types
            else:
                prepared_args[key] = value
        
        # Store temp files list
        prepared_args['_temp_files'] = temp_files
        
        return prepared_args
    
    def _retrieve_result_from_transfer(self, result: Any) -> Any:
        """
        Retrieve a result that may have been saved to a temp file.
        
        Args:
            result: The result to retrieve
            
        Returns:
            The retrieved result
        """
        import numpy as np
        
        # Handle dictionaries that may contain temp file paths
        if isinstance(result, dict) and '_type' in result and 'path' in result:
            if result['_type'] == 'numpy':
                return np.load(result['path'], allow_pickle=False)
            elif result['_type'] == 'torch':
                import torch
                return torch.load(result['path'], weights_only=True)
            elif result['_type'] == 'list_numpy':
                loaded = np.load(result['path'], allow_pickle=False)
                return [loaded[k] for k in sorted(loaded.files, key=lambda s: int(s.split('_')[-1]))]
            elif result['_type'] == 'list_torch':
                import torch
                return torch.load(result['path'], weights_only=True)
            elif result['_type'] == 'list':
                # Legacy pickle path — refuse; caller should re-run whatever
                # produced this result under the new code path.
                raise RuntimeError(
                    "Refusing to unpickle untrusted IPC payload; "
                    "this temp file was produced by an older build. "
                    "Retrying the isolated call will rewrite it in a safe format."
                )
        
        # Handle tuples that may contain dictionaries
        elif isinstance(result, tuple):
            return tuple(self._retrieve_result_from_transfer(item) for item in result)
        
        # Handle lists that may contain dictionaries
        elif isinstance(result, list):
            return [self._retrieve_result_from_transfer(item) for item in result]
        
        # Return the result as is
        return result
    
    def _cleanup_temp_files(self):
        """Clean up any temporary files created during operations."""
        # Iterate through active processes
        for process_info in self.active_processes.values():
            args = process_info.get('args', {})
            temp_files = args.get('_temp_files', [])
            
            # Remove temp files
            for path in temp_files:
                try:
                    if os.path.exists(path):
                        os.remove(path)
                except Exception as e:
                    logger.error(f"Error removing temp file {path}: {e}")
    
    def cleanup_old_processes(self):
        """
        Clean up processes that have been running for too long.
        """
        current_time = time.time()
        
        # Find processes that have been running for too long
        for process_id, process_info in list(self.active_processes.items()):
            start_time = process_info.get('start_time', 0)
            process = process_info.get('process')
            
            if current_time - start_time > self.max_process_lifetime and process and process.is_alive():
                logger.warning(f"Process {process_id} has been running for too long, terminating")
                
                try:
                    # Try to terminate gracefully
                    process.terminate()
                    process.join(timeout=5)
                    
                    # If still alive, kill with SIGKILL
                    if process.is_alive():
                        logger.warning(f"Process {process_id} did not terminate, sending SIGKILL")
                        os.kill(process_id, signal.SIGKILL)
                except Exception as e:
                    logger.error(f"Error terminating process {process_id}: {e}")
                
                # Remove from active processes
                del self.active_processes[process_id]

# Singleton instance
_lancedb_process_service = None

def get_lancedb_process_service() -> LanceDBProcessService:
    """
    Get the LanceDB process service singleton.
    
    Returns:
        The LanceDB process service
    """
    global _lancedb_process_service
    if _lancedb_process_service is None:
        _lancedb_process_service = LanceDBProcessService()
    return _lancedb_process_service

# Convenience functions for common operations

def add_embeddings_to_lancedb_isolated(
    session_id: str,
    model_name: str,
    embeddings_list: List[Any],
    ids: List[str],
    metadatas: List[Dict[str, Any]]
) -> bool:
    """
    Add embeddings to LanceDB in a separate process.
    
    Args:
        session_id: The session ID
        model_name: The model name
        embeddings_list: List of embeddings to add
        ids: List of IDs for the embeddings
        metadatas: List of metadata dictionaries for the embeddings
        
    Returns:
        True if successful, False otherwise
    """
    service = get_lancedb_process_service()
    return service.run_operation(
        'add_embeddings',
        session_id=session_id,
        model_name=model_name,
        embeddings_list=embeddings_list,
        ids=ids,
        metadatas=metadatas
    )

def query_lancedb_isolated(
    session_id: str,
    model_name: str,
    query_embedding: Any,
    k: int = 10,
    filter_dict: Optional[Dict] = None,
    similarity_threshold: float = 0.2
) -> Tuple[List[Dict], List[float], List[str]]:
    """
    Query LanceDB in a separate process.
    
    Args:
        session_id: The session ID
        model_name: The model name
        query_embedding: The query embedding
        k: The number of results to return
        filter_dict: Optional filter dictionary
        similarity_threshold: Minimum similarity threshold
        
    Returns:
        A tuple of (metadatas, scores, ids)
    """
    service = get_lancedb_process_service()
    return service.run_operation(
        'query',
        session_id=session_id,
        model_name=model_name,
        query_embedding=query_embedding,
        k=k,
        filter_dict=filter_dict,
        similarity_threshold=similarity_threshold
    )

def cleanup_lancedb_isolated(session_id: str) -> bool:
    """
    Clean up LanceDB resources in a separate process.
    
    Args:
        session_id: The session ID
        
    Returns:
        True if successful, False otherwise
    """
    service = get_lancedb_process_service()
    return service.run_operation(
        'cleanup',
        session_id=session_id
    )