"""
IPC (Inter-Process Communication) for Memory-Efficient Data Transfer.

This module provides utilities for transferring data between processes
without excessive memory duplication or serialization overhead.

Security note: no pickle. Data is routed through ``IPCSerializer`` which
accepts only numpy arrays, torch tensors, JSON primitives, and nested
lists/dicts of those. Unsupported types raise ``TypeError`` rather than
falling through to pickle, since loading attacker-controlled pickle is RCE.
Any temp files produced by older builds with ``data_type: "pickle"`` are
rejected on load.
"""

import base64
import os
import io
import time
import json
import tempfile
import mmap
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union, BinaryIO

from src.utils.logger import get_logger

logger = get_logger(__name__)

class SharedMemoryManager:
    """
    Manager for shared memory communication between processes.
    
    Uses memory-mapped files for efficient data transfer, avoiding
    the overhead of serializing and deserializing large data structures.
    """
    
    def __init__(self):
        """Initialize the shared memory manager."""
        self.temp_files = []
        self.mmaps = {}
    
    def write_array(self, array: np.ndarray) -> str:
        """
        Write a numpy array to shared memory.
        
        Args:
            array: Numpy array to write
            
        Returns:
            Identifier for the array in shared memory
        """
        try:
            # Create a temporary file
            fd, path = tempfile.mkstemp(suffix='.npy')
            os.close(fd)
            
            # Save the array to the file
            np.save(path, array)
            
            # Track the file
            self.temp_files.append(path)
            
            return path
        except Exception as e:
            logger.error(f"Error writing array to shared memory: {e}")
            raise
    
    def read_array(self, identifier: str) -> np.ndarray:
        """
        Read a numpy array from shared memory.
        
        Args:
            identifier: Identifier for the array in shared memory
            
        Returns:
            The numpy array
        """
        try:
            # Load the array from the file
            array = np.load(identifier)
            
            return array
        except Exception as e:
            logger.error(f"Error reading array from shared memory: {e}")
            raise
    
    def write_data(self, data: Any) -> str:
        """
        Write data to a temp file, using typed JSON serialization.

        The payload is routed through ``IPCSerializer`` so only explicitly
        supported types (ndarray, Tensor, lists/dicts/JSON primitives) are
        accepted. Unsupported types raise — never pickled.
        """
        try:
            prepared = IPCSerializer.prepare_data(data)
            fd, path = tempfile.mkstemp(suffix='.json')
            os.close(fd)
            with open(path, 'w') as f:
                json.dump(prepared, f)
            self.temp_files.append(path)
            return path
        except Exception as e:
            logger.error(f"Error writing data to shared memory: {e}")
            raise

    def read_data(self, identifier: str) -> Any:
        """Load data previously written by ``write_data``."""
        try:
            with open(identifier, 'r') as f:
                prepared = json.load(f)
            return IPCSerializer.restore_data(prepared)
        except Exception as e:
            logger.error(f"Error reading data from shared memory: {e}")
            raise
    
    def create_shared_memory(self, size: int) -> Tuple[str, mmap.mmap]:
        """
        Create a shared memory region.
        
        Args:
            size: Size of the shared memory region in bytes
            
        Returns:
            Tuple of (identifier, memory map)
        """
        try:
            # Create a temporary file
            fd, path = tempfile.mkstemp(suffix='.shared')
            
            # Resize the file to the requested size
            os.truncate(fd, size)
            
            # Create a memory map
            mmap_obj = mmap.mmap(fd, size, access=mmap.ACCESS_WRITE)
            
            # Close the file descriptor (the mmap keeps it open)
            os.close(fd)
            
            # Track the file and mmap
            self.temp_files.append(path)
            self.mmaps[path] = mmap_obj
            
            return path, mmap_obj
        except Exception as e:
            logger.error(f"Error creating shared memory: {e}")
            raise
    
    def open_shared_memory(self, identifier: str, size: int) -> mmap.mmap:
        """
        Open an existing shared memory region.
        
        Args:
            identifier: Identifier for the shared memory region
            size: Size of the shared memory region in bytes
            
        Returns:
            Memory map for the shared memory region
        """
        try:
            # Open the file
            fd = os.open(identifier, os.O_RDWR)
            
            # Create a memory map
            mmap_obj = mmap.mmap(fd, size, access=mmap.ACCESS_WRITE)
            
            # Close the file descriptor (the mmap keeps it open)
            os.close(fd)
            
            # Track the mmap
            self.mmaps[identifier] = mmap_obj
            
            return mmap_obj
        except Exception as e:
            logger.error(f"Error opening shared memory: {e}")
            raise
    
    def close_shared_memory(self, identifier: str):
        """
        Close a shared memory region.
        
        Args:
            identifier: Identifier for the shared memory region
        """
        try:
            # Close the memory map
            if identifier in self.mmaps:
                self.mmaps[identifier].close()
                del self.mmaps[identifier]
            
            # Remove the file
            if identifier in self.temp_files:
                try:
                    os.remove(identifier)
                except OSError:
                    pass
                self.temp_files.remove(identifier)
        except Exception as e:
            logger.error(f"Error closing shared memory: {e}")
    
    def cleanup(self):
        """Clean up all shared memory resources."""
        # Close all memory maps
        for mmap_obj in self.mmaps.values():
            try:
                mmap_obj.close()
            except Exception:
                pass
        
        self.mmaps.clear()
        
        # Remove all temporary files
        for path in self.temp_files:
            try:
                if os.path.exists(path):
                    os.remove(path)
            except Exception:
                pass
        
        self.temp_files.clear()

class IPCSerializer:
    """
    Serializer for IPC data transfer.
    
    Provides efficient serialization methods for different data types,
    optimized for transfer between processes.
    """
    
    @staticmethod
    def prepare_numpy_array(array: np.ndarray) -> Dict[str, Any]:
        """
        Prepare a numpy array for IPC transfer.
        
        Args:
            array: The numpy array to prepare
            
        Returns:
            Dictionary with serialized array information
        """
        # Save to a bytes buffer then base64-encode so the result is
        # JSON-serializable (callers route through json.dump).
        buffer = io.BytesIO()
        np.save(buffer, array, allow_pickle=False)
        buffer.seek(0)

        return {
            'type': 'numpy.ndarray',
            'dtype': str(array.dtype),
            'shape': list(array.shape),
            'size': int(array.size),
            'bytes_b64': base64.b64encode(buffer.read()).decode('ascii'),
        }
    
    @staticmethod
    def restore_numpy_array(metadata: Dict[str, Any]) -> np.ndarray:
        """
        Restore a numpy array from IPC transfer.
        
        Args:
            metadata: Dictionary with serialized array information
            
        Returns:
            The restored numpy array
        """
        # Legacy entries stored raw bytes under 'bytes'; current entries
        # use base64 under 'bytes_b64'. Support both on read.
        if 'bytes_b64' in metadata:
            raw = base64.b64decode(metadata['bytes_b64'])
        else:
            raw = metadata['bytes']
        buffer = io.BytesIO(raw)
        buffer.seek(0)
        return np.load(buffer, allow_pickle=False)
    
    @staticmethod
    def prepare_torch_tensor(tensor) -> Dict[str, Any]:
        """
        Prepare a torch tensor for IPC transfer.
        
        Args:
            tensor: The torch tensor to prepare
            
        Returns:
            Dictionary with serialized tensor information
        """
        import torch
        
        # Convert to numpy array
        if tensor.requires_grad:
            array = tensor.detach().cpu().numpy()
        else:
            array = tensor.cpu().numpy()
        
        # Use numpy serialization
        return IPCSerializer.prepare_numpy_array(array)
    
    @staticmethod
    def restore_torch_tensor(metadata: Dict[str, Any]):
        """
        Restore a torch tensor from IPC transfer.
        
        Args:
            metadata: Dictionary with serialized tensor information
            
        Returns:
            The restored torch tensor
        """
        import torch
        
        # Restore numpy array
        array = IPCSerializer.restore_numpy_array(metadata)
        
        # Convert to torch tensor
        return torch.from_numpy(array)
    
    @staticmethod
    def prepare_data(data: Any) -> Dict[str, Any]:
        """
        Prepare data for IPC transfer.
        
        Args:
            data: The data to prepare
            
        Returns:
            Dictionary with serialized data information
        """
        import torch
        
        # Handle numpy arrays
        if isinstance(data, np.ndarray):
            return {
                'data_type': 'numpy.ndarray',
                'content': IPCSerializer.prepare_numpy_array(data)
            }
        
        # Handle torch tensors
        elif str(type(data)).startswith("<class 'torch."):
            return {
                'data_type': 'torch.Tensor',
                'content': IPCSerializer.prepare_torch_tensor(data)
            }
        
        # Handle Python primitives that can be directly serialized by JSON
        elif isinstance(data, (str, int, float, bool, type(None))):
            return {
                'data_type': 'json',
                'content': data
            }
        
        # Handle lists
        elif isinstance(data, list):
            if data and isinstance(data[0], np.ndarray):
                # List of numpy arrays
                return {
                    'data_type': 'list.numpy.ndarray',
                    'content': [IPCSerializer.prepare_numpy_array(arr) for arr in data]
                }
            elif data and str(type(data[0])).startswith("<class 'torch."):
                # List of torch tensors
                return {
                    'data_type': 'list.torch.Tensor',
                    'content': [IPCSerializer.prepare_torch_tensor(t) for t in data]
                }
            else:
                # Regular list
                return {
                    'data_type': 'list',
                    'content': [IPCSerializer.prepare_data(item) for item in data]
                }
        
        # Handle dictionaries
        elif isinstance(data, dict):
            return {
                'data_type': 'dict',
                'content': {k: IPCSerializer.prepare_data(v) for k, v in data.items()}
            }
        
        # Reject types we can't serialize safely.
        # Pickle is never used — loading attacker-controlled pickle is RCE.
        else:
            msg = (
                f"IPCSerializer cannot safely serialize {type(data).__name__}; "
                "only numpy arrays, torch tensors, JSON primitives, and lists/dicts "
                "of those are supported."
            )
            logger.error(msg)
            raise TypeError(msg)
    
    @staticmethod
    def restore_data(data_info: Dict[str, Any]) -> Any:
        """
        Restore data from IPC transfer.
        
        Args:
            data_info: Dictionary with serialized data information
            
        Returns:
            The restored data
        """
        try:
            data_type = data_info.get('data_type')
            content = data_info.get('content')
            
            if data_type == 'numpy.ndarray':
                return IPCSerializer.restore_numpy_array(content)
            elif data_type == 'torch.Tensor':
                return IPCSerializer.restore_torch_tensor(content)
            elif data_type == 'json':
                return content
            elif data_type == 'list.numpy.ndarray':
                return [IPCSerializer.restore_numpy_array(arr_info) for arr_info in content]
            elif data_type == 'list.torch.Tensor':
                return [IPCSerializer.restore_torch_tensor(t_info) for t_info in content]
            elif data_type == 'list':
                return [IPCSerializer.restore_data(item_info) for item_info in content]
            elif data_type == 'dict':
                return {k: IPCSerializer.restore_data(v) for k, v in content.items()}
            elif data_type == 'pickle':
                # Legacy entries from older builds. Never restore.
                raise RuntimeError(
                    "Refusing to deserialize IPC payload with legacy 'pickle' "
                    "data_type. Recreate the payload with the current build."
                )
            elif data_type == 'error':
                raise ValueError(f"Received error during serialization: {content}")
            else:
                raise ValueError(f"Unknown data type: {data_type}")
        except Exception as e:
            logger.error(f"Error deserializing data: {e}")
            raise

# Singleton instance of shared memory manager
_shared_memory_manager = None

def get_shared_memory_manager() -> SharedMemoryManager:
    """
    Get the shared memory manager singleton.
    
    Returns:
        The shared memory manager
    """
    global _shared_memory_manager
    if _shared_memory_manager is None:
        _shared_memory_manager = SharedMemoryManager()
    return _shared_memory_manager

# Convenience functions

def to_shared_memory(data: Any) -> str:
    """
    Write data to shared memory.
    
    Args:
        data: The data to write
        
    Returns:
        Identifier for the data in shared memory
    """
    manager = get_shared_memory_manager()
    
    # Handle numpy arrays specially
    if isinstance(data, np.ndarray):
        return manager.write_array(data)
    
    # Handle other types
    return manager.write_data(data)

def from_shared_memory(identifier: str) -> Any:
    """
    Read data from shared memory.
    
    Args:
        identifier: Identifier for the data in shared memory
        
    Returns:
        The data
    """
    manager = get_shared_memory_manager()
    
    # Check file extension to determine how to load
    if identifier.endswith('.npy'):
        return manager.read_array(identifier)

    # Otherwise: safe typed JSON (see SharedMemoryManager.write_data).
    return manager.read_data(identifier)

def cleanup_shared_memory():
    """Clean up all shared memory resources."""
    manager = get_shared_memory_manager()
    manager.cleanup()