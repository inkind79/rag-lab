"""
Process Manager for Memory Isolation.

This module provides a central manager for all isolated processes, handling
lifecycle management, cleanup, and monitoring.
"""

import os
import time
import signal
import threading
import multiprocessing
import psutil
from typing import Dict, List, Any, Optional, Set, Tuple

from src.utils.logger import get_logger

logger = get_logger(__name__)

class ProcessManager:
    """
    Manager for isolated processes used in memory management.
    
    This class coordinates all isolated processes, ensuring they're
    properly terminated and resources are cleaned up.
    """
    
    def __init__(self, 
                 cleanup_interval: int = 60,    # 1 minute
                 max_process_age: int = 300,    # 5 minutes
                 memory_threshold: float = 85): # 85% memory usage
        """
        Initialize the process manager.
        
        Args:
            cleanup_interval: Interval in seconds between cleanup cycles
            max_process_age: Maximum age of a process in seconds before it's terminated
            memory_threshold: Memory threshold percentage to trigger emergency cleanup
        """
        self.cleanup_interval = cleanup_interval
        self.max_process_age = max_process_age
        self.memory_threshold = memory_threshold
        
        # Process tracking
        self.active_processes: Dict[int, Dict[str, Any]] = {}
        self.known_process_ids: Set[int] = set()
        
        # Statistics
        self.cleanup_runs = 0
        self.processes_terminated = 0
        self.emergency_cleanups = 0
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Start background monitoring thread
        self._start_monitoring()
    
    def _start_monitoring(self):
        """Start the background monitoring thread."""
        thread = threading.Thread(
            target=self._monitoring_thread,
            daemon=True
        )
        thread.start()
        logger.info("Started process monitoring thread")
    
    def _monitoring_thread(self):
        """Background thread for monitoring and cleanup."""
        logger.info("Process monitoring thread started")
        
        while True:
            try:
                # Sleep between checks
                time.sleep(self.cleanup_interval)
                
                # Check memory usage
                self._check_memory_usage()
                
                # Clean up old processes
                self._cleanup_old_processes()
                
                # Update statistics
                self.cleanup_runs += 1
                
            except Exception as e:
                logger.error(f"Error in process monitoring thread: {e}")
                # Wait a bit before trying again
                time.sleep(10)
    
    def _check_memory_usage(self):
        """Check system memory usage and perform cleanup if necessary."""
        try:
            # Get memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            if memory_percent > self.memory_threshold:
                logger.warning(f"Memory usage {memory_percent:.1f}% exceeds threshold "
                              f"{self.memory_threshold:.1f}%, performing emergency cleanup")
                
                # Perform emergency cleanup
                self._emergency_cleanup()
                
                # Update statistics
                self.emergency_cleanups += 1
                
        except Exception as e:
            logger.error(f"Error checking memory usage: {e}")
    
    def _emergency_cleanup(self):
        """
        Perform emergency cleanup when memory usage is high.
        This terminates all managed processes to free memory.
        """
        with self.lock:
            # Terminate all active processes
            for pid, info in list(self.active_processes.items()):
                try:
                    self._terminate_process(pid)
                except Exception as e:
                    logger.error(f"Error terminating process {pid} during emergency cleanup: {e}")
            
            # Clear process tracking
            self.active_processes.clear()
            
            # Force garbage collection
            self._force_gc()
            
            logger.info("Emergency cleanup completed")
    
    def _cleanup_old_processes(self):
        """Clean up processes that have been running for too long."""
        current_time = time.time()
        
        with self.lock:
            # Find processes to terminate
            pids_to_terminate = []
            
            for pid, info in list(self.active_processes.items()):
                start_time = info.get('start_time', 0)
                
                # Check if process has exceeded the maximum age
                if current_time - start_time > self.max_process_age:
                    pids_to_terminate.append(pid)
                    
                # Check if process is still running
                elif not self._is_process_alive(pid):
                    logger.info(f"Process {pid} is no longer running, removing from tracking")
                    del self.active_processes[pid]
                    if pid in self.known_process_ids:
                        self.known_process_ids.remove(pid)
            
            # Terminate old processes
            for pid in pids_to_terminate:
                try:
                    logger.info(f"Terminating old process {pid}")
                    self._terminate_process(pid)
                    self.processes_terminated += 1
                except Exception as e:
                    logger.error(f"Error terminating old process {pid}: {e}")
    
    def _is_process_alive(self, pid: int) -> bool:
        """
        Check if a process is still alive.
        
        Args:
            pid: Process ID to check
            
        Returns:
            True if the process is alive, False otherwise
        """
        try:
            # Check if the process exists
            process = psutil.Process(pid)
            return process.is_running()
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            return False
        except Exception as e:
            logger.error(f"Error checking if process {pid} is alive: {e}")
            return False
    
    def _terminate_process(self, pid: int):
        """
        Terminate a process.
        
        Args:
            pid: Process ID to terminate
        """
        try:
            # Get the process
            process = psutil.Process(pid)
            
            # Try SIGTERM first
            process.terminate()
            
            # Wait for process to terminate
            gone, alive = psutil.wait_procs([process], timeout=3)
            
            # If still alive, use SIGKILL
            if process in alive:
                logger.warning(f"Process {pid} did not terminate, using SIGKILL")
                process.kill()
                
                # Wait again
                gone, alive = psutil.wait_procs([process], timeout=3)
                
                if process in alive:
                    logger.error(f"Failed to kill process {pid}")
                    return
            
            logger.info(f"Successfully terminated process {pid}")
            
            # Remove from tracking
            if pid in self.active_processes:
                del self.active_processes[pid]
            
            if pid in self.known_process_ids:
                self.known_process_ids.remove(pid)
                
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            # Process already gone
            if pid in self.active_processes:
                del self.active_processes[pid]
            
            if pid in self.known_process_ids:
                self.known_process_ids.remove(pid)
                
        except Exception as e:
            logger.error(f"Error terminating process {pid}: {e}")
    
    def _force_gc(self):
        """Force garbage collection to free memory."""
        try:
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
            
        except Exception as e:
            logger.error(f"Error during forced garbage collection: {e}")
    
    def register_process(self, pid: int, name: str, description: str = "") -> bool:
        """
        Register a process with the manager.
        
        Args:
            pid: Process ID to register
            name: Name of the process
            description: Description of the process
            
        Returns:
            True if registration was successful, False otherwise
        """
        with self.lock:
            # Check if process exists
            if not self._is_process_alive(pid):
                logger.warning(f"Cannot register non-existent process {pid}")
                return False
            
            # Register the process
            self.active_processes[pid] = {
                'name': name,
                'description': description,
                'start_time': time.time(),
                'registered_time': time.time()
            }
            
            self.known_process_ids.add(pid)
            
            logger.info(f"Registered process {pid} ({name})")
            return True
    
    def register_current_process(self, name: str, description: str = "") -> bool:
        """
        Register the current process with the manager.
        
        Args:
            name: Name of the process
            description: Description of the process
            
        Returns:
            True if registration was successful, False otherwise
        """
        return self.register_process(os.getpid(), name, description)
    
    def unregister_process(self, pid: int, terminate: bool = False) -> bool:
        """
        Unregister a process from the manager.
        
        Args:
            pid: Process ID to unregister
            terminate: Whether to terminate the process
            
        Returns:
            True if unregistration was successful, False otherwise
        """
        with self.lock:
            # Check if process is registered
            if pid not in self.active_processes:
                logger.warning(f"Process {pid} is not registered")
                return False
            
            # Terminate if requested
            if terminate:
                try:
                    self._terminate_process(pid)
                except Exception as e:
                    logger.error(f"Error terminating process {pid}: {e}")
            
            # Unregister
            if pid in self.active_processes:
                del self.active_processes[pid]
            
            if pid in self.known_process_ids:
                self.known_process_ids.remove(pid)
            
            logger.info(f"Unregistered process {pid}")
            return True
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the process manager.
        
        Returns:
            Dictionary of statistics
        """
        with self.lock:
            return {
                'active_processes': len(self.active_processes),
                'known_process_ids': len(self.known_process_ids),
                'cleanup_runs': self.cleanup_runs,
                'processes_terminated': self.processes_terminated,
                'emergency_cleanups': self.emergency_cleanups,
                'process_list': [
                    {
                        'pid': pid,
                        'name': info.get('name', 'unknown'),
                        'description': info.get('description', ''),
                        'uptime': time.time() - info.get('start_time', time.time()),
                        'alive': self._is_process_alive(pid)
                    }
                    for pid, info in self.active_processes.items()
                ]
            }
    
    def terminate_all(self) -> int:
        """
        Terminate all managed processes.
        
        Returns:
            Number of processes terminated
        """
        count = 0
        with self.lock:
            for pid in list(self.active_processes.keys()):
                try:
                    self._terminate_process(pid)
                    count += 1
                except Exception as e:
                    logger.error(f"Error terminating process {pid}: {e}")
        
        return count

# Singleton instance
_process_manager = None

def get_process_manager() -> ProcessManager:
    """
    Get the process manager singleton.
    
    Returns:
        The process manager
    """
    global _process_manager
    if _process_manager is None:
        _process_manager = ProcessManager()
    return _process_manager

def register_isolated_process(pid: int, name: str, description: str = "") -> bool:
    """
    Register an isolated process with the manager.
    
    Args:
        pid: Process ID to register
        name: Name of the process
        description: Description of the process
        
    Returns:
        True if registration was successful, False otherwise
    """
    manager = get_process_manager()
    return manager.register_process(pid, name, description)

def unregister_isolated_process(pid: int, terminate: bool = False) -> bool:
    """
    Unregister an isolated process from the manager.
    
    Args:
        pid: Process ID to unregister
        terminate: Whether to terminate the process
        
    Returns:
        True if unregistration was successful, False otherwise
    """
    manager = get_process_manager()
    return manager.unregister_process(pid, terminate)