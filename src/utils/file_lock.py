"""
File locking utilities for thread-safe file operations.

Provides cross-platform file locking to prevent race conditions
when multiple workers access the same files.
"""

import os
import time
import fcntl
import threading
from contextlib import contextmanager
from src.utils.logger import get_logger

logger = get_logger(__name__)


class FileLock:
    """Cross-platform file locking mechanism."""
    
    def __init__(self, filepath, timeout=10, check_interval=0.1):
        """
        Initialize file lock.
        
        Args:
            filepath: Path to the file to lock
            timeout: Maximum time to wait for lock in seconds
            check_interval: How often to check for lock availability
        """
        self.filepath = filepath
        self.lockfile = f"{filepath}.lock"
        self.timeout = timeout
        self.check_interval = check_interval
        self.lock_handle = None
        
    def acquire(self):
        """Acquire the file lock."""
        start_time = time.time()
        
        while True:
            try:
                # Try to create lock file exclusively
                self.lock_handle = open(self.lockfile, 'x')
                
                # On Unix, use fcntl for additional safety
                try:
                    fcntl.flock(self.lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except IOError:
                    # Another process has the lock
                    self.lock_handle.close()
                    os.remove(self.lockfile)
                    raise
                    
                logger.debug(f"Acquired lock for {self.filepath}")
                return True
                
            except (FileExistsError, IOError):
                # Lock file exists or is locked by another process
                if time.time() - start_time > self.timeout:
                    raise TimeoutError(f"Could not acquire lock for {self.filepath} within {self.timeout} seconds")
                    
                time.sleep(self.check_interval)
                
    def release(self):
        """Release the file lock."""
        if self.lock_handle:
            try:
                # Release fcntl lock
                try:
                    fcntl.flock(self.lock_handle.fileno(), fcntl.LOCK_UN)
                except:
                    pass
                    
                # Close and remove lock file
                self.lock_handle.close()
                self.lock_handle = None
                
                try:
                    os.remove(self.lockfile)
                except FileNotFoundError:
                    pass
                    
                logger.debug(f"Released lock for {self.filepath}")
            except Exception as e:
                logger.error(f"Error releasing lock for {self.filepath}: {e}")
                
    def __enter__(self):
        """Context manager entry."""
        self.acquire()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.release()
        return False


# Global lock registry to prevent multiple locks on same file within process
_lock_registry = {}
_registry_lock = threading.Lock()


@contextmanager
def file_lock(filepath, timeout=10):
    """
    Context manager for file locking.

    Usage:
        with file_lock('/path/to/file.json'):
            # Read or write file safely
            pass

    If a stale ``<filepath>.lock`` from a previously crashed writer blocks
    this process past the timeout, the stale file is removed and acquisition
    is retried once. Stale is determined by the lock file being older than
    the timeout AND the owning process no longer holding an fcntl lock.
    """
    filepath = os.fspath(filepath)
    with _registry_lock:
        if filepath not in _lock_registry:
            _lock_registry[filepath] = threading.Lock()
        process_lock = _lock_registry[filepath]

    with process_lock:
        lock = FileLock(filepath, timeout)
        try:
            try:
                lock.acquire()
            except TimeoutError:
                _try_clear_stale_lock(lock.lockfile, timeout)
                lock.acquire()
            yield
        finally:
            lock.release()


def _try_clear_stale_lock(lockfile: str, age_threshold: float) -> None:
    """Remove a stale lockfile left behind by a crashed writer."""
    try:
        st = os.stat(lockfile)
    except OSError:
        return
    if time.time() - st.st_mtime < age_threshold:
        return
    try:
        with open(lockfile, 'r') as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            # If we got here, no one holds the fcntl lock — safe to delete.
            os.remove(lockfile)
            logger.warning(f"Removed stale lock file: {lockfile}")
    except (IOError, OSError):
        # Another process holds the lock, or the file vanished. Don't touch it.
        return


def safe_json_read(filepath, default=None):
    """
    Safely read JSON file with locking.
    
    Args:
        filepath: Path to JSON file
        default: Default value if file doesn't exist or is invalid
        
    Returns:
        Parsed JSON data or default value
    """
    import json
    
    try:
        with file_lock(filepath):
            if not os.path.exists(filepath):
                return default
                
            with open(filepath, 'r') as f:
                return json.load(f)
    except Exception as e:
        logger.error(f"Error reading JSON file {filepath}: {e}")
        return default


def safe_json_write(filepath, data):
    """
    Safely write JSON file with locking.
    
    Args:
        filepath: Path to JSON file
        data: Data to serialize to JSON
        
    Returns:
        True if successful, False otherwise
    """
    import json
    
    try:
        with file_lock(filepath):
            # Write to temporary file first
            temp_file = f"{filepath}.tmp"
            with open(temp_file, 'w') as f:
                json.dump(data, f, indent=2)
                
            # Atomic rename
            os.rename(temp_file, filepath)
            return True
            
    except Exception as e:
        logger.error(f"Error writing JSON file {filepath}: {e}")
        return False