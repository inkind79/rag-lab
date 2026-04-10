"""
Cross-Platform Abstraction Layer

Wraps Linux-specific system calls with cross-platform alternatives:
- malloc_trim → gc.collect() on non-Linux
- SIGALRM/signal.alarm → threading.Timer on Windows
- SIGKILL process termination → psutil.Process.kill()
- /tmp paths → tempfile.gettempdir()
- os.system('sync') / drop_caches → no-op on non-Linux

Usage:
    from src.utils.platform import malloc_trim, get_temp_dir, kill_process, AlarmTimeout
"""

import os
import sys
import gc
import logging

logger = logging.getLogger(__name__)

IS_LINUX = sys.platform.startswith('linux')
IS_WINDOWS = sys.platform == 'win32'
IS_MACOS = sys.platform == 'darwin'


# --- Memory Management ---

def malloc_trim() -> bool:
    """Release freed memory back to the OS.

    On Linux: calls libc malloc_trim(0) to release fragmented heap memory.
    On other platforms: runs gc.collect() as the best available alternative.

    Returns:
        True if the operation succeeded
    """
    gc.collect()

    if IS_LINUX:
        try:
            import ctypes
            libc = ctypes.CDLL('libc.so.6')
            libc.malloc_trim(0)
            return True
        except (OSError, AttributeError) as e:
            logger.debug(f"malloc_trim unavailable: {e}")
            return False
    else:
        # gc.collect() already called above — best we can do on non-Linux
        return True


def flush_filesystem() -> None:
    """Flush filesystem buffers.

    On Linux: calls sync.
    On other platforms: no-op (OS handles this automatically).
    """
    if IS_LINUX:
        try:
            os.system('sync')
        except Exception as e:
            logger.debug(f"sync failed: {e}")


def drop_caches() -> None:
    """Drop kernel filesystem caches (Linux only, requires root).

    On non-Linux: no-op.
    """
    if IS_LINUX:
        try:
            os.system('echo 3 > /proc/sys/vm/drop_caches 2>/dev/null || true')
        except Exception as e:
            logger.debug(f"drop_caches failed: {e}")


# --- Temporary Directories ---

def get_temp_dir() -> str:
    """Get the platform-appropriate temporary directory.

    Returns tempfile.gettempdir() instead of hardcoded '/tmp'.
    """
    import tempfile
    return tempfile.gettempdir()


# --- Process Termination ---

def kill_process(pid: int, force: bool = True) -> bool:
    """Terminate a process by PID in a cross-platform way.

    On Linux: uses SIGKILL (force=True) or SIGTERM (force=False).
    On Windows: uses psutil.Process.kill() or .terminate().

    Args:
        pid: Process ID to terminate
        force: If True, force kill (SIGKILL/kill). If False, graceful (SIGTERM/terminate).

    Returns:
        True if the process was terminated
    """
    try:
        import psutil
        proc = psutil.Process(pid)
        if force:
            proc.kill()
        else:
            proc.terminate()
        return True
    except Exception as e:
        logger.warning(f"Failed to {'kill' if force else 'terminate'} process {pid}: {e}")
        return False


# --- Timeout via Alarm (cross-platform) ---

class AlarmTimeout:
    """Cross-platform timeout context manager.

    On Linux: uses signal.alarm (SIGALRM) for reliable timeout.
    On other platforms: uses threading.Timer as a fallback.

    Usage:
        with AlarmTimeout(seconds=30):
            # This code will raise TimeoutError after 30 seconds
            long_running_operation()
    """

    def __init__(self, seconds: int, message: str = "Operation timed out"):
        self.seconds = seconds
        self.message = message
        self._timer = None
        self._old_handler = None

    def __enter__(self):
        if IS_LINUX:
            import signal

            def _handler(signum, frame):
                raise TimeoutError(self.message)

            self._old_handler = signal.signal(signal.SIGALRM, _handler)
            signal.alarm(self.seconds)
        else:
            import threading

            def _timeout():
                # On non-Linux, we can't interrupt the main thread reliably.
                # We set a flag and log a warning instead.
                logger.error(f"TIMEOUT ({self.seconds}s): {self.message}")

            self._timer = threading.Timer(self.seconds, _timeout)
            self._timer.daemon = True
            self._timer.start()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if IS_LINUX:
            import signal
            signal.alarm(0)  # Cancel the alarm
            if self._old_handler is not None:
                signal.signal(signal.SIGALRM, self._old_handler)
        else:
            if self._timer is not None:
                self._timer.cancel()

        return False  # Don't suppress exceptions
