"""
Platform Compatibility Shim

Makes Linux-specific system calls safe on non-Linux platforms.
Import this module early in app startup (before any ctypes.CDLL('libc.so.6') calls).

On non-Linux:
- Patches ctypes.CDLL to return a fake libc when 'libc.so.6' is requested
- The fake libc's malloc_trim() calls gc.collect() instead
- signal.SIGALRM is set to a no-op constant if missing
- signal.alarm is set to a no-op function if missing

On Linux:
- Does nothing (all native calls work as-is)
"""

import sys
import gc
import logging

logger = logging.getLogger(__name__)

IS_LINUX = sys.platform.startswith('linux')


def apply_platform_shim():
    """Apply cross-platform compatibility patches.

    Call this once during app startup, before any module tries to
    access libc.so.6 or signal.SIGALRM.
    """
    if IS_LINUX:
        logger.debug("Platform shim: Linux detected, no patches needed")
        return

    _patch_ctypes_libc()
    _patch_signal_alarm()
    logger.info(f"Platform shim applied for {sys.platform}")


def _patch_ctypes_libc():
    """Patch ctypes.CDLL to handle 'libc.so.6' on non-Linux."""
    import ctypes

    _original_cdll_init = ctypes.CDLL.__init__

    class _FakeLibc:
        """Fake libc that provides malloc_trim as gc.collect."""
        def malloc_trim(self, pad=0):
            gc.collect()
            return 1  # Success

    _fake_libc = _FakeLibc()

    _original_CDLL = ctypes.CDLL

    class _PatchedCDLL(ctypes.CDLL):
        def __new__(cls, name, *args, **kwargs):
            if isinstance(name, str) and 'libc.so' in name:
                logger.debug(f"Platform shim: intercepted CDLL('{name}'), returning fake libc")
                return _fake_libc
            return super().__new__(cls)

        def __init__(self, name, *args, **kwargs):
            if isinstance(name, str) and 'libc.so' in name:
                return  # Already handled in __new__
            _original_cdll_init(self, name, *args, **kwargs)

    ctypes.CDLL = _PatchedCDLL


def _patch_signal_alarm():
    """Provide signal.SIGALRM and signal.alarm on Windows."""
    import signal

    if not hasattr(signal, 'SIGALRM'):
        signal.SIGALRM = 14  # Standard Linux value, used as a constant
        logger.debug("Platform shim: added signal.SIGALRM constant")

    if not hasattr(signal, 'alarm'):
        def _noop_alarm(seconds):
            """No-op alarm for non-Linux. Use threading.Timer instead."""
            if seconds > 0:
                logger.debug(f"Platform shim: signal.alarm({seconds}) is a no-op on {sys.platform}")
            return 0
        signal.alarm = _noop_alarm
        logger.debug("Platform shim: added signal.alarm no-op")
