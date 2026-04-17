# logger.py

import logging
import os
from datetime import datetime

# Set up root logger once
_root_logger_configured = False

def reset_log_file(log_file_path='logs/app.log', backup_old_log=True):
    """
    Reset the log file at application startup.

    Args:
        log_file_path (str): Path to the log file to reset
        backup_old_log (bool): Whether to backup the old log file before resetting
    """
    try:
        # Ensure logs directory exists
        log_dir = os.path.dirname(log_file_path)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
            print(f"Created log directory: {log_dir}")
        
        if os.path.exists(log_file_path):
            if backup_old_log:
                # Create a backup with timestamp
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                backup_path = f"{log_file_path}.backup_{timestamp}"

                # Only keep the last 5 backups to avoid accumulating too many files
                backup_dir = os.path.dirname(log_file_path) or '.'
                backup_files = [f for f in os.listdir(backup_dir) if f.startswith(f"{os.path.basename(log_file_path)}.backup_")]
                backup_files.sort(reverse=True)  # Most recent first

                # Remove old backups (keep only 4, so with the new one we'll have 5)
                for old_backup in backup_files[4:]:
                    try:
                        os.remove(os.path.join(backup_dir, old_backup))
                        print(f"Removed old log backup: {old_backup}")
                    except Exception as e:
                        print(f"Warning: Could not remove old backup {old_backup}: {e}")

                # Create new backup
                os.rename(log_file_path, backup_path)
                print(f"Backed up previous log to: {backup_path}")
            else:
                # Just remove the old log file
                os.remove(log_file_path)
                print(f"Removed previous log file: {log_file_path}")

        # Create a fresh log file with startup message
        with open(log_file_path, 'w') as f:
            startup_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            f.write(f"{startup_time} - INFO - logger - Application started - Log file reset\n")

        print(f"Log file reset: {log_file_path}")

    except Exception as e:
        print(f"Warning: Could not reset log file {log_file_path}: {e}")
        # Don't raise the exception - logging should not prevent app startup

def _configure_root_logger():
    """Configure the root logger to avoid duplicate logs"""
    global _root_logger_configured

    if not _root_logger_configured:
        # Configure the root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)

        # Console handler - only log INFO and above to console
        c_handler = logging.StreamHandler()
        c_handler.setLevel(logging.INFO)
        c_format = logging.Formatter('[%(asctime)s] [%(levelname)8s] %(filename)s:%(lineno)d - %(message)s',
                                    datefmt='%Y-%m-%d %H:%M:%S')
        c_handler.setFormatter(c_format)

        # Filter out memory cleaning logs
        class MemoryCleaningFilter(logging.Filter):
            def filter(self, record):
                # Filter out memory-related logs (case insensitive)
                memory_patterns = ['memory', 'cleanup', 'cleaned', 'cleared', 'MEMORY']
                message = record.getMessage()
                if any(pattern in message for pattern in memory_patterns):
                    # Allow ERROR level memory logs to pass through
                    return record.levelno >= logging.ERROR
                return True

        c_handler.addFilter(MemoryCleaningFilter())

        # File handler - log INFO and above to file, filter memory cleaning logs
        # Ensure logs directory exists
        os.makedirs('logs', exist_ok=True)
        f_handler = logging.FileHandler('logs/app.log')
        f_handler.setLevel(logging.INFO)  # Changed from DEBUG to INFO
        f_format = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
        f_handler.setFormatter(f_format)
        f_handler.addFilter(MemoryCleaningFilter())

        # Redact known secrets (Authorization headers, hf_/sk- tokens, and any
        # values registered via register_secret) before they hit the handlers.
        # Filters attached to the *handlers* (rather than to the root logger)
        # because Python's logging only re-applies handler filters when records
        # propagate up — logger filters wouldn't catch child-logger records.
        from src.utils.log_redaction import RedactingFilter, register_secret
        c_handler.addFilter(RedactingFilter())
        f_handler.addFilter(RedactingFilter())

        # Pull current secrets from config so registered values mask out
        # automatically. Lazy import: src.api.config imports the logger, so
        # eager import would loop.
        try:
            from src.api import config as _cfg
            for value in (_cfg.JWT_SECRET, _cfg.OLLAMA_API_KEY):
                register_secret(value)
        except Exception:
            pass  # config not importable yet (e.g. in tooling) — fine

        # Add handlers to root logger
        root_logger.addHandler(c_handler)
        root_logger.addHandler(f_handler)

        # Mark as configured
        _root_logger_configured = True

def get_logger(name):
    """
    Creates a logger with the specified name.

    Args:
        name (str): The name of the logger.

    Returns:
        Logger: Configured logger instance.
    """
    # Make sure root logger is configured first
    _configure_root_logger()

    # Get a logger with the specified name, inheriting root logger's handlers
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Don't add handlers to this logger - it will use the root logger's handlers
    # This prevents duplicate logging

    return logger
