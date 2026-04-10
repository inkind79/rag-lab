"""
Memory usage tracking and logging module.

This module provides functions for tracking memory usage statistics
and logging them to a dedicated memory log file for monitoring.
"""

import os
import time
import logging
import psutil
import torch
import gc
from typing import Dict, Any, Optional, Tuple

# Create a dedicated memory logger
memory_logger = logging.getLogger("memory_tracker")
memory_logger.setLevel(logging.WARNING)  # Changed from INFO to WARNING to reduce logging

# Create a file handler specifically for memory logs
log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'logs')
os.makedirs(log_dir, exist_ok=True)
memory_log_file = os.path.join(log_dir, 'memory_usage.log')
file_handler = logging.FileHandler(memory_log_file)
file_handler.setLevel(logging.WARNING)  # Changed from INFO to WARNING

# Create a formatter and add it to the handler
formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s')
file_handler.setFormatter(formatter)

# Add the handler to the logger
memory_logger.addHandler(file_handler)

# Ensure memory_logger output isn't duplicated to root logger
memory_logger.propagate = False

def log_memory_usage(label: str, session_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Log current memory usage with an explanatory label.
    
    Args:
        label: Description of where/why this measurement is being taken
        session_id: Optional session ID for correlation
        
    Returns:
        dict: Memory usage information including RAM and GPU metrics
    """
    # System memory (RAM)
    process = psutil.Process(os.getpid())
    ram_usage = process.memory_info().rss / (1024 * 1024 * 1024)  # in GB
    system_memory = psutil.virtual_memory()
    system_ram_percent = system_memory.percent
    
    # GPU memory if available
    gpu_usage = None
    gpu_total = None
    gpu_percentage = None
    
    if torch.cuda.is_available():
        try:
            gpu_usage = torch.cuda.memory_allocated() / (1024 * 1024 * 1024)  # in GB
            gpu_reserved = torch.cuda.memory_reserved() / (1024 * 1024 * 1024)  # in GB
            gpu_total = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024 * 1024)  # in GB
            gpu_percentage = (gpu_usage / gpu_total) * 100 if gpu_total > 0 else 0
        except Exception as e:
            memory_logger.error(f"Error getting GPU memory: {e}")
    
    # Format the memory usage message
    message = f"MEMORY [{label}]"
    if session_id:
        message += f" [Session: {session_id}]"
    
    message += f" RAM: {ram_usage:.2f} GB ({system_ram_percent:.1f}%)"
    
    if gpu_usage is not None:
        message += f", GPU: {gpu_usage:.2f} GB ({gpu_percentage:.1f}%)"
    
    # Log the message
    memory_logger.info(message)
    
    # Return the memory usage statistics
    return {
        "ram_usage_gb": round(ram_usage, 2),
        "system_ram_percent": round(system_ram_percent, 2),
        "gpu_usage_gb": round(gpu_usage, 2) if gpu_usage is not None else None,
        "gpu_total_gb": round(gpu_total, 2) if gpu_total is not None else None,
        "gpu_percentage": round(gpu_percentage, 2) if gpu_percentage is not None else None,
        "label": label,
        "session_id": session_id,
        "timestamp": time.time()
    }

def log_memory_comparison(before: Dict[str, Any], after: Dict[str, Any], operation: str, session_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Log a comparison of memory usage before and after an operation.
    
    Args:
        before: Memory usage statistics from before the operation
        after: Memory usage statistics from after the operation
        operation: Description of the operation performed
        session_id: Optional session ID for correlation
        
    Returns:
        dict: Memory difference statistics
    """
    # Calculate differences
    ram_before = before.get("ram_usage_gb", 0) or 0
    ram_after = after.get("ram_usage_gb", 0) or 0
    ram_diff = ram_before - ram_after
    
    gpu_before = before.get("gpu_usage_gb", 0) or 0
    gpu_after = after.get("gpu_usage_gb", 0) or 0
    gpu_diff = gpu_before - gpu_after
    
    # Format the memory comparison message
    message = f"MEMORY DIFF [{operation}]"
    if session_id:
        message += f" [Session: {session_id}]"
    
    message += f" RAM Change: {ram_diff:.2f} GB ({ram_before:.2f} → {ram_after:.2f})"
    
    if gpu_before is not None and gpu_after is not None:
        message += f", GPU Change: {gpu_diff:.2f} GB ({gpu_before:.2f} → {gpu_after:.2f})"
    
    # Log the message
    if ram_diff > 0 or (gpu_diff is not None and gpu_diff > 0):
        memory_logger.info(message)  # Positive change (memory freed)
    elif ram_diff < -0.1 or (gpu_diff is not None and gpu_diff < -0.1):
        memory_logger.warning(message)  # Negative change (memory used)
    else:
        memory_logger.debug(message)  # No significant change
    
    # Return the memory difference statistics
    return {
        "ram_diff": round(ram_diff, 2),
        "ram_before": round(ram_before, 2),
        "ram_after": round(ram_after, 2),
        "gpu_diff": round(gpu_diff, 2) if gpu_before is not None and gpu_after is not None else None,
        "gpu_before": round(gpu_before, 2) if gpu_before is not None else None,
        "gpu_after": round(gpu_after, 2) if gpu_after is not None else None,
        "operation": operation,
        "session_id": session_id,
        "timestamp": time.time()
    }