"""
Memory diagnostics tools for RAG Lab.

This module provides functions to diagnose memory issues and identify memory usage.
"""

import os
import gc
import sys
import torch
import psutil
import numpy as np
from collections import defaultdict
import logging

from src.utils.logger import get_logger

logger = get_logger(__name__)

def get_size(obj, seen=None):
    """
    Recursively find the size of objects in memory

    Args:
        obj: The object to size
        seen: Set of already seen objects (for recursion)

    Returns:
        Size in bytes
    """
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0

    # Mark as seen
    seen.add(obj_id)

    # Handle containers
    if isinstance(obj, dict):
        size += sum(get_size(k, seen) + get_size(v, seen) for k, v in obj.items())
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        try:
            size += sum(get_size(i, seen) for i in obj)
        except TypeError:
            pass

    return size

def find_large_objects(threshold_mb=10):
    """
    Find large objects in memory that might be causing memory leaks

    Args:
        threshold_mb: Minimum size in MB to report

    Returns:
        Dictionary mapping object types to their total memory usage
    """
    gc.collect()  # Collect garbage first

    objects = defaultdict(int)
    tensor_shapes = defaultdict(list)
    count_by_type = defaultdict(int)

    # Get all objects
    for obj in gc.get_objects():
        try:
            # Skip modules, types, and functions - they're not typically our memory issue
            if isinstance(obj, (type, type(sys), type(find_large_objects))):
                continue

            # Get object type and size
            obj_type = type(obj).__name__
            size = 0

            # Special handling for torch tensors to get more details
            if obj_type == 'Tensor' and hasattr(obj, 'shape'):
                try:
                    shape_str = str(tuple(obj.shape))
                    dtype_str = str(obj.dtype)
                    tensor_id = f"Tensor[{shape_str}][{dtype_str}]"
                    size = obj.element_size() * obj.nelement()
                    tensor_shapes[tensor_id].append(size)
                except:
                    size = sys.getsizeof(obj)
            # Special handling for numpy arrays
            elif obj_type == 'ndarray' and hasattr(obj, 'shape'):
                try:
                    shape_str = str(tuple(obj.shape))
                    dtype_str = str(obj.dtype)
                    tensor_id = f"ndarray[{shape_str}][{dtype_str}]"
                    size = obj.itemsize * obj.size
                    tensor_shapes[tensor_id].append(size)
                except:
                    size = sys.getsizeof(obj)
            # Default size calculation for non-tensor objects
            else:
                size = sys.getsizeof(obj)

            objects[obj_type] += size
            count_by_type[obj_type] += 1
        except:
            # Some objects can't be safely examined - skip them
            continue

    # Convert byte sizes to MB and filter by threshold
    result = {
        f"{t} (count: {count_by_type[t]})": s / (1024 * 1024)
        for t, s in objects.items()
        if s / (1024 * 1024) >= threshold_mb
    }

    # Also include tensor shapes that meet the threshold
    tensor_results = {}
    for shape, sizes in tensor_shapes.items():
        total_size = sum(sizes)
        count = len(sizes)
        if total_size / (1024 * 1024) >= threshold_mb:
            tensor_results[f"{shape} (count: {count})"] = total_size / (1024 * 1024)

    return {
        "large_objects_by_type": result,
        "large_tensors_by_shape": tensor_results
    }

def memory_usage_report():
    """
    Generate a detailed memory usage report

    Returns:
        Dictionary with memory usage information
    """
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()

    # Get process memory usage
    ram_usage = mem_info.rss / (1024 * 1024)  # MB
    ram_percent = process.memory_percent()

    # Get GPU memory if available
    gpu_usage = 0
    gpu_percent = 0
    gpu_allocated = 0
    gpu_reserved = 0
    if torch.cuda.is_available():
        gpu_allocated = torch.cuda.memory_allocated() / (1024 * 1024)  # MB
        gpu_reserved = torch.cuda.memory_reserved() / (1024 * 1024)  # MB

        try:
            gpu_stats = torch.cuda.memory_stats()
            active_alloc = gpu_stats.get('active_bytes.all.allocated', 0) / (1024 * 1024)
            active_freed = gpu_stats.get('active_bytes.all.freed', 0) / (1024 * 1024)
            inactive_alloc = gpu_stats.get('inactive_split_bytes.all.allocated', 0) / (1024 * 1024)
            inactive_freed = gpu_stats.get('inactive_split_bytes.all.freed', 0) / (1024 * 1024)
        except:
            active_alloc = active_freed = inactive_alloc = inactive_freed = 0

    # Get the LanceDB cache stats
    try:
        from src.models.vector_stores.lancedb_manager import _lancedb_connections
        lancedb_cache_size = len(_lancedb_connections)
        lancedb_cache_keys = list(_lancedb_connections.keys())
        lancedb_cache_stats = {
            "size": lancedb_cache_size,
            "keys": lancedb_cache_keys
        }
    except:
        lancedb_cache_stats = {"error": "Could not access LanceDB cache stats"}

    # Find large objects
    large_objects = find_large_objects(threshold_mb=10)

    # Get Python garbage collector stats
    gc_stats = {
        "objects": len(gc.get_objects()),
        "garbage": len(gc.garbage),
        "collections": gc.get_count(),
        "thresholds": gc.get_threshold(),
    }

    # Check if there are any memory leaks with the garbage collector
    gc.collect()
    unreachable = len(gc.get_objects())

    # Put it all together
    return {
        "process": {
            "pid": os.getpid(),
            "ram_usage_mb": ram_usage,
            "ram_percent": ram_percent,
        },
        "gpu": {
            "available": torch.cuda.is_available(),
            "allocated_mb": gpu_allocated,
            "reserved_mb": gpu_reserved,
            "active_allocated_mb": active_alloc if torch.cuda.is_available() else 0,
            "active_freed_mb": active_freed if torch.cuda.is_available() else 0,
            "inactive_allocated_mb": inactive_alloc if torch.cuda.is_available() else 0,
            "inactive_freed_mb": inactive_freed if torch.cuda.is_available() else 0,
        },
        "python": {
            "gc_stats": gc_stats,
            "unreachable_objects": unreachable,
        },
        "lancedb_cache": lancedb_cache_stats,
        "large_objects": large_objects,
    }

def log_memory_report():
    """
    Generate and log a detailed memory usage report
    """
    try:
        report = memory_usage_report()

        # Log the basic information
        logger.debug("=== MEMORY USAGE REPORT ===")
        logger.debug(f"Process RAM: {report['process']['ram_usage_mb']:.2f} MB ({report['process']['ram_percent']:.2f}%)")

        if report['gpu']['available']:
            logger.debug(f"GPU Memory: Allocated {report['gpu']['allocated_mb']:.2f} MB, Reserved {report['gpu']['reserved_mb']:.2f} MB")

        # Log LanceDB cache info
        if 'error' not in report['lancedb_cache']:
            logger.debug(f"LanceDB Cache: {report['lancedb_cache']['size']} connections in cache")
            if report['lancedb_cache']['size'] > 0:
                logger.debug(f"LanceDB Cache Keys: {report['lancedb_cache']['keys']}")

        # Log large objects by type
        if report['large_objects']['large_objects_by_type']:
            logger.debug("Large Objects by Type (>= 10MB):")
            for obj_type, size_mb in sorted(report['large_objects']['large_objects_by_type'].items(), key=lambda x: x[1], reverse=True):
                logger.debug(f"  {obj_type}: {size_mb:.2f} MB")

        # Log large tensors by shape
        if report['large_objects']['large_tensors_by_shape']:
            logger.debug("Large Tensors by Shape (>= 10MB):")
            for shape, size_mb in sorted(report['large_objects']['large_tensors_by_shape'].items(), key=lambda x: x[1], reverse=True):
                logger.debug(f"  {shape}: {size_mb:.2f} MB")

        # Log Python GC stats
        logger.debug(f"Python GC: {report['python']['unreachable_objects']} unreachable objects after collection")
        logger.debug(f"Python GC: {report['python']['gc_stats']['objects']} total objects tracked")

        logger.debug("=== END MEMORY REPORT ===")

        return report
    except Exception as e:
        logger.error(f"Error generating memory report: {e}")
        return {"error": str(e)}

if __name__ == '__main__':
    # When run directly, print a full memory report
    log_memory_report()
