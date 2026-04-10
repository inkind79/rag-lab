# Memory Management for RAG Lab

This module provides a simple but effective solution for managing memory usage in the RAG Lab application, particularly for FAISS-based vector retrieval operations.

## Problem Addressed

When using FAISS for vector retrieval in RAG applications, memory usage can build up over time due to the following issues:

1. FAISS indexes remain in memory even after the response has been returned
2. PyTorch tensors and GPU memory may not be released automatically
3. Garbage collection may not be triggered frequently enough

This can lead to:
- Increased memory usage over time
- Eventual out-of-memory errors
- Degraded performance as available memory decreases

## Solution

The memory management module implements a simple cleanup mechanism that:

1. Explicitly removes FAISS indexes from the cache after each response
2. Forces garbage collection to reclaim unused memory
3. Empties the CUDA cache if GPU is being used
4. Provides detailed memory usage statistics for monitoring

## Integration Points

The cleanup is triggered at two key points in the application flow:

1. **Response Generator**: After a response is generated, memory is released
2. **Chat Routes**: After processing a query, memory is released again as a redundancy measure

This ensures that memory is properly cleared after each user interaction.

## Usage

To manually trigger the memory cleanup:

```python
from utils.memory_management import cleanup_after_response

# Perform general cleanup
cleanup_stats = cleanup_after_response()

# Perform targeted cleanup for a specific session and model
cleanup_stats = cleanup_after_response(
    session_id="session-uuid", 
    user_id="user-id",
    faiss_model_name="colqwen25"
)

# Get memory usage statistics
print(f"RAM freed: {cleanup_stats['freed']['ram']:.2f}MB")
print(f"GPU freed: {cleanup_stats['freed']['gpu']:.2f}MB")
```

## Memory Usage Monitoring

The module logs detailed memory usage statistics before and after cleanup. This can help identify memory leaks and monitor the effectiveness of the cleanup process.

## Future Improvements

Potential future enhancements:
- Scheduled memory cleanup for long-running sessions
- Per-model memory budgets to prevent any single model from using too much memory
- Automatic LRU caching for FAISS indexes based on frequency of use
