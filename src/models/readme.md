# Retriever System Architecture

## Overview

This document explains the current retriever architecture and the refactoring process to move from function-based to class-based implementations while maintaining backward compatibility.

## Architecture Components

### Base Interface

- `models/retriever_base.py`: Defines the abstract `BaseRetriever` class that all retrievers must implement.

### Class Implementations

- `models/rag_retriever.py`: Implements `RAGRetriever` class that extends `BaseRetriever`.
- `models/hybrid_rag_retriever.py`: Implements `HybridRAGRetriever` class that extends `BaseRetriever`.

### Adapter Functions (Backward Compatibility)

- `models/retriever.py`: Function-based adapter that delegates to `RAGRetriever`.
- `models/hybrid_retriever.py`: Function-based adapter that delegates to `HybridRAGRetriever`.

## Architectural Pattern

The system uses the **Adapter Pattern** to maintain compatibility with existing code that expects the function-based API while allowing migration to a cleaner, class-based implementation.

```
External System → Function-Based API → Adapter Functions → Class-Based Implementation
```

## Implementation Notes

1. The singleton pattern is used for retriever instances:
   - `rag_retriever = RAGRetriever()` in `models/rag_retriever.py`
   - `hybrid_rag_retriever = HybridRAGRetriever()` in `models/hybrid_rag_retriever.py`

2. Adapter functions maintain the original method signatures to ensure compatibility with existing code.

3. The adapter pattern means that there are now two ways to use the retrievers:
   - Legacy way (function-based): `from models.retriever import retrieve_documents`
   - New way (class-based): `from models.rag_retriever import rag_retriever`

## Future Refactoring

In the future, a complete refactoring would involve:

1. Updating `retriever_manager.py` to directly use the class instances.
2. Removing the adapter functions once all code is migrated to use the class-based API.
3. Standardizing error handling and parameter validation across all retriever implementations.

## Diagram

```
┌────────────────────┐     ┌─────────────────┐     ┌────────────────────┐
│  External Systems  │     │ Function Adapter │     │ Class Implementation│
│ (retriever_manager)│────▶│ retrieve_docs()  │────▶│  RAGRetriever      │
└────────────────────┘     └─────────────────┘     └────────────────────┘
                                   │                           ▲
                                   │                           │
                                   │                           │
                                   │                ┌────────────────────┐
                                   └───────────────▶│   BaseRetriever    │
                                                    │  (Abstract Class)  │
                                                    └────────────────────┘
```

## Testing

When testing the retriever system, test both:

1. The function-based API to ensure backward compatibility
2. The class-based implementation to validate the new architecture