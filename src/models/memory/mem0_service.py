"""
Mem0 memory service — automatic conversation memory using local Ollama models.

Replaces the previous custom ChromaDB + embedding + topic detection system
with Mem0's built-in memory extraction and semantic search.
All processing is local (Ollama LLM + embeddings, ChromaDB storage).
"""

import os
from typing import Optional
from src.utils.logger import get_logger

logger = get_logger(__name__)

_mem0_instance = None
_mem0_init_failed = False


def _get_mem0():
    """Lazy-initialize the Mem0 Memory instance with local Ollama config."""
    global _mem0_instance, _mem0_init_failed
    if _mem0_instance is not None:
        return _mem0_instance
    if _mem0_init_failed:
        return None

    try:
        from mem0 import Memory

        config = {
            "llm": {
                "provider": "ollama",
                "config": {
                    "model": os.environ.get("MEM0_LLM_MODEL", "gemma3:4b"),
                    "temperature": 0,
                    "max_tokens": 1000,
                    "ollama_base_url": os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
                },
            },
            "embedder": {
                "provider": "ollama",
                "config": {
                    "model": os.environ.get("MEM0_EMBED_MODEL", "nomic-embed-text:latest"),
                    "ollama_base_url": os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
                },
            },
            "vector_store": {
                "provider": "chroma",
                "config": {
                    "collection_name": "raglab_memory",
                    "path": os.environ.get("MEM0_CHROMA_PATH", "./data/mem0_chroma"),
                },
            },
        }

        _mem0_instance = Memory.from_config(config)
        logger.info("Mem0 memory service initialized (Ollama + ChromaDB)")
        return _mem0_instance

    except Exception as e:
        logger.warning(f"Mem0 initialization failed (memory features disabled): {e}")
        _mem0_init_failed = True
        return None


def add_conversation(user_id: str, messages: list[dict], session_id: Optional[str] = None):
    """Store a conversation exchange in Mem0.

    Args:
        user_id: User identifier
        messages: List of {role, content} dicts (typically a user + assistant pair)
        session_id: Optional session ID for metadata
    """
    m = _get_mem0()
    if not m:
        return

    try:
        metadata = {}
        if session_id:
            metadata["session_id"] = session_id

        m.add(messages, user_id=user_id, metadata=metadata)
        logger.debug(f"Mem0: stored {len(messages)} messages for user {user_id}")
    except Exception as e:
        logger.warning(f"Mem0: failed to store conversation: {e}")


def get_relevant_memories(user_id: str, query: str, limit: int = 5) -> list[str]:
    """Retrieve memories relevant to the current query.

    Args:
        user_id: User identifier
        query: Current user query
        limit: Max number of memories to return

    Returns:
        List of memory strings, most relevant first
    """
    m = _get_mem0()
    if not m:
        return []

    try:
        results = m.search(query, user_id=user_id, limit=limit)

        # Extract memory text from results
        memories = []
        if isinstance(results, dict) and "results" in results:
            for item in results["results"]:
                text = item.get("memory", "")
                if text:
                    memories.append(text)
        elif isinstance(results, list):
            for item in results:
                text = item.get("memory", "") if isinstance(item, dict) else str(item)
                if text:
                    memories.append(text)

        if memories:
            logger.debug(f"Mem0: retrieved {len(memories)} memories for query '{query[:50]}...'")
        return memories[:limit]

    except Exception as e:
        logger.warning(f"Mem0: memory search failed: {e}")
        return []


def clear_user_memories(user_id: str):
    """Delete all memories for a user."""
    m = _get_mem0()
    if not m:
        return

    try:
        m.delete_all(user_id=user_id)
        logger.info(f"Mem0: cleared all memories for user {user_id}")
    except Exception as e:
        logger.warning(f"Mem0: failed to clear memories: {e}")


def get_all_memories(user_id: str) -> list[dict]:
    """Get all stored memories for a user (for debugging/admin)."""
    m = _get_mem0()
    if not m:
        return []

    try:
        result = m.get_all(user_id=user_id)
        if isinstance(result, dict) and "results" in result:
            return result["results"]
        elif isinstance(result, list):
            return result
        return []
    except Exception as e:
        logger.warning(f"Mem0: failed to get memories: {e}")
        return []
