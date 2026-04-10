"""
Chat Router — SSE streaming

POST /api/v1/chat/stream
"""

import asyncio
import json
import queue
from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.api.deps import get_session_id, get_rag_models
from src.api.deps import get_current_user
from src.api import config
from src.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()


class ChatRequest(BaseModel):
    query: str
    session_uuid: Optional[str] = None
    is_rag_mode: bool = True
    is_batch_mode: bool = False
    has_pasted_images: bool = False
    pasted_images: List[str] = []


@router.post("/chat/stream")
async def stream_chat(
    req: ChatRequest,
    session_id: str = Depends(get_session_id),
    user_id: str = Depends(get_current_user),
):
    """Stream a chat response via Server-Sent Events.

    Handles pasted images correctly:
    1. Decodes base64 images from client
    2. Saves to disk as files
    3. Passes file paths to generate_streaming_response()
    """
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="No query provided")

    # Load session data fresh (not via Depends — the generator needs its own copy)
    from src.services.session_manager.manager import load_session
    session_data = load_session(config.SESSION_FOLDER, session_id)
    if not session_data:
        raise HTTPException(status_code=404, detail="Session not found")
    if session_data.get('user_id') != user_id:
        raise HTTPException(status_code=403, detail="Not authorized")

    # Decode pasted images to disk BEFORE starting the generator
    saved_pasted_image_paths: list[str] = []
    if req.has_pasted_images and req.pasted_images:
        try:
            from src.utils.pasted_image_utils import decode_pasted_images_for_streaming
            saved_pasted_image_paths = decode_pasted_images_for_streaming(
                req.pasted_images,
                session_id,
                config.STATIC_FOLDER,
            )
            logger.info(f"Saved {len(saved_pasted_image_paths)} pasted images for streaming")
        except Exception as e:
            logger.error(f"Error processing pasted images: {e}")
            raise HTTPException(status_code=500, detail="Failed to process pasted images")

    generation_model = session_data.get('generation_model', 'ollama-gemma3n-vision-fp16')
    rag_models = get_rag_models()

    # Batch mode is driven by the UI toggle, passed through to the generator
    if req.is_batch_mode and req.is_rag_mode:
        session_data['_force_batch'] = True

    async def event_stream():
        """Run the sync streaming generator in a thread pool, yield SSE chunks."""
        chunk_queue: queue.Queue = queue.Queue()
        sentinel = object()

        def _run_sync():
            from src.services.response_generator.generator import generate_streaming_response
            from src.services.session_manager.manager import load_session, save_session
            from datetime import datetime
            import uuid as _uuid

            # For batch mode, collect per-document responses
            # For non-batch, these act as a single-entry list
            doc_responses: list[dict] = []
            current_doc_content = ""
            current_doc_images: list = []
            current_doc_name = ""
            response_model = generation_model
            is_batch = req.is_batch_mode

            try:
                for chunk in generate_streaming_response(
                    query=req.query,
                    session_id=session_id,
                    generation_model=generation_model,
                    session_data=session_data,
                    is_rag_mode=req.is_rag_mode,
                    user_id=user_id,
                    pasted_images_paths=saved_pasted_image_paths or None,
                    rag_models=rag_models,
                ):
                    chunk_queue.put(chunk)
                    # Collect response data for history
                    if isinstance(chunk, dict):
                        chunk_type = chunk.get('type')
                        if chunk_type == 'response':
                            current_doc_content += chunk.get('content', '')
                        elif chunk_type == 'images':
                            current_doc_images.extend(chunk.get('images', []))
                        elif chunk_type == 'model_info':
                            response_model = chunk.get('model', generation_model)
                        elif chunk_type == 'doc_start':
                            # Reset per-doc accumulators
                            current_doc_content = ""
                            current_doc_images = []
                            current_doc_name = chunk.get('doc_name', '')
                        elif chunk_type == 'doc_complete':
                            # Finalize this document's entry
                            if current_doc_content:
                                doc_responses.append({
                                    'content': f"### {current_doc_name}\n\n{current_doc_content}" if current_doc_name else current_doc_content,
                                    'images': current_doc_images,
                                    'doc_name': current_doc_name,
                                })
            except Exception as e:
                logger.error(f"Streaming error: {e}", exc_info=True)
                chunk_queue.put({'type': 'error', 'message': 'An internal error occurred'})
            finally:
                chunk_queue.put(sentinel)

                # For non-batch, wrap remaining content as a single entry
                if not is_batch and current_doc_content:
                    doc_responses.append({
                        'content': current_doc_content,
                        'images': current_doc_images,
                    })

                # Save chat history to session after streaming completes
                if doc_responses:
                    try:
                        sd = load_session(config.SESSION_FOLDER, session_id)
                        if sd:
                            history = sd.get('chat_history', [])
                            history.append({
                                'role': 'user',
                                'content': req.query,
                                'timestamp': datetime.now().isoformat(),
                            })
                            # One assistant entry per document
                            for doc in doc_responses:
                                history.append({
                                    'role': 'assistant',
                                    'content': doc['content'],
                                    'images': doc.get('images', []),
                                    'model': response_model,
                                    'id': str(_uuid.uuid4()),
                                    'timestamp': datetime.now().isoformat(),
                                })
                            sd['chat_history'] = history
                            # Auto-name session on first message
                            user_msgs = [m for m in history if m.get('role') == 'user']
                            if len(user_msgs) == 1:
                                sd['session_name'] = req.query[:50]
                            save_session(config.SESSION_FOLDER, session_id, sd)
                            logger.info(f"Saved chat history ({len(history)} messages) to session {session_id}")

                            # Store conversation in Mem0 for cross-session memory
                            try:
                                from src.models.memory.mem0_service import add_conversation
                                mem0_messages = [{"role": "user", "content": req.query}]
                                for doc in doc_responses:
                                    mem0_messages.append({"role": "assistant", "content": doc['content'][:500]})
                                add_conversation(user_id, mem0_messages, session_id=session_id)
                            except Exception as mem_err:
                                logger.debug(f"Mem0 storage skipped: {mem_err}")

                    except Exception as e:
                        logger.error(f"Error saving chat history: {e}")

        # Start sync generator in thread pool
        task = asyncio.get_running_loop().run_in_executor(None, _run_sync)

        try:
            while True:
                try:
                    chunk = chunk_queue.get(timeout=0.05)
                except queue.Empty:
                    await asyncio.sleep(0.01)
                    continue

                if chunk is sentinel:
                    break
                if isinstance(chunk, str):
                    yield chunk
                elif isinstance(chunk, dict):
                    yield f"data: {json.dumps(chunk)}\n\n"
        finally:
            # Ensure the thread finishes even if client disconnects
            if not task.done():
                await task

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
