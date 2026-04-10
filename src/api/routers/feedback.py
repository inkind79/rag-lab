"""
Feedback Router — Feedback submission and query optimization

POST /api/v1/feedback/submit
POST /api/v1/feedback/optimize
GET  /api/v1/feedback/optimization/{run_id}/status
GET  /api/v1/feedback/optimization/{run_id}/results
"""

import json
import threading
from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from src.api.deps import get_session_id
from src.api.deps import get_current_user
from src.api import config
from src.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()


class FeedbackSubmit(BaseModel):
    message_id: str
    relevant_images: List[str] = []
    response_feedback: str = ""
    expected_response: str = ""


class OptimizeRequest(BaseModel):
    message_id: str
    iteration_count: int = 3
    expected_response: str = ""
    relevant_images: List[str] = []


@router.post("/feedback/submit")
async def submit_feedback(
    body: FeedbackSubmit,
    session_id: str = Depends(get_session_id),
    user_id: str = Depends(get_current_user),
):
    from src.models.feedback_db import save_feedback

    feedback_data = {
        'user_id': user_id,
        'session_id': session_id,
        'message_id': body.message_id,
        'relevant_images': body.relevant_images,
        'response_feedback': body.response_feedback,
        'expected_response': body.expected_response,
    }

    feedback_id = save_feedback(feedback_data)
    if not feedback_id:
        raise HTTPException(status_code=500, detail="Failed to save feedback")

    return {"success": True, "data": {"feedback_id": feedback_id}}


@router.post("/feedback/optimize")
async def start_optimization(
    body: OptimizeRequest,
    session_id: str = Depends(get_session_id),
    user_id: str = Depends(get_current_user),
):
    from src.models.feedback_db import create_optimization_run, update_optimization_run_status
    from src.services.session_manager.manager import load_session

    session_data = load_session(config.SESSION_FOLDER, session_id)
    if not session_data:
        raise HTTPException(status_code=404, detail="Session not found")

    chat_history = session_data.get('chat_history', [])

    # Find the message to optimize
    target_msg = None
    target_query = None
    for i, msg in enumerate(chat_history):
        if msg.get('id') == body.message_id or str(i) == body.message_id:
            target_msg = msg
            # Find the preceding user query
            for j in range(i - 1, -1, -1):
                if chat_history[j].get('role') == 'user':
                    target_query = chat_history[j].get('content', '')
                    break
            break

    if not target_query:
        raise HTTPException(status_code=400, detail="Could not find query for this message")

    run_id = create_optimization_run(
        user_id=user_id,
        session_id=session_id,
        iteration_count=body.iteration_count,
        template_name=session_data.get('selected_template_id', ''),
    )

    if not run_id:
        raise HTTPException(status_code=500, detail="Failed to create optimization run")

    # Run optimization in background thread
    def run_optimization():
        try:
            from src.services.prompt_optimizer import PromptOptimizer
            optimizer = PromptOptimizer(session_id=session_id, user_id=user_id)
            result = optimizer.optimize(
                query=target_query,
                run_id=run_id,
                iteration_count=body.iteration_count,
                expected_response=body.expected_response,
                relevant_images=body.relevant_images,
                session_data=session_data,
            )

            if result and result.get('success') and result.get('best_template'):
                best_template = result['best_template']
                saved_template_id = best_template.get('template_id')
                if saved_template_id:
                    from src.services.session_manager.manager import save_session
                    session_data['selected_template_id'] = saved_template_id
                    save_session(config.SESSION_FOLDER, session_id, session_data)
        except Exception as e:
            logger.error(f"Optimization error: {e}")
            update_optimization_run_status(run_id, 'failed')

    thread = threading.Thread(target=run_optimization, daemon=True)
    thread.start()

    return {"success": True, "data": {"optimization_run_id": run_id}}


@router.get("/feedback/optimization/{run_id}/status")
async def get_optimization_status(run_id: str, user_id: str = Depends(get_current_user)):
    from src.models.feedback_db import get_db_connection

    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT status, current_iteration, iteration_count, template_name,
                   optimization_log, created_at, completed_at
            FROM optimization_runs
            WHERE id = ? AND user_id = ?
        ''', (run_id, user_id))

        row = cursor.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Optimization run not found")

        result = dict(row)
        if result.get('optimization_log'):
            try:
                result['optimization_log'] = json.loads(result['optimization_log'])
            except Exception:
                result['optimization_log'] = []

        return {"success": True, "data": result}


@router.get("/feedback/optimization/{run_id}/results")
async def get_optimization_results(run_id: str, user_id: str = Depends(get_current_user)):
    from src.models.feedback_db import get_db_connection

    with get_db_connection() as conn:
        cursor = conn.cursor()

        # Get run details
        cursor.execute('''
            SELECT id, status, current_iteration, iteration_count, template_name,
                   optimization_log, created_at, completed_at
            FROM optimization_runs
            WHERE id = ? AND user_id = ?
        ''', (run_id, user_id))

        run_row = cursor.fetchone()
        if not run_row:
            raise HTTPException(status_code=404, detail="Optimization run not found")

        run_data = dict(run_row)

        # Get iterations
        cursor.execute('''
            SELECT iteration_number, prompt_variant, retrieval_results, response_text,
                   evaluation_score, evaluation_notes, llm_evaluation, evaluator_model,
                   optimized_query, timestamp
            FROM optimization_iterations
            WHERE optimization_run_id = ?
            ORDER BY iteration_number
        ''', (run_id,))

        iterations = []
        for row in cursor.fetchall():
            it = dict(row)
            for json_field in ('prompt_variant', 'retrieval_results', 'llm_evaluation'):
                if it.get(json_field):
                    try:
                        it[json_field] = json.loads(it[json_field])
                    except Exception:
                        it[json_field] = {} if json_field != 'retrieval_results' else []
            iterations.append(it)

        optimization_log = []
        if run_data.get('optimization_log'):
            try:
                optimization_log = json.loads(run_data['optimization_log'])
            except Exception:
                pass

        results = {
            'run_info': {
                'id': run_data['id'],
                'status': run_data['status'],
                'template_name': run_data['template_name'],
                'created_at': run_data['created_at'],
                'completed_at': run_data['completed_at'],
                'iteration_count': run_data['iteration_count'],
                'current_iteration': run_data['current_iteration'],
            },
            'iterations': iterations,
            'optimization_log': optimization_log,
            'summary': {
                'total_iterations': len(iterations),
                'best_score': max((it.get('evaluation_score', 0) for it in iterations), default=0),
                'best_iteration': max(iterations, key=lambda x: x.get('evaluation_score', 0)) if iterations else None,
            }
        }

        return {"success": True, "results": results}
