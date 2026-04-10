"""
Feedback Database Manager

This module provides database models and functions for storing and retrieving
user feedback on RAG retrieval results and response quality.
"""

import os
import sqlite3
import json
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from contextlib import contextmanager

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Database path
FEEDBACK_DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'feedback.db')

def init_feedback_database():
    """Initialize the feedback database with required tables."""
    try:
        # Ensure the data directory exists
        os.makedirs(os.path.dirname(FEEDBACK_DB_PATH), exist_ok=True)

        with get_db_connection() as conn:
            cursor = conn.cursor()

            # Run database migrations first
            _run_database_migrations(cursor)
            
            # Create feedback table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS feedback (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    message_id TEXT NOT NULL,
                    query TEXT NOT NULL,
                    relevant_images TEXT,  -- JSON array of image paths
                    response_feedback TEXT,  -- Legacy field for backward compatibility
                    expected_response TEXT,  -- New field for user's ideal response
                    original_prompt TEXT,
                    retrieval_model TEXT,
                    generation_model TEXT,
                    timestamp REAL NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create optimization_runs table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS optimization_runs (
                    id TEXT PRIMARY KEY,
                    feedback_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    iteration_count INTEGER NOT NULL,
                    template_name TEXT,  -- AI-generated name, initially placeholder
                    status TEXT DEFAULT 'pending',  -- pending, running, completed, failed
                    current_iteration INTEGER DEFAULT 0,
                    best_prompt TEXT,
                    optimization_log TEXT,  -- JSON array of iteration results
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    completed_at DATETIME,
                    FOREIGN KEY (feedback_id) REFERENCES feedback (id)
                )
            ''')
            
            # Create optimization_iterations table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS optimization_iterations (
                    id TEXT PRIMARY KEY,
                    optimization_run_id TEXT NOT NULL,
                    iteration_number INTEGER NOT NULL,
                    prompt_variant TEXT NOT NULL,
                    retrieval_results TEXT,  -- JSON array of retrieved images
                    response_text TEXT,
                    evaluation_score REAL,
                    evaluation_notes TEXT,
                    llm_evaluation TEXT,  -- JSON object containing LLM judge evaluation
                    evaluator_model TEXT,  -- Model used for evaluation
                    timestamp REAL NOT NULL,
                    FOREIGN KEY (optimization_run_id) REFERENCES optimization_runs (id)
                )
            ''')
            
            # Create indexes for better performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_feedback_user_id ON feedback (user_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_feedback_session_id ON feedback (session_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_feedback_message_id ON feedback (message_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_optimization_runs_feedback_id ON optimization_runs (feedback_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_optimization_runs_user_id ON optimization_runs (user_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_optimization_iterations_run_id ON optimization_iterations (optimization_run_id)')
            
            conn.commit()
            logger.info("Feedback database initialized successfully")
            
    except Exception as e:
        logger.error(f"Error initializing feedback database: {e}")
        raise

@contextmanager
def get_db_connection():
    """Get a database connection with proper error handling."""
    conn = None
    try:
        conn = sqlite3.connect(FEEDBACK_DB_PATH)
        conn.row_factory = sqlite3.Row  # Enable dict-like access to rows
        yield conn
    except Exception as e:
        if conn:
            conn.rollback()
        logger.error(f"Database error: {e}")
        raise
    finally:
        if conn:
            conn.close()

def store_feedback(user_id: str, session_id: str, message_id: str, query: str,
                  relevant_images: List[str], response_feedback: str = "",
                  expected_response: str = "", original_prompt: str = "",
                  retrieval_model: str = "", generation_model: str = "") -> str:
    """
    Store user feedback in the database.

    Args:
        user_id: The ID of the user providing feedback
        session_id: The session ID where the feedback was given
        message_id: The ID of the message being rated
        query: The original query that was asked
        relevant_images: List of image paths marked as relevant
        response_feedback: User's feedback on the response quality (legacy)
        expected_response: User's ideal/expected response (new approach)
        original_prompt: The original prompt used for generation
        retrieval_model: The model used for retrieval
        generation_model: The model used for generation

    Returns:
        The ID of the stored feedback record
    """
    try:
        feedback_id = str(uuid.uuid4())
        timestamp = datetime.now().timestamp()
        
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO feedback (
                    id, user_id, session_id, message_id, query,
                    relevant_images, response_feedback, expected_response, original_prompt,
                    retrieval_model, generation_model, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                feedback_id, user_id, session_id, message_id, query,
                json.dumps(relevant_images), response_feedback, expected_response, original_prompt,
                retrieval_model, generation_model, timestamp
            ))
            conn.commit()
            
        logger.info(f"Stored feedback {feedback_id} for user {user_id}")
        return feedback_id
        
    except Exception as e:
        logger.error(f"Error storing feedback: {e}")
        raise

def get_feedback(feedback_id: str) -> Optional[Dict[str, Any]]:
    """
    Retrieve feedback by ID.
    
    Args:
        feedback_id: The ID of the feedback to retrieve
        
    Returns:
        Dictionary containing feedback data or None if not found
    """
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM feedback WHERE id = ?', (feedback_id,))
            row = cursor.fetchone()
            
            if row:
                feedback = dict(row)
                # Parse JSON fields
                feedback['relevant_images'] = json.loads(feedback['relevant_images'])
                return feedback
            return None
            
    except Exception as e:
        logger.error(f"Error retrieving feedback {feedback_id}: {e}")
        return None

def create_optimization_run(feedback_id: str, user_id: str, iteration_count: int,
                           template_name: str) -> str:
    """
    Create a new optimization run.
    
    Args:
        feedback_id: The ID of the feedback to optimize
        user_id: The user ID
        iteration_count: Number of iterations to run
        template_name: Name for the resulting template
        
    Returns:
        The ID of the optimization run
    """
    try:
        run_id = str(uuid.uuid4())
        
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO optimization_runs (
                    id, feedback_id, user_id, iteration_count, template_name
                ) VALUES (?, ?, ?, ?, ?)
            ''', (run_id, feedback_id, user_id, iteration_count, template_name))
            conn.commit()
            
        logger.info(f"Created optimization run {run_id} for feedback {feedback_id}")
        return run_id
        
    except Exception as e:
        logger.error(f"Error creating optimization run: {e}")
        raise

def update_optimization_run_status(run_id: str, status: str, current_iteration: int = None,
                                  best_prompt: str = None, optimization_log: List[Dict] = None,
                                  final_template_name: str = None):
    """
    Update the status of an optimization run.

    Args:
        run_id: The optimization run ID
        status: New status (pending, running, completed, failed)
        current_iteration: Current iteration number
        best_prompt: Best prompt found so far
        optimization_log: Log of optimization iterations
        final_template_name: Final AI-generated template name
    """
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            update_fields = ['status = ?']
            values = [status]
            
            if current_iteration is not None:
                update_fields.append('current_iteration = ?')
                values.append(current_iteration)
                
            if best_prompt is not None:
                update_fields.append('best_prompt = ?')
                values.append(best_prompt)
                
            if optimization_log is not None:
                update_fields.append('optimization_log = ?')
                values.append(json.dumps(optimization_log))

            if final_template_name is not None:
                update_fields.append('template_name = ?')
                values.append(final_template_name)

            if status == 'completed':
                update_fields.append('completed_at = CURRENT_TIMESTAMP')
                
            values.append(run_id)
            
            query = f'UPDATE optimization_runs SET {", ".join(update_fields)} WHERE id = ?'
            cursor.execute(query, values)
            conn.commit()
            
        logger.info(f"Updated optimization run {run_id} status to {status}")
        
    except Exception as e:
        logger.error(f"Error updating optimization run {run_id}: {e}")
        raise

def store_optimization_iteration(run_id: str, iteration_number: int, prompt_variant: str,
                               retrieval_results: List[str], response_text: str,
                               evaluation_score: float, evaluation_notes: str = "",
                               llm_evaluation: Dict[str, Any] = None, evaluator_model: str = "",
                               optimized_query: str = "") -> str:
    """
    Store the results of an optimization iteration.

    Args:
        run_id: The optimization run ID
        iteration_number: The iteration number
        prompt_variant: The prompt variant tested
        retrieval_results: List of retrieved image paths
        response_text: The generated response
        evaluation_score: Score for this iteration
        evaluation_notes: Additional evaluation notes
        llm_evaluation: LLM judge evaluation results (JSON)
        evaluator_model: Model used for evaluation
        optimized_query: The optimized query used for this iteration

    Returns:
        The ID of the stored iteration
    """
    try:
        iteration_id = str(uuid.uuid4())
        timestamp = datetime.now().timestamp()
        
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO optimization_iterations (
                    id, optimization_run_id, iteration_number, prompt_variant,
                    retrieval_results, response_text, evaluation_score,
                    evaluation_notes, llm_evaluation, evaluator_model, optimized_query, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                iteration_id, run_id, iteration_number, prompt_variant,
                json.dumps(retrieval_results), response_text, evaluation_score,
                evaluation_notes, json.dumps(llm_evaluation) if llm_evaluation else None,
                evaluator_model, optimized_query, timestamp
            ))
            conn.commit()
            
        logger.info(f"Stored optimization iteration {iteration_id} for run {run_id}")
        return iteration_id
        
    except Exception as e:
        logger.error(f"Error storing optimization iteration: {e}")
        raise

def get_user_feedback_history(user_id: str, limit: int = 50) -> List[Dict[str, Any]]:
    """
    Get feedback history for a user.
    
    Args:
        user_id: The user ID
        limit: Maximum number of records to return
        
    Returns:
        List of feedback records
    """
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM feedback 
                WHERE user_id = ? 
                ORDER BY timestamp DESC 
                LIMIT ?
            ''', (user_id, limit))
            
            rows = cursor.fetchall()
            feedback_list = []
            
            for row in rows:
                feedback = dict(row)
                feedback['relevant_images'] = json.loads(feedback['relevant_images'])
                feedback_list.append(feedback)
                
            return feedback_list
            
    except Exception as e:
        logger.error(f"Error retrieving feedback history for user {user_id}: {e}")
        return []


def _run_database_migrations(cursor):
    """Run database migrations to update schema for new features."""
    try:
        # Check if expected_response column exists in feedback table
        cursor.execute("PRAGMA table_info(feedback)")
        columns = [column[1] for column in cursor.fetchall()]

        if 'expected_response' not in columns:
            logger.info("Adding expected_response column to feedback table")
            cursor.execute("ALTER TABLE feedback ADD COLUMN expected_response TEXT")

        # Check if llm_evaluation and evaluator_model columns exist in optimization_iterations table
        cursor.execute("PRAGMA table_info(optimization_iterations)")
        columns = [column[1] for column in cursor.fetchall()]

        if 'llm_evaluation' not in columns:
            logger.info("Adding llm_evaluation column to optimization_iterations table")
            cursor.execute("ALTER TABLE optimization_iterations ADD COLUMN llm_evaluation TEXT")

        if 'evaluator_model' not in columns:
            logger.info("Adding evaluator_model column to optimization_iterations table")
            cursor.execute("ALTER TABLE optimization_iterations ADD COLUMN evaluator_model TEXT")

        if 'optimized_query' not in columns:
            logger.info("Adding optimized_query column to optimization_iterations table")
            cursor.execute("ALTER TABLE optimization_iterations ADD COLUMN optimized_query TEXT")

        logger.info("Database migrations completed successfully")

    except Exception as e:
        logger.error(f"Error running database migrations: {e}")
        # Don't raise the error - let the database initialization continue


# Initialize the database when the module is imported
try:
    init_feedback_database()
except Exception as e:
    logger.error(f"Failed to initialize feedback database: {e}")
