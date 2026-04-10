"""
Prompt Optimization Engine

This module implements an agentic loop that uses user feedback to iteratively
improve prompts for better RAG retrieval and response generation.
"""

import json
import asyncio
import time
import os
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

from src.utils.logger import get_logger
from src.models.feedback_db import (
    get_feedback, update_optimization_run_status, store_optimization_iteration
)
from src.models.model_loader import load_rag_model
from src.models.retriever_manager import select_retriever
from src.services.session_manager.manager import load_session
from src.models.prompt_templates import create_template
from src.services.llm_judge import get_llm_judge
from src.models.responder import generate_response

logger = get_logger(__name__)

class PromptOptimizer:
    """
    Agentic prompt optimization engine that uses feedback to improve prompts.
    """
    
    def __init__(self, app_config: Dict[str, Any], app=None):
        self.app_config = app_config
        self.app = app
        self.optimization_prompt = self._get_optimization_prompt()
    
    def _get_optimization_prompt(self) -> str:
        """Get the system prompt for the optimization agent."""
        return """You are an expert prompt optimization agent. Your task is to improve RAG (Retrieval-Augmented Generation) systems based on user feedback.

Given:
1. Original query and prompt
2. User's expected response
3. Retrieved documents that were marked as relevant/irrelevant by the user
4. User feedback on the response quality
5. Previous optimization attempts (if any)

Your goal is to optimize BOTH the query and prompt template to:
- Generate better queries for document retrieval
- Retrieve more relevant documents
- Generate responses that better match user expectations
- Address the specific issues mentioned in the feedback

Guidelines:
- Optimize user queries to find documents that would help produce the expected response
- Focus on clarity and specificity in prompts
- Consider both retrieval and generation aspects
- Make incremental improvements based on feedback
- Maintain the core intent while improving effectiveness

Respond with a JSON object containing:
{
    "optimized_query": "Improved query for better document retrieval",
    "improved_system_prompt": "The improved system prompt",
    "improved_query_prefix": "Improved query prefix if needed",
    "improved_query_suffix": "Improved query suffix if needed",
    "reasoning": "Explanation of changes made and why",
    "confidence_score": 0.85
}"""

    async def optimize_prompt(self, run_id: str, feedback_id: str, user_id: str, 
                            session_id: str, iteration_count: int) -> Dict[str, Any]:
        """
        Run the optimization process for a given feedback.
        
        Args:
            run_id: The optimization run ID
            feedback_id: The feedback ID to optimize
            user_id: The user ID
            session_id: The session ID
            iteration_count: Number of iterations to run
            
        Returns:
            Dictionary containing optimization results
        """
        logger.info(f"🚀 OPTIMIZATION START: run_id={run_id}, feedback_id={feedback_id}, user={user_id}, session={session_id}, iterations={iteration_count}")

        # Store run_id as instance variable for use in template saving
        self.run_id = run_id

        try:
            # Update status to running
            logger.info(f"📊 Updating optimization run status to 'running'")
            update_optimization_run_status(run_id, 'running', 0)

            # Get feedback data
            logger.info(f"📋 Retrieving feedback data for feedback_id={feedback_id}")
            feedback = get_feedback(feedback_id)
            if not feedback:
                logger.error(f"❌ Feedback {feedback_id} not found")
                raise ValueError(f"Feedback {feedback_id} not found")

            logger.info(f"📋 Feedback retrieved: query='{feedback.get('query', 'N/A')[:50]}...', expected_response='{feedback.get('expected_response', 'N/A')[:50]}...', relevant_images={len(feedback.get('relevant_images', []))}")
            
            # Load session data
            logger.info(f"📁 Loading session data for session_id={session_id}")
            session_data = load_session(self.app_config['SESSION_FOLDER'], session_id)
            if not session_data:
                logger.error(f"❌ Session {session_id} not found")
                raise ValueError(f"Session {session_id} not found")

            logger.info(f"📁 Session loaded: generation_model={session_data.get('generation_model', 'N/A')}, retrieval_model={session_data.get('retrieval_model', 'N/A')}")

            # Initialize optimization context
            optimization_log = []
            best_prompt = None
            best_score = 0.0
            logger.info(f"🔧 Optimization context initialized")

            # Get the original prompt context
            logger.info(f"🎯 Extracting original context from feedback and session")
            original_context = self._extract_original_context(feedback, session_data, session_id, user_id)
            logger.info(f"🎯 Original context extracted: query='{original_context.get('query', 'N/A')[:50]}...', model='{original_context.get('generation_model', 'N/A')}'")

            for iteration in range(1, iteration_count + 1):
                logger.info(f"🔄 ITERATION {iteration}/{iteration_count} START")

                # Update current iteration
                logger.info(f"📊 Updating optimization run status: iteration {iteration}")
                update_optimization_run_status(run_id, 'running', iteration)

                # Generate optimized query
                logger.info(f"🔍 Generating optimized query for iteration {iteration}")
                optimized_query = await self._generate_optimized_query(
                    original_context, feedback, optimization_log
                )
                logger.info(f"🔍 Generated optimized query: '{optimized_query}'")

                # Generate improved prompt
                logger.info(f"🧠 Generating improved prompt for iteration {iteration}")
                improved_prompt = await self._generate_improved_prompt(
                    original_context, feedback, optimization_log
                )
                logger.info(f"🧠 Generated prompt: system_prompt='{improved_prompt.get('system_prompt', 'N/A')[:50]}...', confidence={improved_prompt.get('confidence_score', 'N/A')}")

                # Test the improved prompt with optimized query
                logger.info(f"🧪 Testing improved prompt with optimized query for iteration {iteration}")
                test_session_data = {
                    **session_data,
                    'original_query': original_context['query'],
                    'optimized_query': optimized_query,
                    'session_id': session_id,
                    'user_id': user_id
                }
                test_results = await self._test_prompt(
                    improved_prompt, feedback, test_session_data
                )
                logger.info(f"🧪 Test completed: response_length={len(test_results.get('response_text', ''))}, retrieved_images={len(test_results.get('retrieved_images', []))}")

                # Evaluate the results using LLM judge
                logger.info(f"⚖️ Evaluating results with LLM judge for iteration {iteration}")
                evaluation_result = await self._evaluate_results_with_llm(
                    test_results, feedback, original_context, session_data
                )
                evaluation_score = evaluation_result.get('overall_score', 0.0)
                # Safe formatting for logging
                try:
                    score_str = f"{float(evaluation_score):.2f}" if evaluation_score != 'N/A' else 'N/A'
                    confidence = evaluation_result.get('confidence', 'N/A')
                    conf_str = f"{float(confidence):.2f}" if confidence != 'N/A' and confidence != 0 else str(confidence)
                except (ValueError, TypeError):
                    score_str = str(evaluation_score)
                    conf_str = str(evaluation_result.get('confidence', 'N/A'))

                logger.info(f"⚖️ Evaluation completed: score={score_str}/10, method={evaluation_result.get('parsing_method', 'N/A')}, confidence={conf_str}")
                
                # Store iteration results with LLM evaluation
                logger.info(f"💾 Storing iteration {iteration} results to database")
                iteration_id = store_optimization_iteration(
                    run_id=run_id,
                    iteration_number=iteration,
                    prompt_variant=json.dumps(improved_prompt),
                    retrieval_results=test_results.get('retrieved_images', []),
                    response_text=test_results.get('response_text', ''),
                    evaluation_score=evaluation_score,
                    evaluation_notes=test_results.get('evaluation_notes', ''),
                    llm_evaluation=evaluation_result,
                    evaluator_model=session_data.get('generation_model', 'unknown'),
                    optimized_query=optimized_query
                )
                logger.info(f"💾 Iteration {iteration} stored with ID: {iteration_id}")
                
                # Update best prompt if this iteration is better
                if evaluation_score > best_score:
                    logger.info(f"🏆 NEW BEST SCORE: {evaluation_score:.2f} (previous: {best_score:.2f}) in iteration {iteration}")
                    best_score = evaluation_score
                    best_prompt = improved_prompt
                else:
                    logger.info(f"📊 Score {evaluation_score:.2f} (best remains: {best_score:.2f})")

                # Add to optimization log with optimized query for iterative learning
                optimization_log.append({
                    'iteration': iteration,
                    'prompt': improved_prompt,
                    'score': evaluation_score,
                    'results': test_results,
                    'optimized_query': optimized_query,  # Add for easy access in query optimization
                    'test_results': test_results  # Ensure test_results is accessible
                })

                logger.info(f"🔄 ITERATION {iteration}/{iteration_count} COMPLETE")
                
                # Small delay between iterations
                await asyncio.sleep(1)
            
            # Mark as completed
            logger.info(f"🏁 OPTIMIZATION COMPLETE: run_id={run_id}, best_score={best_score:.2f}/10")
            final_template_name = best_prompt.get('template_name', 'AI-Generated Template') if best_prompt else 'AI-Generated Template'
            update_optimization_run_status(
                run_id, 'completed', iteration_count,
                json.dumps(best_prompt), optimization_log,
                final_template_name=final_template_name
            )

            # Save the best template as an optimized template
            saved_template_id = None
            if best_prompt and best_score > 0:
                # Find the optimized query that corresponds to the best template
                best_optimized_query = None
                for log_entry in optimization_log:
                    if log_entry.get('score', 0) == best_score:
                        best_optimized_query = log_entry.get('optimized_query')
                        break

                # If no exact match, use the query from the best scoring iteration
                if not best_optimized_query and optimization_log:
                    best_log_entry = max(optimization_log, key=lambda x: x.get('score', 0))
                    best_optimized_query = best_log_entry.get('optimized_query')

                logger.info(f"💾 Saving optimized template with query: '{best_optimized_query}'")
                saved_template_id = await self._save_optimized_template(best_prompt, feedback, original_context, best_score, best_optimized_query)
                if saved_template_id:
                    best_prompt['template_id'] = saved_template_id

            logger.info(f"✅ OPTIMIZATION SUCCESS: {iteration_count} iterations completed, best template saved")

            return {
                'success': True,
                'best_template': best_prompt,
                'best_score': best_score,
                'optimization_log': optimization_log
            }
            
        except Exception as e:
            logger.error(f"❌ OPTIMIZATION FAILED: run_id={run_id}, error={str(e)}", exc_info=True)
            update_optimization_run_status(run_id, 'failed')
            raise

    async def _save_optimized_template(self, best_template: Dict[str, Any],
                                     feedback: Dict[str, Any],
                                     original_context: Dict[str, Any],
                                     best_score: float,
                                     optimized_query: str = None) -> Optional[str]:
        """Save the optimized template to the user's template collection."""
        try:
            from src.models.prompt_templates import save_user_template

            # Get user info
            session_data = original_context.get('session_data', {})
            user_id = session_data.get('user_id', 'admin')

            # Create the template structure for saving
            optimized_template = {
                'id': best_template.get('template_id', f'optimized-{int(time.time())}'),
                'name': best_template.get('template_name', 'Optimized Template'),
                'description': f"{best_template.get('template_description', 'AI-optimized template')} (Score: {best_score:.2f}/10)",
                'is_default': False,
                'template_type': 'optimized',
                'system_prompt': best_template.get('system_prompt', ''),
                'query_prefix': best_template.get('query_prefix', ''),
                'query_suffix': best_template.get('query_suffix', ''),
                'optimized_query': optimized_query,  # Store the optimized query
                'optimization_run_id': self.run_id,  # Store the optimization run ID for accessing results later
                'optimization_info': {
                    'base_template_id': best_template.get('base_template_id', 'unknown'),
                    'optimization_score': best_score,
                    'optimization_date': datetime.now().isoformat(),
                    'source_query': feedback.get('query', ''),
                    'expected_response': feedback.get('expected_response', ''),
                    'reasoning': best_template.get('reasoning', ''),
                    'gap_analysis': best_template.get('gap_analysis', ''),
                    'extraction_strategy': best_template.get('extraction_strategy', ''),
                    'optimized_query': optimized_query,  # Also store in optimization_info for reference
                    'optimization_run_id': self.run_id  # Also store in optimization_info for easy access
                }
            }

            # Save the template
            success = save_user_template(user_id, optimized_template)

            if success:
                logger.info(f"💾 OPTIMIZED TEMPLATE SAVED: {optimized_template['name']} (ID: {optimized_template['id']})")
                logger.info(f"📋 Template available for future use with score: {best_score:.2f}/10")
                return optimized_template['id']
            else:
                logger.error(f"❌ Failed to save optimized template")
                return None

        except Exception as e:
            logger.error(f"❌ Error saving optimized template: {e}", exc_info=True)
            return None
    
    def _extract_original_context(self, feedback: Dict[str, Any],
                                session_data: Dict[str, Any],
                                session_id: str,
                                user_id: str) -> Dict[str, Any]:
        """Extract the original context from feedback and session data."""
        # Use current session's generation model, not the one from feedback
        current_generation_model = session_data.get('generation_model', 'ollama-gemma3n-vision-fp16')

        logger.info(f"🎯 Using current session's generation model: {current_generation_model} (user: {user_id}, session: {session_id})")

        return {
            'query': feedback['query'],
            'original_prompt': feedback.get('original_prompt', ''),
            'retrieval_model': feedback.get('retrieval_model', ''),
            'generation_model': current_generation_model,  # Use current session model
            'relevant_images': feedback['relevant_images'],
            'response_feedback': feedback['response_feedback'],
            'session_data': {  # Add session_data for easy access in optimization methods
                'generation_model': current_generation_model,
                'user_id': user_id,  # Use passed user_id
                'session_id': session_id,  # Use passed session_id
                'retrieval_count': session_data.get('retrieval_count', 3),
                'use_ocr': session_data.get('use_ocr', False),
                'use_score_slope': session_data.get('use_score_slope', True)
            },
            'session_settings': {
                'retrieval_count': session_data.get('retrieval_count', 3),
                'use_ocr': session_data.get('use_ocr', False),
                'use_score_slope': session_data.get('use_score_slope', True)
            }
        }
    
    async def _generate_optimized_query(self, original_context: Dict[str, Any],
                                       feedback: Dict[str, Any],
                                       optimization_log: List[Dict[str, Any]]) -> str:
        """Generate an optimized query for better document retrieval."""
        try:
            iteration_num = len(optimization_log) + 1
            logger.info(f"🔍 Generating optimized query for iteration {iteration_num}")

            original_query = original_context.get('query', '')
            expected_response = feedback.get('expected_response', '')

            # Create enhanced query optimization prompt with iterative learning
            query_optimization_prompt = f"""You are an expert at optimizing search queries for document retrieval systems with iterative learning capabilities.

CRITICAL TASK: Analyze previous query attempts and their retrieval performance to generate a progressively better query that will retrieve documents containing the information needed to produce the expected response.

ORIGINAL QUERY: "{original_query}"

EXPECTED RESPONSE: "{expected_response}"

{self._format_previous_query_attempts(optimization_log)}

ITERATIVE QUERY OPTIMIZATION STRATEGY:

1. LEARN FROM PREVIOUS ATTEMPTS:
   - Which queries achieved higher scores and why?
   - What retrieval patterns led to better results?
   - Which specific terms or concepts in successful queries should be preserved?
   - What weaknesses in previous queries should be avoided?

2. ANALYZE RETRIEVAL GAPS:
   - Did previous queries retrieve the right types of documents?
   - What key information from the expected response was missing in retrieved documents?
   - What terminology from the expected response should be emphasized in the query?

3. PROGRESSIVE REFINEMENT:
   - Build upon successful elements from the best-performing previous query
   - Address specific weaknesses identified in the evaluation feedback
   - Incorporate improvement suggestions from previous iterations
   - Avoid query patterns that led to declining performance

4. QUERY OPTIMIZATION PRINCIPLES:
   - Use specific terminology that would appear in documents containing the expected information
   - Include key concepts, entities, and data points mentioned in the expected response
   - Consider synonyms and alternative phrasings that might appear in relevant documents
   - Balance specificity with breadth to avoid overly narrow results
   - Focus on terms that would help distinguish relevant from irrelevant documents

MANDATORY REQUIREMENTS:
- Your new query MUST demonstrate clear learning from previous attempts
- Address at least one specific weakness identified in previous query evaluations
- Incorporate successful elements from the best-performing previous query (if any)
- Avoid repeating query patterns that led to poor retrieval performance

Respond with ONLY the optimized query text, nothing else."""

            # Use AI to generate optimized query
            if self.app:
                with self.app.app_context():
                    from src.models.responder import generate_response

                    session_data = original_context.get('session_data', {})
                    generation_model = session_data.get('generation_model', 'ollama-gemma3n-vision-fp16')
                    user_id = session_data.get('user_id', 'admin')
                    session_id = session_data.get('session_id', 'optimization')

                    ai_response, _ = generate_response(
                        images=[],
                        query=query_optimization_prompt,
                        session_id=f"{session_id}_query_opt",
                        model_choice=generation_model,
                        user_id=user_id,
                        chat_history=[]
                    )

                    # Clean up the response to get just the query
                    optimized_query = ai_response.strip()

                    # Basic validation - if response is too long or seems like explanation, use original
                    if len(optimized_query) > 200 or '\n' in optimized_query:
                        logger.warning("AI query optimization produced explanation instead of query, using original")
                        optimized_query = original_query

                    logger.info(f"🔍 Generated optimized query: '{optimized_query}'")
                    return optimized_query
            else:
                logger.warning("⚠️ No app context for query optimization, using original query")
                return original_query

        except Exception as e:
            logger.error(f"❌ Query optimization failed: {e}")
            return original_context.get('query', '')

    def _format_previous_query_attempts(self, optimization_log: List[Dict[str, Any]]) -> str:
        """Format previous query optimization attempts with detailed retrieval analysis for iterative learning."""
        if not optimization_log:
            return "None - this is the first optimization attempt."

        attempts = []
        attempts.append("DETAILED PREVIOUS QUERY ATTEMPTS & RETRIEVAL ANALYSIS:")
        attempts.append("=" * 60)

        for i, log_entry in enumerate(optimization_log, 1):  # Show all attempts for comprehensive learning
            test_results = log_entry.get('test_results', {})
            evaluation_data = test_results.get('evaluation_data', {})

            # Extract query and performance data
            optimized_query = log_entry.get('optimized_query', 'N/A')
            score = log_entry.get('score', 0)
            retrieved_docs = test_results.get('retrieved_images', [])

            # Extract evaluation feedback
            dimension_scores = evaluation_data.get('dimension_scores', {})
            strengths = evaluation_data.get('strengths', [])
            weaknesses = evaluation_data.get('weaknesses', [])
            improvement_suggestions = evaluation_data.get('improvement_suggestions', [])

            attempts.append(f"\nQUERY ATTEMPT {i}:")
            attempts.append(f"Query Used: '{optimized_query}'")
            attempts.append(f"Overall Score: {score:.1f}/10")

            # Add dimension scores for detailed analysis
            if dimension_scores:
                attempts.append("Dimension Scores:")
                for dim, dim_score in dimension_scores.items():
                    attempts.append(f"  - {dim.replace('_', ' ').title()}: {dim_score:.1f}/10")

            # Add retrieval analysis
            attempts.append(f"Documents Retrieved: {len(retrieved_docs)} documents")
            if retrieved_docs:
                # Show first few retrieved documents for analysis
                doc_names = [doc.split('/')[-1] for doc in retrieved_docs[:3]]
                attempts.append(f"  Top Retrieved: {', '.join(doc_names)}")

            # Add evaluation feedback for query improvement
            if strengths:
                attempts.append("✅ What Worked:")
                for strength in strengths[:2]:
                    attempts.append(f"  • {strength}")

            if weaknesses:
                attempts.append("❌ What Didn't Work:")
                for weakness in weaknesses[:2]:
                    attempts.append(f"  • {weakness}")

            if improvement_suggestions:
                attempts.append("💡 Specific Improvement Suggestions:")
                for suggestion in improvement_suggestions[:2]:
                    attempts.append(f"  • {suggestion}")

            # Add performance trend analysis
            if i > 1:
                prev_score = optimization_log[i-2].get('score', 0)
                trend = "↗️ IMPROVING" if score > prev_score else "↘️ DECLINING" if score < prev_score else "➡️ STABLE"
                attempts.append(f"Performance Trend: {trend} (Previous: {prev_score:.1f} → Current: {score:.1f})")

            attempts.append("-" * 40)

        # Add query learning summary
        if len(optimization_log) > 1:
            attempts.append(self._generate_query_learning_summary(optimization_log))

        return '\n'.join(attempts)

    def _generate_query_learning_summary(self, optimization_log: List[Dict[str, Any]]) -> str:
        """Generate a learning summary specifically for query optimization."""
        try:
            summary = ["\nQUERY OPTIMIZATION LEARNING SUMMARY:"]
            summary.append("=" * 40)

            # Analyze query performance trends
            queries = []
            scores = []
            for log_entry in optimization_log:
                query = log_entry.get('optimized_query', '')
                score = log_entry.get('score', 0)
                if query and query != 'N/A':
                    queries.append(query)
                    scores.append(score)

            if len(scores) > 1:
                best_idx = scores.index(max(scores))
                worst_idx = scores.index(min(scores))

                summary.append(f"\n🎯 QUERY PERFORMANCE ANALYSIS:")
                summary.append(f"  Best Query (Score: {scores[best_idx]:.1f}): '{queries[best_idx]}'")
                summary.append(f"  Worst Query (Score: {scores[worst_idx]:.1f}): '{queries[worst_idx]}'")

                # Analyze what made the best query successful
                if best_idx < len(optimization_log):
                    best_eval = optimization_log[best_idx].get('test_results', {}).get('evaluation_data', {})
                    best_strengths = best_eval.get('strengths', [])
                    if best_strengths:
                        summary.append(f"\n✅ SUCCESS PATTERNS FROM BEST QUERY:")
                        for strength in best_strengths[:2]:
                            summary.append(f"  • {strength}")

                # Analyze what made queries fail
                if worst_idx < len(optimization_log):
                    worst_eval = optimization_log[worst_idx].get('test_results', {}).get('evaluation_data', {})
                    worst_weaknesses = worst_eval.get('weaknesses', [])
                    if worst_weaknesses:
                        summary.append(f"\n❌ FAILURE PATTERNS TO AVOID:")
                        for weakness in worst_weaknesses[:2]:
                            summary.append(f"  • {weakness}")

            # Strategic recommendations for next query
            summary.append(f"\n🎯 STRATEGIC QUERY RECOMMENDATIONS:")
            if len(optimization_log) >= 2:
                latest_eval = optimization_log[-1].get('test_results', {}).get('evaluation_data', {})
                latest_suggestions = latest_eval.get('improvement_suggestions', [])

                if latest_suggestions:
                    summary.append(f"  Based on latest evaluation:")
                    for suggestion in latest_suggestions[:2]:
                        if 'query' in suggestion.lower() or 'search' in suggestion.lower() or 'retriev' in suggestion.lower():
                            summary.append(f"  • {suggestion}")

                # Add retrieval-specific recommendations
                summary.append(f"  General query improvement strategies:")
                summary.append(f"  • Use more specific terminology from the expected response")
                summary.append(f"  • Include key concepts that would appear in relevant documents")
                summary.append(f"  • Avoid overly broad or generic terms")
                summary.append(f"  • Consider synonyms and alternative phrasings")

            summary.append("=" * 40)
            return '\n'.join(summary)

        except Exception as e:
            logger.error(f"Error generating query learning summary: {e}")
            return "\nQUERY LEARNING SUMMARY: Error generating summary"

    async def _generate_improved_prompt(self, original_context: Dict[str, Any],
                                      feedback: Dict[str, Any],
                                      optimization_log: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate an improved TEMPLATE using AI-based optimization."""
        try:
            iteration_num = len(optimization_log) + 1
            logger.info(f"🧠 Generating AI-improved TEMPLATE for iteration {iteration_num}")

            # Get the current template being used
            current_template = await self._get_current_template(original_context, feedback)
            logger.info(f"📋 Current template: {current_template.get('name', 'Unknown')}")

            # Use AI to generate improved template
            if self.app:
                with self.app.app_context():
                    improved_template = await self._ai_generate_improved_template(
                        current_template, original_context, feedback, optimization_log
                    )
            else:
                # Fallback to rule-based if no app context
                logger.warning("⚠️ No app context, using rule-based template improvement")
                improved_template = self._rule_based_template_improvement(
                    current_template, original_context, feedback, optimization_log
                )

            logger.info(f"🧠 Generated template with reasoning: {improved_template.get('reasoning', 'N/A')[:100]}...")
            return improved_template

        except Exception as e:
            logger.error(f"❌ Error generating improved template: {e}")
            # Return a fallback template
            return {
                'system_prompt': 'You are a helpful document analysis assistant.',
                'query_prefix': 'Based on the provided documents, ',
                'query_suffix': ' Please provide a detailed answer.',
                'reasoning': f'Fallback template due to generation error: {str(e)}',
                'confidence_score': 0.5,
                'template_name': 'Fallback Template',
                'template_id': 'fallback-template'
            }

    async def _get_current_template(self, original_context: Dict[str, Any],
                                  feedback: Dict[str, Any]) -> Dict[str, Any]:
        """Get the current template being used for this optimization."""
        try:
            from src.models.prompt_templates import load_user_templates

            # Get session data to find user and template info
            session_data = original_context.get('session_data', {})
            user_id = session_data.get('user_id', 'admin')

            # Load user templates
            templates = load_user_templates(user_id)

            # Try to find the template used for the original query
            # For now, use the default template or first available
            current_template = None
            for template in templates:
                if template.get('is_default', False):
                    current_template = template
                    break

            if not current_template and templates:
                current_template = templates[0]  # Use first template as fallback

            if not current_template:
                # Create a basic template structure
                current_template = {
                    'id': 'basic-template',
                    'name': 'Basic Template',
                    'system_prompt': 'You are a document analysis assistant.',
                    'query_prefix': 'Based on the provided documents, ',
                    'query_suffix': ' Please provide a detailed answer.',
                    'description': 'Basic template for optimization'
                }

            logger.info(f"📋 Using template: {current_template['name']} (ID: {current_template.get('id', 'unknown')})")
            return current_template

        except Exception as e:
            logger.error(f"❌ Error loading current template: {e}")
            # Return basic template
            return {
                'id': 'error-fallback',
                'name': 'Error Fallback Template',
                'system_prompt': 'You are a document analysis assistant.',
                'query_prefix': 'Based on the provided documents, ',
                'query_suffix': ' Please provide a detailed answer.',
                'description': 'Fallback template due to loading error'
            }

    async def _ai_generate_improved_template(self, current_template: Dict[str, Any],
                                           original_context: Dict[str, Any],
                                           feedback: Dict[str, Any],
                                           optimization_log: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Use AI to generate an improved TEMPLATE based on feedback and previous attempts."""
        from src.models.responder import generate_response

        # Apply progressive learning to enhance the current template before optimization
        enhanced_template = self._apply_progressive_learning(current_template, optimization_log)

        logger.info(f"🧠 Applied progressive learning from {len(optimization_log)} previous iterations")

        # Prepare the template optimization prompt
        optimization_prompt = self._build_template_optimization_prompt(
            enhanced_template, original_context, feedback, optimization_log
        )

        logger.info(f"🤖 Requesting AI template optimization")

        # Use the same generation model to optimize the template
        session_data = original_context.get('session_data', {})
        generation_model = session_data.get('generation_model', 'ollama-gemma3n-vision-fp16')
        user_id = session_data.get('user_id', 'admin')
        session_id = session_data.get('session_id', 'optimization')

        try:
            # Generate improved template using AI
            ai_response, _ = generate_response(
                images=[],  # No images needed for template optimization
                query=optimization_prompt,
                session_id=f"{session_id}_template_opt",
                model_choice=generation_model,
                user_id=user_id,
                chat_history=[]
            )

            logger.info(f"🤖 AI template optimization response length: {len(ai_response)}")

            # Parse the AI response to extract template components
            return self._parse_ai_template_response(ai_response, current_template, len(optimization_log) + 1)

        except Exception as e:
            logger.error(f"❌ AI template generation failed: {e}")
            # Fallback to rule-based
            return self._rule_based_template_improvement(current_template, original_context, feedback, optimization_log)

    def _extract_key_components_from_expected_response(self, expected_response: str, query: str, generation_model: str = None) -> Dict[str, Any]:
        """Extract key components/facts from the expected response for targeted optimization."""
        logger.info(f"🔍 EXTRACTING COMPONENTS: query='{query[:50]}...', expected_response='{expected_response[:100]}...'")

        try:
            from src.models.responder import generate_response

            extraction_prompt = f"""Analyze the following expected response and extract key components that should be captured in an AI-generated response.

QUERY: {query}

EXPECTED RESPONSE: {expected_response}

Extract the following components as JSON:
1. **key_facts**: List of specific facts, numbers, or claims that must be present
2. **required_elements**: List of required information types (e.g., "dollar amount", "date", "percentage")
3. **context_clues**: List of document context or source indicators mentioned
4. **format_requirements**: Any specific formatting or structure requirements
5. **priority_focus**: The single most important aspect to get right

Return ONLY this JSON structure:
{{
    "key_facts": ["fact1", "fact2", "fact3"],
    "required_elements": ["element1", "element2"],
    "context_clues": ["clue1", "clue2"],
    "format_requirements": ["format1", "format2"],
    "priority_focus": "most important aspect"
}}"""

            # Use the same model as inference for consistency
            model_to_use = generation_model or "ollama-gemma3n-vision-fp16"  # Use passed model or fallback

            logger.info(f"🤖 Calling component extraction with inference model: {model_to_use}")
            response, _ = generate_response(
                images=[],
                query=extraction_prompt,
                session_id="component_extraction",
                model_choice=model_to_use,
                user_id="system",
                chat_history=[]
            )

            logger.info(f"🤖 Component extraction response: '{response[:200]}...'")

            # Parse the response
            import json
            import re

            # Try to extract JSON
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                logger.info(f"📝 Found JSON: '{json_str[:200]}...'")
                components = json.loads(json_str)
                logger.info(f"✅ Successfully extracted {len(components.get('key_facts', []))} key facts from expected response")
                logger.info(f"📊 Components: {components}")
                return components
            else:
                logger.warning("⚠️ No JSON found in response, using fallback")
                return self._fallback_component_extraction(expected_response, query)

        except Exception as e:
            logger.error(f"❌ Component extraction failed: {e}", exc_info=True)
            return self._fallback_component_extraction(expected_response, query)

    def _fallback_component_extraction(self, expected_response: str, query: str) -> Dict[str, Any]:
        """Fallback component extraction using simple heuristics."""
        logger.info(f"🔄 Using fallback component extraction")
        import re

        # Extract numbers and dollar amounts
        numbers = re.findall(r'\$?[\d,]+\.?\d*', expected_response)
        logger.info(f"🔢 Found numbers: {numbers}")

        # Extract key phrases
        sentences = expected_response.split('.')
        key_facts = [s.strip() for s in sentences if len(s.strip()) > 10][:3]
        logger.info(f"📝 Key facts: {key_facts}")

        # Determine focus based on query
        priority_focus = "accurate information extraction"
        if any(word in query.lower() for word in ['amount', 'income', 'cost', 'price']):
            priority_focus = "numerical accuracy"
        elif any(word in query.lower() for word in ['date', 'when', 'time']):
            priority_focus = "temporal accuracy"

        fallback_components = {
            "key_facts": key_facts,
            "required_elements": numbers if numbers else ["specific information"],
            "context_clues": ["document-based information"],
            "format_requirements": ["clear and specific"],
            "priority_focus": priority_focus
        }

        logger.info(f"🔄 Fallback components: {fallback_components}")
        return fallback_components

    def _build_template_optimization_prompt(self, current_template: Dict[str, Any],
                                          original_context: Dict[str, Any],
                                          feedback: Dict[str, Any],
                                          optimization_log: List[Dict[str, Any]]) -> str:
        """Build the prompt for AI-based TEMPLATE optimization with component-based guidance."""

        iteration_num = len(optimization_log) + 1
        query = original_context.get('query', '')
        expected_response = feedback.get('expected_response', '')

        # Extract key components from expected response
        generation_model = original_context.get('generation_model', 'ollama-gemma3n-vision-fp16')
        components = self._extract_key_components_from_expected_response(expected_response, query, generation_model)

        # Current template info
        current_name = current_template.get('name', 'Unknown Template')
        current_system = current_template.get('system_prompt', '')
        current_prefix = current_template.get('query_prefix', '')
        current_suffix = current_template.get('query_suffix', '')

        # Get the original response that failed (if available)
        original_response = ""
        retrieved_documents = []
        if optimization_log:
            # Get the most recent response for comparison
            last_iteration = optimization_log[-1]
            test_results = last_iteration.get('test_results', {})
            original_response = test_results.get('response_text', 'No previous response available')
            retrieved_documents = test_results.get('retrieved_images', [])
        else:
            # For first iteration, we might not have a previous response
            original_response = "No previous response available - this is the first optimization attempt"
            retrieved_documents = []

        # Get user-selected relevant documents for comparison
        user_relevant_docs = feedback.get('relevant_images', [])

        # Determine optimization mode and testing methodology
        optimization_mode = "extraction" if user_relevant_docs else "retrieval"

        if optimization_mode == "extraction":
            # Extraction optimization: User selected relevant pages, focus on improving extraction
            retrieval_analysis = {
                'missing_documents': [],  # User provided the right documents
                'irrelevant_documents': [],
                'retrieval_accuracy': 1.0,  # User-selected docs are assumed correct
                'total_relevant': len(user_relevant_docs),
                'total_retrieved': len(user_relevant_docs),
                'correctly_retrieved': len(user_relevant_docs),
                'testing_note': 'EXTRACTION OPTIMIZATION: Template tested with user-selected relevant documents to improve information extraction capabilities',
                'optimization_focus': 'Improve system prompt and query suffix to better extract and analyze information from the correct documents'
            }
        else:
            # Retrieval optimization: No pages selected, focus on improving retrieval
            retrieval_analysis = {
                'missing_documents': ['Unknown - user did not specify which documents contain the expected information'],
                'irrelevant_documents': ['Current retrieval may not be finding the right documents'],
                'retrieval_accuracy': 0.0,  # Current retrieval is not working
                'total_relevant': 1,  # Assume at least one relevant document exists
                'total_retrieved': 0,  # Current query is not retrieving correctly
                'correctly_retrieved': 0,
                'testing_note': 'RETRIEVAL OPTIMIZATION: Template tested with RAG-retrieved documents using enhanced query to improve retrieval capabilities',
                'optimization_focus': 'Improve query prefix and suffix to retrieve documents that contain the expected information'
            }

        # Build comprehensive previous attempts context with detailed evaluation feedback
        previous_attempts = ""
        if optimization_log:
            previous_attempts = "\n\nDETAILED PREVIOUS OPTIMIZATION ATTEMPTS & LEARNINGS:\n"
            previous_attempts += "=" * 60 + "\n\n"

            # Include all previous iterations for comprehensive learning
            for i, log_entry in enumerate(optimization_log, 1):
                score = log_entry.get('score', 0)
                template_info = log_entry.get('prompt', {})
                test_results = log_entry.get('test_results', {})
                evaluation_data = test_results.get('evaluation_data', {})

                # Extract template components
                system_prompt = template_info.get('system_prompt', 'N/A')
                query_prefix = template_info.get('query_prefix', 'N/A')
                query_suffix = template_info.get('query_suffix', 'N/A')

                # Extract evaluation details
                dimension_scores = evaluation_data.get('dimension_scores', {})
                strengths = evaluation_data.get('strengths', [])
                weaknesses = evaluation_data.get('weaknesses', [])
                improvement_suggestions = evaluation_data.get('improvement_suggestions', [])
                generated_response = test_results.get('response_text', 'No response')

                previous_attempts += f"ITERATION {i} ANALYSIS:\n"
                previous_attempts += f"Overall Score: {score:.1f}/10\n"

                # Add dimension scores for detailed analysis
                if dimension_scores:
                    previous_attempts += f"Dimension Scores:\n"
                    for dim, dim_score in dimension_scores.items():
                        previous_attempts += f"  - {dim.replace('_', ' ').title()}: {dim_score:.1f}/10\n"

                previous_attempts += f"\nTemplate Used:\n"
                previous_attempts += f"  System Prompt: {system_prompt[:200]}{'...' if len(system_prompt) > 200 else ''}\n"
                previous_attempts += f"  Query Prefix: {query_prefix}\n"
                previous_attempts += f"  Query Suffix: {query_suffix}\n"

                previous_attempts += f"\nGenerated Response: {generated_response[:300]}{'...' if len(generated_response) > 300 else ''}\n"

                # Add detailed evaluation feedback
                if strengths:
                    previous_attempts += f"\n✅ WHAT WORKED (Strengths):\n"
                    for strength in strengths[:3]:  # Top 3 strengths
                        previous_attempts += f"  • {strength}\n"

                if weaknesses:
                    previous_attempts += f"\n❌ WHAT DIDN'T WORK (Weaknesses):\n"
                    for weakness in weaknesses[:3]:  # Top 3 weaknesses
                        previous_attempts += f"  • {weakness}\n"

                if improvement_suggestions:
                    previous_attempts += f"\n💡 SPECIFIC IMPROVEMENT SUGGESTIONS:\n"
                    for suggestion in improvement_suggestions[:3]:  # Top 3 suggestions
                        previous_attempts += f"  • {suggestion}\n"

                # Add performance trend analysis
                if i > 1:
                    prev_score = optimization_log[i-2].get('score', 0)
                    trend = "↗️ IMPROVING" if score > prev_score else "↘️ DECLINING" if score < prev_score else "➡️ STABLE"
                    previous_attempts += f"\nPerformance Trend: {trend} (Previous: {prev_score:.1f} → Current: {score:.1f})\n"

                previous_attempts += "\n" + "-" * 50 + "\n\n"

            # Add cumulative learning summary
            if len(optimization_log) > 1:
                previous_attempts += self._generate_cumulative_learning_summary(optimization_log)

        optimization_prompt = f"""You are an expert RAG template optimization agent specializing in prompt engineering, information extraction, and document retrieval optimization. Your task is to analyze both RETRIEVAL and RESPONSE gaps to create improved PROMPT TEMPLATES.

CRITICAL ANALYSIS REQUIRED:

CURRENT TEMPLATE PERFORMANCE:
- Template Name: "{current_name}"
- System Prompt: "{current_system}"
- Query Prefix: "{current_prefix}"
- Query Suffix: "{current_suffix}"

TEMPLATE TESTING METHODOLOGY:
- User-Selected Relevant Documents: {user_relevant_docs}
- Testing Approach: Templates are tested using user-selected relevant documents to focus on EXTRACTION optimization
- Note: {retrieval_analysis.get('testing_note', 'Direct document testing for extraction evaluation')}

EXTRACTION FOCUS:
Since templates are tested with the correct documents, focus on optimizing the EXTRACTION of specific facts from those documents rather than retrieval optimization.

RESPONSE GAP ANALYSIS:
- User Query: "{query}"
- Expected Response: "{expected_response}"
- Original Response: "{original_response}"

KEY COMPONENTS TO CAPTURE (extracted from expected response):
- Key Facts: {components.get('key_facts', [])}
- Required Elements: {components.get('required_elements', [])}
- Context Clues: {components.get('context_clues', [])}
- Format Requirements: {components.get('format_requirements', [])}
- Priority Focus: {components.get('priority_focus', 'accurate extraction')}

STEP 1 - COMPONENT-BASED GAP IDENTIFICATION:
For each key component above, analyze what's missing from the original response:
- Which key facts are missing or incorrect?
- Which required elements are not captured?
- Are context clues being ignored?
- Are format requirements not being followed?
- Is the priority focus being addressed?

STEP 2 - ROOT CAUSE ANALYSIS:
Identify WHY the current template failed to extract the required information from the provided documents:
- Does the system prompt lack domain expertise?
- Are the instructions too generic?
- Is the template missing specific extraction guidance?
- Does it fail to direct attention to the right document sections?

{previous_attempts}

STEP 3 - ITERATIVE LEARNING ANALYSIS:
CRITICAL: Use the detailed previous iteration analysis above to guide your optimization strategy.

LEARNING FROM PREVIOUS ITERATIONS:
1. IDENTIFY SUCCESSFUL PATTERNS:
   - Which template components (system prompt, query prefix/suffix) led to higher scores?
   - What specific strengths were consistently mentioned across iterations?
   - Which approaches showed improvement trends (↗️ IMPROVING)?

2. AVOID FAILED APPROACHES:
   - Which template changes led to declining scores (↘️ DECLINING)?
   - What recurring weaknesses must be addressed?
   - Which specific improvement suggestions from previous evaluations haven't been implemented yet?

3. BUILD ON CUMULATIVE LEARNINGS:
   - Apply the strategic recommendations from the cumulative learning summary
   - Address recurring weaknesses that appear across multiple iterations
   - Preserve and enhance the recurring strengths identified

4. PROGRESSIVE REFINEMENT:
   - Don't repeat the same mistakes from previous iterations
   - Build incrementally on what worked in the best-performing iteration
   - Implement specific improvement suggestions from the latest evaluation

MANDATORY: Your new template MUST demonstrate clear learning from previous attempts by:
- Explicitly addressing at least 2 recurring weaknesses mentioned in previous iterations
- Building upon at least 1 recurring strength from previous iterations
- Implementing at least 1 specific improvement suggestion from the latest evaluation
- Avoiding template patterns that led to declining scores in previous iterations

STEP 4 - PROMPT ENGINEERING STRATEGY:
Apply these proven techniques to improve information extraction from the provided documents:

A) DOMAIN SPECIALIZATION TECHNIQUES:
- Add specific domain expertise to system prompt (e.g., "Schedule K-1 specialist", "tax form expert")
- Include knowledge of document structure and common locations of information
- Use domain-specific terminology and concepts

B) SPECIFICITY TECHNIQUES:
- Use general domain terminology (e.g., "Schedule K-1 form" instead of "document")
- Request exact formats (e.g., "provide the dollar amount as $XX")
- Ask to examine relevant document sections without specifying exact box numbers
- CRITICAL: Never reference specific box numbers or sections unless you can verify they exist in the actual document

C) INFORMATION EXTRACTION TECHNIQUES:
- Use step-by-step instructions for finding information
- Request citations of specific document locations
- Ask for verification of found information

D) OUTPUT FORMATTING:
- Specify exact response format that matches expected response
- Use templates like "According to [document section], the [item] is [value]"
- Request structured responses when appropriate

STEP 5 - COMPONENT-FOCUSED TEMPLATE OPTIMIZATION:
Create an improved template that specifically targets the key components identified above:

COMPONENT OPTIMIZATION STRATEGY:
- System Prompt: Add expertise focused on the priority focus and required elements
- Query Prefix: Set context that primes for the specific key facts and context clues
- Query Suffix: Add targeted instructions to extract the required elements in the expected format

COMPONENT-FOCUSED EXAMPLES:
- If key_facts include "$1,234", suffix should ask for "exact dollar amounts"
- If context_clues mention "Schedule K-1", prefix should reference "tax form analysis"
- If priority_focus is "numerical accuracy", system prompt should emphasize precision
- If required_elements include "dates", suffix should specify date format requirements

RESPONSE FORMAT:
Return a JSON object with these fields:
{{
    "system_prompt": "Improved system prompt with domain expertise and specific extraction instructions",
    "query_prefix": "Improved prefix that helps focus on relevant document sections",
    "query_suffix": "Improved suffix that COMPLEMENTS the user's question with additional instructions, format requirements, or context. DO NOT include a complete question - only add instructions that enhance the user's original query.",
    "template_name": "Descriptive name reflecting the specialized domain/task",
    "template_description": "Brief description of what specific information this template extracts",
    "reasoning": "Detailed explanation of what gaps you identified and how your improvements address them",
    "confidence_score": 0.85,
    "gap_analysis": "Summary of what was missing from original response and why",
    "extraction_strategy": "Specific prompt engineering techniques used to improve information extraction"
}}

IMPORTANT TEMPLATE STRUCTURE:
- query_prefix: Sets context (e.g., "Based on the provided documents,")
- USER'S QUERY: The actual question (e.g., "what's the interest income?")
- query_suffix: Adds instructions/requirements (e.g., ". Look for interest income amounts on the Schedule K-1 form and provide the exact dollar amount.")

The final prompt will be: prefix + user_query + suffix
Example: "Based on the provided documents, what's the interest income? Look for interest income amounts on the Schedule K-1 form and provide the exact dollar amount."

CRITICAL: Do NOT reference specific box numbers, sections, or document elements unless you can verify they exist in the actual document being analyzed.

OPTIMIZATION FOCUS: {retrieval_analysis['optimization_focus']}

TESTING METHODOLOGY: {retrieval_analysis['testing_note']}

{f"EXTRACTION MODE: The template will be tested with user-selected relevant documents. Focus on improving system prompt and query suffix to better extract and analyze the expected information from the correct documents." if optimization_mode == "extraction" else "RETRIEVAL MODE: The template will be tested by performing RAG retrieval with the enhanced query. Focus on improving query prefix and suffix to retrieve documents that contain the expected information. The enhanced query must be specific enough to find the right documents."}"""

        return optimization_prompt

    def _generate_cumulative_learning_summary(self, optimization_log: List[Dict[str, Any]]) -> str:
        """Generate a cumulative learning summary from all previous iterations."""
        try:
            summary = "CUMULATIVE LEARNING SUMMARY:\n"
            summary += "=" * 40 + "\n\n"

            # Analyze score progression
            scores = [log.get('score', 0) for log in optimization_log]
            if len(scores) > 1:
                initial_score = scores[0]
                latest_score = scores[-1]
                best_score = max(scores)
                worst_score = min(scores)

                summary += f"📊 SCORE PROGRESSION:\n"
                summary += f"  Initial Score: {initial_score:.1f}/10\n"
                summary += f"  Latest Score: {latest_score:.1f}/10\n"
                summary += f"  Best Score: {best_score:.1f}/10\n"
                summary += f"  Improvement: {latest_score - initial_score:+.1f} points\n\n"

            # Collect all strengths and weaknesses
            all_strengths = []
            all_weaknesses = []
            all_suggestions = []

            for log_entry in optimization_log:
                test_results = log_entry.get('test_results', {})
                evaluation_data = test_results.get('evaluation_data', {})

                all_strengths.extend(evaluation_data.get('strengths', []))
                all_weaknesses.extend(evaluation_data.get('weaknesses', []))
                all_suggestions.extend(evaluation_data.get('improvement_suggestions', []))

            # Find recurring patterns
            if all_strengths:
                summary += f"🔄 RECURRING STRENGTHS (Keep doing):\n"
                strength_counts = {}
                for strength in all_strengths:
                    # Simple keyword matching for recurring themes
                    for word in strength.lower().split():
                        if len(word) > 4:  # Skip short words
                            strength_counts[word] = strength_counts.get(word, 0) + 1

                # Get top recurring strength themes
                top_strengths = sorted(strength_counts.items(), key=lambda x: x[1], reverse=True)[:3]
                for theme, count in top_strengths:
                    if count > 1:
                        summary += f"  • {theme.title()} (mentioned {count} times)\n"
                summary += "\n"

            if all_weaknesses:
                summary += f"⚠️ RECURRING WEAKNESSES (Must address):\n"
                weakness_counts = {}
                for weakness in all_weaknesses:
                    # Simple keyword matching for recurring themes
                    for word in weakness.lower().split():
                        if len(word) > 4:  # Skip short words
                            weakness_counts[word] = weakness_counts.get(word, 0) + 1

                # Get top recurring weakness themes
                top_weaknesses = sorted(weakness_counts.items(), key=lambda x: x[1], reverse=True)[:3]
                for theme, count in top_weaknesses:
                    if count > 1:
                        summary += f"  • {theme.title()} (mentioned {count} times)\n"
                summary += "\n"

            # Strategic recommendations
            summary += f"🎯 STRATEGIC RECOMMENDATIONS FOR NEXT ITERATION:\n"
            if len(optimization_log) >= 2:
                latest_eval = optimization_log[-1].get('test_results', {}).get('evaluation_data', {})
                latest_suggestions = latest_eval.get('improvement_suggestions', [])

                if latest_suggestions:
                    summary += f"  Based on latest evaluation:\n"
                    for suggestion in latest_suggestions[:2]:
                        summary += f"  • {suggestion}\n"
                else:
                    summary += f"  • Focus on addressing recurring weaknesses identified above\n"
                    summary += f"  • Build upon the recurring strengths\n"

            summary += "\n" + "=" * 40 + "\n\n"
            return summary

        except Exception as e:
            logger.error(f"Error generating cumulative learning summary: {e}")
            return "\nCUMULATIVE LEARNING SUMMARY: Error generating summary\n\n"

    def _extract_best_iteration_learnings(self, optimization_log: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract key learnings from the best-performing iteration."""
        try:
            if not optimization_log:
                return {}

            # Find the best iteration
            best_iteration = max(optimization_log, key=lambda x: x.get('score', 0))
            best_score = best_iteration.get('score', 0)

            # Extract successful template components
            best_template = best_iteration.get('prompt', {})
            best_evaluation = best_iteration.get('test_results', {}).get('evaluation_data', {})

            learnings = {
                'best_score': best_score,
                'successful_template': {
                    'system_prompt': best_template.get('system_prompt', ''),
                    'query_prefix': best_template.get('query_prefix', ''),
                    'query_suffix': best_template.get('query_suffix', ''),
                },
                'key_strengths': best_evaluation.get('strengths', [])[:3],
                'dimension_scores': best_evaluation.get('dimension_scores', {}),
                'successful_patterns': []
            }

            # Identify successful patterns from the best iteration
            if best_evaluation.get('strengths'):
                for strength in best_evaluation.get('strengths', []):
                    if 'specific' in strength.lower() or 'detailed' in strength.lower():
                        learnings['successful_patterns'].append('specificity_works')
                    if 'format' in strength.lower() or 'structure' in strength.lower():
                        learnings['successful_patterns'].append('formatting_works')
                    if 'accurate' in strength.lower() or 'correct' in strength.lower():
                        learnings['successful_patterns'].append('accuracy_focus_works')

            logger.info(f"Extracted learnings from best iteration (score: {best_score:.1f}): {len(learnings['successful_patterns'])} patterns identified")
            return learnings

        except Exception as e:
            logger.error(f"Error extracting best iteration learnings: {e}")
            return {}

    def _apply_progressive_learning(self, current_template: Dict[str, Any],
                                  optimization_log: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Apply progressive learning from previous iterations to enhance the current template."""
        try:
            if not optimization_log:
                return current_template

            # Get learnings from the best iteration
            best_learnings = self._extract_best_iteration_learnings(optimization_log)

            if not best_learnings:
                return current_template

            enhanced_template = current_template.copy()

            # If current iteration is performing worse than the best, incorporate best practices
            latest_score = optimization_log[-1].get('score', 0) if optimization_log else 0
            best_score = best_learnings.get('best_score', 0)

            if latest_score < best_score:
                logger.info(f"Applying progressive learning: latest score ({latest_score:.1f}) < best score ({best_score:.1f})")

                # Incorporate successful template elements
                best_template = best_learnings.get('successful_template', {})
                successful_patterns = best_learnings.get('successful_patterns', [])

                # Apply successful patterns to enhance current template
                if 'specificity_works' in successful_patterns:
                    # Enhance with more specific language
                    if 'document' in enhanced_template.get('system_prompt', '').lower():
                        enhanced_template['system_prompt'] = enhanced_template.get('system_prompt', '').replace(
                            'document', 'specific document type'
                        )

                if 'formatting_works' in successful_patterns:
                    # Ensure formatting instructions are included
                    if 'format' not in enhanced_template.get('query_suffix', '').lower():
                        current_suffix = enhanced_template.get('query_suffix', '')
                        enhanced_template['query_suffix'] = current_suffix + ' Provide your response in a clear, structured format.'

                if 'accuracy_focus_works' in successful_patterns:
                    # Enhance accuracy focus
                    if 'accurate' not in enhanced_template.get('system_prompt', '').lower():
                        current_system = enhanced_template.get('system_prompt', '')
                        enhanced_template['system_prompt'] = current_system + ' Focus on providing accurate, verifiable information.'

                logger.info(f"Applied {len(successful_patterns)} successful patterns from best iteration")

            return enhanced_template

        except Exception as e:
            logger.error(f"Error applying progressive learning: {e}")
            return current_template

    def _analyze_retrieval_gaps(self, retrieved_documents: List[str], user_relevant_docs: List[str]) -> Dict[str, Any]:
        """Analyze gaps between retrieved documents and user-selected relevant documents."""
        try:
            # Normalize paths for comparison
            def normalize_path(path):
                if isinstance(path, str):
                    return path.replace('\\', '/').split('/')[-1].lower()  # Get filename only
                return str(path).lower()

            retrieved_normalized = [normalize_path(doc) for doc in retrieved_documents]
            relevant_normalized = [normalize_path(doc) for doc in user_relevant_docs]

            # Find missing documents (user marked as relevant but not retrieved)
            missing_docs = []
            for i, relevant_doc in enumerate(relevant_normalized):
                if relevant_doc not in retrieved_normalized:
                    missing_docs.append(user_relevant_docs[i])

            # Find irrelevant documents (retrieved but user didn't mark as relevant)
            irrelevant_docs = []
            for i, retrieved_doc in enumerate(retrieved_normalized):
                if retrieved_doc not in relevant_normalized:
                    irrelevant_docs.append(retrieved_documents[i])

            # Calculate retrieval accuracy
            total_relevant = len(user_relevant_docs)
            correctly_retrieved = len([doc for doc in relevant_normalized if doc in retrieved_normalized])
            retrieval_accuracy = (correctly_retrieved / total_relevant) if total_relevant > 0 else 0.0

            return {
                'missing_documents': missing_docs,
                'irrelevant_documents': irrelevant_docs,
                'retrieval_accuracy': retrieval_accuracy,
                'total_relevant': total_relevant,
                'total_retrieved': len(retrieved_documents),
                'correctly_retrieved': correctly_retrieved
            }

        except Exception as e:
            logger.error(f"Error analyzing retrieval gaps: {e}")
            return {
                'missing_documents': [],
                'irrelevant_documents': [],
                'retrieval_accuracy': 0.0,
                'total_relevant': len(user_relevant_docs),
                'total_retrieved': len(retrieved_documents),
                'correctly_retrieved': 0
            }

    def _parse_ai_template_response(self, ai_response: str, current_template: Dict[str, Any], iteration_num: int) -> Dict[str, Any]:
        """Parse the AI response to extract TEMPLATE components."""
        try:
            # Try to extract JSON from the response
            import re
            json_match = re.search(r'\{.*\}', ai_response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                template_data = json.loads(json_str)

                # Validate required fields
                if 'system_prompt' in template_data:
                    logger.info(f"✅ Successfully parsed AI-generated template")

                    # Create optimized template structure
                    optimized_template = {
                        'system_prompt': template_data.get('system_prompt', ''),
                        'query_prefix': template_data.get('query_prefix', ''),
                        'query_suffix': template_data.get('query_suffix', ''),
                        'template_name': template_data.get('template_name', f"Optimized {current_template.get('name', 'Template')} v{iteration_num}"),
                        'template_description': template_data.get('template_description', f'AI-optimized template (iteration {iteration_num})'),
                        'reasoning': template_data.get('reasoning', f'AI-generated template improvement for iteration {iteration_num}'),
                        'confidence_score': float(template_data.get('confidence_score', 0.7)),
                        'template_id': f"optimized-{current_template.get('id', 'template')}-v{iteration_num}",
                        'is_optimized': True,
                        'base_template_id': current_template.get('id', 'unknown'),
                        'gap_analysis': template_data.get('gap_analysis', 'No gap analysis provided'),
                        'extraction_strategy': template_data.get('extraction_strategy', 'No extraction strategy provided')
                    }

                    return optimized_template

            # If JSON parsing fails, try to extract template manually
            logger.warning("⚠️ JSON parsing failed, attempting manual template extraction")
            return self._manual_extract_template(ai_response, current_template, iteration_num)

        except Exception as e:
            logger.error(f"❌ Failed to parse AI template response: {e}")
            return self._manual_extract_template(ai_response, current_template, iteration_num)

    def _parse_ai_prompt_response(self, ai_response: str, iteration_num: int) -> Dict[str, Any]:
        """Parse the AI response to extract prompt components."""
        try:
            # Try to extract JSON from the response
            import re
            json_match = re.search(r'\{.*\}', ai_response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                prompt_data = json.loads(json_str)

                # Validate required fields
                if 'system_prompt' in prompt_data:
                    logger.info(f"✅ Successfully parsed AI-generated prompt")
                    return {
                        'system_prompt': prompt_data.get('system_prompt', ''),
                        'query_prefix': prompt_data.get('query_prefix', ''),
                        'query_suffix': prompt_data.get('query_suffix', ''),
                        'reasoning': prompt_data.get('reasoning', f'AI-generated improvement for iteration {iteration_num}'),
                        'confidence_score': float(prompt_data.get('confidence_score', 0.7))
                    }

            # If JSON parsing fails, try to extract system prompt manually
            logger.warning("⚠️ JSON parsing failed, attempting manual extraction")
            return self._manual_extract_prompt(ai_response, iteration_num)

        except Exception as e:
            logger.error(f"❌ Failed to parse AI prompt response: {e}")
            return self._manual_extract_prompt(ai_response, iteration_num)

    def _manual_extract_prompt(self, ai_response: str, iteration_num: int) -> Dict[str, Any]:
        """Manually extract prompt from AI response when JSON parsing fails."""
        # Look for system prompt in the response
        lines = ai_response.split('\n')
        system_prompt = ""

        for line in lines:
            if 'system_prompt' in line.lower() or 'prompt' in line.lower():
                # Try to extract the prompt content
                if ':' in line:
                    system_prompt = line.split(':', 1)[1].strip().strip('"')
                    break

        if not system_prompt:
            # Use the entire response as a system prompt if we can't parse it
            system_prompt = f"You are a document analysis assistant specialized in financial documents. {ai_response[:200]}..."

        return {
            'system_prompt': system_prompt,
            'query_prefix': '',
            'query_suffix': '',
            'reasoning': f'Manual extraction from AI response (iteration {iteration_num})',
            'confidence_score': 0.6
        }

    def _manual_extract_template(self, ai_response: str, current_template: Dict[str, Any], iteration_num: int) -> Dict[str, Any]:
        """Manually extract template from AI response when JSON parsing fails."""
        # Look for system prompt in the response
        lines = ai_response.split('\n')
        system_prompt = ""

        for line in lines:
            if 'system_prompt' in line.lower() or 'system prompt' in line.lower():
                # Try to extract the prompt content
                if ':' in line:
                    system_prompt = line.split(':', 1)[1].strip().strip('"')
                    break

        if not system_prompt:
            # Use the entire response as a system prompt if we can't parse it
            system_prompt = f"You are a specialized document analysis assistant. {ai_response[:200]}..."

        return {
            'system_prompt': system_prompt,
            'query_prefix': current_template.get('query_prefix', 'Based on the provided documents, '),
            'query_suffix': current_template.get('query_suffix', ' Please provide a detailed answer.'),
            'template_name': f"Manual Extract {current_template.get('name', 'Template')} v{iteration_num}",
            'template_description': f'Manually extracted template (iteration {iteration_num})',
            'reasoning': f'Manual extraction from AI response (iteration {iteration_num})',
            'confidence_score': 0.6,
            'template_id': f"manual-{current_template.get('id', 'template')}-v{iteration_num}",
            'is_optimized': True,
            'base_template_id': current_template.get('id', 'unknown')
        }

    def _rule_based_template_improvement(self, current_template: Dict[str, Any],
                                       original_context: Dict[str, Any],
                                       feedback: Dict[str, Any],
                                       optimization_log: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Fallback rule-based template improvement when AI generation fails."""
        iteration_num = len(optimization_log) + 1

        # Get current template components
        base_system = current_template.get('system_prompt', 'You are a document analysis assistant.')
        base_prefix = current_template.get('query_prefix', 'Based on the provided documents, ')
        base_suffix = current_template.get('query_suffix', ' Please provide a detailed answer.')

        # Apply rule-based improvements based on expected response
        expected = feedback.get('expected_response', '').lower()

        # Domain-specific improvements
        if 'schedule k-1' in expected or 'k-1' in expected:
            system_prompt = f"{base_system}\n\nYou specialize in analyzing Schedule K-1 tax forms. When analyzing K-1 documents, focus on finding specific line items and numerical values. Always reference the specific form section when providing financial information."
            query_prefix = "Based on the Schedule K-1 tax form, "
            query_suffix = " Please provide the specific amount and reference the form section."
        elif 'financial' in expected or 'income' in expected or '$' in expected:
            system_prompt = f"{base_system}\n\nYou specialize in financial document analysis. Focus on finding specific numerical values and clearly state the source document and section."
            query_prefix = "Based on the financial documents, "
            query_suffix = " Please provide specific amounts with source references."
        else:
            # Generic improvement
            system_prompt = f"{base_system}\n\nBe more specific and detailed in your analysis. Always reference the source document when providing information."
            query_prefix = base_prefix
            query_suffix = base_suffix

        return {
            'system_prompt': system_prompt,
            'query_prefix': query_prefix,
            'query_suffix': query_suffix,
            'template_name': f"Rule-based {current_template.get('name', 'Template')} v{iteration_num}",
            'template_description': f'Rule-based template improvement (iteration {iteration_num})',
            'reasoning': f'Rule-based improvement focusing on domain-specific requirements (iteration {iteration_num})',
            'confidence_score': min(0.5 + (iteration_num * 0.1), 0.8),
            'template_id': f"rule-{current_template.get('id', 'template')}-v{iteration_num}",
            'is_optimized': True,
            'base_template_id': current_template.get('id', 'unknown')
        }
    
    def _rule_based_prompt_improvement(self, original_context: Dict[str, Any],
                                      feedback: Dict[str, Any],
                                      optimization_log: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Fallback rule-based prompt improvement when AI generation fails."""
        iteration_num = len(optimization_log) + 1

        improved_prompt = {
            'system_prompt': self._improve_system_prompt_rules(original_context, feedback, optimization_log),
            'query_prefix': self._improve_query_prefix(original_context, feedback),
            'query_suffix': self._improve_query_suffix(original_context, feedback),
            'reasoning': f'Rule-based improvement for iteration {iteration_num}',
            'confidence_score': min(0.5 + (iteration_num * 0.1), 0.8)
        }

        return improved_prompt

    def _improve_system_prompt_rules(self, context: Dict[str, Any],
                                   feedback: Dict[str, Any],
                                   optimization_log: List[Dict[str, Any]] = None) -> str:
        """Improve the system prompt based on feedback and previous attempts."""
        base_prompt = """You are a document analysis assistant. Analyze the provided documents and answer questions based only on their content.

Key guidelines:
- Only use information explicitly present in the documents
- Never hallucinate or make up information
- If information isn't in the documents, clearly state this fact
- Be concise and clear in your responses"""

        # Analyze feedback more comprehensively
        feedback_text = feedback.get('response_feedback', '').lower()

        # Track what improvements we've already tried
        tried_improvements = set()
        if optimization_log:
            for log_entry in optimization_log:
                if 'improvements' in log_entry:
                    tried_improvements.update(log_entry['improvements'])

        current_improvements = []

        if feedback_text:
            # Check for requests for more context/source information
            if any(keyword in feedback_text for keyword in ['context', 'source', 'document', 'where', 'from']):
                if 'context_reference' not in tried_improvements:
                    base_prompt += "\n- Always provide context about which part of the document contains the information"
                    base_prompt += "\n- Reference specific sections, pages, or document elements when possible"
                    current_improvements.append('context_reference')
                elif 'enhanced_context' not in tried_improvements:
                    base_prompt += "\n- Always provide detailed context about which part of the document contains the information"
                    base_prompt += "\n- Reference specific sections, pages, or document elements with precise locations"
                    base_prompt += "\n- Include relevant background information and surrounding context from the source document"
                    current_improvements.append('enhanced_context')

            # Check for requests for more detail
            if any(keyword in feedback_text for keyword in ['more detail', 'detailed', 'elaborate', 'explain', 'expand']):
                if 'basic_detail' not in tried_improvements:
                    base_prompt += "\n- Provide detailed explanations when relevant information is found"
                    base_prompt += "\n- Include supporting details and examples from the documents"
                    current_improvements.append('basic_detail')
                elif 'comprehensive_detail' not in tried_improvements:
                    base_prompt += "\n- Provide comprehensive and thorough explanations with multiple supporting details"
                    base_prompt += "\n- Include specific examples, data points, and contextual information from the documents"
                    base_prompt += "\n- Explain the significance and implications of the information found"
                    current_improvements.append('comprehensive_detail')

            # Check for requests for better structure
            if any(keyword in feedback_text for keyword in ['structure', 'organize', 'format', 'list', 'bullet']):
                if 'basic_structure' not in tried_improvements:
                    base_prompt += "\n- Organize information using bullet points or numbered lists"
                    base_prompt += "\n- Use clear headings and sections when appropriate"
                    current_improvements.append('basic_structure')
                elif 'advanced_structure' not in tried_improvements:
                    base_prompt += "\n- Organize information using clear hierarchical structure with headings, subheadings, and bullet points"
                    base_prompt += "\n- Group related information together and use consistent formatting"
                    base_prompt += "\n- Provide clear transitions between different topics or sections"
                    current_improvements.append('advanced_structure')

            # Check for requests for examples
            if any(keyword in feedback_text for keyword in ['example', 'instance', 'case', 'sample']):
                if 'basic_examples' not in tried_improvements:
                    base_prompt += "\n- Provide specific examples from the documents when possible"
                    base_prompt += "\n- Use concrete instances to illustrate points"
                    current_improvements.append('basic_examples')
                elif 'detailed_examples' not in tried_improvements:
                    base_prompt += "\n- Provide multiple specific examples from the documents with detailed explanations"
                    base_prompt += "\n- Use concrete instances with context to illustrate and support all major points"
                    current_improvements.append('detailed_examples')

        # Store improvements for next iteration
        if hasattr(self, '_current_improvements'):
            self._current_improvements = current_improvements

        return base_prompt
    
    def _improve_query_prefix(self, context: Dict[str, Any], 
                            feedback: Dict[str, Any]) -> str:
        """Improve the query prefix based on feedback."""
        return "Based on the provided document images, "
    
    def _improve_query_suffix(self, context: Dict[str, Any],
                            feedback: Dict[str, Any]) -> str:
        """Improve the query suffix based on feedback."""
        suffix = ""
        feedback_text = feedback.get('response_feedback', '').lower()

        if feedback_text:
            # Add specific instructions based on feedback
            if any(keyword in feedback_text for keyword in ['context', 'source', 'document', 'where', 'from']):
                suffix += "\n\nPlease include information about the source and context of your answer, referencing specific parts of the document."

            if any(keyword in feedback_text for keyword in ['example', 'instance', 'case', 'sample']):
                suffix += "\n\nProvide specific examples from the documents when possible."

            if any(keyword in feedback_text for keyword in ['summary', 'summarize', 'overview']):
                suffix += "\n\nSummarize the key points clearly."

            if any(keyword in feedback_text for keyword in ['detail', 'detailed', 'elaborate', 'explain']):
                suffix += "\n\nProvide a detailed explanation with supporting information from the documents."

        return suffix
    
    async def _test_prompt(self, improved_template: Dict[str, Any],
                         feedback: Dict[str, Any],
                         session_data: Dict[str, Any]) -> Dict[str, Any]:
        """Test the improved TEMPLATE by generating a real response using the RAG system."""
        try:
            original_query = session_data.get('original_query', '')
            session_id = session_data.get('session_id', '')
            user_id = session_data.get('user_id', '')
            generation_model = session_data.get('generation_model', 'ollama-gemma3n-vision-fp16')

            logger.info(f"Testing improved prompt with model: {generation_model}")

            # Get relevant images from feedback for testing
            relevant_images = feedback.get('relevant_images', [])
            logger.info(f"📁 Relevant images from feedback: {relevant_images}")

            # Determine optimization mode based on page selection
            if relevant_images:
                optimization_mode = "extraction"
                logger.info("🎯 EXTRACTION OPTIMIZATION MODE: User selected relevant pages - optimizing template extraction capabilities")
            else:
                optimization_mode = "retrieval"
                logger.info("🔍 RETRIEVAL OPTIMIZATION MODE: No pages selected - optimizing query for better retrieval")

            # Convert relative paths to full paths that actually exist
            full_path_images = []
            for img_path in relevant_images:
                if isinstance(img_path, str):
                    # Try different path combinations to find the actual file
                    possible_paths = [
                        img_path,  # Original path
                        os.path.join('static', img_path),  # Add static prefix
                        os.path.abspath(os.path.join('static', img_path)),  # Absolute static path
                    ]

                    for full_path in possible_paths:
                        if os.path.exists(full_path):
                            full_path_images.append(full_path)
                            logger.info(f"✅ Found valid image path: {full_path}")
                            break
                    else:
                        logger.warning(f"⚠️ Image not found at any path: {img_path}")

            logger.info(f"📁 Valid image paths for testing: {full_path_images}")

            # Always perform retrieval with optimized query (enhanced optimization)
            logger.info("🔍 Performing RAG retrieval with optimized query and enhanced template")
            try:
                # Use optimized query if available, otherwise fall back to original
                optimized_query = session_data.get('optimized_query', original_query)

                # Create enhanced query for retrieval testing by combining optimized query with template
                enhanced_query_for_retrieval = optimized_query
                if improved_template.get('query_prefix'):
                    enhanced_query_for_retrieval = f"{improved_template['query_prefix']} {enhanced_query_for_retrieval}"
                if improved_template.get('query_suffix'):
                    enhanced_query_for_retrieval = f"{enhanced_query_for_retrieval} {improved_template['query_suffix']}"

                logger.info(f"🔍 Testing retrieval with optimized query: '{optimized_query}'")
                logger.info(f"🔍 Enhanced query for retrieval: '{enhanced_query_for_retrieval[:100]}...'")

                # Perform RAG retrieval using the enhanced query
                from src.models.rag_retriever import rag_retriever
                from src.models.model_loader import load_rag_model

                # Load RAG model for retrieval testing
                # Get the indexer model from session data
                indexer_model = session_data.get('indexer_model', 'athrael-soju/colqwen3.5-4.5B-v3')
                logger.info(f"🔍 Loading RAG model for retrieval testing: {indexer_model}")
                rag_model = load_rag_model(indexer_model)
                if rag_model:
                    logger.info("✅ RAG model loaded successfully for optimization testing")
                    # Get selected documents from session for filtering
                    selected_docs = session_data.get('selected_docs', [])

                    # Use the original session ID for retrieval (not the temporary optimization session)
                    original_session_id = session_data.get('session_id', session_id)
                    logger.info(f"🔍 Using original session ID for retrieval: {original_session_id}")

                    # Perform retrieval with enhanced query
                    try:
                        rag_retriever.set_embedding_adapter(rag_model)
                        retrieval_results = rag_retriever.retrieve_documents(
                            enhanced_query_for_retrieval,
                            original_session_id,  # Use original session ID
                            k=5,  # Get top 5 pages for testing (increased from 3)
                            selected_filenames=selected_docs
                        )
                        logger.info(f"✅ Retrieval completed with enhanced query")
                    except Exception as retrieval_error:
                        logger.error(f"❌ Enhanced query retrieval failed: {retrieval_error}")
                        # Fallback to original query
                        logger.info("🔄 Falling back to original query for retrieval")
                        try:
                            rag_retriever.set_embedding_adapter(rag_model)
                            retrieval_results = rag_retriever.retrieve_documents(
                                optimized_query,  # Use just the optimized query without template additions
                                original_session_id,
                                k=5,
                                selected_filenames=selected_docs
                            )
                            logger.info(f"✅ Fallback retrieval completed with original query")
                        except Exception as fallback_error:
                            logger.error(f"❌ Fallback retrieval also failed: {fallback_error}")
                            retrieval_results = []

                    # Convert retrieval results to image paths for testing
                    logger.info(f"🔍 Retrieval results: {len(retrieval_results) if retrieval_results else 0} documents found")
                    if retrieval_results:
                        retrieved_images_for_testing = []
                        for result in retrieval_results:
                            logger.debug(f"🔍 Processing retrieval result: {result}")
                            # Handle both 'path' and 'image_path' keys for compatibility
                            img_path = result.get('image_path') or result.get('path', '')
                            if img_path:
                                # Try different path combinations
                                possible_paths = [
                                    img_path,
                                    os.path.join('static', img_path),
                                    os.path.abspath(os.path.join('static', img_path)),
                                ]

                                for full_path in possible_paths:
                                    if os.path.exists(full_path):
                                        retrieved_images_for_testing.append(full_path)
                                        logger.info(f"✅ Retrieved image for testing: {full_path}")
                                        break
                                else:
                                    logger.warning(f"⚠️ Retrieved image path not found: {img_path}")

                        logger.info(f"🔍 Retrieved {len(retrieved_images_for_testing)} valid images for optimization testing")

                        # For retrieval mode, use retrieved images; for extraction mode, use user-selected images
                        if optimization_mode == "retrieval":
                            full_path_images = retrieved_images_for_testing
                        # For extraction mode, we already have full_path_images from user selection
                    else:
                        logger.warning("⚠️ No images retrieved with enhanced query - trying fallback approach")
                        # Check if documents exist in the session and use them directly
                        session_folder = os.path.join(self.app_config.get('SESSION_FOLDER', 'sessions'), original_session_id)
                        if os.path.exists(session_folder):
                            logger.info(f"📁 Session folder exists: {session_folder}")
                            # Try to use selected documents directly as a fallback
                            if selected_docs and optimization_mode == "retrieval":
                                logger.info("🔄 Using selected documents as fallback for retrieval testing")
                                fallback_images = []
                                for doc in selected_docs[:3]:  # Use first 3 selected documents
                                    # Look for PDF pages in the session folder
                                    doc_base = os.path.splitext(doc)[0]
                                    for i in range(1, 6):  # Check first 5 pages
                                        page_file = f"{doc_base}_page_{i}.png"
                                        page_path = os.path.join(session_folder, page_file)
                                        if os.path.exists(page_path):
                                            fallback_images.append(page_path)
                                            logger.info(f"✅ Found fallback document page: {page_path}")
                                            break

                                if fallback_images:
                                    full_path_images = fallback_images
                                    logger.info(f"🔄 Using {len(fallback_images)} fallback images for optimization testing")
                                else:
                                    full_path_images = []
                            else:
                                full_path_images = []

                            # List files in session folder for debugging
                            try:
                                files = os.listdir(session_folder)
                                logger.info(f"📁 Files in session folder: {files[:10]}...")  # Show first 10 files
                            except Exception as e:
                                logger.error(f"❌ Error listing session folder: {e}")
                        else:
                            logger.error(f"❌ Session folder not found: {session_folder}")
                            if optimization_mode == "retrieval":
                                full_path_images = []
                else:
                    logger.error("❌ Could not load RAG model for retrieval testing")
                    if optimization_mode == "retrieval":
                        full_path_images = []

            except Exception as e:
                logger.error(f"❌ Error during retrieval optimization testing: {e}")
                # Fall back to empty images - will test template's ability to handle no relevant content
                if optimization_mode == "retrieval":
                    full_path_images = []

            # Create a query using the improved template
            enhanced_query = original_query
            if improved_template.get('query_prefix'):
                enhanced_query = f"{improved_template['query_prefix']} {enhanced_query}"
            if improved_template.get('query_suffix'):
                enhanced_query = f"{enhanced_query} {improved_template['query_suffix']}"

            logger.info(f"🧪 Testing TEMPLATE with enhanced query: '{enhanced_query[:100]}...'")
            logger.info(f"📋 Template: {improved_template.get('template_name', 'Unknown')}")

            # Generate response using the improved template
            # Create a temporary template and apply it during testing

            # Use application context if available
            if self.app:
                logger.info(f"🔗 Using Flask app context for response generation")
                with self.app.app_context():
                    # Temporarily create and save the optimized template for testing
                    temp_template_id = f"temp-test-{int(time.time())}"
                    temp_template = {
                        'id': temp_template_id,
                        'name': improved_template.get('template_name', 'Test Template'),
                        'description': 'Temporary template for optimization testing',
                        'is_default': False,
                        'template_type': 'test',
                        'system_prompt': improved_template.get('system_prompt', ''),
                        'query_prefix': improved_template.get('query_prefix', ''),
                        'query_suffix': improved_template.get('query_suffix', '')
                    }

                    # Save temporary template
                    from src.models.prompt_templates import save_user_template
                    save_user_template(user_id, temp_template)
                    logger.info(f"🧪 Created temporary template for testing: {temp_template_id}")

                    # Load and modify session to use the test template
                    from src.services.session_manager.manager import load_session, save_session
                    session_folder = self.app_config.get('SESSION_FOLDER', 'sessions')
                    original_session_data = load_session(session_folder, session_id)
                    if original_session_data:
                        # Backup original template selection
                        original_template_id = original_session_data.get('selected_template_id')

                        # Temporarily set the test template
                        original_session_data['selected_template_id'] = temp_template_id
                        save_session(session_folder, session_id, original_session_data)
                        logger.info(f"🔄 Temporarily applied test template {temp_template_id} to session")

                        try:
                            # Generate response with the test template
                            response_text, used_images = generate_response(
                                images=full_path_images,  # Use validated full path images
                                query=enhanced_query,
                                session_id=session_id,  # Use session with test template
                                model_choice=generation_model,
                                user_id=user_id,
                                chat_history=[]  # No chat history for testing
                            )
                            logger.info(f"✅ Template test response generated with optimized template: length={len(response_text)}, images_used={len(used_images)}")
                        finally:
                            # Restore original template selection
                            if original_template_id:
                                original_session_data['selected_template_id'] = original_template_id
                            else:
                                original_session_data.pop('selected_template_id', None)
                            save_session(session_folder, session_id, original_session_data)
                            logger.info(f"🔄 Restored original template selection")

                            # Clean up temporary template
                            from src.models.prompt_templates import delete_template
                            delete_template(user_id, temp_template_id)
                            logger.info(f"🧹 Cleaned up temporary template {temp_template_id}")
                    else:
                        logger.warning(f"⚠️ Could not load session {session_id} for template testing")
                        # Fallback to basic response generation
                        response_text, used_images = generate_response(
                            images=full_path_images,
                            query=enhanced_query,
                            session_id=session_id,
                            model_choice=generation_model,
                            user_id=user_id,
                            chat_history=[]
                        )
                        logger.info(f"✅ Template test response generated (fallback): length={len(response_text)}, images_used={len(used_images)}")
            else:
                # Fallback: create a mock response if no app context available
                logger.warning("⚠️ No app context available, creating mock response for template testing")
                response_text = f"Mock response for template query: {enhanced_query[:100]}... (Generated with {generation_model})"
                used_images = relevant_images[:3] if relevant_images else []
                logger.info(f"🔄 Mock template response created: length={len(response_text)}, mock_images={len(used_images)}")

            return {
                'retrieved_images': used_images,
                'response_text': response_text,
                'enhanced_query': enhanced_query,
                'optimized_query': session_data.get('optimized_query', original_query),
                'template_info': {
                    'name': improved_template.get('template_name', 'Unknown'),
                    'id': improved_template.get('template_id', 'unknown'),
                    'confidence': improved_template.get('confidence_score', 0.5)
                },
                'evaluation_notes': f'Generated using optimized template: {improved_template.get("template_name", "Unknown")} (confidence: {improved_template.get("confidence_score", 0.5)})'
            }

        except Exception as e:
            logger.error(f"Error testing prompt: {e}", exc_info=True)
            return {
                'retrieved_images': [],
                'response_text': f'Error testing prompt: {str(e)}',
                'optimized_query': session_data.get('optimized_query', session_data.get('original_query', '')),
                'evaluation_notes': f'Error: {str(e)}'
            }
    
    async def _evaluate_results_with_llm(self, test_results: Dict[str, Any],
                                       feedback: Dict[str, Any],
                                       original_context: Dict[str, Any],
                                       session_data: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate the quality of the test results using LLM-as-judge."""
        try:
            # Get the expected response from feedback
            expected_response = feedback.get('expected_response', '')
            if not expected_response:
                # Fall back to legacy response_feedback if expected_response is not available
                expected_response = feedback.get('response_feedback', '')

            if not expected_response:
                logger.warning("No expected response found in feedback, using fallback evaluation")
                return self._get_fallback_evaluation_result()

            # Get the generated response
            generated_response = test_results.get('response_text', '')
            retrieved_context = test_results.get('retrieved_images', [])

            # Get evaluation parameters
            query = original_context.get('query', '')
            generation_model = session_data.get('generation_model', 'ollama-gemma3n-vision-fp16')
            session_id = session_data.get('session_id', '')
            user_id = session_data.get('user_id', '')

            # Use component-based LLM judge for evaluation
            logger.info(f"🎯 STARTING COMPONENT-BASED EVALUATION")
            llm_judge = get_llm_judge(app=self.app)

            # Extract components if not already done
            if not hasattr(self, '_current_components'):
                logger.info(f"🔍 Extracting key components from expected response")
                self._current_components = self._extract_key_components_from_expected_response(
                    expected_response, query, generation_model
                )
                logger.info(f"✅ Extracted components: {self._current_components}")

            logger.info(f"🎯 Calling component-based evaluation with {len(self._current_components.get('key_facts', []))} key facts")
            evaluation_result = await llm_judge.evaluate_response_component_based(
                query=query,
                expected_response=expected_response,
                generated_response=generated_response,
                retrieved_context=retrieved_context,
                generation_model=generation_model,
                session_id=session_id,
                user_id=user_id,
                key_components=self._current_components
            )

            logger.info(f"LLM evaluation completed with score: {evaluation_result.get('overall_score', 'N/A')}")
            return evaluation_result

        except Exception as e:
            logger.error(f"Error in LLM evaluation: {e}", exc_info=True)
            return self._get_fallback_evaluation_result(str(e))

    def _get_fallback_evaluation_result(self, error_msg: str = "") -> Dict[str, Any]:
        """Return a fallback evaluation result when LLM evaluation fails."""
        return {
            'overall_score': 5.0,  # Neutral score
            'dimension_scores': {
                'content_accuracy': 5.0,
                'completeness': 5.0,
                'clarity_structure': 5.0,
                'relevance': 5.0,
                'detail_level': 5.0
            },
            'strengths': ['Evaluation could not be completed'],
            'weaknesses': ['LLM evaluation failed'],
            'improvement_suggestions': ['Manual review recommended'],
            'confidence': 0.0,
            'error': error_msg or 'LLM evaluation failed',
            'parsing_method': 'fallback'
        }

# Global optimizer instance
_optimizer = None

def get_prompt_optimizer(app_config: Dict[str, Any], app=None) -> PromptOptimizer:
    """Get or create the global prompt optimizer instance."""
    global _optimizer
    if _optimizer is None:
        _optimizer = PromptOptimizer(app_config, app=app)
    return _optimizer
