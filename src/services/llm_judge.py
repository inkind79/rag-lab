"""
LLM-as-Judge Service

This module implements an LLM-based evaluation system that uses the currently
selected generation model to judge the quality of RAG responses against
user-specified expected responses.
"""

import json
import re
import asyncio
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

from src.utils.logger import get_logger
from src.models.responder import generate_response
from src.models.model_loader import load_model
from src.services.session_manager.manager import load_session

logger = get_logger(__name__)


class LLMJudge:
    """
    LLM-based judge for evaluating RAG response quality.

    Uses the currently selected generation model to compare generated responses
    against user-specified expected responses and provide detailed scoring.
    """

    def __init__(self, app=None):
        self.app = app
        self.evaluation_prompt_template = self._get_evaluation_prompt_template()
    
    def _get_evaluation_prompt_template(self) -> str:
        """Get the system prompt template for LLM evaluation."""
        return """You are an expert evaluator. Your PRIMARY task is to compare an AI-generated response against a user's expected ideal response and score how well they align.

EVALUATION DIMENSIONS (0-10 scale):
1. Response Similarity: How closely the generated response matches the expected response in content, structure, and key points (MOST IMPORTANT - 40% weight)
2. Content Accuracy: Factual correctness and relevance to the query (25% weight)
3. Completeness: Coverage of expected content and key information (20% weight)
4. Clarity: Organization, readability, and communication effectiveness (10% weight)
5. Detail Level: Appropriate depth and specificity of information (5% weight)

CRITICAL EVALUATION CRITERIA:
- RESPONSE ALIGNMENT: The generated response should closely match the expected response in content, tone, and structure. Responses that address the same topic but differ significantly in approach or content should receive lower similarity scores.
- CONTENT MATCHING: Score higher when the generated response covers the same key points, uses similar terminology, and reaches similar conclusions as the expected response.
- HALLUCINATION PENALTY: If the generated response references specific document elements (like box numbers, sections, or fields) that don't actually exist in the document, severely penalize the content accuracy score.
- FACTUAL VERIFICATION: Ensure the response only claims information that can be verified from the actual document content.
- FALSE SPECIFICITY: Responses that make up specific details not present in the document should receive very low scores.

SCORING GUIDELINES:
- Response Similarity 9-10: Generated response is nearly identical in content and approach to expected response
- Response Similarity 7-8: Generated response covers most key points from expected response with similar approach
- Response Similarity 5-6: Generated response addresses the topic but with different focus or missing key elements
- Response Similarity 3-4: Generated response is somewhat related but significantly different from expected response
- Response Similarity 0-2: Generated response is completely different or unrelated to expected response

INSTRUCTIONS:
- PRIORITIZE response similarity - this is the most important factor
- Compare the generated response to the expected response as ground truth
- Consider different ways of expressing the same information, but prioritize content alignment
- Be objective and constructive in evaluation
- Heavily penalize any hallucinated or fabricated document references

REQUIRED OUTPUT FORMAT - Respond with ONLY this JSON structure:
{
    "overall_score": 7.2,
    "dimension_scores": {
        "response_similarity": 8.5,
        "content_accuracy": 8.0,
        "completeness": 7.0,
        "clarity_structure": 7.5,
        "detail_level": 6.5
    },
    "strengths": ["Closely matches expected response", "Accurate information"],
    "weaknesses": ["Missing some key details", "Different structure"],
    "improvement_suggestions": ["Include specific points from expected response", "Match expected tone"],
    "confidence": 0.8
}

IMPORTANT: Return ONLY the JSON object, no other text."""

    async def evaluate_response_component_based(
        self,
        query: str,
        expected_response: str,
        generated_response: str,
        retrieved_context: List[str],
        generation_model: str,
        session_id: str,
        user_id: str,
        key_components: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Evaluate response using component-based boolean scoring for better accuracy.

        Args:
            query: The original user query
            expected_response: User-specified ideal response (ground truth)
            generated_response: AI-generated response to evaluate
            retrieved_context: List of retrieved document/image paths
            generation_model: The model used for generation (also used for judging)
            session_id: Current session ID
            user_id: Current user ID
            key_components: Extracted components from expected response

        Returns:
            Dictionary containing component-based evaluation scores and feedback
        """
        try:
            logger.info(f"🎯 COMPONENT-BASED EVALUATION CALLED: query='{query[:50]}...', expected_len={len(expected_response)}, generated_len={len(generated_response)}")

            # If no components provided, extract them
            if not key_components:
                logger.info(f"🔍 No components provided, extracting from expected response")
                key_components = await self._extract_key_components(query, expected_response, generation_model, session_id, user_id)
            else:
                logger.info(f"✅ Using provided components: {key_components}")

            # Prepare component-based evaluation
            evaluation_query = self._prepare_component_evaluation_query(
                query, expected_response, generated_response, retrieved_context, key_components
            )

            logger.info(f"🎯 COMPONENT-BASED LLM JUDGE START: model={generation_model}, components={len(key_components.get('key_facts', []))}")

            from src.models.responder import generate_response

            evaluation_response, _ = generate_response(
                images=[],
                query=evaluation_query,
                session_id=f"{session_id}_comp_eval",
                model_choice=generation_model,
                user_id=user_id,
                chat_history=[]
            )

            logger.info(f"🎯 Component evaluation response received: length={len(evaluation_response)}")

            logger.info(f"🎯 Component evaluation response: '{evaluation_response[:300]}...'")

            # Parse the component-based evaluation response
            evaluation_result = self._parse_component_evaluation_response(evaluation_response, key_components)

            # Add metadata
            evaluation_result['evaluation_timestamp'] = datetime.now().isoformat()
            evaluation_result['evaluator_model'] = generation_model
            evaluation_result['evaluation_method'] = 'component_based'
            evaluation_result['key_components'] = key_components
            evaluation_result['query'] = query
            evaluation_result['expected_response'] = expected_response
            evaluation_result['generated_response'] = generated_response

            logger.info(f"✅ COMPONENT-BASED JUDGE COMPLETE: score={evaluation_result.get('overall_score', 0):.2f}/10, method=component_based")
            logger.info(f"📊 Component evaluation result: {evaluation_result}")

            return evaluation_result

        except Exception as e:
            logger.error(f"Error in component-based LLM evaluation: {e}", exc_info=True)
            # Fallback to traditional evaluation
            return await self.evaluate_response(query, expected_response, generated_response, retrieved_context, generation_model, session_id, user_id)

    async def evaluate_response(
        self,
        query: str,
        expected_response: str,
        generated_response: str,
        retrieved_context: List[str],
        generation_model: str,
        session_id: str,
        user_id: str
    ) -> Dict[str, Any]:
        """
        Evaluate a generated response against the expected response using LLM-as-judge.
        
        Args:
            query: The original user query
            expected_response: User-specified ideal response (ground truth)
            generated_response: AI-generated response to evaluate
            retrieved_context: List of retrieved document/image paths
            generation_model: The model used for generation (also used for judging)
            session_id: Current session ID
            user_id: Current user ID
            
        Returns:
            Dictionary containing evaluation scores and feedback
        """
        try:
            # Prepare the evaluation prompt
            evaluation_query = self._prepare_evaluation_query(
                query, expected_response, generated_response, retrieved_context
            )
            
            logger.info(f"🤖 LLM JUDGE START: model={generation_model}, query='{query[:50]}...', expected_length={len(expected_response)}, generated_length={len(generated_response)}")

            # Use the same model for evaluation as was used for generation
            # This ensures consistency and uses the currently selected model

            evaluation_response, _ = generate_response(
                images=[],
                query=evaluation_query,
                session_id=f"{session_id}_eval",
                model_choice=generation_model,
                user_id=user_id,
                chat_history=[]
            )

            logger.info(f"LLM evaluation response received: length={len(evaluation_response)}")
            
            # Parse the evaluation response
            logger.info(f"🔍 Parsing LLM evaluation response")
            evaluation_result = self._parse_evaluation_response(evaluation_response)

            # Add metadata
            evaluation_result['evaluation_timestamp'] = datetime.now().isoformat()
            evaluation_result['evaluator_model'] = generation_model
            evaluation_result['query'] = query
            evaluation_result['expected_response'] = expected_response
            evaluation_result['generated_response'] = generated_response

            # Safe formatting for logging
            overall_score = evaluation_result.get('overall_score', 0)
            confidence = evaluation_result.get('confidence', 0)
            method = evaluation_result.get('parsing_method', 'N/A')

            # Ensure numeric values for formatting
            try:
                score_str = f"{float(overall_score):.2f}" if overall_score != 'N/A' else 'N/A'
                conf_str = f"{float(confidence):.2f}" if confidence != 'N/A' else 'N/A'
            except (ValueError, TypeError):
                score_str = str(overall_score)
                conf_str = str(confidence)

            logger.info(f"✅ LLM JUDGE COMPLETE: score={score_str}/10, method={method}, confidence={conf_str}")

            if evaluation_result.get('strengths'):
                logger.info(f"💪 Strengths identified: {evaluation_result['strengths'][:2]}")
            if evaluation_result.get('weaknesses'):
                logger.info(f"⚠️ Weaknesses identified: {evaluation_result['weaknesses'][:2]}")

            return evaluation_result
            
        except Exception as e:
            logger.error(f"Error in LLM evaluation: {e}", exc_info=True)
            # Return a fallback evaluation
            return self._get_fallback_evaluation(str(e))
    
    def _prepare_evaluation_query(
        self,
        query: str,
        expected_response: str,
        generated_response: str,
        retrieved_context: List[str]
    ) -> str:
        """Prepare the evaluation query for the LLM judge."""

        context_info = f"Retrieved {len(retrieved_context)} documents/images" if retrieved_context else "No documents retrieved"

        evaluation_query = f"""EVALUATION TASK:

Query: {query}

Expected Response: {expected_response}

Generated Response: {generated_response}

Context: {context_info}

Evaluate how well the generated response matches the expected response. Return only the JSON evaluation object as specified in your instructions."""

        return evaluation_query

    async def _extract_key_components(self, query: str, expected_response: str, generation_model: str, session_id: str, user_id: str) -> Dict[str, Any]:
        """Extract key components from expected response for component-based evaluation."""
        try:
            from src.models.responder import generate_response

            extraction_prompt = f"""Extract key components from this expected response for evaluation purposes.

QUERY: {query}
EXPECTED RESPONSE: {expected_response}

Extract these components as JSON:
{{
    "key_facts": ["specific facts that must be present"],
    "required_elements": ["types of information needed"],
    "format_requirements": ["formatting expectations"],
    "priority_focus": "most important aspect"
}}

Return ONLY the JSON object."""

            response, _ = generate_response(
                images=[],
                query=extraction_prompt,
                session_id=f"{session_id}_extract",
                model_choice=generation_model,  # Use same model as inference
                user_id=user_id,
                chat_history=[]
            )

            import json
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                return self._fallback_component_extraction(expected_response)

        except Exception as e:
            logger.error(f"Component extraction failed: {e}")
            return self._fallback_component_extraction(expected_response)

    def _fallback_component_extraction(self, expected_response: str) -> Dict[str, Any]:
        """Simple fallback for component extraction."""
        import re
        numbers = re.findall(r'\$?[\d,]+\.?\d*', expected_response)
        sentences = expected_response.split('.')[:3]

        return {
            "key_facts": [s.strip() for s in sentences if len(s.strip()) > 5],
            "required_elements": numbers if numbers else ["specific information"],
            "format_requirements": ["clear and accurate"],
            "priority_focus": "accurate information"
        }

    def _prepare_component_evaluation_query(self, query: str, expected_response: str, generated_response: str, retrieved_context: List[str], key_components: Dict[str, Any]) -> str:
        """Prepare component-based evaluation query."""

        context_info = f"Retrieved {len(retrieved_context)} documents/images" if retrieved_context else "No documents retrieved"

        evaluation_query = f"""COMPONENT-BASED EVALUATION TASK:

You are evaluating how well a generated response captures specific key components from an expected response.

QUERY: {query}
EXPECTED RESPONSE: {expected_response}
GENERATED RESPONSE: {generated_response}
CONTEXT: {context_info}

KEY COMPONENTS TO EVALUATE:
- Key Facts: {key_components.get('key_facts', [])}
- Required Elements: {key_components.get('required_elements', [])}
- Format Requirements: {key_components.get('format_requirements', [])}
- Priority Focus: {key_components.get('priority_focus', 'accuracy')}

EVALUATION METHOD:
For each key component, determine if it's correctly captured in the generated response (PASS/FAIL).

Return ONLY this JSON structure:
{{
    "component_scores": {{
        "key_facts_captured": [true, false, true],
        "required_elements_captured": [true, true],
        "format_requirements_met": [true, false],
        "priority_focus_addressed": true
    }},
    "component_details": {{
        "key_facts_analysis": ["fact 1: correctly captured", "fact 2: missing", "fact 3: correctly captured"],
        "required_elements_analysis": ["element 1: present", "element 2: present"],
        "format_analysis": ["format 1: correct", "format 2: incorrect"],
        "priority_analysis": "priority focus is adequately addressed"
    }},
    "overall_pass_rate": 0.75,
    "strengths": ["accurate key facts", "good format"],
    "weaknesses": ["missing some details"],
    "confidence": 0.9
}}"""

        return evaluation_query

    def _parse_component_evaluation_response(self, response: str, key_components: Dict[str, Any]) -> Dict[str, Any]:
        """Parse component-based evaluation response."""
        logger.info(f"🔍 Parsing component evaluation response: '{response[:200]}...'")
        try:
            import json
            import re

            # Extract JSON
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                logger.info(f"📝 Found JSON in component evaluation: '{json_str[:300]}...'")
                eval_data = json.loads(json_str)
                logger.info(f"📊 Parsed component evaluation data: {eval_data}")

                # Calculate overall score from component pass rates
                component_scores = eval_data.get('component_scores', {})
                total_components = 0
                passed_components = 0

                for component_type, scores in component_scores.items():
                    if isinstance(scores, list):
                        total_components += len(scores)
                        passed_components += sum(1 for score in scores if score)
                    elif isinstance(scores, bool):
                        total_components += 1
                        passed_components += 1 if scores else 0

                pass_rate = passed_components / total_components if total_components > 0 else 0
                overall_score = pass_rate * 10  # Convert to 0-10 scale

                logger.info(f"🎯 Component scoring: {passed_components}/{total_components} passed, rate={pass_rate:.2f}, score={overall_score:.2f}")

                result = {
                    'overall_score': overall_score,
                    'component_scores': component_scores,
                    'component_details': eval_data.get('component_details', {}),
                    'pass_rate': pass_rate,
                    'total_components': total_components,
                    'passed_components': passed_components,
                    'strengths': eval_data.get('strengths', []),
                    'weaknesses': eval_data.get('weaknesses', []),
                    'improvement_suggestions': eval_data.get('weaknesses', []),  # Use weaknesses as suggestions
                    'confidence': eval_data.get('confidence', 0.8),
                    'parsing_method': 'component_based'
                }

                logger.info(f"✅ Component evaluation parsed successfully: score={overall_score:.2f}")
                return result
            else:
                logger.warning("⚠️ No JSON found in component evaluation response")
                return self._get_fallback_evaluation("JSON parsing failed")

        except Exception as e:
            logger.error(f"❌ Error parsing component evaluation: {e}", exc_info=True)
            return self._get_fallback_evaluation(str(e))

    def _parse_evaluation_response(self, response: str) -> Dict[str, Any]:
        """Parse the LLM evaluation response into structured data."""
        try:
            # Clean the response
            response = response.strip()

            # Try multiple JSON extraction methods
            json_str = None

            # Method 1: Look for complete JSON object
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1

            if start_idx != -1 and end_idx > start_idx:
                json_str = response[start_idx:end_idx]

                # Try to parse
                try:
                    evaluation_data = json.loads(json_str)
                    if 'overall_score' in evaluation_data:
                        logger.info("Successfully parsed JSON evaluation response")
                        # Apply weighted scoring calculation
                        evaluation_data = self._apply_weighted_scoring(evaluation_data)
                        return evaluation_data
                except json.JSONDecodeError:
                    pass

            # Method 2: Try to find JSON in code blocks
            import re
            json_blocks = re.findall(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
            for block in json_blocks:
                try:
                    evaluation_data = json.loads(block)
                    if 'overall_score' in evaluation_data:
                        logger.info("Successfully parsed JSON from code block")
                        # Apply weighted scoring calculation
                        evaluation_data = self._apply_weighted_scoring(evaluation_data)
                        return evaluation_data
                except json.JSONDecodeError:
                    continue

            # Method 3: Try the entire response as JSON
            try:
                evaluation_data = json.loads(response)
                if 'overall_score' in evaluation_data:
                    logger.info("Successfully parsed entire response as JSON")
                    # Apply weighted scoring calculation
                    evaluation_data = self._apply_weighted_scoring(evaluation_data)
                    return evaluation_data
            except json.JSONDecodeError:
                pass

            # If all JSON parsing fails, use manual extraction
            logger.warning("JSON parsing failed, using manual extraction")
            return self._extract_scores_manually(response)

        except Exception as e:
            logger.error(f"Error parsing evaluation response: {e}")
            return self._extract_scores_manually(response)
    
    def _extract_scores_manually(self, response: str) -> Dict[str, Any]:
        """Manually extract scores from evaluation response if JSON parsing fails."""
        try:
            import re

            # Try to find overall score with various patterns
            score_patterns = [
                r'overall[_\s]*score[:\s]*(\d+\.?\d*)',
                r'total[_\s]*score[:\s]*(\d+\.?\d*)',
                r'final[_\s]*score[:\s]*(\d+\.?\d*)',
                r'score[:\s]*(\d+\.?\d*)[/\s]*10',
                r'(\d+\.?\d*)[/\s]*10'
            ]

            overall_score = 5.0  # Default neutral score
            for pattern in score_patterns:
                match = re.search(pattern, response, re.IGNORECASE)
                if match:
                    score = float(match.group(1))
                    if score <= 10:  # Valid score range
                        overall_score = score
                        break

            # Try to extract dimension scores
            dimension_scores = {}
            dimension_patterns = {
                'response_similarity': r'(?:response|similarity)[_\s]*(?:score)?[:\s]*(\d+\.?\d*)',
                'content_accuracy': r'(?:content|accuracy)[_\s]*(?:score)?[:\s]*(\d+\.?\d*)',
                'completeness': r'completeness[_\s]*(?:score)?[:\s]*(\d+\.?\d*)',
                'clarity_structure': r'(?:clarity|structure)[_\s]*(?:score)?[:\s]*(\d+\.?\d*)',
                'detail_level': r'(?:detail|level)[_\s]*(?:score)?[:\s]*(\d+\.?\d*)'
            }

            for dim, pattern in dimension_patterns.items():
                match = re.search(pattern, response, re.IGNORECASE)
                if match:
                    score = float(match.group(1))
                    if score <= 10:
                        dimension_scores[dim] = score
                else:
                    dimension_scores[dim] = overall_score

            # Extract strengths and weaknesses more carefully
            strengths = self._extract_evaluation_points(response, ['strength', 'good', 'positive', 'well'])
            weaknesses = self._extract_evaluation_points(response, ['weakness', 'issue', 'problem', 'improve', 'lacking'])
            suggestions = self._extract_evaluation_points(response, ['suggest', 'recommend', 'could', 'should'])

            manual_result = {
                'overall_score': min(max(overall_score, 0), 10),
                'dimension_scores': dimension_scores,
                'strengths': strengths[:3],
                'weaknesses': weaknesses[:3],
                'improvement_suggestions': suggestions[:3],
                'confidence': 0.6,  # Lower confidence for manual extraction
                'parsing_method': 'manual_extraction'
            }

            # Apply weighted scoring to manual extraction results as well
            return self._apply_weighted_scoring(manual_result)

        except Exception as e:
            logger.error(f"Error in manual score extraction: {e}")
            return self._get_fallback_evaluation(f"Manual extraction failed: {str(e)}")
    
    def _extract_evaluation_points(self, text: str, keywords: List[str]) -> List[str]:
        """Extract evaluation points (strengths, weaknesses, suggestions) from text."""
        items = []
        lines = text.split('\n')

        for line in lines:
            line = line.strip()
            if any(keyword in line.lower() for keyword in keywords):
                # Clean up the line - remove bullets, numbers, etc.
                cleaned = re.sub(r'^[-*•\d\.\)\s]+', '', line).strip()
                cleaned = re.sub(r'^(strength|weakness|suggestion|good|bad|improve)[s]?[:\s]*', '', cleaned, flags=re.IGNORECASE).strip()

                # Only keep meaningful items
                if cleaned and len(cleaned) > 15 and len(cleaned) < 150:
                    items.append(cleaned)

        # If no specific items found, try to extract from sentences
        if not items:
            sentences = re.split(r'[.!?]+', text)
            for sentence in sentences:
                sentence = sentence.strip()
                if any(keyword in sentence.lower() for keyword in keywords) and len(sentence) > 20:
                    items.append(sentence[:100])

        return items[:5]
    
    def _get_fallback_evaluation(self, error_msg: str) -> Dict[str, Any]:
        """Return a fallback evaluation when LLM evaluation fails."""
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
            'error': error_msg,
            'parsing_method': 'fallback'
        }

    def _get_fallback_evaluation_with_similarity(
        self,
        expected_response: str,
        generated_response: str
    ) -> Dict[str, Any]:
        """Return a fallback evaluation using similarity scoring."""
        similarity_score = self.calculate_similarity_score(expected_response, generated_response)

        return {
            'overall_score': similarity_score,
            'dimension_scores': {
                'content_accuracy': similarity_score,
                'completeness': similarity_score * 0.8,  # Slightly lower for completeness
                'clarity_structure': similarity_score * 0.9,
                'relevance': similarity_score,
                'detail_level': similarity_score * 0.7
            },
            'strengths': ['Response generated successfully'],
            'weaknesses': ['Limited evaluation due to context constraints'],
            'improvement_suggestions': ['Consider running with full application context'],
            'confidence': 0.6,  # Lower confidence for similarity-based evaluation
            'parsing_method': 'similarity_fallback'
        }
    
    def calculate_similarity_score(
        self,
        expected_response: str,
        generated_response: str
    ) -> float:
        """
        Calculate a simple similarity score between expected and generated responses.
        This is used as a backup scoring method.
        """
        try:
            # Simple word overlap similarity
            expected_words = set(expected_response.lower().split())
            generated_words = set(generated_response.lower().split())
            
            if not expected_words:
                return 0.0
            
            overlap = len(expected_words.intersection(generated_words))
            similarity = overlap / len(expected_words)
            
            return min(similarity * 10, 10.0)  # Scale to 0-10
            
        except Exception as e:
            logger.error(f"Error calculating similarity score: {e}")
            return 5.0  # Neutral score on error

    def _apply_weighted_scoring(self, evaluation_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply weighted scoring calculation to emphasize response similarity."""
        try:
            dimension_scores = evaluation_data.get('dimension_scores', {})

            # Define weights for each dimension (must sum to 1.0)
            weights = {
                'response_similarity': 0.40,  # 40% - Most important
                'content_accuracy': 0.25,    # 25%
                'completeness': 0.20,        # 20%
                'clarity_structure': 0.10,   # 10%
                'detail_level': 0.05         # 5%
            }

            # Calculate weighted overall score
            weighted_score = 0.0
            total_weight = 0.0

            for dimension, weight in weights.items():
                if dimension in dimension_scores:
                    score = float(dimension_scores[dimension])
                    weighted_score += score * weight
                    total_weight += weight
                    logger.debug(f"Dimension {dimension}: score={score:.2f}, weight={weight:.2f}, contribution={score*weight:.2f}")

            # If we have dimension scores, use weighted calculation
            if total_weight > 0:
                final_score = weighted_score
                # Ensure score is within valid range
                final_score = max(0.0, min(10.0, final_score))

                # Store original overall score for comparison
                original_score = evaluation_data.get('overall_score', 0)
                evaluation_data['original_overall_score'] = original_score
                evaluation_data['overall_score'] = final_score

                logger.info(f"Applied weighted scoring: original={original_score:.2f}, weighted={final_score:.2f}")
                logger.info(f"Dimension contributions: {[(dim, dimension_scores.get(dim, 0) * weights.get(dim, 0)) for dim in weights.keys() if dim in dimension_scores]}")
            else:
                logger.warning("No dimension scores found for weighted calculation, keeping original overall score")

            return evaluation_data

        except Exception as e:
            logger.error(f"Error applying weighted scoring: {e}")
            return evaluation_data  # Return original data on error


# Global instance
_llm_judge = None

def get_llm_judge(app=None) -> LLMJudge:
    """Get the global LLM judge instance."""
    global _llm_judge
    if _llm_judge is None:
        _llm_judge = LLMJudge(app=app)
    return _llm_judge
