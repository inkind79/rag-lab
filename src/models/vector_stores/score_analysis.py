"""
Score analysis utilities for improving retrieval quality.

This module provides functions to analyze similarity scores and determine
optimal cutoff points based on score distribution patterns and token budgets.
"""
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from src.utils.logger import get_logger
from src.utils.token_utils import estimate_tokens, get_model_context_window, get_document_text

logger = get_logger(__name__)

def analyze_score_distribution(
    results: List[Dict[str, Any]],
    rel_drop_threshold: float = 0.65,
    abs_score_threshold: float = 0.25,
    min_results: int = 3,
    max_results: int = 20,  # Add a maximum limit to prevent too many results
    model_name: str = None
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Analyzes the score distribution of retrieval results and determines
    the optimal cutoff point based on the "elbow" in the score curve.

    Args:
        results: List of retrieval results, each containing at least a 'score' key
        rel_drop_threshold: Threshold for relative score drop (score_i / score_0)
        abs_score_threshold: Absolute minimum score threshold
        min_results: Minimum number of results to return, regardless of thresholds
        max_results: Maximum number of results to return, even if all pass thresholds
        model_name: Optional model name for model-specific adjustments

    Returns:
        Tuple containing:
        - Filtered list of results (only those above the determined threshold)
        - Dictionary with analysis metadata (original_count, filtered_count, cutoff_index, etc.)
    """
    if not results:
        return [], {"original_count": 0, "filtered_count": 0, "cutoff_index": 0, "cutoff_reason": "empty_results"}

    # Ensure results are sorted by score (descending)
    sorted_results = sorted(results, key=lambda x: x.get('score', 0), reverse=True)

    # Extract scores
    scores = [item.get('score', 0) for item in sorted_results]

    # Initialize analysis metadata
    analysis = {
        "original_count": len(scores),
        "max_score": scores[0] if scores else 0,
        "min_score": min(scores) if scores else 0,
        "mean_score": np.mean(scores) if scores else 0,
        "median_score": np.median(scores) if scores else 0
    }

    # Find cutoff index based on relative drop
    cutoff_index = len(scores)  # Default to keeping all
    cutoff_reason = "no_cutoff"

    # Only analyze if we have a valid maximum score
    if scores and scores[0] > 0:
        max_score = scores[0]

        # Apply model-specific adjustments if needed
        is_colqwen25 = model_name and "colqwen2.5" in model_name.lower()
        is_colnomic = model_name and "colnomic" in model_name.lower()
        is_colpali = is_colqwen25 or is_colnomic
        
        # Use the thresholds provided by the user
        effective_rel_drop_threshold = rel_drop_threshold
        effective_abs_score_threshold = abs_score_threshold

        # For ColBERT sum-similarity scores, we need to convert UI thresholds to internal thresholds
        # These scores are in the range [0, m] where m is the number of query tokens
        if is_colpali:
            # For relative drop threshold, we need model-specific handling
            if is_colnomic:
                # ColNomic has more compressed scores, so we need to be more sensitive
                # Adjust the threshold to be more aggressive for ColNomic
                # A 0.65 threshold becomes 0.85 for ColNomic (less aggressive)
                # This accounts for the narrower score distribution
                effective_rel_drop_threshold = 0.85 + (rel_drop_threshold - 0.65) * 0.5
                logger.info(f"Using ColNomic-specific relative drop threshold adjustment")
            else:
                # For ColQwen2.5, use the UI value directly since it's a ratio
                effective_rel_drop_threshold = rel_drop_threshold

            # For absolute score threshold, we need to handle normalized ColBERT scores
            # In lancedb_manager.py, the scores are now normalized by dividing by query_vector_count
            # This gives us scores in a more predictable range, typically [-1, 1] for normalized scores
            
            # For normalized ColBERT scores, we need model-specific handling
            # ColNomic produces scores in a narrower positive range compared to ColQwen2.5
            if is_colnomic:
                # ColNomic normalized scores tend to be in range [0.2, 0.4]
                # Map UI threshold [0,1] to a more appropriate range for ColNomic
                # UI 0.0 → 0.15 (very loose)
                # UI 0.25 → 0.20 (default)
                # UI 0.5 → 0.25 (moderate)
                # UI 1.0 → 0.35 (strict)
                internal_abs_threshold = 0.15 + (abs_score_threshold * 0.2)
                effective_abs_score_threshold = internal_abs_threshold
                logger.info(f"Using ColNomic-specific absolute threshold mapping")
            else:
                # For ColQwen2.5 and others, normalized scores can be in range [-1, 1]
                # Map from [0,1] to [-0.5,0.5] for normalized scores
                internal_abs_threshold = abs_score_threshold - 0.5
                effective_abs_score_threshold = internal_abs_threshold

            logger.info(f"Using thresholds for ColPali with normalized ColBERT scores:")
            logger.info(f"  - Relative drop: {rel_drop_threshold:.4f} (from UI setting)")
            logger.info(f"  - Absolute score: {abs_score_threshold:.4f} (UI) → {effective_abs_score_threshold:.4f} (normalized)")
        # Removed Byaldi-specific logging

        # Calculate relative drops from max score
        rel_drops_from_max = [score / max_score for score in scores]

        # Calculate relative drops between consecutive items
        # This helps identify "plateaus" where scores are similar
        rel_drops_consecutive = []
        for i in range(1, len(scores)):
            if scores[i-1] > 0:  # Avoid division by zero
                rel_drops_consecutive.append(scores[i] / scores[i-1])
            else:
                rel_drops_consecutive.append(0)

        # Add a dummy value for the first item to align indices
        rel_drops_consecutive.insert(0, 1.0)

        # Find first index where relative drop goes below threshold
        for i in range(1, len(scores)):
            # Skip the first result (it's our reference point)
            if i == 0:
                continue

            # Get relative drop from max score
            rel_drop_from_max = rel_drops_from_max[i]

            # Get relative drop from previous item
            rel_drop_consecutive = rel_drops_consecutive[i]

            # Check if we've hit a significant drop from max score
            if rel_drop_from_max < effective_rel_drop_threshold:
                # Before cutting off, check if the current item is very close to the previous one
                # This helps identify "plateaus" where scores are similar
                # For ColNomic, use a stricter plateau threshold since scores are compressed
                plateau_threshold = 0.98 if is_colnomic else 0.95
                
                if rel_drop_consecutive > plateau_threshold:
                    # This item is very close to the previous one, so include it
                    logger.info(f"Including index {i} despite rel_drop={rel_drop_from_max:.4f} < threshold={effective_rel_drop_threshold} because it's very close to previous item (rel_drop_consecutive={rel_drop_consecutive:.4f})")
                    continue

                cutoff_index = i
                cutoff_reason = "relative_drop"
                logger.info(f"Score-slope cutoff at index {i}: rel_drop={rel_drop_from_max:.4f} < threshold={effective_rel_drop_threshold}")
                break

            # Check absolute score threshold
            if scores[i] < effective_abs_score_threshold:
                cutoff_index = i
                cutoff_reason = "absolute_score"
                logger.info(f"Score-slope cutoff at index {i}: score={scores[i]:.4f} < threshold={effective_abs_score_threshold}")
                break

    # Ensure we have at least min_results if available
    if cutoff_index < min_results and len(scores) >= min_results:
        original_cutoff = cutoff_index
        cutoff_index = min_results
        cutoff_reason = "min_results_enforced"
        logger.info(f"Enforcing minimum of {min_results} results (original cutoff was {original_cutoff})")

    # Enforce maximum results limit if specified
    if max_results > 0 and cutoff_index > max_results:
        original_cutoff = cutoff_index
        cutoff_index = max_results
        cutoff_reason = "max_results_enforced"
        logger.info(f"Enforcing maximum of {max_results} results (original cutoff was {original_cutoff})")

    # Ensure we don't exceed the length of scores
    if cutoff_index > len(scores):
        cutoff_index = len(scores)
        cutoff_reason = "adjusted_to_available"
        logger.info(f"Adjusted cutoff to available results: {cutoff_index}")

    # Update analysis with cutoff information
    analysis.update({
        "cutoff_index": cutoff_index,
        "cutoff_reason": cutoff_reason,
        "filtered_count": cutoff_index,
        "cutoff_score": scores[cutoff_index-1] if cutoff_index > 0 and cutoff_index <= len(scores) else None
    })

    # Log detailed analysis
    logger.info(f"Score distribution analysis: {analysis}")
    if cutoff_index < len(scores):
        logger.info(f"Trimming results from {len(scores)} to {cutoff_index} based on score-slope")

        # Log score details for debugging
        score_details = [f"{i}:{score:.4f}" for i, score in enumerate(scores[:min(10, len(scores))])]
        logger.debug(f"Top scores: {', '.join(score_details)}")

    # Return filtered results and analysis metadata
    return sorted_results[:cutoff_index], analysis

def apply_token_budget_filter(
    results: List[Dict[str, Any]],
    model_name: str,
    session_data: Dict[str, Any],
    ocr_cache: Optional[Dict[str, Any]] = None,
    reserved_tokens: int = 1000,
    budget_percentage: float = 0.8
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Filter results based on token budget to ensure they fit within the model's context window.

    Args:
        results: List of retrieval results, already filtered by relevance
        model_name: Name of the current model
        session_data: Session data containing model configurations
        ocr_cache: Optional cache of OCR results for documents
        reserved_tokens: Tokens to reserve for query and response
        budget_percentage: Percentage of context window to use for documents

    Returns:
        Tuple containing:
        - Filtered list of results that fit within the token budget
        - Dictionary with analysis metadata
    """
    if not results:
        return [], {"original_count": 0, "filtered_count": 0, "total_tokens": 0, "budget": 0}

    # Get model's context window size
    context_window = get_model_context_window(model_name, session_data)

    # Calculate token budget
    token_budget = int(context_window * budget_percentage)
    available_budget = token_budget - reserved_tokens

    # Initialize result list and token counter
    budget_filtered_results = []
    total_tokens = 0

    # Greedy packing algorithm
    for doc in results:
        # Get document text
        doc_text = get_document_text(doc, ocr_cache)

        # Count tokens in document
        doc_tokens = estimate_tokens(doc_text, model_name)

        # Store token count in document for future reference
        doc['token_count'] = doc_tokens

        # Check if adding this document would exceed the budget
        if total_tokens + doc_tokens <= available_budget:
            budget_filtered_results.append(doc)
            total_tokens += doc_tokens
        else:
            # We've reached our token budget
            break

    # Prepare analysis metadata
    analysis = {
        "original_count": len(results),
        "filtered_count": len(budget_filtered_results),
        "total_tokens": total_tokens,
        "budget": available_budget,
        "context_window": context_window,
        "model_name": model_name
    }

    # Log detailed analysis
    logger.info(f"Token budget filtering: {len(results)} docs → {len(budget_filtered_results)} docs, {total_tokens}/{available_budget} tokens used ({(total_tokens/available_budget)*100:.1f}%)")

    return budget_filtered_results, analysis


def apply_score_slope_analysis(
    scores: List[float],
    rel_drop_threshold: float = 0.65,
    abs_score_threshold: float = 0.25,
    min_results: int = 3,
    max_results: int = 20,
    model_name: str = None
) -> List[int]:
    """
    Analyzes a list of scores and returns the indices of items to keep based on score-slope analysis.

    This is a simplified wrapper around analyze_score_distribution that works with raw scores
    instead of document dictionaries, making it easier to use in contexts like batch processing.

    Args:
        scores: List of similarity scores (higher is better)
        rel_drop_threshold: Threshold for relative score drop (score_i / score_0)
        abs_score_threshold: Absolute minimum score threshold
        min_results: Minimum number of results to return, regardless of thresholds
        max_results: Maximum number of results to return, even if all pass thresholds
        model_name: Optional model name for model-specific adjustments

    Returns:
        List of indices corresponding to items that should be kept
    """
    if not scores:
        return []

    # Create dummy results with the scores
    dummy_results = [{'score': score, 'index': i} for i, score in enumerate(scores)]

    # Apply score-slope analysis
    filtered_results, _ = analyze_score_distribution(
        dummy_results,
        rel_drop_threshold=rel_drop_threshold,
        abs_score_threshold=abs_score_threshold,
        min_results=min_results,
        max_results=max_results,
        model_name=model_name
    )

    # Extract the original indices from the filtered results
    filtered_indices = [item.get('index', 0) for item in filtered_results]

    logger.info(f"Score-slope analysis: {len(scores)} scores → {len(filtered_indices)} filtered indices")

    return filtered_indices
