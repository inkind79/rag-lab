"""
LLM handler utilities for common operations.

This module provides utility functions that are used by multiple LLM handlers to reduce code duplication.
"""
import os
import re
import logging
from typing import List, Dict, Any, Tuple, Optional, Union

logger = logging.getLogger(__name__)

def parse_conversation_history(conversation_context: str, message_format: str = 'generic') -> List[Dict[str, Any]]:
    """
    Parse a formatted conversation history string into a list of message objects.

    Handles conversation history with "User:" and "Assistant:" or "Assistant (Model):" prefixes
    and converts them into a standardized message format for different models.

    Args:
        conversation_context: String containing the formatted conversation history.
        message_format: Format to use for output messages ('generic', 'openai', 'ollama', 'gemini').
                        'generic' returns [{'role': role, 'content': content}, ...] for all handlers.

    Returns:
        List of message objects in the format appropriate for the specified handler type.
    """
    if not conversation_context or not conversation_context.strip():
        logger.info("Empty conversation context provided to parse_conversation_history.")
        return []

    try:
        formatted_messages = []
        current_role = None
        current_content = []
        lines = conversation_context.strip().split("\n")
        i = 0

        # Add system message based on message format
        if message_format == 'openai':
            formatted_messages.append({
                "role": "system",
                "content": "You are having a conversation with the user. The conversation history includes responses from different AI models (marked in parentheses). Be consistent with previous answers, even if they were given by different models."
            })

        while i < len(lines):
            line = lines[i].strip()
            if not line:  # Skip empty lines
                i += 1
                continue

            # Check for system message markers
            if line.startswith("System:"):
                if current_role and current_content:
                    formatted_messages.append({
                        "role": current_role,
                        "content": "\n".join(current_content).strip()
                    })
                current_role = "system"
                content_part = line[7:].strip()
                current_content = [content_part] if content_part else []
                logger.info(f"Found system prompt: {content_part[:50]}..." if len(content_part) > 50 else f"Found system prompt: {content_part}")

            # Check for user message markers
            elif line.startswith("User:"):
                if current_role and current_content:
                    formatted_messages.append({
                        "role": current_role,
                        "content": "\n".join(current_content).strip()
                    })
                current_role = "user"
                content_part = line[5:].strip()
                current_content = [content_part] if content_part else []

            # Check for assistant message markers
            elif line.startswith("Assistant"):
                if current_role and current_content:
                    formatted_messages.append({
                        "role": current_role,
                        "content": "\n".join(current_content).strip()
                    })
                current_role = "assistant"
                # Extract content after "Assistant:" or "Assistant (Model):"
                match = re.match(r'Assistant(?:\s*\([^)]*\))?\s*:\s*(.*)', line, re.DOTALL)
                content_part = match.group(1).strip() if match else ""
                current_content = [content_part] if content_part else []

            # Append continuation lines
            elif current_role is not None:
                current_content.append(line)

            i += 1

        # Add the last message
        if current_role and current_content:
            formatted_messages.append({
                "role": current_role,
                "content": "\n".join(current_content).strip()
            })

        logger.info(f"Successfully processed conversation into {len(formatted_messages)} messages")
        return formatted_messages

    except Exception as e:
        logger.warning(f"Error parsing conversation history: {e}")
        logger.warning("Falling back to simple format")

        # Fallback for different message formats
        if message_format == 'openai':
            return [
                {
                    "role": "system",
                    "content": f"Previous conversation history:\n{conversation_context}\n\nContinue the conversation based on this history."
                }
            ]
        else:
            # Return a single user message with the whole context
            return [{
                "role": "user",
                "content": f"Previous conversation history:\n{conversation_context}\n\nContinue the conversation based on this history."
            }]

def format_message_for_model(parsed_messages: List[Dict[str, Any]], model_type: str) -> Any:
    """
    Format parsed messages according to the specific requirements of different model APIs.

    This is a helper function that converts the generic message format to the specific format
    required by different LLM APIs.

    Args:
        parsed_messages: List of message objects with 'role' and 'content' fields.
        model_type: Model type ('openai', 'ollama', 'gemini').

    Returns:
        Formatted messages in the structure expected by the specified model API.
    """
    if model_type == 'openai':
        # OpenAI format is already the same as our generic format
        return parsed_messages

    elif model_type == 'ollama':
        # Ollama expects the same format
        return parsed_messages

    elif model_type == 'gemini':
        # Convert to Gemini format if needed
        # Note: Gemini might have specific requirements depending on the API version
        return parsed_messages

    else:
        logger.warning(f"Unknown model type for message formatting: {model_type}")
        return parsed_messages

def get_image_file_path(image_path: str, make_absolute: bool = False, verify_exists: bool = False, session_id: Optional[str] = None) -> str:
    """
    Standardize image paths across the application.

    Handles both relative and absolute paths, ensuring proper formatting
    with respect to the 'static' folder.

    Args:
        image_path: Original image path (could be relative or absolute)
        make_absolute: If True, convert to absolute path; if False, ensure relative to 'static'
        verify_exists: If True, verify the file exists and return None if it doesn't
        session_id: Optional session ID for logging or path construction

    Returns:
        Standardized image path or None if verify_exists=True and the file doesn't exist
    """
    try:
        # First, normalize path separators
        normalized_path = image_path.replace('\\', '/')

        # Determine if path is absolute or relative
        is_absolute = os.path.isabs(normalized_path)

        # Verify file exists if requested
        if verify_exists:
            if is_absolute:
                file_exists = os.path.exists(normalized_path)
            else:
                # For relative paths, check both with and without 'static/' prefix
                if normalized_path.startswith('static/'):
                    file_exists = os.path.exists(normalized_path)
                else:
                    file_exists = os.path.exists(os.path.join('static', normalized_path))

            if not file_exists:
                logger.warning(f"Image file does not exist: {normalized_path}")
                return None

        # Process based on whether we need absolute or relative path
        if is_absolute:
            # Path is already absolute
            if make_absolute:
                # Return as-is
                return normalized_path
            else:
                # Make it relative to static
                static_folder_abs = os.path.abspath('static')
                try:
                    rel_path = os.path.relpath(normalized_path, static_folder_abs).replace('\\', '/')
                    # Ensure it doesn't go up directories
                    if rel_path.startswith('../'):
                        # Fallback to using just the basename
                        logger.warning(f"Path {normalized_path} is outside static folder, using basename")
                        return os.path.basename(normalized_path)
                    return rel_path
                except Exception as e:
                    logger.warning(f"Error making path relative: {e}")
                    return os.path.basename(normalized_path)
        else:
            # Path is relative
            if normalized_path.startswith('static/'):
                # Remove 'static/' prefix for consistent relative paths
                clean_path = normalized_path.replace('static/', '', 1)
                if make_absolute:
                    # Convert to absolute with static prefix
                    return os.path.abspath(os.path.join('static', clean_path)).replace('\\', '/')
                else:
                    # Return without 'static/' prefix
                    return clean_path
            else:
                # Path is already relative without 'static/' prefix
                if make_absolute:
                    # Add 'static/' prefix and make absolute
                    return os.path.abspath(os.path.join('static', normalized_path)).replace('\\', '/')
                else:
                    # Return as-is (already relative without 'static/' prefix)
                    return normalized_path

    except Exception as e:
        logger.error(f"Error processing image path {image_path}: {e}")
        # Return original as fallback
        return image_path

def get_session_image_dir(session_id: str, create_if_missing: bool = True) -> str:
    """
    Get the path to the session's image directory.

    Args:
        session_id: Session identifier
        create_if_missing: If True, create the directory if it doesn't exist

    Returns:
        Absolute path to the session's image directory
    """
    session_images_folder = os.path.join('static', 'images', session_id)
    if create_if_missing:
        os.makedirs(session_images_folder, exist_ok=True)
    return os.path.abspath(session_images_folder).replace('\\', '/')

def normalize_path(path: str) -> str:
    """
    Normalize a path by replacing backslashes with forward slashes.

    Args:
        path: Path to normalize

    Returns:
        Normalized path with consistent separators
    """
    return path.replace('\\', '/')

def absolute_to_relative_path(abs_path: str, base_dir: str = 'static') -> str:
    """
    Convert an absolute path to a path relative to the specified base directory.

    Args:
        abs_path: Absolute path to convert
        base_dir: Base directory to make the path relative to (default: 'static')

    Returns:
        Relative path or original path with 'static/' prefix removed if conversion fails
    """
    try:
        # Ensure paths are normalized
        abs_path = normalize_path(abs_path)
        base_dir_abs = os.path.abspath(base_dir).replace('\\', '/')

        # Handle case where the path is already relative
        if not os.path.isabs(abs_path):
            # Remove 'static/' prefix if present for consistency
            if abs_path.startswith(f"{base_dir}/"):
                return abs_path[len(base_dir)+1:]
            return abs_path

        # Make the path relative
        rel_path = os.path.relpath(abs_path, base_dir_abs).replace('\\', '/')

        # Check if the relative path goes outside the base directory
        if rel_path.startswith('../'):
            logger.warning(f"Path {abs_path} is outside {base_dir} directory, using basename")
            return os.path.basename(abs_path)

        return rel_path

    except Exception as e:
        logger.error(f"Error converting absolute path to relative: {e}")
        # Fallback: try to strip 'static/' prefix if present
        if base_dir == 'static' and abs_path.startswith('static/'):
            return abs_path[7:]  # Strip 'static/'
        return os.path.basename(abs_path)  # Last resort: just use the filename

def get_image_paths_for_template(image_paths: List[str]) -> List[str]:
    """
    Convert a list of image paths to relative paths suitable for templates.

    This function processes a list of image paths (that may be absolute or relative,
    with or without 'static/' prefix) and converts them to consistent relative paths
    for use in HTML templates (relative to the 'static' folder, without 'static/' prefix).

    Args:
        image_paths: List of image paths to convert

    Returns:
        List of relative paths suitable for use in templates
    """
    relative_paths = []
    for path in image_paths:
        rel_path = absolute_to_relative_path(path)
        relative_paths.append(rel_path)
    return relative_paths
