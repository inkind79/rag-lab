"""
Template Parser Utilities for RAG Lab

This module provides utilities to extract, validate, and parse prompt templates
from LLM responses, which may not always be in perfect JSON format.
"""

import json
import re
import logging

logger = logging.getLogger(__name__)

def extract_structured_json(text):
    """
    Extract structured JSON from an LLM response text.
    Attempts multiple extraction strategies to parse JSON from various formats.
    
    Args:
        text (str): The text response from the LLM
        
    Returns:
        dict or None: The parsed JSON object or None if extraction fails
    """
    if not text:
        return None
    
    try:
        # Try direct JSON parse first
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Not valid JSON, proceed with extraction
            pass
        
        # Try to extract from structured-output format
        structured_match = re.search(r'<structured-output>([\s\S]*?)</structured-output>', text)
        if structured_match and structured_match.group(1):
            json_content = structured_match.group(1).strip()
            return json.loads(json_content)
        
        # Try to extract from code blocks
        codeblock_match = re.search(r'```(?:json)?\s*([\s\S]*?)```', text)
        if codeblock_match and codeblock_match.group(1):
            json_content = codeblock_match.group(1).strip()
            return json.loads(json_content)
        
        # Try to find JSON between { and }
        json_match = re.search(r'(\{[\s\S]*\})', text)
        if json_match:
            json_content = json_match.group(0).strip()
            return json.loads(json_content)
        
        return None
    except Exception as e:
        logger.error(f"Error extracting JSON from LLM response: {e}")
        return None

def validate_template(template):
    """
    Validates a template object to ensure it has all required fields.
    
    Args:
        template (dict): The template object to validate
        
    Returns:
        dict: An object with valid flag and error message if invalid
    """
    if not template:
        return {"valid": False, "error": "Template is null or undefined"}
    
    required_fields = ["name", "template_type", "system_prompt"]
    missing_fields = [field for field in required_fields 
                     if field not in template or not template[field]]
    
    if missing_fields:
        return {
            "valid": False,
            "error": f"Template missing required fields: {', '.join(missing_fields)}"
        }
    
    return {"valid": True}

def normalize_template(template):
    """
    Ensures a template has all the optional fields with default values if missing.
    
    Args:
        template (dict): The template object to normalize
        
    Returns:
        dict: The normalized template with all fields
    """
    if not template:
        return None
    
    normalized_template = template.copy()
    
    # Set default values for missing optional fields
    if "description" not in normalized_template or not normalized_template["description"]:
        normalized_template["description"] = f"Template: {normalized_template.get('name', 'Untitled')}"
    
    if "query_prefix" not in normalized_template:
        normalized_template["query_prefix"] = ""
    
    if "query_suffix" not in normalized_template:
        normalized_template["query_suffix"] = ""
    
    return normalized_template
