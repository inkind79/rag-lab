"""
Prompt Templates Module for RAG Lab

This module provides functionality for managing prompt templates used in the RAG system.
Templates can be created, retrieved, updated, and deleted, and are stored per user.
"""

import os
import json
import uuid
import logging
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

# Define the base directory for storing prompt templates
PROMPT_TEMPLATES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'prompt_templates')

# Ensure the directory exists
try:
    os.makedirs(PROMPT_TEMPLATES_DIR, exist_ok=True)
    logger.info(f"Prompt templates directory created/verified at: {PROMPT_TEMPLATES_DIR}")
except Exception as e:
    logger.error(f"Failed to create prompt templates directory: {e}")

# Define default prompt templates
DEFAULT_PROMPT_TEMPLATES = [
    {
        "id": "general-rag",
        "name": "General Purpose",
        "description": "General-purpose document analysis template",
        "is_default": True,
        "template_type": "general",
        "system_prompt": "You are a document analysis assistant. Analyze the provided documents and answer questions based only on their content.\n\nKey guidelines:\n- Only use information explicitly present in the documents\n- Never hallucinate or make up information\n- If information isn't in the documents, clearly state this fact\n- Be concise and clear in your responses\n- When appropriate, organize information using bullet points or lists",
        "query_prefix": "Based on the provided documents, ",
        "query_suffix": "Provide a detailed answer using information from the documents."
    },
    {
        "id": "document-summary",
        "name": "Document Summary",
        "description": "Summarize each selected document individually (batch mode)",
        "is_default": False,
        "template_type": "general",
        "system_prompt": "You are an expert document summarization assistant. Your task is to analyze the provided documents and create a comprehensive, well-structured summary that captures the essence of the content while maintaining accuracy and clarity.\n\nGUIDELINES TO PREVENT HALLUCINATIONS:\n- Your summary must be based EXCLUSIVELY on content that appears in the provided documents\n- NEVER add information from your general knowledge that doesn't appear in the documents\n- Do not make assumptions about the document's context, purpose, or missing information\n- If the documents contain unclear, ambiguous, or contradictory information, acknowledge this in your summary\n- Use phrases like \"the document states\" or \"according to the document\" to make clear you're reporting document content\n- Avoid drawing conclusions beyond what's explicitly stated in the documents\n- If the documents are incomplete or missing crucial information, note this in your summary rather than filling in gaps\n- Focus on capturing what IS in the document rather than speculating about what ISN'T\n\nSUMMARY STRUCTURE:\n- Start with a brief overview of the document type and purpose\n- Organize content into logical sections\n- Highlight key findings, data, or conclusions\n- Note any actionable items or recommendations\n- End with main takeaways",
        "query_prefix": "Based on all pages of the document, create a comprehensive summary. ",
        "query_suffix": "\n\nYour summary should include:\n1. Document Overview (type, purpose, scope)\n2. Main Topics and Key Points\n3. Important Findings or Data\n4. Conclusions and Recommendations\n5. Notable Insights or Implications\n\nEnsure the summary is well-organized, professional, and captures both details and the overall narrative.",
        "bypass_page_restriction": True,
        "hide_retrieved_pages": True
    }
]

def get_user_templates_file(user_id: str) -> str:
    """
    Get the file path for a user's prompt templates.

    Args:
        user_id: The ID of the user

    Returns:
        File path for the user's prompt templates
    """
    return os.path.join(PROMPT_TEMPLATES_DIR, f"{user_id}_templates.json")

def initialize_user_templates(user_id: str) -> bool:
    """
    Initialize prompt templates for a new user with default templates.

    Args:
        user_id: The ID of the user

    Returns:
        True if successful, False otherwise
    """
    templates_file = get_user_templates_file(user_id)

    # If the file already exists, don't overwrite it
    if os.path.exists(templates_file):
        return True

    try:
        # Create a copy of the default templates
        user_templates = DEFAULT_PROMPT_TEMPLATES.copy()

        # Save the templates
        with open(templates_file, 'w') as f:
            json.dump(user_templates, f, indent=2)

        logger.info(f"Initialized prompt templates for user {user_id}")
        return True
    except Exception as e:
        logger.error(f"Error initializing prompt templates for user {user_id}: {e}")
        return False

def get_user_templates(user_id: str) -> List[Dict[str, Any]]:
    """
    Get all prompt templates for a user.

    Args:
        user_id: The ID of the user

    Returns:
        List of prompt templates
    """
    # Make sure the templates directory exists
    os.makedirs(PROMPT_TEMPLATES_DIR, exist_ok=True)

    # Initialize templates if they don't exist
    templates_file = get_user_templates_file(user_id)
    if not os.path.exists(templates_file):
        logger.info(f"Initializing templates for user {user_id}")
        initialize_user_templates(user_id)

    try:
        with open(templates_file, 'r') as f:
            templates = json.load(f)
        logger.info(f"Loaded {len(templates)} templates for user {user_id}")
        return templates
    except Exception as e:
        logger.error(f"Error loading prompt templates for user {user_id}: {e}")
        # Return default templates as fallback
        logger.info(f"Returning default templates as fallback for user {user_id}")
        return DEFAULT_PROMPT_TEMPLATES.copy()

def get_template_by_id(user_id: str, template_id: str) -> Optional[Dict[str, Any]]:
    """
    Get a specific prompt template by ID.

    Args:
        user_id: The ID of the user
        template_id: The ID of the template to retrieve

    Returns:
        The template if found, None otherwise
    """
    templates = get_user_templates(user_id)

    for template in templates:
        if template.get('id') == template_id:
            return template

    return None

def get_default_template(user_id: str) -> Dict[str, Any]:
    """
    Get the default prompt template for a user.

    Args:
        user_id: The ID of the user

    Returns:
        The default template
    """
    templates = get_user_templates(user_id)

    # Find the template marked as default
    for template in templates:
        if template.get('is_default', False):
            return template

    # If no default is found, return the first template
    if templates:
        return templates[0]

    # If no templates exist, return the first default template
    return DEFAULT_PROMPT_TEMPLATES[0]

def create_template(user_id: str, template_data: Dict[str, Any]) -> Optional[str]:
    """
    Create a new prompt template for a user.

    Args:
        user_id: The ID of the user
        template_data: The template data

    Returns:
        The ID of the created template if successful, None otherwise
    """
    templates = get_user_templates(user_id)

    # Normalize template field names
    normalized_data = normalize_template_fields(template_data)

    # Generate a new ID if not provided
    if 'id' not in normalized_data:
        normalized_data['id'] = str(uuid.uuid4())

    # Add the new template
    templates.append(normalized_data)

    try:
        with open(get_user_templates_file(user_id), 'w') as f:
            json.dump(templates, f, indent=2)

        logger.info(f"Created prompt template {normalized_data['id']} for user {user_id}")
        return normalized_data['id']
    except Exception as e:
        logger.error(f"Error creating prompt template for user {user_id}: {e}")
        return None

def normalize_template_fields(template_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize template field names to ensure consistency.
    Converts hyphenated field names to standard field names.
    Automatically injects anti-hallucination guidance into system prompts for new templates.

    Args:
        template_data: The template data to normalize

    Returns:
        Normalized template data
    """
    # Define field name mappings (hyphenated to standard)
    field_mappings = {
        'template-id': 'id',
        'template-name': 'name',
        'template-type': 'template_type',
        'template-description': 'description',
        'system-prompt': 'system_prompt',
        'query-prefix': 'query_prefix',
        'query-suffix': 'query_suffix',
        'is-default': 'is_default'
    }

    # Create a new dict with normalized field names
    normalized_data = {}

    # First, copy any fields that are already using standard names
    for key, value in template_data.items():
        if key in ['id', 'name', 'template_type', 'description', 'system_prompt',
                  'query_prefix', 'query_suffix', 'is_default', 'optimized_query', 'optimization_info', 'optimization_run_id']:
            normalized_data[key] = value

    # Then, convert any hyphenated field names
    for old_key, new_key in field_mappings.items():
        if old_key in template_data and new_key not in normalized_data:
            normalized_data[new_key] = template_data[old_key]

    # Ensure required fields exist
    if 'id' not in normalized_data and 'template-id' in template_data:
        normalized_data['id'] = template_data['template-id']

    # Add default values for missing fields
    if 'description' not in normalized_data:
        normalized_data['description'] = 'Custom template'

    if 'query_prefix' not in normalized_data:
        normalized_data['query_prefix'] = ''

    if 'query_suffix' not in normalized_data:
        normalized_data['query_suffix'] = ''

    # Standard anti-hallucination text to add to system prompts
    # This ensures ALL new templates include anti-hallucination guidance
    anti_hallucination_text = """

Key guidelines:
- Only use information explicitly present in the documents
- Never hallucinate or make up information
- If information isn't in the documents, clearly state this fact
- Be concise and clear in your responses
"""

    # Check if system_prompt exists and doesn't already have anti-hallucination guidance
    if 'system_prompt' in normalized_data:
        system_prompt = normalized_data['system_prompt']

        # Check if the system prompt already contains anti-hallucination keywords
        has_anti_hallucination = any(keyword in system_prompt.lower() for keyword
                                     in ['hallucination', 'make up', 'invent', 'never guess'])

        # If it doesn't have anti-hallucination guidance, add it
        if not has_anti_hallucination:
            normalized_data['system_prompt'] = system_prompt + anti_hallucination_text
            logger.info("Added anti-hallucination guidance to new template system prompt")
    else:
        # If no system prompt exists, create one with anti-hallucination guidance
        normalized_data['system_prompt'] = "You are a document analysis assistant." + anti_hallucination_text
        logger.info("Created system prompt with anti-hallucination guidance for new template")

    # Ensure query_prefix has a meaningful value if it's empty
    if 'query_prefix' in normalized_data and not normalized_data['query_prefix'].strip():
        normalized_data['query_prefix'] = "Based on the provided documents, "
        logger.info("Added default query prefix")

    # Add a reminder in the query suffix if it doesn't already have anti-hallucination keywords
    # or similar instructions about using document information
    if 'query_suffix' in normalized_data:
        query_suffix = normalized_data['query_suffix']
        # Check for anti-hallucination keywords or phrases about using document information
        if not any(keyword in query_suffix.lower() for keyword
                   in ['hallucination', 'make up', 'base your answer on', 'only use information',
                       'using information from the document', 'from the documents', 'based on the document']):
            # Only add the suffix if it's not empty and doesn't end with punctuation
            if query_suffix.strip():
                # Add a space or newline depending on whether the suffix ends with punctuation
                if query_suffix.rstrip()[-1] in ['.', '!', '?', ':', ';']:
                    normalized_data['query_suffix'] = query_suffix.rstrip() + " Provide a detailed answer using information from the documents."
                else:
                    normalized_data['query_suffix'] = query_suffix.rstrip() + ". Provide a detailed answer using information from the documents."
            else:
                normalized_data['query_suffix'] = "Provide a detailed answer using information from the documents."
            logger.info("Added concise reminder to query suffix")

    logger.debug(f"Normalized template fields: {normalized_data.keys()}")
    return normalized_data

def update_template(user_id: str, template_id: str, template_data: Dict[str, Any]) -> bool:
    """
    Update an existing prompt template.

    Args:
        user_id: The ID of the user
        template_id: The ID of the template to update
        template_data: The updated template data

    Returns:
        True if successful, False otherwise
    """
    templates = get_user_templates(user_id)

    # Normalize template field names
    normalized_data = normalize_template_fields(template_data)

    # Ensure the ID is preserved
    normalized_data['id'] = template_id

    # Find the template to update
    for i, template in enumerate(templates):
        if template.get('id') == template_id:
            # Preserve is_default if it exists in the original template but not in the update
            if 'is_default' in template and 'is_default' not in normalized_data:
                normalized_data['is_default'] = template['is_default']

            # Update the template
            templates[i] = normalized_data

            try:
                with open(get_user_templates_file(user_id), 'w') as f:
                    json.dump(templates, f, indent=2)

                logger.info(f"Updated prompt template {template_id} for user {user_id}")
                return True
            except Exception as e:
                logger.error(f"Error updating prompt template {template_id} for user {user_id}: {e}")
                return False

    logger.warning(f"Template {template_id} not found for user {user_id}")
    return False

def delete_template(user_id: str, template_id: str) -> bool:
    """
    Delete a prompt template.

    Args:
        user_id: The ID of the user
        template_id: The ID of the template to delete

    Returns:
        True if successful, False otherwise
    """
    templates = get_user_templates(user_id)

    # Find the template to delete
    for i, template in enumerate(templates):
        if template.get('id') == template_id:
            # Check if it's the default template or a protected template
            if template.get('is_default', False):
                logger.warning(f"Cannot delete default template {template_id} for user {user_id}")
                return False
            # Prevent deletion of Document Summary template
            if template.get('id') == 'document-summary' or template.get('name') == 'Document Summary':
                logger.warning(f"Cannot delete Document Summary template {template_id} for user {user_id}")
                return False

            # Remove the template
            templates.pop(i)

            try:
                with open(get_user_templates_file(user_id), 'w') as f:
                    json.dump(templates, f, indent=2)

                logger.info(f"Deleted prompt template {template_id} for user {user_id}")
                return True
            except Exception as e:
                logger.error(f"Error deleting prompt template {template_id} for user {user_id}: {e}")
                return False

    logger.warning(f"Template {template_id} not found for user {user_id}")
    return False

def set_default_template(user_id: str, template_id: str) -> bool:
    """
    Set a template as the default for a user.

    Args:
        user_id: The ID of the user
        template_id: The ID of the template to set as default

    Returns:
        True if successful, False otherwise
    """
    templates = get_user_templates(user_id)

    # Find the template to set as default
    template_found = False

    for template in templates:
        if template.get('id') == template_id:
            template['is_default'] = True
            template_found = True
        else:
            template['is_default'] = False

    if not template_found:
        logger.warning(f"Template {template_id} not found for user {user_id}")
        return False

    try:
        with open(get_user_templates_file(user_id), 'w') as f:
            json.dump(templates, f, indent=2)

        logger.info(f"Set template {template_id} as default for user {user_id}")
        return True
    except Exception as e:
        logger.error(f"Error setting default template for user {user_id}: {e}")
        return False

def update_system_default_template() -> None:
    """
    Updates the default template for all existing users to match the current system default.

    This ensures all users have the latest default template, even if they already have templates.
    """
    try:
        # Get the current system default template
        system_default = next((t for t in DEFAULT_PROMPT_TEMPLATES if t.get('is_default', False)), DEFAULT_PROMPT_TEMPLATES[0])
        system_default_id = system_default.get('id')

        logger.info(f"Updating all users with system default template: {system_default_id}")

        # Get all user template files
        template_files = [f for f in os.listdir(PROMPT_TEMPLATES_DIR) if f.endswith('_templates.json')]

        for template_file in template_files:
            try:
                # Extract user_id from filename
                user_id = template_file.replace('_templates.json', '')

                # Load user templates
                user_templates = get_user_templates(user_id)

                # Check if the system default template exists for this user
                system_template_exists = any(t.get('id') == system_default_id for t in user_templates)

                if system_template_exists:
                    # If it exists, make sure it's set as default
                    for template in user_templates:
                        if template.get('id') == system_default_id:
                            template['is_default'] = True
                            # Update template content to match current system default
                            template.update({
                                'name': system_default['name'],
                                'description': system_default['description'],
                                'template_type': system_default['template_type'],
                                'system_prompt': system_default['system_prompt'],
                                'query_prefix': system_default['query_prefix'],
                                'query_suffix': system_default['query_suffix']
                            })
                        else:
                            template['is_default'] = False
                else:
                    # If it doesn't exist, add it and set as default
                    new_template = system_default.copy()
                    new_template['is_default'] = True

                    # Set all other templates as non-default
                    for template in user_templates:
                        template['is_default'] = False

                    # Add the new template
                    user_templates.append(new_template)

                # Save updated templates
                with open(get_user_templates_file(user_id), 'w') as f:
                    json.dump(user_templates, f, indent=2)

                logger.info(f"Updated default template for user {user_id}")

            except Exception as e:
                logger.error(f"Error updating templates for user file {template_file}: {e}")
                continue

        logger.info("Completed system default template update for all users")
    except Exception as e:
        logger.error(f"Error in update_system_default_template: {e}")



def update_all_user_templates() -> None:
    """
    Updates all user templates to ensure system default templates are current.
    This function will:
    1. Update existing system templates (general-rag, document-summary) to match current versions
    2. Add any missing system templates
    3. Preserve all user-created templates (custom and AI-generated)
    4. Maintain user's default template selection
    """
    try:
        # Get the current system templates
        system_templates = DEFAULT_PROMPT_TEMPLATES.copy()
        system_template_ids = {t['id'] for t in system_templates}

        # Get all user template files
        template_files = [f for f in os.listdir(PROMPT_TEMPLATES_DIR) if f.endswith('_templates.json')]

        for template_file in template_files:
            try:
                # Extract user_id from filename
                user_id = template_file.replace('_templates.json', '')

                # Load user templates
                user_templates = []
                try:
                    with open(get_user_templates_file(user_id), 'r') as f:
                        user_templates = json.load(f)
                except Exception as e:
                    logger.error(f"Error loading templates for user {user_id}: {e}")
                    continue

                # Find the current default template for this user
                default_template = next((t for t in user_templates if t.get('is_default', False)), None)

                # Create updated templates list
                updated_templates = []
                user_template_ids = set()

                # First, process existing user templates
                for user_template in user_templates:
                    template_id = user_template.get('id')
                    user_template_ids.add(template_id)

                    if template_id in system_template_ids:
                        # This is a system template - update it to current version
                        system_template = next(t for t in system_templates if t['id'] == template_id)
                        updated_template = system_template.copy()

                        # Preserve user's default status if this was their default
                        if user_template.get('is_default', False):
                            updated_template['is_default'] = True

                        updated_templates.append(updated_template)
                        logger.debug(f"Updated system template {template_id} for user {user_id}")
                    else:
                        # This is a user-created template - preserve it as-is
                        updated_templates.append(user_template)
                        logger.debug(f"Preserved user template {template_id} for user {user_id}")

                # Add any missing system templates
                for system_template in system_templates:
                    if system_template['id'] not in user_template_ids:
                        new_template = system_template.copy()
                        # Don't make it default unless user has no default template
                        if not default_template and system_template.get('is_default', False):
                            new_template['is_default'] = True
                        else:
                            new_template['is_default'] = False

                        updated_templates.append(new_template)
                        logger.debug(f"Added missing system template {system_template['id']} for user {user_id}")

                # Save updated templates
                with open(get_user_templates_file(user_id), 'w') as f:
                    json.dump(updated_templates, f, indent=2)

                logger.info(f"Updated templates for user {user_id}: preserved {len(updated_templates) - len(system_templates)} user templates, updated {len(system_templates)} system templates")

            except Exception as e:
                logger.error(f"Error updating templates for user file {template_file}: {e}")
                continue

        logger.info("Completed update of all user templates - user templates preserved")
    except Exception as e:
        logger.error(f"Error in update_all_user_templates: {e}")


def get_session_template(user_id: str, session_id: str, session_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get the prompt template for a specific session.

    Args:
        user_id: The ID of the user
        session_id: The ID of the session
        session_data: The session data

    Returns:
        The prompt template for the session
    """
    # Log the session ID for debugging purposes
    logger.debug(f"Getting template for session {session_id}")
    # Check if the session has a selected template
    template_id = session_data.get('selected_template_id')

    if template_id:
        # Try to get the selected template
        template = get_template_by_id(user_id, template_id)
        if template:
            return template

    # Fall back to the default template
    return get_default_template(user_id)


def load_user_templates(user_id: str) -> List[Dict[str, Any]]:
    """
    Load user templates (alias for get_user_templates for compatibility).

    Args:
        user_id: The ID of the user

    Returns:
        List of prompt templates for the user
    """
    return get_user_templates(user_id)


def save_user_template(user_id: str, template_data: Dict[str, Any]) -> bool:
    """
    Save a new template or update an existing one for a user.

    Args:
        user_id: The ID of the user
        template_data: The template data to save

    Returns:
        True if successful, False otherwise
    """
    try:
        # Check if template already exists
        template_id = template_data.get('id')
        if template_id and get_template_by_id(user_id, template_id):
            # Update existing template
            return update_template(user_id, template_id, template_data)
        else:
            # Create new template
            result = create_template(user_id, template_data)
            return result is not None
    except Exception as e:
        logger.error(f"Error saving template for user {user_id}: {e}")
        return False
