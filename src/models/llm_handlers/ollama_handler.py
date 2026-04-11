# models/llm_handlers/ollama_handler.py
import os
import json
from typing import List, Tuple, Optional, Any, Dict
import ollama
from PIL import Image
import re # Import re
import requests # Import requests for raw API calls

from .base_handler import BaseLLMHandler
from src.utils.logger import get_logger
from src.utils.llm_utils import parse_conversation_history, get_image_file_path, get_session_image_dir
# Assuming ocr_retriever might be needed for OCR formatting, adjust if moved
# from src.models.ocr_retriever import format_ocr_context

logger = get_logger(__name__)

from src.utils.logger import get_logger

logger = get_logger(__name__)

# --- Configuration Loading ---
CONFIG_FILE_PATH = "config/model_configs.json"
# Models that do NOT support multiple images and need the collage workaround.
# Everything else defaults to multi-image (the modern standard).
OLLAMA_SINGLE_IMAGE_MODELS: list = []
OLLAMA_TEXT_ONLY_MODELS = []

def _load_ollama_config():
    """Loads single-image model denylist from config."""
    global OLLAMA_SINGLE_IMAGE_MODELS
    try:
        if os.path.exists(CONFIG_FILE_PATH):
            with open(CONFIG_FILE_PATH, 'r') as f:
                config_data = json.load(f)
            OLLAMA_SINGLE_IMAGE_MODELS = config_data.get("single_image_models", [])
            logger.info(f"Loaded config: {len(OLLAMA_SINGLE_IMAGE_MODELS)} single-image models (collage required).")
        else:
            logger.warning(f"Config not found at {CONFIG_FILE_PATH}. Defaulting to multi-image for all models.")
            OLLAMA_SINGLE_IMAGE_MODELS = []
    except Exception as e:
        logger.error(f"Error loading config from {CONFIG_FILE_PATH}: {e}", exc_info=True)
        OLLAMA_SINGLE_IMAGE_MODELS = []

# Load config when the module is imported
_load_ollama_config()

# Define models known to be text-only (can still be hardcoded or loaded from config)
# Ensure consistency if loading from config above
OLLAMA_TEXT_ONLY_MODELS = ['ollama-phi4', 'ollama-phi4-mini', 'ollama-phi4-mini-fp16', 'ollama-phi4-mini-reasoning', 'ollama-phi4-mini-reasoning-fp16', 'ollama-phi4-mini-reasoning:3.8b-fp16', 'ollama-phi4-q8', 'ollama-phi4-fp16', 'ollama-phi4-reasoning:plus', 'ollama-phi4-mini-reasoning:latest', 'ollama-phi4-reasoning:14b-plus-q8_0', 'ollama-phi4-reasoning:14b-plus-fp16', 'ollama-olmo2', 'ollama-qwq', 'ollama-exaone-deep:32b']


class OllamaHandler(BaseLLMHandler):
    """
    Handler for Ollama models (vision and text-only).
    """

    def __init__(self, model_choice: str, model_obj: Any, session_model_params: Dict):
        """
        Initialize the handler.

        Args:
            model_choice: The specific model identifier (e.g., 'ollama-llama-vision').
            model_obj: Tuple containing (_, _, _, ollama_model_name string).
                       The first three elements are ignored for Ollama handler.
            session_model_params: The 'model_params' dictionary loaded from the user's session/defaults.
        """
        # Pass session_model_params up to the base class initializer
        super().__init__(model_choice, model_obj, session_model_params)
        # Unpack the model object - Ollama loader returns a tuple, we need the 4th element
        if isinstance(model_obj, tuple) and len(model_obj) >= 4:
            # The actual model name used by the ollama library API
            self.api_model_name: str = model_obj[3]
        else:
            logger.error(f"OllamaHandler received unexpected model_obj format: {model_obj}. Expected tuple with at least 4 elements.")
            # Fallback: try to extract from model_choice if it follows 'ollama-...' pattern
            if model_choice.startswith('ollama-'):
                self.api_model_name = model_choice.split('-', 1)[1] # e.g., 'llama-vision'
                logger.warning(f"Using fallback model name from model_choice: {self.api_model_name}")
            else:
                raise ValueError("Ollama model_obj format incorrect and cannot infer model name.")

        # Determine if the model is text-only based on the original model_choice
        self.is_text_only = model_choice in OLLAMA_TEXT_ONLY_MODELS

        # Default to multi-image (modern standard). Only use collage for known single-image models.
        self.supports_multiple_images = self.api_model_name not in OLLAMA_SINGLE_IMAGE_MODELS
        if self.supports_multiple_images:
            logger.info(f"Model {self.api_model_name} supports multiple images directly.")
        else:
            logger.info(f"Model {self.api_model_name} requires collage for multiple images.")


    def _create_collage(self, image_paths: List[str], session_id: str) -> Optional[str]:
        """Creates a vertical collage from a list of image paths.

        This is especially important for models like Llama 3.2 Vision that don't support multiple images.
        The function will combine all images into a single vertical collage.

        Args:
            image_paths: List of paths to images to include in the collage
            session_id: The session ID to use for the collage filename

        Returns:
            Path to the collage image, or None if collage creation failed
        """
        if not image_paths:
            logger.warning("No image paths provided for collage creation")
            return None

        # Get the session image directory using utility
        session_dir = get_session_image_dir(session_id)
        collage_path = os.path.join(session_dir, f"temp_collage_{session_id}.jpg").replace('\\', '/')

        logger.info(f"Creating collage for {len(image_paths)} images in session {session_id}")

        try:
            opened_images = []
            for img_path in image_paths:
                # Ensure path is absolute and verify it exists
                abs_path = get_image_file_path(img_path, make_absolute=True, verify_exists=True)

                if abs_path:
                    try:
                        img = Image.open(abs_path).convert('RGB')
                        opened_images.append(img)
                    except Exception as e_open:
                        logger.error(f"Failed to open image {abs_path} for collage: {e_open}")
                else:
                    logger.warning(f"Image file not found or invalid path: {img_path}")

            if not opened_images:
                logger.error("Could not open any images to create collage.")
                return None

            if len(opened_images) == 1:
                 # No need for collage, return the single image path directly
                 # However, the current logic expects a collage path, so we save it anyway
                 # opened_images[0].save(collage_path) # Save the single image to the expected path
                 # logger.info(f"Only one image provided, saving it to collage path: {collage_path}")
                 # return collage_path
                 # --- OR --- return the original path if downstream handles it
                 logger.info(f"Only one image provided: {image_paths[0]}")
                 return image_paths[0] # Return the original path for single image


            # Calculate the size of the collage
            max_width = max(img.width for img in opened_images)
            total_height = sum(img.height for img in opened_images)

            # Create a new image with the total size
            collage = Image.new('RGB', (max_width, total_height))

            # Paste the images into the collage
            y_offset = 0
            for img in opened_images:
                collage.paste(img, (0, y_offset))
                y_offset += img.height

            collage.save(collage_path)
            logger.info(f"Created collage of {len(opened_images)} images at {collage_path}")
            return collage_path

        except Exception as e:
            logger.error(f"Error creating collage: {e}", exc_info=True)
            # Fallback: maybe return the first image path? Or None?
            if opened_images:
                 logger.warning(f"Collage creation failed, falling back to first image: {image_paths[0]}")
                 return image_paths[0] # Fallback to first image path
            return None


    def _get_ocr_context(self, image_paths: List[str], direct_ocr_results: Optional[Dict[str, Any]], session_id: str, use_ocr: bool) -> str:
        """Extracts or formats OCR context using the base handler's prepare_ocr_context method."""
        return self.prepare_ocr_context(
            image_paths=image_paths,
            direct_ocr_results=direct_ocr_results,
            session_id=session_id,
            use_ocr=use_ocr
        )


    def _prepare_ollama_messages(self, query: str, conversation_context: str, image_paths_for_api: List[str], ocr_context: str, img_count: int) -> List[Dict[str, Any]]:
        """Prepares the list of messages for the Ollama API call."""
        formatted_messages = []

        # 1. Parse Conversation History using utility function
        if conversation_context:
            # Check if this is a summarized conversation (starts with "Previous conversation summary:")
            if conversation_context.strip().startswith("Previous conversation summary:"):
                # This is a summarized conversation, add it as a system message
                formatted_messages.append({
                    "role": "system",
                    "content": conversation_context.strip()
                })
                logger.info(f"Added summarized conversation as system message: {conversation_context[:100]}...")
            else:
                # This is a regular conversation history, parse it normally
                parsed_messages = parse_conversation_history(conversation_context, message_format='ollama')

                # Log the parsed messages for debugging
                logger.info(f"===== PARSED CONVERSATION MESSAGES =====")
                for i, msg in enumerate(parsed_messages):
                    role = msg.get('role', 'unknown')
                    content = msg.get('content', '')
                    # Log the full content for testing prompt engineering
                    logger.info(f"Message {i+1}: Role={role}")
                    logger.info(f"Content: {content}")
                logger.info(f"===== END PARSED MESSAGES =====")

                # Check if the last message is a user message with the same content as the current query
                # If so, we don't need to add the current query again later
                has_current_query = False
                if parsed_messages and len(parsed_messages) > 0:
                    last_msg = parsed_messages[-1]
                    if last_msg.get('role') == 'user' and last_msg.get('content') == query:
                        has_current_query = True
                        logger.info(f"Current query already present in conversation history, will not add it again")

                formatted_messages.extend(parsed_messages)
                logger.info(f"Processed conversation into {len(parsed_messages)} Ollama messages using utility function")

        # 2. Check if we have a system message and handle it specially
        system_message = None
        for msg in formatted_messages:
            if msg.get('role') == 'system':
                system_message = msg
                formatted_messages.remove(msg)
                logger.info(f"Found system message: {msg.get('content', '')[:50]}..." if len(msg.get('content', '')) > 50 else f"Found system message: {msg.get('content', '')}")
                break

        # If we have a system message, add it as the first message
        if system_message:
            # Ollama expects system messages in a specific format
            system_content = system_message.get('content', '')
            formatted_messages.insert(0, {
                "role": "system",
                "content": system_content
            })
            logger.info(f"===== SYSTEM MESSAGE =====")
            logger.info(f"{system_content}")
            logger.info(f"===== END SYSTEM MESSAGE =====")
            logger.info(f"Added system message as the first message in the conversation")

        # Check if we need to add the current query
        # If the last message in formatted_messages is already the current query, we don't need to add it again
        has_current_query = False
        if formatted_messages and len(formatted_messages) > 0:
            last_msg = formatted_messages[-1]
            if last_msg.get('role') == 'user' and last_msg.get('content') == query:
                has_current_query = True
                logger.info(f"Current query already present as last message, will not add it again")

        # Only add the current query if it's not already present
        if not has_current_query:
            # 3. Construct Current User Message (Query + Image + OCR)
            # Base query content
            current_content = query

            # Enhance content based on images and OCR
            if not self.is_text_only and image_paths_for_api:
                # Vision model with image(s)
                display_img_count = len(image_paths_for_api) if len(image_paths_for_api) == img_count else img_count # Use actual count if collage failed/skipped
                image_desc = f"I'm showing you {display_img_count} image(s). " if display_img_count > 0 else ""
                current_content = f"{image_desc}{query}"
                if ocr_context:
                    current_content += f"\n\nHere is extracted text from the image(s):\n{ocr_context}\n\nPlease use both the image(s) and the text."
                    logger.info("Enhanced Ollama vision prompt with OCR text context")
                else:
                     logger.info("Using Ollama vision prompt without OCR text.")
            elif self.is_text_only and ocr_context:
                # Text-only model with OCR
                current_content = f"{query}\n\nHere's the extracted text from the documents:\n\n{ocr_context}\n\nPlease use this text to provide a detailed answer based on the extracted content only."
                logger.info("Using text-only model with OCR context")
            elif self.is_text_only:
                 # Text-only model without OCR
                 logger.info("Using text-only model without OCR context.")
                 current_content = query # Just the query
            else:
                 # Vision model but no images provided/processed (shouldn't happen if called correctly)
                 logger.warning("Ollama vision model called without valid images?")
                 current_content = query


            # Special handling for specific models (e.g., Granite)
            if self.api_model_name == "granite3.2-vision" and not self.is_text_only:
                granite_prefix = "The image contains a document that you need to analyze. "
                if "document" not in current_content.lower() and "image" in current_content.lower():
                     current_content = current_content.replace("image(s)", "document(s)", 1)
                if not current_content.startswith(granite_prefix):
                     current_content = granite_prefix + current_content
                logger.warning(f"Enhanced Granite Vision prompt: {current_content[:150]}...")


            # 3. Create the final message object
            current_message = {
                "role": "user",
                "content": current_content
            }
            # Add images only if it's not a text-only model and we have image paths
            if not self.is_text_only and image_paths_for_api:
                current_message["images"] = image_paths_for_api
                img_basenames = [p.split('/')[-1] if '/' in p else p for p in image_paths_for_api]
                logger.info(f"SENDING TO OLLAMA ({self.api_model_name}): images {img_basenames} with content {len(current_content)} chars")
                logger.info(f"===== FINAL PROMPT TO LLM =====")
                logger.info(f"Model: {self.api_model_name}")
                logger.info(f"Images: {img_basenames}")
                logger.info(f"Content:\n{current_content}")

                # Log all messages being sent to the LLM for complete visibility
                logger.info(f"===== COMPLETE MESSAGE SEQUENCE =====")
                for i, msg in enumerate(formatted_messages):
                    role = msg.get('role', 'unknown')
                    content = msg.get('content', '')
                    logger.info(f"Message {i+1}: Role={role}")
                    logger.info(f"Content: {content}")
                    if 'images' in msg:
                        logger.info(f"Images: {msg['images']}")
                # Add the current message that will be appended
                logger.info(f"Message {len(formatted_messages)+1}: Role=user")
                logger.info(f"Content: {current_content}")
                logger.info(f"Images: {img_basenames}")
                logger.info(f"===== END COMPLETE MESSAGE SEQUENCE =====")

                logger.info(f"===== END FINAL PROMPT =====")
            else:
                 logger.info(f"SENDING TO OLLAMA ({self.api_model_name}): text-only content {len(current_content)} chars")
                 logger.info(f"===== FINAL PROMPT TO LLM =====")
                 logger.info(f"Model: {self.api_model_name}")
                 logger.info(f"Content:\n{current_content}")

                 # Log all messages being sent to the LLM for complete visibility
                 logger.info(f"===== COMPLETE MESSAGE SEQUENCE =====")
                 for i, msg in enumerate(formatted_messages):
                     role = msg.get('role', 'unknown')
                     content = msg.get('content', '')
                     logger.info(f"Message {i+1}: Role={role}")
                     logger.info(f"Content: {content}")
                 # Add the current message that will be appended
                 logger.info(f"Message {len(formatted_messages)+1}: Role=user")
                 logger.info(f"Content: {current_content}")
                 logger.info(f"===== END COMPLETE MESSAGE SEQUENCE =====")

                 logger.info(f"===== END FINAL PROMPT =====")

            # Add the current message to the formatted messages
            formatted_messages.append(current_message)

        return formatted_messages


    def _is_thinking_mode_requested(self, query: str) -> bool:
        """
        Detects if the user is requesting thinking mode in their query.

        Args:
            query: The user's query

        Returns:
            bool: True if thinking mode is requested, False otherwise
        """
        thinking_phrases = [
            "thinking mode",
            "show your thinking",
            "show your work",
            "think step by step",
            "think through this",
            "show me your reasoning",
            "show reasoning",
            "show your reasoning",
            "explain your thinking",
            "think aloud",
            "use reasoning"
        ]

        # Check if any of the thinking phrases are in the query (case insensitive)
        query_lower = query.lower()
        for phrase in thinking_phrases:
            if phrase in query_lower:
                logger.info(f"Thinking mode requested: Found phrase '{phrase}' in query")
                return True

        return False

    def _handle_granite_streaming(self, query: str, conversation_context: str, options: dict):
        """
        Handle granite streaming with raw HTTP API and control message.
        Yields reasoning progressively instead of one shot.
        """
        logger.info(f"Starting granite streaming for {self.api_model_name}")

        # Prepare messages with control message
        messages = []
        messages.append({"role": "control", "content": "thinking"})

        # Parse conversation history if provided
        if conversation_context:
            if conversation_context.strip().startswith("Previous conversation summary:"):
                messages.append({"role": "system", "content": conversation_context.strip()})
            else:
                parsed_messages = parse_conversation_history(conversation_context, message_format='ollama')
                if parsed_messages:
                    # Check if last message is current query
                    last_msg = parsed_messages[-1] if parsed_messages else None
                    if last_msg and last_msg.get('role') == 'user' and last_msg.get('content') == query:
                        # Remove duplicate query
                        parsed_messages = parsed_messages[:-1]
                    messages.extend(parsed_messages)

        # Add current query
        messages.append({"role": "user", "content": query})

        # Prepare payload for raw HTTP API
        payload = {
            "model": self.api_model_name,
            "messages": messages,
            "options": options,
            "stream": True
        }

        try:
            import requests
            from src.utils.ollama_client import get_ollama_url, get_ollama_headers
            _model = payload.get("model", "")
            _url = f"{get_ollama_url(_model)}/api/chat"
            _headers = get_ollama_headers(_model)
            response = requests.post(_url, json=payload, headers=_headers, stream=True, timeout=1800)

            if response.status_code == 200:
                buffer = ""
                in_thinking = False
                in_response = False

                for line in response.iter_lines():
                    if line:
                        try:
                            chunk = json.loads(line)
                            if 'message' in chunk and 'content' in chunk['message']:
                                content = chunk['message']['content']
                                buffer += content

                                # Process buffer for thinking and response tags
                                while True:
                                    processed = False

                                    # Check for <think> tag
                                    if '<think>' in buffer and not in_thinking:
                                        think_start = buffer.find('<think>') + len('<think>')
                                        buffer = buffer[think_start:]
                                        in_thinking = True
                                        processed = True
                                        continue

                                    # Check for </think> tag
                                    if in_thinking and '</think>' in buffer:
                                        think_end = buffer.find('</think>')
                                        thinking_content = buffer[:think_end]
                                        if thinking_content:  # Don't strip - preserve whitespace as per Ollama guide
                                            yield {'type': 'reasoning', 'content': thinking_content}
                                        buffer = buffer[think_end + len('</think>'):]
                                        in_thinking = False
                                        processed = True
                                        continue

                                    # Check for <response> tag
                                    if '<response>' in buffer and not in_response:
                                        response_start = buffer.find('<response>') + len('<response>')
                                        buffer = buffer[response_start:]
                                        in_response = True
                                        processed = True
                                        continue

                                    # Check for </response> tag
                                    if in_response and '</response>' in buffer:
                                        response_end = buffer.find('</response>')
                                        response_content = buffer[:response_end]
                                        if response_content:  # Don't strip - preserve whitespace as per Ollama guide
                                            yield {'type': 'response', 'content': response_content}
                                        buffer = buffer[response_end + len('</response>'):]
                                        in_response = False
                                        processed = True
                                        continue

                                    # Yield thinking content progressively
                                    if in_thinking and len(buffer) > 50:
                                        # Look for natural break points
                                        last_break = max(
                                            buffer.rfind('\n'),
                                            buffer.rfind('. '),
                                            buffer.rfind('? '),
                                            buffer.rfind('! ')
                                        )
                                        if last_break > 0:
                                            chunk_content = buffer[:last_break + 1]
                                            yield {'type': 'reasoning', 'content': chunk_content}
                                            buffer = buffer[last_break + 1:]
                                            processed = True

                                    # Yield response content progressively
                                    if in_response and len(buffer) > 20:
                                        # Look for natural break points
                                        last_break = max(
                                            buffer.rfind('\n'),
                                            buffer.rfind('. '),
                                            buffer.rfind('? '),
                                            buffer.rfind('! ')
                                        )
                                        if last_break > 0:
                                            chunk_content = buffer[:last_break + 1]
                                            yield {'type': 'response', 'content': chunk_content}
                                            buffer = buffer[last_break + 1:]
                                            processed = True

                                    if not processed:
                                        break

                        except json.JSONDecodeError:
                            logger.warning(f"Failed to decode granite streaming chunk: {line}")
                            continue

                # Yield any remaining content
                if buffer.strip():
                    if in_thinking:
                        yield {'type': 'reasoning', 'content': buffer.strip()}
                    elif in_response:
                        yield {'type': 'response', 'content': buffer.strip()}
                    else:
                        # If no tags, treat as response
                        yield {'type': 'response', 'content': buffer.strip()}

                # Signal that reasoning is complete and should be cleared
                yield {'type': 'reasoning_complete'}
                yield {'type': 'complete'}

            else:
                logger.error(f"Granite raw API error: {response.status_code} - {response.text}")
                yield {'type': 'error', 'error': f'Granite API error: {response.status_code}'}

        except Exception as e:
            logger.error(f"Error in granite streaming: {e}")
            yield {'type': 'error', 'error': f'Granite streaming error: {str(e)}'}

    def _remove_phi4_reasoning(self, text):
        """
        Remove content inside <think>...</think> tags from Phi-4 reasoning models.
        Works with both properly closed tags and unclosed tags.

        Args:
            text: The text to process

        Returns:
            Text with reasoning content removed
        """
        if not text or not isinstance(text, str):
            return text

        original_length = len(text)

        # First try the standard approach for properly closed tags
        if '<think>' in text and '</think>' in text:
            cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)

        # Handle the case where there's an opening tag but no closing tag
        # (as seen with phi4-reasoning:plus)
        elif '<think>' in text:
            # Return everything before the first <think> tag
            parts = text.split('<think>', 1)
            cleaned = parts[0] if len(parts) > 1 else text
        else:
            # No think tags found, return the original text
            return text

        # Clean up any extra whitespace
        cleaned = cleaned.strip()

        # Log the change
        new_length = len(cleaned)
        if original_length != new_length:
            logger.info(f"Removed Phi-4 reasoning content: {original_length} chars -> {new_length} chars")

        return cleaned

    def _handle_granite_thinking_mode(self, query: str, conversation_context: str = "") -> str:
        """
        Handles thinking mode for Granite 3.3 models using raw API calls.

        Args:
            query: The user's query
            conversation_context: Optional conversation history

        Returns:
            str: The generated response with thinking process removed
        """
        logger.info(f"--- OllamaHandler: Handling Granite thinking mode for {self.api_model_name} ---")

        # Prepare messages with thinking control message
        messages = []

        # Add thinking control message
        messages.append({"role": "control", "content": "thinking"})

        # Initialize has_current_query to False
        has_current_query = False

        # Parse conversation history if provided
        if conversation_context:
            # Check if this is a summarized conversation (starts with "Previous conversation summary:")
            if conversation_context.strip().startswith("Previous conversation summary:"):
                # This is a summarized conversation, add it as a system message
                messages.append({
                    "role": "system",
                    "content": conversation_context.strip()
                })
                logger.info(f"Added summarized conversation as system message: {conversation_context[:100]}...")
            else:
                # This is a regular conversation history, parse it normally
                parsed_messages = parse_conversation_history(conversation_context, message_format='ollama')

                # Check if the last message is already the current query
                if parsed_messages and len(parsed_messages) > 0:
                    last_msg = parsed_messages[-1]
                    if last_msg.get('role') == 'user' and last_msg.get('content') == query:
                        has_current_query = True
                        logger.info(f"Current query already present in conversation history, will not add it again")

                # Add all messages from the conversation history
                for msg in parsed_messages:
                    messages.append(msg)
                logger.info(f"Added {len(parsed_messages)} messages from conversation history")

        # Add user message only if it's not already the last message in the conversation history
        if not has_current_query:
            messages.append({"role": "user", "content": query})

        # Get API options
        options = self._get_ollama_options()

        # Prepare payload
        payload = {
            "model": self.api_model_name,
            "messages": messages,
            "options": options
        }

        try:
            logger.info(f"Making raw API call to Ollama for {self.api_model_name} with thinking mode")
            from src.utils.ollama_client import get_ollama_url, get_ollama_headers
            _url = f"{get_ollama_url(self.api_model_name)}/api/chat"
            _headers = get_ollama_headers(self.api_model_name)
            response = requests.post(_url, json=payload, headers=_headers, stream=True)

            if response.status_code == 200:
                full_response = ""
                for line in response.iter_lines():
                    if line:
                        try:
                            chunk = json.loads(line)
                            if 'message' in chunk and 'content' in chunk['message']:
                                full_response += chunk['message']['content']
                        except json.JSONDecodeError:
                            logger.error(f"Error decoding JSON: {line}")

                logger.info(f"Raw response from Granite thinking mode: {len(full_response)} chars")

                # Extract thinking and response parts
                thinking = ""
                response_text = ""

                # Try to extract thinking and response parts
                if "<think>" in full_response and "</think>" in full_response:
                    thinking_start = full_response.find("<think>") + len("<think>")
                    thinking_end = full_response.find("</think>")
                    thinking = full_response[thinking_start:thinking_end].strip()
                    logger.info(f"Extracted thinking process: {len(thinking)} chars")

                # Check for response tags
                if "<response>" in full_response and "</response>" in full_response:
                    response_start = full_response.find("<response>") + len("<response>")
                    response_end = full_response.find("</response>")
                    response_text = full_response[response_start:response_end].strip()
                    logger.info(f"Extracted final response: {len(response_text)} chars")
                    return response_text
                else:
                    # If no response tags found, try to extract the response in a different way
                    logger.warning(f"No <response> tags found in Granite thinking mode response")

                    # If we can't extract a clean response, apply regex to remove thinking blocks
                    pattern_remove = r"<think>.*?</think>"
                    cleaned_text = re.sub(pattern_remove, '', full_response, flags=re.DOTALL | re.MULTILINE).strip()
                    if cleaned_text != full_response:
                        logger.info(f"Removed thinking block using regex: {len(cleaned_text)} chars")
                        return cleaned_text

                    # If all else fails, return the full response
                    logger.warning(f"Could not extract thinking or response parts from Granite thinking mode response. Returning full response.")
                    return full_response
            else:
                error_msg = f"Error in Granite thinking mode API call: {response.status_code} - {response.text}"
                logger.error(error_msg)
                return error_msg
        except Exception as e:
            error_msg = f"Exception in Granite thinking mode API call: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return error_msg

    def _get_ollama_options(self) -> Dict[str, Any]:
        """Returns model-specific options from the loaded config for the Ollama API call."""
        # Log the full configuration for debugging
        logger.info(f"Full config for {self.api_model_name} before processing: {self.config}")

        # Extract relevant options from self.config loaded by BaseLLMHandler
        # Define the keys Ollama API accepts in the 'options' dictionary
        valid_ollama_options = {
            "num_keep", "seed", "num_predict", "top_k", "top_p",
            "tfs_z", "typical_p", "repeat_last_n", "temperature",
            "repeat_penalty", "presence_penalty", "frequency_penalty",
            "mirostat", "mirostat_tau", "mirostat_eta", "penalize_newline",
            "stop", "numa", "num_ctx", "num_batch", "num_gpu", "main_gpu",
            "low_vram", "f16_kv", "vocab_only", "use_mmap", "use_mlock",
            "num_thread"
        }

        # Log which valid options are present in the config
        present_options = {k: self.config[k] for k in valid_ollama_options if k in self.config}
        logger.info(f"Valid options found in config: {present_options}")

        options = {}
        for key, value in self.config.items():
            if key in valid_ollama_options:
                options[key] = value
                logger.info(f"Setting option {key} = {value} from config")

        # Set num_gpu to -1 to ensure all available GPU memory is used, but don't override other settings
        if "num_gpu" not in options:
            options["num_gpu"] = -1
            logger.info("Adding default num_gpu = -1 (use all available GPU memory)")
        else:
            logger.info(f"Using num_gpu = {options['num_gpu']} from config")

        # Log the final options being used
        logger.info(f"Final Ollama options for {self.api_model_name}: {options}")
        return options


    def generate_response(
        self,
        images: List[str], # Validated full paths
        query: str,
        session_id: str,
        chat_history: Optional[List[Dict[str, Any]]] = None,
        direct_ocr_results: Optional[Dict[str, Any]] = None,
        original_query: Optional[str] = None,
        use_ocr: bool = False,
        conversation_context: str = "",
        use_score_slope: bool = False,
        retrieved_text_context: str = ""
    ) -> Tuple[str, List[str]]:
        """
        Generates a response using Ollama models. Handles collage creation.
        """
        logger.info(f"--- OllamaHandler: Generating response for session {session_id} using {self.model_choice} ({self.api_model_name}) ---")

        # Determine images to use based on retrieval count (fetch from session settings)
        retrieval_count = 3 # Default
        try:
            session_file = os.path.join('sessions', f"{session_id}.json")
            if os.path.exists(session_file):
                with open(session_file, 'r') as f:
                    session_data = json.load(f)

                # Check if this is a summary template
                is_summary_template = False
                template_id = session_data.get('selected_template_id')
                if template_id:
                    # Try to load the template to check if it's a summary template
                    user_id = session_data.get('user_id')
                    if user_id:
                        try:
                            from src.models.prompt_templates import get_template_by_id
                            template = get_template_by_id(user_id, template_id)
                            if template and (template.get('template_type') == 'summary' or 
                                           template.get('id') == 'document-summary' or 
                                           template.get('name') == 'Document Summary'):
                                is_summary_template = True
                                logger.info(f"Summary template detected - using all available images")
                        except Exception as template_e:
                            logger.warning(f"Error checking template type: {template_e}")

                # Check if token budget filtering has been applied and score-slope is enabled
                token_budget_analysis = session_data.get('token_budget_analysis', None)

                # Use the use_score_slope parameter passed to the function
                # If it's not provided, fall back to the session data
                session_use_score_slope = session_data.get('use_score_slope', False)
                effective_use_score_slope = use_score_slope or session_use_score_slope
                logger.info(f"Score-slope enabled: {effective_use_score_slope} (param: {use_score_slope}, session: {session_use_score_slope})")

                # If score-slope is enabled, use all images provided
                # These have already been filtered by relevance and token budget
                if effective_use_score_slope:
                    retrieval_count = len(images)
                    logger.info(f"Using all {retrieval_count} images from score-slope analysis")
                elif token_budget_analysis:
                    # Token budget filtering has been applied, use all images provided
                    retrieval_count = len(images)
                    logger.info(f"Using all {retrieval_count} images from token budget filtering")
                elif is_summary_template:
                    # Use all images up to a reasonable limit (e.g., 20)
                    retrieval_count = min(len(images), 20)
                    logger.info(f"Using summary template retrieval count: {retrieval_count}")
                else:
                    # Use the configured retrieval count from session
                    retrieval_count = int(session_data.get('retrieval_count', 3))
                    logger.info(f"Using user-configured retrieval count: {retrieval_count}")
            else:
                logger.info(f"No session file found, using default retrieval count: {retrieval_count}")
        except Exception as e:
            logger.warning(f"Error loading retrieval count from session: {e}, using default: {retrieval_count}")

        # Select the images based on the count
        images_to_process = images[:min(len(images), retrieval_count)]
        img_count = len(images_to_process)
        logger.info(f"Attempting to process {img_count} images for Ollama.")

        # Prepare image input for the API
        image_paths_for_api = []
        if not self.is_text_only and images_to_process:
            if self.supports_multiple_images:
                # Use original paths directly if model supports multiple images
                image_paths_for_api = images_to_process
                logger.info(f"Sending {len(image_paths_for_api)} image paths directly to Ollama {self.api_model_name}.")
            else:
                # Use collage approach for models that don't support multiple images (like Llama 3.2 Vision)
                logger.info(f"Model {self.api_model_name} doesn't support multiple images directly - creating collage from {len(images_to_process)} images")
                collage_or_single_path = self._create_collage(images_to_process, session_id)
                if collage_or_single_path:
                    # If _create_collage returned a single path (because only one image was input),
                    # or if it returned a collage path, wrap it in a list for the API.
                    image_paths_for_api = [collage_or_single_path]
                    if "temp_collage_" in collage_or_single_path:
                         logger.info(f"SUCCESS: Created and using collage image for Ollama {self.api_model_name}: {collage_or_single_path}")
                    else:
                         logger.info(f"Using single image for Ollama {self.api_model_name} (no collage needed): {collage_or_single_path}")

                else:
                    logger.error("Failed to prepare image input (collage or single image).")
                    # Return original images attempted, even if processing failed
                    return "Error preparing image input for Ollama.", images_to_process

        # Get OCR context - will wait for results if OCR is enabled
        ocr_context_str = self._get_ocr_context(images_to_process, direct_ocr_results, session_id, use_ocr)

        # Check if this is a Granite 3.3 model and enable thinking mode by default
        is_granite = self.api_model_name in ['granite3.3', 'granite3.3:2b', 'granite3.3:8b']
        thinking_requested = self._is_thinking_mode_requested(query)

        # For Granite models, always use thinking mode to show reasoning
        # But only for text-only mode - thinking mode doesn't work with images yet
        if is_granite and not image_paths_for_api:
            logger.info(f"Using special Granite thinking mode for {self.api_model_name} in generate_response")
            response_text = self._handle_granite_thinking_mode(query, conversation_context)
            return response_text, images_to_process

        # Prepare messages payload
        messages = self._prepare_ollama_messages(query, conversation_context, image_paths_for_api, ocr_context_str, img_count)

        # Get API options
        options = self._get_ollama_options()

        try:
            # Create a client with a longer timeout (30 minutes)
            client = ollama.Client(timeout=1800.0)

            # Make the API call with the client
            response = client.chat(
                model=self.api_model_name,
                messages=messages,
                options=options
            )

            generated_text = response.get('message', {}).get('content', '')
            logger.info(f"Raw response generated using Ollama {self.api_model_name} model.")

            # --- Parse and remove reasoning for specific models ---
            if self.api_model_name in ['exaone-deep:32b', 'qwq', 'qwen3:30b-a3b', 'qwen3:30b-a3b-q8_0', 'qwen3:30b-a3b-fp16', 'deepcoder:14b-preview-q8_0', 'deepcoder:14b-preview-fp16', 'granite3.3', 'granite3.3:2b', 'granite3.3:8b', 'phi4', 'phi4:14b-q8_0', 'phi4:14b-fp16', 'phi4-mini-reasoning', 'phi4-mini-reasoning:3.8b-fp16', 'phi4-reasoning:plus', 'phi4-mini-reasoning:latest', 'phi4-reasoning:14b-plus-q8_0', 'phi4-reasoning:14b-plus-fp16', 'deepseek-r1:8b-0528-qwen3-q4_K_M', 'deepseek-r1:8b-0528-qwen3-q8_0', 'deepseek-r1:8b-0528-qwen3-fp16', 'magistral:24b', 'magistral:24b-small-2506-q8_0', 'magistral:24b-small-2506-fp16']:
                # First check for <response> tags
                response_pattern = r"<response>(.*?)</response>"
                response_match = re.search(response_pattern, generated_text, re.DOTALL)
                if response_match:
                    # Extract just the content inside the response tags
                    generated_text = response_match.group(1).strip()
                    logger.info(f"Extracted content from <response> tags in {self.api_model_name} response.")

                # Remove \boxed{} patterns from Phi-4 Mini Reasoning and Magistral models
                if ('phi4-mini-reasoning' in self.api_model_name or self.api_model_name == 'phi4-mini-reasoning:latest' or self.model_choice == 'ollama-phi4-mini-reasoning:latest' or
                    self.api_model_name in ['magistral:24b', 'magistral:24b-small-2506-q8_0', 'magistral:24b-small-2506-fp16']):
                    # For Magistral models, completely remove \boxed{} patterns as they often duplicate content
                    if self.api_model_name in ['magistral:24b', 'magistral:24b-small-2506-q8_0', 'magistral:24b-small-2506-fp16']:
                        # Remove the entire \boxed{} expression and its content
                        # Use a more comprehensive pattern that handles nested braces and \text{}
                        # This pattern matches \boxed{ followed by any content (including \text{}) until the matching }
                        boxed_comprehensive_pattern = r"\\boxed\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}"
                        generated_text = re.sub(boxed_comprehensive_pattern, '', generated_text, flags=re.DOTALL | re.MULTILINE)
                        
                        # Also try a simpler pattern for any remaining \boxed
                        simple_boxed_pattern = r"\\boxed\{.*?\}"
                        generated_text = re.sub(simple_boxed_pattern, '', generated_text, flags=re.DOTALL)
                        
                        # Clean up any extra whitespace or newlines left behind
                        generated_text = re.sub(r'\n\s*\n', '\n\n', generated_text)
                        generated_text = generated_text.strip()
                        
                        logger.info(f"Removed \\boxed{{}} patterns from Magistral model response.")
                    else:
                        # For Phi-4 models, extract content from \boxed{} pattern
                        boxed_pattern = r"\\boxed\{(.*?)\}"
                        boxed_match = re.search(boxed_pattern, generated_text, re.DOTALL)
                        if boxed_match:
                            # Extract just the content inside the boxed tags
                            boxed_content = boxed_match.group(1).strip()
                            # Replace ~ with spaces (common in boxed content)
                            boxed_content = boxed_content.replace('~', ' ')
                            logger.info(f"Extracted and cleaned content from \\boxed{{}} in {self.api_model_name} response.")
                            # Replace the \boxed{} expression with its content
                            generated_text = re.sub(boxed_pattern, boxed_content, generated_text, flags=re.DOTALL)

                    # Also try to find and remove any markdown code blocks with reasoning
                    markdown_reasoning_pattern = r"```(?:reasoning|thinking|thought|rationale)[\s\S]*?```"
                    generated_text = re.sub(markdown_reasoning_pattern, '', generated_text, flags=re.DOTALL)

                    # Also try to remove any text between "Reasoning:" and "Answer:" or "Response:"
                    reasoning_section_pattern = r"(?:Reasoning|Rationale|Thinking):\s*.*?(?=(?:Answer|Response|Final Answer|Output):)"
                    generated_text = re.sub(reasoning_section_pattern, '', generated_text, flags=re.DOTALL | re.IGNORECASE)

                # Use the dedicated method to remove Phi-4 reasoning content
                if 'phi4-reasoning' in self.api_model_name or 'phi4-mini-reasoning' in self.api_model_name:
                    original_text = generated_text
                    generated_text = self._remove_phi4_reasoning(generated_text)
                    if generated_text != original_text:
                        logger.info(f"Used _remove_phi4_reasoning method to clean {self.api_model_name} response.")
                    else:
                        logger.warning(f"_remove_phi4_reasoning did not find any reasoning content to remove.")

                # For other models, still use the standard reasoning tag removal
                else:
                    # Regex to remove reasoning blocks and surrounding whitespace
                    # Matches <thought>...</thought> OR <think>...</think>, including content
                    pattern_remove = r"\s*<(?:thought|think)>[\s\S]*?</(?:thought|think)>\s*"
                    cleaned_text = re.sub(pattern_remove, '', generated_text, flags=re.DOTALL | re.MULTILINE).strip()

                    if cleaned_text != generated_text:
                        logger.info(f"Removed reasoning block from {self.api_model_name} response.")
                        generated_text = cleaned_text
            # --- End parsing ---


            # IMPORTANT: Return the original images that were intended for processing,
            # not the collage path.
            logger.info(f"Ollama response generated. Returning original image paths intended: {images_to_process}")
            return generated_text, images_to_process

        except Exception as e:
            logger.error(f"Error during Ollama API call: {str(e)}", exc_info=True)
            # Return the original images attempted
            return f"An error occurred while processing the request with Ollama {self.api_model_name}: {str(e)}", images_to_process


    def handle_direct_chat(
        self,
        query: str,
        conversation_context: str = ""
    ) -> Tuple[str, List[str]]:
        """
        Handles direct chat using Ollama models (no images).
        """
        logger.info(f"--- OllamaHandler: Handling direct chat using {self.model_choice} ({self.api_model_name}) ---")

        # Check if this is a Granite 3.3 model and thinking mode is requested
        is_granite = self.api_model_name in ['granite3.3', 'granite3.3:2b', 'granite3.3:8b']
        thinking_requested = self._is_thinking_mode_requested(query)

        if is_granite and thinking_requested:
            logger.info(f"Using special Granite thinking mode for {self.api_model_name}")
            response_text = self._handle_granite_thinking_mode(query, conversation_context)
            return response_text, [] # No images used in direct chat

        # Standard flow for non-Granite models or when thinking mode is not requested
        # Prepare messages payload (no images, no OCR needed for direct chat)
        messages = self._prepare_ollama_messages(query, conversation_context, [], "", 0)

        # Get API options
        options = self._get_ollama_options()

        try:
            # Create a client with a longer timeout (30 minutes)
            client = ollama.Client(timeout=1800.0)

            # Make the API call with the client
            response = client.chat(
                model=self.api_model_name,
                messages=messages,
                options=options
            )
            generated_text = response.get('message', {}).get('content', '')
            logger.info(f"Raw direct chat response generated using Ollama {self.api_model_name}.")

            # --- Parse and remove reasoning for specific models ---
            if self.api_model_name in ['exaone-deep:32b', 'qwq', 'qwen3:30b-a3b', 'qwen3:30b-a3b-q8_0', 'qwen3:30b-a3b-fp16', 'deepcoder:14b-preview-q8_0', 'deepcoder:14b-preview-fp16', 'granite3.3', 'granite3.3:2b', 'granite3.3:8b', 'phi4', 'phi4:14b-q8_0', 'phi4:14b-fp16', 'phi4-mini-reasoning', 'phi4-mini-reasoning:3.8b-fp16', 'phi4-reasoning:plus', 'phi4-mini-reasoning:latest', 'phi4-reasoning:14b-plus-q8_0', 'phi4-reasoning:14b-plus-fp16', 'deepseek-r1:8b-0528-qwen3-q4_K_M', 'deepseek-r1:8b-0528-qwen3-q8_0', 'deepseek-r1:8b-0528-qwen3-fp16', 'magistral:24b', 'magistral:24b-small-2506-q8_0', 'magistral:24b-small-2506-fp16']:
                # First check for <response> tags
                response_pattern = r"<response>(.*?)</response>"
                response_match = re.search(response_pattern, generated_text, re.DOTALL)
                if response_match:
                    # Extract just the content inside the response tags
                    generated_text = response_match.group(1).strip()
                    logger.info(f"Extracted content from <response> tags in {self.api_model_name} direct chat response.")

                # Remove \boxed{} patterns from Phi-4 Mini Reasoning and Magistral models
                if ('phi4-mini-reasoning' in self.api_model_name or self.api_model_name == 'phi4-mini-reasoning:latest' or self.model_choice == 'ollama-phi4-mini-reasoning:latest' or
                    self.api_model_name in ['magistral:24b', 'magistral:24b-small-2506-q8_0', 'magistral:24b-small-2506-fp16']):
                    # For Magistral models, completely remove \boxed{} patterns as they often duplicate content
                    if self.api_model_name in ['magistral:24b', 'magistral:24b-small-2506-q8_0', 'magistral:24b-small-2506-fp16']:
                        # Remove the entire \boxed{} expression and its content
                        # Use a more comprehensive pattern that handles nested braces and \text{}
                        # This pattern matches \boxed{ followed by any content (including \text{}) until the matching }
                        boxed_comprehensive_pattern = r"\\boxed\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}"
                        generated_text = re.sub(boxed_comprehensive_pattern, '', generated_text, flags=re.DOTALL | re.MULTILINE)
                        
                        # Also try a simpler pattern for any remaining \boxed
                        simple_boxed_pattern = r"\\boxed\{.*?\}"
                        generated_text = re.sub(simple_boxed_pattern, '', generated_text, flags=re.DOTALL)
                        
                        # Clean up any extra whitespace or newlines left behind
                        generated_text = re.sub(r'\n\s*\n', '\n\n', generated_text)
                        generated_text = generated_text.strip()
                        
                        logger.info(f"Removed \\boxed{{}} patterns from Magistral model direct chat response.")
                    else:
                        # For Phi-4 models, extract content from \boxed{} pattern
                        boxed_pattern = r"\\boxed\{(.*?)\}"
                        boxed_match = re.search(boxed_pattern, generated_text, re.DOTALL)
                        if boxed_match:
                            # Extract just the content inside the boxed tags
                            boxed_content = boxed_match.group(1).strip()
                            # Replace ~ with spaces (common in boxed content)
                            boxed_content = boxed_content.replace('~', ' ')
                            logger.info(f"Extracted and cleaned content from \\boxed{{}} in {self.api_model_name} direct chat response.")
                            # Replace the \boxed{} expression with its content
                            generated_text = re.sub(boxed_pattern, boxed_content, generated_text, flags=re.DOTALL)

                    # Also try to find and remove any markdown code blocks with reasoning
                    markdown_reasoning_pattern = r"```(?:reasoning|thinking|thought|rationale)[\s\S]*?```"
                    generated_text = re.sub(markdown_reasoning_pattern, '', generated_text, flags=re.DOTALL)

                    # Also try to remove any text between "Reasoning:" and "Answer:" or "Response:"
                    reasoning_section_pattern = r"(?:Reasoning|Rationale|Thinking):\s*.*?(?=(?:Answer|Response|Final Answer|Output):)"
                    generated_text = re.sub(reasoning_section_pattern, '', generated_text, flags=re.DOTALL | re.IGNORECASE)

                # Use the dedicated method to remove Phi-4 reasoning content
                if 'phi4-reasoning' in self.api_model_name or 'phi4-mini-reasoning' in self.api_model_name:
                    original_text = generated_text
                    generated_text = self._remove_phi4_reasoning(generated_text)
                    if generated_text != original_text:
                        logger.info(f"Used _remove_phi4_reasoning method to clean {self.api_model_name} direct chat response.")
                    else:
                        logger.warning(f"_remove_phi4_reasoning did not find any reasoning content to remove.")

                # For other models, still use the standard reasoning tag removal
                else:
                    # Regex to remove reasoning blocks and surrounding whitespace
                    # Matches <thought>...</thought> OR <think>...</think>, including content
                    pattern_remove = r"\s*<(?:thought|think)>[\s\S]*?</(?:thought|think)>\s*"
                    cleaned_text = re.sub(pattern_remove, '', generated_text, flags=re.DOTALL | re.MULTILINE).strip()

                    if cleaned_text != generated_text:
                        logger.info(f"Removed reasoning block from {self.api_model_name} direct chat response.")
                        generated_text = cleaned_text
            # --- End parsing ---

            return generated_text, [] # No images used in direct chat
        except Exception as e:
            logger.error(f"Error during Ollama direct chat API call: {str(e)}", exc_info=True)
            return f"An error occurred during direct chat with Ollama {self.api_model_name}: {str(e)}", []

    def handle_direct_chat_with_format(
        self,
        query: str,
        conversation_context: str = "",
        format_schema: Dict = None
    ) -> str:
        """
        Handles direct chat using Ollama models with format parameter for structured output.

        Args:
            query: The query to send to the model
            conversation_context: Optional conversation history
            format_schema: JSON schema for structured output

        Returns:
            str: The generated response
        """
        logger.info(f"--- OllamaHandler: Handling direct chat with format using {self.model_choice} ({self.api_model_name}) ---")
        logger.info(f"Using format parameter: {format_schema}")

        # Check if the model supports format parameter
        # All models should support format parameter for template generation
        format_supported = True

        if not format_supported:
            logger.warning(f"Model {self.api_model_name} does not support format parameter. Using standard direct chat.")
            response, _ = self.handle_direct_chat(query, conversation_context)
            return response

        # Check if this is a Granite 3.3 model and thinking mode is requested
        is_granite = self.api_model_name in ['granite3.3', 'granite3.3:2b', 'granite3.3:8b']
        thinking_requested = self._is_thinking_mode_requested(query)

        # For Granite models with thinking mode, we need to use raw API calls
        if is_granite and thinking_requested:
            logger.info(f"Using special Granite thinking mode for {self.api_model_name} in handle_direct_chat_with_format")
            response_text = self._handle_granite_thinking_mode(query, conversation_context)
            return response_text

        # Prepare messages payload (no images, no OCR needed for direct chat)
        messages = self._prepare_ollama_messages(query, conversation_context, [], "", 0)

        # Get API options
        options = self._get_ollama_options()

        try:
            # Create a client with a longer timeout (30 minutes)
            client = ollama.Client(timeout=1800.0)

            # Using structured logging format - format schema is captured here
            logger.info(f"Using Ollama with format schema for {self.api_model_name}")
            # Note: Message content is already logged in _prepare_ollama_messages method

            # Make the API call with the client and format parameter
            response = client.chat(
                model=self.api_model_name,
                messages=messages,
                options=options,
                format=format_schema  # Add format parameter
            )

            generated_text = response.get('message', {}).get('content', '')
            logger.info(f"Raw formatted response generated using Ollama {self.api_model_name}.")

            # --- Parse and remove reasoning for specific models ---
            if self.api_model_name in ['exaone-deep:32b', 'qwq', 'qwen3:30b-a3b', 'qwen3:30b-a3b-q8_0', 'qwen3:30b-a3b-fp16', 'deepcoder:14b-preview-q8_0', 'deepcoder:14b-preview-fp16', 'granite3.3', 'granite3.3:2b', 'phi4', 'phi4:14b-q8_0', 'phi4:14b-fp16', 'phi4-mini-reasoning', 'phi4-mini-reasoning:3.8b-fp16', 'phi4-reasoning:plus', 'phi4-mini-reasoning:latest', 'phi4-reasoning:14b-plus-q8_0', 'phi4-reasoning:14b-plus-fp16', 'deepseek-r1:8b-0528-qwen3-q4_K_M', 'deepseek-r1:8b-0528-qwen3-q8_0', 'deepseek-r1:8b-0528-qwen3-fp16', 'magistral:24b', 'magistral:24b-small-2506-q8_0', 'magistral:24b-small-2506-fp16']:
                # First check for <response> tags
                response_pattern = r"<response>(.*?)</response>"
                response_match = re.search(response_pattern, generated_text, re.DOTALL)
                if response_match:
                    # Extract just the content inside the response tags
                    generated_text = response_match.group(1).strip()
                    logger.info(f"Extracted content from <response> tags in {self.api_model_name} formatted response.")

                # Remove \boxed{} patterns from Phi-4 Mini Reasoning and Magistral models
                if ('phi4-mini-reasoning' in self.api_model_name or self.api_model_name == 'phi4-mini-reasoning:latest' or
                    self.api_model_name in ['magistral:24b', 'magistral:24b-small-2506-q8_0', 'magistral:24b-small-2506-fp16']):
                    # For Magistral models, completely remove \boxed{} patterns as they often duplicate content
                    if self.api_model_name in ['magistral:24b', 'magistral:24b-small-2506-q8_0', 'magistral:24b-small-2506-fp16']:
                        # Remove the entire \boxed{} expression and its content
                        # Use a more comprehensive pattern that handles nested braces and \text{}
                        # This pattern matches \boxed{ followed by any content (including \text{}) until the matching }
                        boxed_comprehensive_pattern = r"\\boxed\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}"
                        generated_text = re.sub(boxed_comprehensive_pattern, '', generated_text, flags=re.DOTALL | re.MULTILINE)
                        
                        # Also try a simpler pattern for any remaining \boxed
                        simple_boxed_pattern = r"\\boxed\{.*?\}"
                        generated_text = re.sub(simple_boxed_pattern, '', generated_text, flags=re.DOTALL)
                        
                        # Clean up any extra whitespace or newlines left behind
                        generated_text = re.sub(r'\n\s*\n', '\n\n', generated_text)
                        generated_text = generated_text.strip()
                        
                        logger.info(f"Removed \\boxed{{}} patterns from Magistral model formatted response.")
                    else:
                        # For Phi-4 models, extract content from \boxed{} pattern
                        boxed_pattern = r"\\boxed\{(.*?)\}"
                        boxed_match = re.search(boxed_pattern, generated_text, re.DOTALL)
                        if boxed_match:
                            # Extract just the content inside the boxed tags
                            boxed_content = boxed_match.group(1).strip()
                            # Replace ~ with spaces (common in boxed content)
                            boxed_content = boxed_content.replace('~', ' ')
                            logger.info(f"Extracted and cleaned content from \\boxed{{}} in {self.api_model_name} formatted response.")
                            # Replace the \boxed{} expression with its content
                            generated_text = re.sub(boxed_pattern, boxed_content, generated_text, flags=re.DOTALL)

                    # Also try to find and remove any markdown code blocks with reasoning
                    markdown_reasoning_pattern = r"```(?:reasoning|thinking|thought|rationale)[\s\S]*?```"
                    generated_text = re.sub(markdown_reasoning_pattern, '', generated_text, flags=re.DOTALL)

                    # Also try to remove any text between "Reasoning:" and "Answer:" or "Response:"
                    reasoning_section_pattern = r"(?:Reasoning|Rationale|Thinking):\s*.*?(?=(?:Answer|Response|Final Answer|Output):)"
                    generated_text = re.sub(reasoning_section_pattern, '', generated_text, flags=re.DOTALL | re.IGNORECASE)

                # Use the dedicated method to remove Phi-4 reasoning content
                if 'phi4-reasoning' in self.api_model_name or 'phi4-mini-reasoning' in self.api_model_name:
                    original_text = generated_text
                    generated_text = self._remove_phi4_reasoning(generated_text)
                    if generated_text != original_text:
                        logger.info(f"Used _remove_phi4_reasoning method to clean {self.api_model_name} formatted response.")
                    else:
                        logger.warning(f"_remove_phi4_reasoning did not find any reasoning content to remove.")

                # For other models, still use the standard reasoning tag removal
                else:
                    # Regex to remove reasoning blocks and surrounding whitespace
                    # Matches <thought>...</thought> OR <think>...</think>, including content
                    pattern_remove = r"\s*<(?:thought|think)>[\s\S]*?</(?:thought|think)>\s*"
                    cleaned_text = re.sub(pattern_remove, '', generated_text, flags=re.DOTALL | re.MULTILINE).strip()

                    if cleaned_text != generated_text:
                        logger.info(f"Removed reasoning block from {self.api_model_name} formatted response.")
                        generated_text = cleaned_text
            # --- End parsing ---

            return generated_text

        except Exception as e:
            logger.error(f"Error during Ollama direct chat with format API call: {str(e)}", exc_info=True)
            # Fall back to standard direct chat
            logger.warning(f"Falling back to standard direct chat without format parameter")
            response, _ = self.handle_direct_chat(query, conversation_context)
            return response

    def generate_streaming_response(
        self,
        images: List[str],
        query: str,
        session_id: str,
        chat_history: Optional[List[Dict[str, Any]]] = None,
        conversation_context: str = "",
        direct_ocr_results: Optional[Dict[str, Any]] = None,
        original_query: Optional[str] = None,
        use_ocr: bool = False,
        use_score_slope: bool = False,
        is_pasted_images: bool = False,
        retrieved_text_context: str = ""
    ):
        """
        Generate a streaming response with reasoning display support.

        Yields chunks in the format:
        - {'type': 'reasoning', 'content': '...'} for reasoning content
        - {'type': 'response', 'content': '...'} for actual response
        """
        logger.info(f"--- OllamaHandler: Starting streaming response for {self.api_model_name} ---")

        # Get API options early for granite models
        options = self._get_ollama_options()

        # Check if this is a Granite model and enable thinking mode by default
        # Always use special handling for granite models to show reasoning
        is_granite = self.api_model_name in ['granite3.3', 'granite3.3:2b', 'granite3.3:8b']
        thinking_requested = self._is_thinking_mode_requested(query)

        # For Granite models, we need to use raw HTTP API with control message
        # This will stream reasoning progressively instead of one shot
        if is_granite and not images:
            logger.info(f"Granite thinking mode enabled for {self.api_model_name} - using streaming with control message")
            logger.info(f"Using granite reasoning for query: {query}")

            # Use special granite streaming handler
            for chunk_data in self._handle_granite_streaming(query, conversation_context, options):
                yield chunk_data
            return

        # Check if this is a reasoning model
        # NOTE: DeepCoder IS a code reasoning model per the documentation
        reasoning_models = [
            'exaone-deep:32b', 'qwq', 'qwen3:30b-a3b', 'qwen3:30b-a3b-q8_0', 'qwen3:30b-a3b-fp16',
            'deepcoder:14b-preview-q8_0', 'deepcoder:14b-preview-fp16',  # Code reasoning models
            'granite3.3', 'granite3.3:2b', 'granite3.3:8b', 'granite3.3:latest',  # Granite reasoning models
            'phi4', 'phi4:14b-q8_0', 'phi4:14b-fp16',
            'phi4-mini-reasoning', 'phi4-mini-reasoning:3.8b-fp16',
            'phi4-reasoning:plus', 'phi4-mini-reasoning:latest',
            'phi4-reasoning:14b-plus-q8_0', 'phi4-reasoning:14b-plus-fp16',
            'deepseek-r1:8b', 'deepseek-r1:latest',  # Base DeepSeek R1 models
            'deepseek-r1:8b-0528-qwen3-q4_K_M', 'deepseek-r1:8b-0528-qwen3-q8_0', 'deepseek-r1:8b-0528-qwen3-fp16',  # DeepSeek R1 reasoning models
            'magistral:24b', 'magistral:24b-small-2506-q8_0', 'magistral:24b-small-2506-fp16'  # Mistral Magistral reasoning models
        ]

        is_reasoning_model = self.api_model_name in reasoning_models

        # Determine images to use based on retrieval count (same logic as generate_response)
        retrieval_count = 3 # Default
        
        # If these are pasted images, use all of them regardless of retrieval count
        if is_pasted_images:
            images_to_process = images
            logger.info(f"Using all {len(images)} pasted images (bypassing retrieval count)")
        else:
            # Only apply retrieval count logic for RAG mode
            try:
                session_file = os.path.join('sessions', f"{session_id}.json")
                if os.path.exists(session_file):
                    with open(session_file, 'r') as f:
                        session_data = json.load(f)

                    # Check if this is a summary template
                    is_summary_template = False
                    template_id = session_data.get('selected_template_id')
                    if template_id:
                        # Try to load the template to check if it's a summary template
                        user_id = session_data.get('user_id')
                        if user_id:
                            try:
                                from src.models.prompt_templates import get_template_by_id
                                template = get_template_by_id(user_id, template_id)
                                if template and (template.get('template_type') == 'summary' or 
                                               template.get('id') == 'document-summary' or 
                                               template.get('name') == 'Document Summary'):
                                    is_summary_template = True
                                    logger.info(f"Summary template detected - using all available images")
                            except Exception as template_e:
                                logger.warning(f"Error checking template type: {template_e}")

                    # Check if token budget filtering has been applied and score-slope is enabled
                    token_budget_analysis = session_data.get('token_budget_analysis', None)

                    # Use the use_score_slope parameter passed to the function
                    # If it's not provided, fall back to the session data
                    session_use_score_slope = session_data.get('use_score_slope', False)
                    effective_use_score_slope = use_score_slope or session_use_score_slope
                    logger.info(f"Score-slope enabled: {effective_use_score_slope} (param: {use_score_slope}, session: {session_use_score_slope})")

                    # If score-slope is enabled, use all images provided
                    # These have already been filtered by relevance and token budget
                    if effective_use_score_slope:
                        retrieval_count = len(images)
                        logger.info(f"Using all {retrieval_count} images from score-slope analysis")
                    elif token_budget_analysis:
                        # Token budget filtering has been applied, use all images provided
                        retrieval_count = len(images)
                        logger.info(f"Using all {retrieval_count} images from token budget filtering")
                    elif is_summary_template:
                        # Use all images up to a reasonable limit (e.g., 20)
                        retrieval_count = min(len(images), 20)
                        logger.info(f"Using summary template retrieval count: {retrieval_count}")
                    else:
                        # Use the configured retrieval count from session
                        retrieval_count = int(session_data.get('retrieval_count', 3))
                        logger.info(f"Using user-configured retrieval count: {retrieval_count}")
                else:
                    logger.info(f"No session file found, using default retrieval count: {retrieval_count}")
            except Exception as e:
                logger.warning(f"Error loading retrieval count from session: {e}, using default: {retrieval_count}")

            # Select the images based on the count
            images_to_process = images[:min(len(images), retrieval_count)]
        img_count = len(images_to_process)
        logger.info(f"Attempting to process {img_count} images for Ollama streaming.")

        # Prepare image input for the API (CRITICAL: Add collaging logic for streaming)
        image_paths_for_api = []
        if not self.is_text_only and images_to_process:
            if self.supports_multiple_images:
                # Use original paths directly if model supports multiple images
                image_paths_for_api = images_to_process
                logger.info(f"Sending {len(image_paths_for_api)} image paths directly to Ollama streaming {self.api_model_name}.")
            else:
                # Use collage approach for models that don't support multiple images (like Llama 3.2 Vision)
                logger.info(f"Model {self.api_model_name} doesn't support multiple images directly - creating collage from {len(images_to_process)} images for streaming")
                collage_or_single_path = self._create_collage(images_to_process, session_id)
                if collage_or_single_path:
                    # If _create_collage returned a single path (because only one image was input),
                    # or if it returned a collage path, wrap it in a list for the API.
                    image_paths_for_api = [collage_or_single_path]
                    if "temp_collage_" in collage_or_single_path:
                         logger.info(f"SUCCESS: Created and using collage image for Ollama streaming {self.api_model_name}: {collage_or_single_path}")
                    else:
                         logger.info(f"Using single image for Ollama streaming {self.api_model_name} (no collage needed): {collage_or_single_path}")

                else:
                    logger.error("Failed to prepare image input (collage or single image) for streaming.")
                    # Return error message via streaming
                    yield {'type': 'error', 'error': 'Error preparing image input for Ollama streaming.'}
                    return

        # Get OCR context if needed (same as regular method)
        ocr_context_str = self._get_ocr_context(images_to_process, direct_ocr_results, session_id, use_ocr)

        # Prepare messages
        messages = self._prepare_ollama_messages(
            query, conversation_context, image_paths_for_api, ocr_context_str, img_count
        )

        # Remove the problematic control message for now
        # Will handle granite models with special streaming approach

        # API options already obtained earlier

        try:
            # Create a client with a longer timeout
            client = ollama.Client(timeout=1800.0)

            # Make the streaming API call
            response_stream = client.chat(
                model=self.api_model_name,
                messages=messages,
                options=options,
                stream=True  # Enable streaming
            )

            # Initialize streaming state variables
            buffer = ""  # Buffer to accumulate content
            in_reasoning = False  # Track if we're inside reasoning tags
            in_response = False  # Track if we're inside response tags
            reasoning_tag_stack = []  # Stack to track nested reasoning tags

            for chunk in response_stream:
                if 'message' in chunk and 'content' in chunk['message']:
                    content = chunk['message']['content']

                    if not content:
                        continue

                    if not is_reasoning_model:
                        # Non-reasoning model, stream directly
                        yield {'type': 'response', 'content': content}
                    else:
                        # For reasoning models, we need to parse tags progressively
                        buffer += content

                        # Process the buffer to extract complete chunks
                        while buffer:
                            processed = False

                            # Check for <response> tag
                            if not in_response and '<response>' in buffer:
                                # Found response tag start
                                response_start = buffer.find('<response>')

                                # Yield any reasoning content before <response>
                                pre_response = buffer[:response_start]
                                if pre_response and not in_reasoning:
                                    # Clean reasoning tags from the content but preserve whitespace
                                    pre_response = re.sub(r'</?(?:think|thought)>', '', pre_response)
                                    if pre_response.strip():  # Only check if content exists, but don't strip
                                        yield {'type': 'reasoning', 'content': pre_response}

                                # Update buffer and set in_response flag
                                buffer = buffer[response_start + len('<response>'):]
                                in_response = True
                                processed = True

                            # Check for </response> tag
                            elif in_response and '</response>' in buffer:
                                # Found response tag end
                                response_end = buffer.find('</response>')
                                response_content = buffer[:response_end]  # Don't strip - preserve whitespace

                                if response_content:
                                    # For Magistral models, remove \boxed{} patterns before yielding
                                    if self.api_model_name in ['magistral:24b', 'magistral:24b-small-2506-q8_0', 'magistral:24b-small-2506-fp16']:
                                        # Remove the entire \boxed{} expression and its content
                                        boxed_comprehensive_pattern = r"\\boxed\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}"
                                        response_content = re.sub(boxed_comprehensive_pattern, '', response_content, flags=re.DOTALL | re.MULTILINE)
                                        simple_boxed_pattern = r"\\boxed\{.*?\}"
                                        response_content = re.sub(simple_boxed_pattern, '', response_content, flags=re.DOTALL)
                                        response_content = response_content.strip()
                                    yield {'type': 'response', 'content': response_content}

                                # Update buffer and reset in_response flag
                                buffer = buffer[response_end + len('</response>'):]
                                in_response = False
                                processed = True

                            # If we're in response mode, yield content progressively
                            elif in_response:
                                # Look for incomplete </response> tag at the end
                                if buffer.endswith('<') or buffer.endswith('</') or buffer.endswith('</r') or \
                                   buffer.endswith('</re') or buffer.endswith('</res') or buffer.endswith('</resp') or \
                                   buffer.endswith('</respo') or buffer.endswith('</respon') or buffer.endswith('</respons'):
                                    # Wait for more content
                                    break
                                else:
                                    # Yield accumulated response content
                                    # For Magistral models, we'll clean boxed patterns in the final buffer
                                    # Don't clean it here during progressive streaming to avoid breaking patterns
                                    yield {'type': 'response', 'content': buffer}
                                    buffer = ""
                                    processed = True

                            # Check for reasoning tags (think/thought)
                            elif not in_response:
                                # Check for <think> or <thought> tags
                                think_start = buffer.find('<think>')
                                thought_start = buffer.find('<thought>')

                                # Find the earliest tag
                                tag_positions = [(think_start, '<think>', '</think>'),
                                               (thought_start, '<thought>', '</thought>')]
                                tag_positions = [(pos, open_tag, close_tag) for pos, open_tag, close_tag in tag_positions if pos != -1]

                                if tag_positions:
                                    # Sort by position to get the earliest tag
                                    tag_positions.sort(key=lambda x: x[0])
                                    pos, open_tag, close_tag = tag_positions[0]

                                    # Check if we have the complete reasoning block
                                    if close_tag in buffer[pos:]:
                                        # Extract the complete reasoning block
                                        close_pos = buffer.find(close_tag, pos)
                                        reasoning_content = buffer[pos + len(open_tag):close_pos].strip()

                                        if reasoning_content:
                                            yield {'type': 'reasoning', 'content': reasoning_content}

                                        # Update buffer
                                        buffer = buffer[close_pos + len(close_tag):]
                                        processed = True
                                    else:
                                        # Incomplete reasoning block, but check if we have enough content to stream progressively
                                        reasoning_start = pos + len(open_tag)
                                        current_reasoning = buffer[reasoning_start:].strip()

                                        # If we have substantial reasoning content, yield it progressively
                                        if len(current_reasoning) > 500:
                                            # Look for natural break points (double newlines or sentence endings)
                                            last_break = max(
                                                current_reasoning.rfind('\n\n'),
                                                current_reasoning.rfind('. '),
                                                current_reasoning.rfind('? '),
                                                current_reasoning.rfind('! ')
                                            )

                                            if last_break > 0:
                                                # Yield content up to the last break
                                                reasoning_chunk = current_reasoning[:last_break + 1].strip()
                                                if reasoning_chunk:
                                                    yield {'type': 'reasoning', 'content': reasoning_chunk}
                                                # Keep the rest in buffer
                                                buffer = buffer[:reasoning_start] + current_reasoning[last_break + 1:]
                                                processed = True
                                            elif len(current_reasoning) > 2000:
                                                # For models like Magistral that have very long reasoning, increase the chunk size
                                                # and avoid truncating mid-thought if possible
                                                chunk_size = 3000 if self.api_model_name in ['magistral:24b', 'magistral:24b-small-2506-q8_0', 'magistral:24b-small-2506-fp16'] else 1500
                                                
                                                # Try to find a sentence ending near the chunk size
                                                chunk_end = chunk_size
                                                for i in range(chunk_size - 100, min(chunk_size + 100, len(current_reasoning))):
                                                    if i < len(current_reasoning) and current_reasoning[i] in '.!?':
                                                        chunk_end = i + 1
                                                        break
                                                
                                                reasoning_chunk = current_reasoning[:chunk_end]
                                                yield {'type': 'reasoning', 'content': reasoning_chunk}
                                                buffer = buffer[:reasoning_start] + current_reasoning[chunk_end:]
                                                processed = True

                                        if not processed:
                                            # Wait for more content
                                            break
                                else:
                                    # No reasoning tags found
                                    # Special handling for phi4-mini-reasoning models that output reasoning without tags
                                    # Note: Regular phi4 and phi4-mini use <think> tags, but phi4-mini-reasoning doesn't
                                    if 'phi4-mini-reasoning' in self.api_model_name or (self.api_model_name.startswith('phi4-reasoning') and 'phi4-reasoning' in self.api_model_name):
                                        # These models output untagged reasoning followed by the actual response
                                        # We need to detect when the reasoning ends and response begins

                                        # Look for patterns that indicate the start of the actual response
                                        response_indicators = [
                                            "\n\nOnce upon",  # Story beginnings
                                            "\n\nHere's",     # Direct answers
                                            "\n\nThe ",       # Direct answers
                                            "\n\nA ",         # Story beginnings
                                            "\n\nIn ",        # Story beginnings
                                            "\n\nLet me",     # Meta responses
                                            "\n\n\"",         # Quoted responses
                                            "\n\n*",          # Formatted responses
                                            "\n\n1.",         # Numbered lists
                                            "\n\n-",          # Bullet lists
                                            "\n\nCertainly",  # Polite beginnings
                                            "\n\nSure",       # Casual beginnings
                                            "\n\nHello",      # Greetings
                                            "\n\nHi",         # Greetings
                                        ]

                                        # Check if any response indicator is in the buffer
                                        response_start = -1
                                        for indicator in response_indicators:
                                            pos = buffer.find(indicator)
                                            if pos != -1 and (response_start == -1 or pos < response_start):
                                                response_start = pos

                                        if response_start != -1:
                                            # Found a response indicator
                                            # Everything before it is reasoning
                                            reasoning_content = buffer[:response_start].strip()
                                            if reasoning_content:
                                                yield {'type': 'reasoning', 'content': reasoning_content}

                                            # Everything after is response
                                            response_content = buffer[response_start:].strip()
                                            yield {'type': 'response', 'content': response_content}
                                            buffer = ""
                                            processed = True
                                        else:
                                            # No response indicator found yet
                                            # Stream reasoning content progressively
                                            # Look for natural break points (double newlines)
                                            if '\n\n' in buffer and len(buffer) > 500:
                                                # Find the last double newline
                                                last_break = buffer.rfind('\n\n')
                                                if last_break > 0:
                                                    # Yield content up to the last break
                                                    reasoning_chunk = buffer[:last_break].strip()
                                                    if reasoning_chunk:
                                                        yield {'type': 'reasoning', 'content': reasoning_chunk}
                                                    # Keep the rest in buffer
                                                    buffer = buffer[last_break:].lstrip()
                                                    processed = True
                                            elif len(buffer) > 5000:
                                                # Buffer is getting too long, yield what we have
                                                yield {'type': 'reasoning', 'content': buffer.strip()}
                                                buffer = ""
                                                processed = True
                                            else:
                                                # Wait for more content
                                                break
                                    else:
                                        # For other models (like DeepCoder) that use tags properly
                                        # Check for incomplete tags at the end
                                        incomplete_tags = ['<', '<t', '<th', '<thi', '<thin', '<think',
                                                         '<tho', '<thou', '<thoug', '<though', '<thought',
                                                         '<r', '<re', '<res', '<resp', '<respo', '<respon', '<respons', '<response']

                                        has_incomplete = any(buffer.endswith(tag) for tag in incomplete_tags)

                                        if has_incomplete:
                                            # Wait for more content
                                            break
                                        else:
                                            # No tags detected, yield as response
                                            if buffer.strip():
                                                yield {'type': 'response', 'content': buffer}
                                                buffer = ""
                                                processed = True

                            # If nothing was processed, break to wait for more content
                            if not processed:
                                break

            # Process any remaining buffer content
            if buffer:  # Don't strip here - check the original content
                if in_response or not is_reasoning_model:
                    # For Magistral models, remove \boxed{} patterns from final buffer
                    if self.api_model_name in ['magistral:24b', 'magistral:24b-small-2506-q8_0', 'magistral:24b-small-2506-fp16']:
                        # Remove the entire \boxed{} expression and its content
                        boxed_comprehensive_pattern = r"\\boxed\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}"
                        buffer = re.sub(boxed_comprehensive_pattern, '', buffer, flags=re.DOTALL | re.MULTILINE)
                        simple_boxed_pattern = r"\\boxed\{.*?\}"
                        buffer = re.sub(simple_boxed_pattern, '', buffer, flags=re.DOTALL)
                        buffer = buffer.strip()
                    yield {'type': 'response', 'content': buffer}  # Preserve whitespace
                else:
                    # Special handling for phi4-mini-reasoning and Magistral models that don't use tags properly
                    if ('phi4-mini-reasoning' in self.api_model_name or 
                        (self.api_model_name.startswith('phi4-reasoning') and 'phi4-reasoning' in self.api_model_name) or
                        self.api_model_name in ['magistral:24b', 'magistral:24b-small-2506-q8_0', 'magistral:24b-small-2506-fp16']):
                        # This is likely the final response part
                        # For Magistral models, remove \boxed{} patterns
                        if self.api_model_name in ['magistral:24b', 'magistral:24b-small-2506-q8_0', 'magistral:24b-small-2506-fp16']:
                            # Remove the entire \boxed{} expression and its content
                            boxed_comprehensive_pattern = r"\\boxed\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}"
                            buffer = re.sub(boxed_comprehensive_pattern, '', buffer, flags=re.DOTALL | re.MULTILINE)
                            simple_boxed_pattern = r"\\boxed\{.*?\}"
                            buffer = re.sub(simple_boxed_pattern, '', buffer, flags=re.DOTALL)
                            buffer = buffer.strip()
                        yield {'type': 'response', 'content': buffer}  # Preserve whitespace
                    else:
                        # Check for incomplete reasoning tags that never closed
                        if '<think>' in buffer and '</think>' not in buffer:
                            # Extract reasoning content after the opening tag
                            think_start = buffer.find('<think>') + len('<think>')
                            reasoning_content = buffer[think_start:]  # Don't strip
                            if reasoning_content.strip():  # Only check for existence, but preserve content
                                logger.info(f"Found incomplete <think> tag, extracting reasoning content: {len(reasoning_content)} chars")
                                yield {'type': 'reasoning', 'content': reasoning_content}
                            return  # Don't yield anything else
                        elif '<thought>' in buffer and '</thought>' not in buffer:
                            # Extract reasoning content after the opening tag
                            thought_start = buffer.find('<thought>') + len('<thought>')
                            reasoning_content = buffer[thought_start:]  # Don't strip
                            if reasoning_content.strip():  # Only check for existence, but preserve content
                                logger.info(f"Found incomplete <thought> tag, extracting reasoning content: {len(reasoning_content)} chars")
                                yield {'type': 'reasoning', 'content': reasoning_content}
                            return  # Don't yield anything else
                        else:
                            # Check if it's reasoning content with tags
                            cleaned = re.sub(r'</?(?:think|thought|response)>', '', buffer).strip()
                            if cleaned:
                                # For Magistral models, also remove \boxed{} patterns
                                if self.api_model_name in ['magistral:24b', 'magistral:24b-small-2506-q8_0', 'magistral:24b-small-2506-fp16']:
                                    # Remove the entire \boxed{} expression and its content
                                    boxed_comprehensive_pattern = r"\\boxed\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}"
                                    cleaned = re.sub(boxed_comprehensive_pattern, '', cleaned, flags=re.DOTALL | re.MULTILINE)
                                    simple_boxed_pattern = r"\\boxed\{.*?\}"
                                    cleaned = re.sub(simple_boxed_pattern, '', cleaned, flags=re.DOTALL)
                                    cleaned = cleaned.strip()
                                yield {'type': 'response', 'content': cleaned}

            # Signal that reasoning is complete and should be cleared
            if is_reasoning_model:
                yield {'type': 'reasoning_complete'}
            yield {'type': 'complete'}

        except Exception as e:
            logger.error(f"Error during Ollama streaming: {str(e)}", exc_info=True)
            yield {'type': 'error', 'error': str(e)}
