"""
VLM-based OCR wrapper using Ollama
"""
import os
import time
import base64
import requests
import json
from PIL import Image
from io import BytesIO
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Global model cache
_vlm_session = None
_vlm_last_used = 0

def get_vlm_session():
    """
    Get or create a requests session for VLM OCR
    """
    global _vlm_session, _vlm_last_used
    
    # Return cached session if available and used recently (last 5 minutes)
    if _vlm_session is not None and time.time() - _vlm_last_used < 300:
        _vlm_last_used = time.time()
        return _vlm_session
    
    # Create new session
    _vlm_session = requests.Session()
    _vlm_last_used = time.time()
    return _vlm_session

def extract_text_with_vlm_ocr(image_path, vlm_model='llama3.2-vision:11b', ollama_url='http://localhost:11434'):
    """
    Extract text from an image using Vision Language Model through Ollama
    
    Args:
        image_path: Path to the image file
        vlm_model: Ollama model to use for VLM OCR
        ollama_url: URL to Ollama service
        
    Returns:
        Extracted text as a string, or empty string on failure
    """
    logger.info(f"extract_text_with_vlm_ocr called for: {image_path}")
    logger.info(f"Using VLM model: {vlm_model}")
    
    try:
        # Check if image exists
        if not os.path.exists(image_path):
            logger.error(f"Image file not found: {image_path}")
            return ""
        
        logger.info(f"Image file exists: {image_path}, size: {os.path.getsize(image_path)} bytes")
        
        # Load and encode image
        with open(image_path, 'rb') as img_file:
            image_data = img_file.read()
            base64_image = base64.b64encode(image_data).decode('utf-8')
        
        # Create the prompt for text extraction
        prompt = """Please extract all text from this image. 
Output only the extracted text, preserving the original layout and formatting as much as possible.
Do not add any explanations or descriptions - just the raw extracted text."""
        
        # Prepare the request payload
        payload = {
            "model": vlm_model,
            "prompt": prompt,
            "images": [base64_image],
            "stream": False,
            "options": {
                "temperature": 0.1,  # Low temperature for consistent extraction
                "top_p": 0.9,
                "max_tokens": 4096
            }
        }
        
        # Get session
        session = get_vlm_session()
        
        # Make request to Ollama
        logger.info(f"Sending request to Ollama at {ollama_url}/api/generate")
        response = session.post(
            f"{ollama_url}/api/generate",
            json=payload,
            timeout=1800  # 30 minute timeout for VLM processing
        )
        
        if response.status_code != 200:
            logger.error(f"Ollama API error: {response.status_code} - {response.text}")
            return ""
        
        # Parse response
        result = response.json()
        extracted_text = result.get('response', '')
        
        logger.info(f"Extracted {len(extracted_text)} characters of text from image using VLM OCR")
        if extracted_text:
            logger.info(f"First 200 chars of extracted text: {extracted_text[:200]}")
        
        return extracted_text.strip()
        
    except requests.exceptions.Timeout:
        logger.error("VLM OCR request timed out after 60 seconds")
        return ""
    except requests.exceptions.ConnectionError:
        logger.error(f"Failed to connect to Ollama at {ollama_url}. Is Ollama running?")
        return ""
    except Exception as e:
        logger.error(f"Error in VLM OCR text extraction: {e}", exc_info=True)
        return ""

def check_ollama_available(ollama_url='http://localhost:11434'):
    """
    Check if Ollama is available and running
    
    Args:
        ollama_url: URL to Ollama service
        
    Returns:
        True if Ollama is available, False otherwise
    """
    try:
        response = requests.get(f"{ollama_url}/api/version", timeout=5)
        return response.status_code == 200
    except:
        return False

def list_available_vlm_models(ollama_url='http://localhost:11434'):
    """
    List available VLM models in Ollama
    
    Args:
        ollama_url: URL to Ollama service
        
    Returns:
        List of available vision model names
    """
    try:
        response = requests.get(f"{ollama_url}/api/tags", timeout=5)
        if response.status_code == 200:
            data = response.json()
            models = data.get('models', [])
            # Filter for vision models (heuristic: models with "vision" in name or known vision models)
            vision_models = []
            known_vision_models = ['llama3.2-vision', 'llava', 'bakllava', 'minicpm-v', 'qwen2.5-vision', 'molmo', 'gemma-vision']
            
            for model in models:
                model_name = model.get('name', '')
                # Check if it's a known vision model or has 'vision' in the name
                for known in known_vision_models:
                    if known in model_name.lower() or 'vision' in model_name.lower():
                        vision_models.append(model_name)
                        break
            
            return vision_models
        return []
    except Exception as e:
        logger.error(f"Error listing Ollama models: {e}")
        return []