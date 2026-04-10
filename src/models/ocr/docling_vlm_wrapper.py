"""
Docling VLM OCR wrapper using Ollama backend
"""
import os
import time
import tempfile
from pathlib import Path
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    ApiVlmOptions,
    VlmPipelineOptions,
    ResponseFormat
)
from docling.document_converter import DocumentConverter, PdfFormatOption, ImageFormatOption
from docling.pipeline.vlm_pipeline import VlmPipeline
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Global converter cache
_vlm_converter = None
_vlm_converter_last_used = 0
_vlm_converter_model = None

def get_vlm_converter(vlm_model='llama3.2-vision:11b', ollama_url='http://localhost:11434'):
    """
    Get or create a Docling VLM converter
    
    Args:
        vlm_model: Ollama model to use for VLM OCR
        ollama_url: URL to Ollama service
        
    Returns:
        DocumentConverter instance configured for VLM
    """
    global _vlm_converter, _vlm_converter_last_used, _vlm_converter_model
    
    # Return cached converter if available, used recently, and same model
    if (_vlm_converter is not None and 
        time.time() - _vlm_converter_last_used < 300 and
        _vlm_converter_model == vlm_model):
        logger.info(f"Using cached Docling VLM converter for model {vlm_model}")
        _vlm_converter_last_used = time.time()
        return _vlm_converter
    
    try:
        logger.info(f"Initializing Docling VLM converter with model {vlm_model}")
        
        # Configure VLM options for Ollama using the updated API
        vlm_options = ApiVlmOptions(
            addr=ollama_url,
            model_name=vlm_model,
            endpoint='/v1/chat/completions',  # Ollama's OpenAI-compatible endpoint
            request_timeout=1800,
            response_format=ResponseFormat.MARKDOWN,
            prompt="Extract all text from this image. Include all readable text, preserving the structure where possible. Output only the text content without explanations."
        )
        
        # Configure pipeline options
        pipeline_options = VlmPipelineOptions(
            vlm_model=vlm_options,
            enable_remote_services=True  # Required for remote VLM
        )
        
        # Create document converter with VLM pipeline
        _vlm_converter = DocumentConverter(
            format_options={
                # Configure PDF format to use the VLM pipeline
                PdfFormatOption: PdfFormatOption(
                    pipeline=VlmPipeline(pipeline_options=pipeline_options)
                ),
                # Configure IMAGE format to use the VLM pipeline  
                ImageFormatOption: ImageFormatOption(
                    pipeline=VlmPipeline(pipeline_options=pipeline_options)
                )
            }
        )
        
        _vlm_converter_last_used = time.time()
        _vlm_converter_model = vlm_model
        logger.info("Docling VLM converter initialized successfully")
        return _vlm_converter
        
    except ImportError as e:
        logger.error(f"Docling VLM dependencies not installed: {e}")
        return None
    except Exception as e:
        logger.error(f"Error initializing Docling VLM converter: {e}", exc_info=True)
        return None

def extract_text_with_docling_vlm(image_path, vlm_model='llama3.2-vision:11b', ollama_url='http://localhost:11434'):
    """
    Extract text from an image using Docling's VLM pipeline
    
    Args:
        image_path: Path to the image file
        vlm_model: Ollama model to use for VLM OCR
        ollama_url: URL to Ollama service
        
    Returns:
        Extracted text as a string, or empty string on failure
    """
    logger.info(f"extract_text_with_docling_vlm called for: {image_path}")
    logger.info(f"Using VLM model: {vlm_model}")
    
    try:
        # Check if image exists
        if not os.path.exists(image_path):
            logger.error(f"Image file not found: {image_path}")
            return ""
        
        logger.info(f"Image file exists: {image_path}, size: {os.path.getsize(image_path)} bytes")
        
        # Get VLM converter
        converter = get_vlm_converter(vlm_model, ollama_url)
        if converter is None:
            logger.error("Failed to initialize Docling VLM converter")
            return ""
        
        # Convert image to Path object
        image_path_obj = Path(image_path)
        
        # For images, we might need to handle JPG/PNG differently
        # If the file is not a PDF, we may need to convert it
        if image_path.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
            # Images are treated as single-page documents
            logger.info(f"Processing image file with Docling VLM: {image_path}")
        
        # Process the image
        logger.info(f"Converting image with Docling VLM: {image_path}")
        try:
            result = converter.convert(str(image_path_obj))
            
            # Check if conversion was successful
            if result and result.document:
                # Export text from the document
                extracted_text = result.document.export_to_markdown()
                logger.info(f"Extracted {len(extracted_text)} characters of text from image using Docling VLM")
                if extracted_text:
                    logger.info(f"First 200 chars of extracted text: {extracted_text[:200]}")
                return extracted_text.strip()
            else:
                logger.error("Conversion completed but no document was produced")
                return ""
                
        except Exception as e:
            logger.error(f"Error during conversion: {e}")
            return ""
        
    except Exception as e:
        logger.error(f"Error in Docling VLM text extraction: {e}", exc_info=True)
        return ""

def unload_docling_vlm_converter():
    """Explicitly unload Docling VLM converter to free memory"""
    global _vlm_converter, _vlm_converter_last_used, _vlm_converter_model
    if _vlm_converter is not None:
        logger.info("Unloading Docling VLM converter to free memory")
        _vlm_converter = None
        _vlm_converter_last_used = 0
        _vlm_converter_model = None
        
        # Force garbage collection
        import gc
        gc.collect()

def check_ollama_available(ollama_url='http://localhost:11434'):
    """
    Check if Ollama is available and running
    
    Args:
        ollama_url: URL to Ollama service
        
    Returns:
        True if Ollama is available, False otherwise
    """
    try:
        import requests
        response = requests.get(f"{ollama_url}/api/version", timeout=5)
        return response.status_code == 200
    except:
        return False

def list_available_vlm_models(ollama_url='http://localhost:11434'):
    """
    List available VLM models in Ollama that can be used with Docling
    
    Args:
        ollama_url: URL to Ollama service
        
    Returns:
        List of available vision model names
    """
    try:
        import requests
        response = requests.get(f"{ollama_url}/api/tags", timeout=5)
        if response.status_code == 200:
            data = response.json()
            models = data.get('models', [])
            # Filter for vision models (heuristic: models with "vision" in name or known vision models)
            vision_models = []
            known_vision_models = ['llama3.2-vision', 'llava', 'bakllava', 'minicpm-v', 
                                   'qwen2.5-vision', 'molmo', 'gemma-vision']
            
            for model in models:
                model_name = model.get('name', '')
                # Check if it's a known vision model or has 'vision' in the name
                for known in known_vision_models:
                    if known in model_name.lower() or 'vision' in model_name.lower():
                        vision_models.append(model_name)
                        break
            
            logger.info(f"Found {len(vision_models)} vision models in Ollama: {vision_models}")
            return vision_models
        return []
    except Exception as e:
        logger.error(f"Error listing Ollama models: {e}")
        return []