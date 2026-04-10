import os
import time
import numpy as np
import torch
import easyocr # Added for EasyOCR
# from PIL import Image # No longer explicitly needed here
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Global OCR model cache
_ocr_model = None
_ocr_last_used = 0

def get_ocr_model(use_gpu=True): # Default to attempting GPU usage
    """
    Load EasyOCR Reader model with memory-efficient caching.

    Args:
        use_gpu: Whether to use GPU for inference (if available).

    Returns:
        easyocr.Reader instance or None on failure.
    """
    global _ocr_model, _ocr_last_used

    # Return cached model if available and used recently (last 5 minutes)
    if _ocr_model is not None and time.time() - _ocr_last_used < 300:
        logger.info("Using cached OCR model")
        _ocr_last_used = time.time()
        return _ocr_model

    # Check if CUDA is available if use_gpu is requested
    if use_gpu and not torch.cuda.is_available():
        logger.warning("GPU requested for OCR but CUDA not available, falling back to CPU")
        use_gpu = False

    try:
        # EasyOCR is already imported at the top

        # Record memory before loading
        if torch.cuda.is_available():
            mem_before = torch.cuda.memory_allocated() / (1024 * 1024)  # MB

        logger.info(f"Initializing EasyOCR Reader (gpu={use_gpu})")
        # Initialize EasyOCR Reader for English
        # It will automatically download models on first use
        _ocr_model = easyocr.Reader(['en'], gpu=use_gpu)

        # Record memory after loading
        if torch.cuda.is_available():
            mem_after = torch.cuda.memory_allocated() / (1024 * 1024)  # MB
            logger.info(f"EasyOCR Reader loaded. GPU Memory usage: {mem_after-mem_before:.2f} MB")

        logger.info("EasyOCR Reader initialized successfully")
        _ocr_last_used = time.time()
        return _ocr_model

    except ImportError as e:
        logger.error(f"EasyOCR not installed or dependencies missing: {e}. Install with 'pip install easyocr'")
        return None
    except Exception as e:
        logger.error(f"Error initializing EasyOCR Reader: {e}", exc_info=True) # Added exc_info
        return None

# SmolDocling is a modern document understanding model from Hugging Face

def extract_text_from_image(image_path, ocr_engine='easyocr', element_type=None, output_format="markdown"):
    """
    Extract text from an image using the specified OCR engine

    Args:
        image_path: Path to the image file
        ocr_engine: OCR engine to use ('easyocr', 'smoldocling', or 'smoldocling-otsl')
        element_type: Optional element type for SmolDocling ("table", "formula", "code", "chart", or None)
        output_format: Output format for SmolDocling ("markdown", "html", "json", "doctags", or "otsl")

    Returns:
        Extracted text as a string, or empty string on failure
    """
    if ocr_engine.lower() == 'smoldocling':
        # Import SmolDocling wrapper when needed
        from src.models.ocr.smoldocling_wrapper import extract_text_with_smoldocling
        return extract_text_with_smoldocling(image_path, element_type=element_type, output_format=output_format)
    elif ocr_engine.lower() == 'smoldocling-otsl':
        # Import SmolDocling wrapper for OTSL output
        from src.models.ocr.smoldocling_wrapper import extract_text_with_smoldocling
        # Force OTSL output format
        return extract_text_with_smoldocling(image_path, element_type=element_type, output_format="otsl")
    else:
        return extract_text_with_easyocr(image_path)

def extract_text_with_easyocr(image_path):
    """
    Extract text from an image using EasyOCR

    Args:
        image_path: Path to the image file

    Returns:
        Extracted text as a string, or empty string on failure
    """
    try:
        ocr = get_ocr_model(use_gpu=True) # Attempt to use GPU if available
        if ocr is None:
            logger.error("Failed to initialize EasyOCR model")
            return ""

        # Read image
        if not os.path.exists(image_path):
            logger.error(f"Image file not found: {image_path}")
            return ""

        # Process image with EasyOCR
        # detail=0 returns only the text
        # paragraph=True groups text into paragraphs
        logger.info(f"Processing image with EasyOCR: {image_path}")
        result = ocr.readtext(image_path, detail=0, paragraph=True)

        # Join the paragraphs/lines returned by EasyOCR
        extracted_text = "\n".join(result)

        logger.info(f"Extracted {len(extracted_text)} characters of text from image using EasyOCR")
        return extracted_text.strip()
    except Exception as e:
        logger.error(f"Error in EasyOCR text extraction: {e}")
        return ""

# Docling functionality moved to onnxtr_wrapper.py

def analyze_image_content(image_path):
    """
    Analyze image to determine if it's text-heavy or contains mixed content

    Args:
        image_path: Path to the image file

    Returns:
        dict: Analysis results with keys:
            - text_heavy: Boolean indicating if image is primarily text
            - text_ratio: Estimated ratio of text area to total area
    """
    try:
        import cv2

        # Read image
        if not os.path.exists(image_path):
            logger.error(f"Image file not found: {image_path}")
            return {"text_heavy": False, "text_ratio": 0}

        # Load image
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Failed to read image: {image_path}")
            return {"text_heavy": False, "text_ratio": 0}

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply adaptive thresholding to identify potential text areas
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 7
        )

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter contours to find text-like regions
        text_like_area = 0
        total_area = gray.shape[0] * gray.shape[1]

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h

            # Text typically has aspect ratio between 0.1 and 10
            # and reasonable size (not too small, not too large)
            min_size = total_area * 0.0001  # Min size threshold
            max_size = total_area * 0.1     # Max size threshold
            area = w * h

            if (0.1 < aspect_ratio < 10) and (min_size < area < max_size):
                text_like_area += area

        # Calculate text ratio
        text_ratio = text_like_area / total_area

        # Consider text-heavy if more than 15% of the image contains text-like regions
        text_heavy = text_ratio > 0.15

        logger.info(f"Image analysis for {image_path}: text_ratio={text_ratio:.2f}, text_heavy={text_heavy}")
        return {
            "text_heavy": text_heavy,
            "text_ratio": text_ratio
        }
    except Exception as e:
        logger.error(f"Error analyzing image content: {e}")
        return {"text_heavy": False, "text_ratio": 0}

def format_ocr_context(ocr_text_list):
    """
    Format a list of OCR text extractions into a single context string

    Args:
        ocr_text_list: List of OCR text strings extracted from images

    Returns:
        Formatted context string
    """
    if not ocr_text_list:
        return ""

    # Join all text with clear separators
    formatted_text = ""
    for i, text in enumerate(ocr_text_list):
        if text.strip():
            formatted_text += f"\n--- Document {i+1} Text Content ---\n{text.strip()}\n"

    logger.info(f"Formatted OCR context with {len(ocr_text_list)} text blocks")
    return formatted_text.strip()

def unload_ocr_models():
    """Explicitly unload all OCR models to free memory"""
    unload_easyocr_model()
    # Import and unload SmolDocling model
    try:
        from src.models.ocr.smoldocling_wrapper import unload_smoldocling_model
        unload_smoldocling_model()
    except ImportError:
        pass

def unload_easyocr_model():
    """Explicitly unload EasyOCR model to free memory"""
    global _ocr_model, _ocr_last_used
    if _ocr_model is not None:
        logger.info("Unloading EasyOCR model to free memory")
        _ocr_model = None
        _ocr_last_used = 0

        # Force garbage collection
        import gc
        gc.collect()

        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# Alias for backward compatibility
unload_ocr_model = unload_ocr_models