"""
Wrapper for OnnxTR OCR functionality
"""
import os
import time
import torch
from PIL import Image
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Global model cache
_onnxtr_model = None
_onnxtr_last_used = 0

def get_onnxtr_model(use_gpu=True):
    """
    Load OnnxTR OCR model with memory-efficient caching.
    
    Args:
        use_gpu: Whether to use GPU for inference (if available).
        
    Returns:
        OnnxTR OCR model instance or None on failure.
    """
    global _onnxtr_model, _onnxtr_last_used
    
    # Return cached model if available and used recently (last 5 minutes)
    if _onnxtr_model is not None and time.time() - _onnxtr_last_used < 300:
        logger.info("Using cached OnnxTR model")
        _onnxtr_last_used = time.time()
        return _onnxtr_model
    
    # Check if CUDA is available if use_gpu is requested
    if use_gpu and not torch.cuda.is_available():
        logger.warning("GPU requested for OnnxTR but CUDA not available, falling back to CPU")
        use_gpu = False
    
    try:
        logger.info(f"Initializing OnnxTR model (gpu={use_gpu})")
        
        # Import OnnxTR
        from onnxtr.io import DocumentFile
        from onnxtr.models import ocr_predictor
        
        # Initialize the predictor
        _onnxtr_model = ocr_predictor(
            det_arch='fast_base',  # Detection architecture
            reco_arch='crnn_vgg16_bn',  # Recognition architecture
            # pretrained parameter doesn't exist in onnxtr
            # use_gpu parameter doesn't exist in onnxtr
        )
        
        logger.info("OnnxTR model initialized successfully")
        _onnxtr_last_used = time.time()
        return _onnxtr_model
        
    except ImportError as e:
        logger.error(f"OnnxTR not installed or dependencies missing: {e}. Install with 'pip install onnxtr'")
        return None
    except Exception as e:
        logger.error(f"Error initializing OnnxTR model: {e}", exc_info=True)
        return None

def extract_text_with_onnxtr(image_path):
    """
    Extract text from an image using OnnxTR
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Extracted text as a string, or empty string on failure
    """
    logger.info(f"extract_text_with_onnxtr called for: {image_path}")
    try:
        # Get OnnxTR model
        logger.info("Getting OnnxTR model...")
        predictor = get_onnxtr_model(use_gpu=True)
        if predictor is None:
            logger.error("Failed to initialize OnnxTR model")
            return ""
        
        # Read image
        if not os.path.exists(image_path):
            logger.error(f"Image file not found: {image_path}")
            return ""
        
        logger.info(f"Image file exists: {image_path}, size: {os.path.getsize(image_path)} bytes")
        
        # Import DocumentFile for image loading
        from onnxtr.io import DocumentFile
        
        try:
            # Load image using OnnxTR's DocumentFile
            doc = DocumentFile.from_images(image_path)
            logger.info(f"Document loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load image with DocumentFile: {e}")
            return ""
        
        # Perform OCR
        logger.info(f"Processing image with OnnxTR: {image_path}")
        result = predictor(doc)
        
        # Extract text from result
        extracted_text = ""
        for page in result.pages:
            for block in page.blocks:
                for line in block.lines:
                    for word in line.words:
                        extracted_text += word.value + " "
                    extracted_text += "\n"
                extracted_text += "\n"
        
        extracted_text = extracted_text.strip()
        
        logger.info(f"Extracted {len(extracted_text)} characters of text from image using OnnxTR")
        if extracted_text:
            logger.info(f"First 200 chars of extracted text: {extracted_text[:200]}")
        return extracted_text
        
    except Exception as e:
        logger.error(f"Error in OnnxTR text extraction: {e}", exc_info=True)
        return ""

def unload_onnxtr_model():
    """Explicitly unload OnnxTR model to free memory"""
    global _onnxtr_model, _onnxtr_last_used
    if _onnxtr_model is not None:
        logger.info("Unloading OnnxTR model to free memory")
        _onnxtr_model = None
        _onnxtr_last_used = 0
        
        # Force garbage collection
        import gc
        gc.collect()
        
        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()