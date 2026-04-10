import os
import json
import hashlib
import threading
from src.utils.logger import get_logger

logger = get_logger(__name__)

class OCRCache:
    """Cache OCR results to avoid repeated processing"""
    
    _instance = None
    _lock = threading.Lock()
    
    @classmethod
    def get_instance(cls, cache_dir="cache/ocr"):
        """Singleton pattern to ensure only one cache instance exists"""
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls(cache_dir)
            return cls._instance
    
    def __init__(self, cache_dir="cache/ocr"):
        """Initialize OCR cache with specified directory"""
        self.cache_dir = cache_dir
        self.cache_lock = threading.Lock()
        os.makedirs(cache_dir, exist_ok=True)
        logger.info(f"Initialized OCR cache at {cache_dir}")
    
    def _get_cache_key(self, image_path):
        """Generate a unique cache key based on image path and modification time"""
        if not os.path.exists(image_path):
            return None
        
        # Use file path and modification time for cache key
        mod_time = os.path.getmtime(image_path)
        key = f"{image_path}_{mod_time}"
        return hashlib.md5(key.encode()).hexdigest()
    
    def get(self, image_path):
        """Get cached OCR result for an image if available"""
        with self.cache_lock:
            cache_key = self._get_cache_key(image_path)
            if not cache_key:
                return None
            
            cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
            if os.path.exists(cache_file):
                try:
                    with open(cache_file, 'r') as f:
                        data = json.load(f)
                    logger.info(f"OCR cache hit for {image_path}")
                    return data
                except Exception as e:
                    logger.error(f"Error reading OCR cache: {e}")
        
        return None
    
    def set(self, image_path, text, analysis=None):
        """
        Cache OCR result for an image
        
        Args:
            image_path: Path to the image
            text: Extracted text from OCR
            analysis: Optional dict with image analysis results
            
        Returns:
            bool: True if successfully cached, False otherwise
        """
        with self.cache_lock:
            cache_key = self._get_cache_key(image_path)
            if not cache_key:
                return False
            
            cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
            try:
                cache_data = {
                    'text': text,
                    'timestamp': os.path.getmtime(image_path)
                }
                
                # Add analysis data if provided
                if analysis:
                    cache_data['analysis'] = analysis
                
                with open(cache_file, 'w') as f:
                    json.dump(cache_data, f)
                logger.info(f"Cached OCR result for {image_path}")
                return True
            except Exception as e:
                logger.error(f"Error writing OCR cache: {e}")
                return False
    
    def clear(self):
        """Clear all cached OCR results"""
        with self.cache_lock:
            try:
                import shutil
                if os.path.exists(self.cache_dir):
                    shutil.rmtree(self.cache_dir)
                os.makedirs(self.cache_dir, exist_ok=True)
                logger.info(f"Cleared OCR cache at {self.cache_dir}")
                return True
            except Exception as e:
                logger.error(f"Error clearing OCR cache: {e}")
                return False