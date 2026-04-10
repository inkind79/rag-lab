"""
OCR Cache cleanup utilities for session deletion.
"""
import os
import hashlib
from typing import List, Set
from src.utils.logger import get_logger

logger = get_logger(__name__)

def get_session_image_paths(session_uuid: str, app_config: dict) -> Set[str]:
    """
    Get all image paths associated with a session.
    
    Args:
        session_uuid: The session UUID
        app_config: Flask app configuration
        
    Returns:
        Set of image paths used by this session
    """
    image_paths = set()
    
    # Check uploaded documents folder
    upload_folder = os.path.join(app_config['UPLOAD_FOLDER'], session_uuid)
    if os.path.exists(upload_folder):
        for root, _, files in os.walk(upload_folder):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp')):
                    image_paths.add(os.path.join(root, file))
    
    # Check static images folder
    images_folder = os.path.join(app_config['STATIC_FOLDER'], 'images', session_uuid)
    if os.path.exists(images_folder):
        for root, _, files in os.walk(images_folder):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp')):
                    image_paths.add(os.path.join(root, file))
    
    return image_paths

def clear_ocr_cache_for_session(session_uuid: str, app_config: dict) -> int:
    """
    Clear OCR cache entries for all images belonging to a session.
    
    Args:
        session_uuid: The session UUID
        app_config: Flask app configuration
        
    Returns:
        Number of cache entries cleared
    """
    try:
        # Get all image paths for this session
        image_paths = get_session_image_paths(session_uuid, app_config)
        if not image_paths:
            logger.info(f"No images found for session {session_uuid}")
            return 0
        
        # Generate cache keys for all images
        cache_keys = set()
        for image_path in image_paths:
            if os.path.exists(image_path):
                try:
                    mod_time = os.path.getmtime(image_path)
                    key = f"{image_path}_{mod_time}"
                    cache_key = hashlib.md5(key.encode()).hexdigest()
                    cache_keys.add(cache_key)
                except Exception as e:
                    logger.warning(f"Error generating cache key for {image_path}: {e}")
        
        # Remove cache files
        cache_dir = "cache/ocr"
        removed_count = 0
        
        if os.path.exists(cache_dir):
            for cache_key in cache_keys:
                cache_file = os.path.join(cache_dir, f"{cache_key}.json")
                if os.path.exists(cache_file):
                    try:
                        os.remove(cache_file)
                        removed_count += 1
                        logger.debug(f"Removed OCR cache file: {cache_file}")
                    except Exception as e:
                        logger.warning(f"Error removing OCR cache file {cache_file}: {e}")
        
        logger.info(f"Cleared {removed_count} OCR cache entries for session {session_uuid}")
        return removed_count
        
    except Exception as e:
        logger.error(f"Error clearing OCR cache for session {session_uuid}: {e}")
        return 0