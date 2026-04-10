"""
Utility functions for handling pasted images in streaming chat.
"""

import os
import base64
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


def decode_pasted_images_for_streaming(
    pasted_images_data: List[Dict[str, Any]], 
    session_uuid: str, 
    static_folder: str
) -> List[str]:
    """
    Decode base64 pasted images and save them to disk for streaming chat.
    
    Args:
        pasted_images_data: List of image data dicts with 'filename', 'data', 'type', 'size'
        session_uuid: Current session UUID
        static_folder: Flask static folder path
        
    Returns:
        List of saved file paths relative to static folder
        
    Raises:
        Exception: If image processing fails
    """
    if not pasted_images_data:
        return []
    
    saved_paths = []
    
    # Create session-specific images folder
    images_folder = os.path.join(static_folder, 'images', session_uuid)
    os.makedirs(images_folder, exist_ok=True)
    
    logger.info(f"Decoding {len(pasted_images_data)} pasted images for session {session_uuid}")
    
    for idx, img_data in enumerate(pasted_images_data):
        try:
            # Handle both formats:
            # 1. Dict: { filename, data, type, size } (legacy Flask)
            # 2. String: "data:image/png;base64,..." (SvelteKit client)
            if isinstance(img_data, str):
                base64_data = img_data
                filename = f'pasted_image_{idx}'
                file_type = 'image/png'
                file_size = 0
            else:
                filename = img_data.get('filename', f'pasted_image_{idx}')
                base64_data = img_data.get('data', '')
                file_type = img_data.get('type', 'image/png')
                file_size = img_data.get('size', 0)

            if not base64_data:
                logger.warning(f"No base64 data for image {filename}, skipping")
                continue
            
            # Remove data URL prefix if present (e.g., "data:image/png;base64,")
            if base64_data.startswith('data:'):
                base64_data = base64_data.split(',', 1)[1]
            
            # Decode base64 data
            try:
                image_bytes = base64.b64decode(base64_data)
            except Exception as e:
                logger.error(f"Failed to decode base64 for {filename}: {e}")
                continue
            
            # Generate safe filename
            safe_filename = generate_safe_filename(filename, idx)
            file_path = os.path.join(images_folder, safe_filename)
            
            # Save the image file
            with open(file_path, 'wb') as f:
                f.write(image_bytes)
            
            # Return path relative to static folder for use in templates
            relative_path = f"images/{session_uuid}/{safe_filename}"
            saved_paths.append(relative_path)
            
            logger.info(f"Saved pasted image: {safe_filename} ({file_size} bytes, {file_type})")
            
        except Exception as e:
            logger.error(f"Error processing pasted image {idx}: {e}")
            continue
    
    logger.info(f"Successfully processed {len(saved_paths)} out of {len(pasted_images_data)} pasted images")
    return saved_paths


def generate_safe_filename(original_filename: str, index: int) -> str:
    """
    Generate a safe filename for saving pasted images.
    
    Args:
        original_filename: Original filename from frontend
        index: Index in the list for uniqueness
        
    Returns:
        Safe filename for filesystem use
    """
    import re
    import uuid
    from datetime import datetime
    
    # If no original filename, generate one
    if not original_filename or original_filename == f'pasted_image_{index}':
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        return f"pasted_image_{timestamp}_{index}.png"
    
    # Clean the original filename
    # Remove or replace unsafe characters
    safe_name = re.sub(r'[<>:"/\\|?*]', '_', original_filename)
    safe_name = safe_name.strip('.')  # Remove leading/trailing dots
    
    # Ensure it has an extension
    if '.' not in safe_name:
        safe_name += '.png'
    
    # Add index prefix to avoid collisions
    name_parts = safe_name.rsplit('.', 1)
    if len(name_parts) == 2:
        name, ext = name_parts
        safe_name = f"{index:02d}_{name}.{ext}"
    else:
        safe_name = f"{index:02d}_{safe_name}"
    
    # Limit length
    if len(safe_name) > 100:
        name_parts = safe_name.rsplit('.', 1)
        if len(name_parts) == 2:
            name, ext = name_parts
            safe_name = f"{name[:90]}.{ext}"
        else:
            safe_name = safe_name[:100]
    
    return safe_name