import os
import imghdr
from PyPDF2 import PdfReader
from src.utils.logger import get_logger

logger = get_logger(__name__)

def index_documents(folder_path):
    indexed_files_info = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        # Handle PDF files
        if filename.lower().endswith('.pdf'):
            try:
                with open(file_path, 'rb') as pdf_file:
                    reader = PdfReader(pdf_file)
                    page_count = len(reader.pages)
            except Exception as e:
                logger.error(f"Error reading PDF {filename}: {e}")
                page_count = None
            indexed_files_info.append({
                'filename': filename,
                'page_count': page_count,
                'file_type': 'pdf'
            })
            logger.info(f"Indexed PDF: {filename} with {page_count if page_count is not None else 'unknown'} pages")
        
        # Handle image files
        elif any(filename.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.gif', '.bmp']):
            # Verify it's actually an image
            try:
                img_type = imghdr.what(file_path)
                if img_type:
                    indexed_files_info.append({
                        'filename': filename,
                        'page_count': 1,  # Each image counts as 1 page
                        'file_type': 'image',
                        'image_type': img_type
                    })
                    logger.info(f"Indexed image: {filename} ({img_type})")
                else:
                    logger.warning(f"File has image extension but is not a valid image: {filename}")
            except Exception as e:
                logger.error(f"Error processing image {filename}: {e}")
        
        # Detect images by content rather than extension (for pasted images without standard extensions)
        elif os.path.isfile(file_path):
            try:
                img_type = imghdr.what(file_path)
                if img_type:
                    indexed_files_info.append({
                        'filename': filename,
                        'page_count': 1,  # Each image counts as 1 page
                        'file_type': 'image',
                        'image_type': img_type
                    })
                    logger.info(f"Indexed image (detected by content): {filename} ({img_type})")
                else:
                    # Handle other file types here if needed
                    logger.info(f"Skipping unknown file type: {filename}")
            except Exception as e:
                logger.error(f"Error examining file {filename}: {e}")
                
    return indexed_files_info
