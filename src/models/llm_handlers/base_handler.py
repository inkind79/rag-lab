# models/llm_handlers/base_handler.py
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Any, Dict
import time

# Import model configuration utilities (only needed for mapping/defaults now)
from src.utils.model_configs import MODEL_MAPPING, DEFAULT_MODEL_CONFIGS
from src.models.retrieval_result import RetrievalResult, ResultType
from src.utils.logger import get_logger
import copy # For deep copying defaults

logger = get_logger(__name__)

class BaseLLMHandler(ABC):
    """
    Abstract base class for Large Language Model handlers.
    Defines the interface for generating responses and handling direct chat.
    """

    def __init__(self, model_choice: str, model_obj: Any, session_model_params: Dict):
        """
        Initialize the handler with model-specific configurations from session data.

        Args:
            model_choice: The specific model identifier (e.g., 'chatgpt', 'ollama-llama-vision').
            model_obj: The loaded model object or client needed by the handler.
            session_model_params: The 'model_params' dictionary loaded from the user's session/defaults.
        """
        self.model_choice = model_choice
        self.model_obj = model_obj

        # Determine provider and API model name
        if model_choice not in MODEL_MAPPING:
            logger.error(f"Model choice '{model_choice}' not found in MODEL_MAPPING. Cannot determine provider/API name.")
            # Handle error appropriately - maybe raise an exception or set defaults
            self.provider = "unknown"
            self.api_model_name = model_choice
            self.config = copy.deepcopy(DEFAULT_MODEL_CONFIGS.get('default', {}))
        else:
            self.provider, self.api_model_name = MODEL_MAPPING[model_choice]

            # Extract the relevant config from the passed-in session_model_params
            # Start with base defaults for the specific model
            base_provider_defaults = DEFAULT_MODEL_CONFIGS.get(self.provider, {})
            base_model_defaults = base_provider_defaults.get('models', {}).get(self.api_model_name, {})
            merged_config = copy.deepcopy(DEFAULT_MODEL_CONFIGS.get('default', {})) # Start with global default
            # Merge provider level base defaults (excluding models)
            for key, val in base_provider_defaults.items():
                 if key != 'models': merged_config[key] = val
            # Merge model specific base defaults
            merged_config.update(base_model_defaults)

            # Now, overlay the session/user specific parameters
            user_provider_params = session_model_params.get(self.provider, {})
            user_model_specific_params = user_provider_params.get('models', {}).get(self.api_model_name, {})

            # Merge user provider level params (excluding models)
            for key, val in user_provider_params.items():
                 if key != 'models': merged_config[key] = val
            # Merge user model specific params
            merged_config.update(user_model_specific_params)

            self.config = merged_config

        # Log configuration values for debugging
        logger.info(f"Initialized {self.__class__.__name__} for {model_choice} with config: {self.config}")

    @abstractmethod
    def generate_response(
        self,
        images: List[str],
        query: str,
        session_id: str,
        chat_history: Optional[List[Dict[str, Any]]] = None,
        direct_ocr_results: Optional[Dict[str, Any]] = None,
        original_query: Optional[str] = None,
        use_ocr: bool = False,
        conversation_context: str = "", # Pass pre-formatted context
        use_score_slope: bool = False, # Flag to indicate if score-slope analysis is enabled
        retrieved_text_context: str = "", # Text context from text-based retrievers
    ) -> Tuple[str, List[str]]:
        """
        Generates a response based on images, query, and context.

        Args:
            images: List of validated FULL image paths.
            query: The text query from the user.
            session_id: Current session identifier.
            chat_history: Optional raw chat history (list of message dicts).
            direct_ocr_results: Optional OCR results.
            original_query: Optional original query if rewritten.
            use_ocr: Boolean flag indicating if OCR should be used.
            conversation_context: Pre-formatted conversation history string.
            use_score_slope: Flag for score-slope analysis.
            retrieved_text_context: Text context from text-based retrievers (BM25, dense).
                When non-empty, this should be included in the prompt alongside/instead of images.

        Returns:
            Tuple[str, List[str]]: (response_text, used_images_full_paths)
        """
        pass

    @staticmethod
    def extract_images_and_text(retrieval_results: List[RetrievalResult]) -> Tuple[List[str], str]:
        """Split retrieval results into image paths and formatted text context.

        Utility for converting RetrievalResult objects into the two formats
        that LLM handlers need: image paths for vision models and text context
        for text-only models or text-based retrieval results.

        Args:
            retrieval_results: List of RetrievalResult objects from any retriever

        Returns:
            Tuple of (image_paths, text_context_string)
        """
        image_paths = []
        text_chunks = []

        for r in retrieval_results:
            if r.result_type == ResultType.IMAGE and r.image_path:
                image_paths.append(r.image_path)
            elif r.result_type == ResultType.TEXT_CHUNK and r.text_content:
                source = r.source_document or r.original_filename or "unknown"
                page = f", Page {r.page_num}" if r.page_num else ""
                text_chunks.append(f"[Source: {source}{page}]\n{r.text_content}")
            elif r.result_type == ResultType.HYBRID:
                if r.image_path:
                    image_paths.append(r.image_path)
                if r.text_content:
                    source = r.source_document or r.original_filename or "unknown"
                    page = f", Page {r.page_num}" if r.page_num else ""
                    text_chunks.append(f"[Source: {source}{page}]\n{r.text_content}")

        text_context = "\n\n---\n\n".join(text_chunks) if text_chunks else ""
        return image_paths, text_context

    @abstractmethod
    def handle_direct_chat(
        self,
        query: str,
        conversation_context: str = "" # Pass pre-formatted context
    ) -> Tuple[str, List[str]]:
        """
        Handles direct chat without document context (RAG).

        Args:
            query: User's current query.
            conversation_context: Pre-formatted conversation history string.

        Returns:
            Tuple[str, List[str]]: (response_text, empty_list_for_images)
                                   The second element is always an empty list for direct chat.
        """
        pass

    # Helper methods common across handlers
    def prepare_ocr_context(self, image_paths: List[str], direct_ocr_results: Optional[Dict[str, Any]] = None, session_id: str = None, use_ocr: bool = False, timeout: int = 1800) -> str:
        """
        Prepare OCR context for the prompt, waiting for results if needed.
        This is called at the last possible moment before sending the prompt to the model.

        Args:
            image_paths: List of image paths to process
            direct_ocr_results: Pre-processed OCR results if available
            session_id: Current session ID for OCR processing
            use_ocr: Whether OCR should be used
            timeout: Maximum time to wait for OCR results in seconds

        Returns:
            OCR context string to include in the prompt
        """
        if not use_ocr or not session_id:
            return ""

        ocr_context = ""
        start_time = time.time()

        # Use provided OCR results if available
        if direct_ocr_results:
            # If pre-formatted context is already available, use it directly
            if 'ocr_context' in direct_ocr_results and direct_ocr_results['ocr_context']:
                ocr_context = direct_ocr_results['ocr_context']
                logger.info(f"Using pre-formatted OCR context: {len(ocr_context)} characters")
            # Otherwise format it using the utility function
            elif 'results' in direct_ocr_results:
                try:
                    from src.utils.ocr_utils import format_ocr_context
                    ocr_context = format_ocr_context(direct_ocr_results)
                    logger.info(f"Formatted OCR context using utility: {len(ocr_context)} characters")
                except Exception as e:
                    logger.error(f"Error formatting OCR results: {e}")
                    return ""
        else:
            # Request synchronous OCR processing with timeout
            try:
                from src.utils.ocr_utils import process_images_with_ocr
                logger.info(f"Waiting for OCR results (timeout: {timeout}s)...")

                # All preparation steps have been done asynchronously up to this point
                # Now we wait for OCR results just before sending the prompt to the model
                ocr_results = process_images_with_ocr(
                    image_paths=image_paths,
                    session_id=session_id,
                    save_results=True,
                    pre_formatted=True,
                    wait_for_results=True,  # This is the key parameter to make it synchronous
                    timeout=timeout  # Add timeout to prevent indefinite waiting
                )

                if 'ocr_context' in ocr_results:
                    ocr_context = ocr_results['ocr_context']
                    logger.info(f"Generated synchronous OCR results: {len(ocr_context)} characters in {time.time() - start_time:.2f}s")
                else:
                    logger.warning("OCR processing completed but no context was returned")
            except Exception as e:
                logger.error(f"Error during synchronous OCR processing: {e}")
                return ""

        # We no longer apply OCR prompt template from the template
        # Instead, each handler implements its own OCR formatting

        return ocr_context
