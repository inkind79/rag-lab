# models/llm_handlers/huggingface_handler.py
import os
import torch
from typing import List, Tuple, Optional, Any, Dict
from PIL import Image
from threading import Thread

from .base_handler import BaseLLMHandler
from src.utils.logger import get_logger
from src.utils.llm_utils import parse_conversation_history, get_image_file_path

logger = get_logger(__name__)


class HuggingFaceHandler(BaseLLMHandler):
    """
    Handler for HuggingFace Transformers vision-language models.
    Supports POINTS-Reader, Kosmos-2.5, Qwen3-VL and other HF models.
    """

    def __init__(self, model_choice: str, model_obj: Any, session_model_params: Dict):
        """
        Initialize the handler.

        Args:
            model_choice: The specific model identifier (e.g., 'huggingface-points-reader').
            model_obj: Tuple containing (model, processor, model_name).
            session_model_params: The 'model_params' dictionary loaded from the user's session/defaults.
        """
        super().__init__(model_choice, model_obj, session_model_params)

        # Unpack the model object - HuggingFace loader returns (model, processor, model_name)
        if isinstance(model_obj, tuple) and len(model_obj) >= 3:
            self.model = model_obj[0]  # The actual HuggingFace model
            self.processor = model_obj[1]  # The processor/tokenizer
            self.hf_model_name = model_obj[2]  # The HuggingFace model identifier
        else:
            logger.error(f"HuggingFaceHandler received unexpected model_obj format: {model_obj}")
            raise ValueError("HuggingFace model_obj format incorrect.")

        logger.info(f"Initialized HuggingFaceHandler for {model_choice} (HF model: {self.hf_model_name})")

    def _prepare_images(self, image_paths: List[str]) -> List[Image.Image]:
        """
        Load and prepare images for the model.

        Args:
            image_paths: List of paths to images

        Returns:
            List of PIL Image objects
        """
        images = []
        for img_path in image_paths:
            abs_path = get_image_file_path(img_path, make_absolute=True, verify_exists=True)
            if abs_path:
                try:
                    img = Image.open(abs_path).convert('RGB')
                    images.append(img)
                    logger.info(f"Loaded image: {abs_path}")
                except Exception as e:
                    logger.error(f"Failed to open image {abs_path}: {e}")
            else:
                logger.warning(f"Image file not found: {img_path}")

        return images

    def _build_prompt(self, query: str, conversation_context: str, ocr_context: str, has_images: bool) -> str:
        """
        Build the prompt for the HuggingFace model.

        Args:
            query: The user's query
            conversation_context: Pre-formatted conversation history
            ocr_context: OCR text context if available
            has_images: Whether images are provided

        Returns:
            The formatted prompt string
        """
        prompt_parts = []

        # Add conversation context if available
        if conversation_context:
            # For HuggingFace models, we'll format conversation history differently
            # Check if this is a summarized conversation
            if conversation_context.strip().startswith("Previous conversation summary:"):
                prompt_parts.append(conversation_context.strip())
            else:
                # Parse and format conversation history
                parsed_messages = parse_conversation_history(conversation_context, message_format='generic')
                if parsed_messages:
                    for msg in parsed_messages:
                        role = msg.get('role', 'unknown')
                        content = msg.get('content', '')
                        if role == 'user':
                            prompt_parts.append(f"User: {content}")
                        elif role == 'assistant':
                            prompt_parts.append(f"Assistant: {content}")

        # Add current query
        if has_images:
            if ocr_context:
                prompt_parts.append(f"User: {query}\n\nExtracted text from images:\n{ocr_context}\n\nPlease analyze both the images and the extracted text.")
            else:
                prompt_parts.append(f"User: {query}")
        else:
            # Text-only query
            if ocr_context:
                prompt_parts.append(f"User: {query}\n\nExtracted text:\n{ocr_context}")
            else:
                prompt_parts.append(f"User: {query}")

        return "\n\n".join(prompt_parts)

    def generate_response(
        self,
        images: List[str],
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
        Generates a response using HuggingFace models.
        """
        logger.info(f"--- HuggingFaceHandler: Generating response for session {session_id} using {self.model_choice} ({self.hf_model_name}) ---")

        try:
            # === PaddleOCR ===
            if "paddleocr" in self.hf_model_name.lower():
                if not images or len(images) == 0:
                    return "PaddleOCR requires document images. Please upload an image first.", []

                image_file = get_image_file_path(images[0], make_absolute=True, verify_exists=True)
                if not image_file:
                    return f"Could not find image file: {images[0]}", images

                # PaddleOCR pipeline ocr method
                try:
                    result = self.model.ocr(image_file, cls=True)

                    # Format the OCR results as markdown
                    output_lines = []
                    if result and len(result) > 0:
                        for line in result[0]:
                            if line:
                                # line format: [[[x1,y1], [x2,y2], [x3,y3], [x4,y4]], (text, confidence)]
                                text = line[1][0]
                                confidence = line[1][1]
                                output_lines.append(text)

                    output = '\n'.join(output_lines) if output_lines else "No text detected"
                    return output, images
                except Exception as e:
                    logger.error(f"PaddleOCR processing failed: {e}", exc_info=True)
                    return f"PaddleOCR processing failed: {str(e)}", images

            # === POINTS-Reader ===
            elif "points-reader" in self.hf_model_name.lower():
                if not images or len(images) == 0:
                    return "POINTS-Reader requires document images. Please upload an image first.", []

                image_file = get_image_file_path(images[0], make_absolute=True, verify_exists=True)
                if not image_file:
                    return f"Could not find image file: {images[0]}", images

                prompt = (
                    'Please extract all the text from the image with the following requirements:\n'
                    '1. Return tables in HTML format.\n'
                    '2. Return all other text in Markdown format.'
                )
                if query and query.strip():
                    prompt = query

                content = [
                    dict(type='image', image=image_file),
                    dict(type='text', text=prompt)
                ]
                messages = [{'role': 'user', 'content': content}]

                generation_config = {
                    'max_new_tokens': self.config.get('max_new_tokens', 2048),
                    'repetition_penalty': 1.05,
                    'temperature': self.config.get('temperature', 0.7),
                    'top_p': self.config.get('top_p', 0.8),
                    'top_k': self.config.get('top_k', 20),
                    'do_sample': True
                }

                tokenizer, image_processor = self.processor
                response = self.model.chat(messages, tokenizer, image_processor, generation_config)
                return response, images

            # === Kosmos-2.5 ===
            elif "kosmos-2.5" in self.hf_model_name.lower():
                if not images or len(images) == 0:
                    return "Kosmos-2.5 requires document images. Please upload an image first.", []

                pil_images = self._prepare_images(images)
                if len(pil_images) == 0:
                    return "Failed to load image", images

                prompt = "<md>"
                inputs = self.processor(
                    text=prompt,
                    images=pil_images[0],
                    return_tensors="pt"
                ).to(self.model.device)

                max_new_tokens = self.config.get('max_new_tokens', 2048)
                with torch.no_grad():
                    generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)

                generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                return generated_text, images

            # === Qwen3-VL and Nanonets-OCR2 (Qwen2.5-VL based) and olmOCR ===
            elif "qwen3-vl" in self.hf_model_name.lower() or "nanonets" in self.hf_model_name.lower() or "olmocr" in self.hf_model_name.lower():
                if not images or len(images) == 0:
                    return "This model requires images. Please upload an image first.", []

                image_file = get_image_file_path(images[0], make_absolute=True, verify_exists=True)
                if not image_file:
                    return f"Could not find image file: {images[0]}", images

                user_prompt = query if query else "Describe this image in detail."
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image_file},
                            {"type": "text", "text": user_prompt},
                        ],
                    }
                ]

                inputs = self.processor.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt"
                )
                inputs = inputs.to(self.model.device)

                max_new_tokens = self.config.get('max_new_tokens', 2048)
                with torch.no_grad():
                    generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)

                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in
                    zip(inputs.input_ids, generated_ids)
                ]
                generated_text = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0]
                return generated_text, images

            # === Standard/Generic HuggingFace ===
            else:
                pil_images = self._prepare_images(images) if images else []
                logger.info(f"Prepared {len(pil_images)} images for processing")

                ocr_context_str = self.prepare_ocr_context(
                    image_paths=images,
                    direct_ocr_results=direct_ocr_results,
                    session_id=session_id,
                    use_ocr=use_ocr
                )

                prompt = self._build_prompt(query, conversation_context, ocr_context_str, len(pil_images) > 0)
                logger.info(f"Built prompt: {len(prompt)} characters")

                if pil_images:
                    inputs = self.processor(
                        text=prompt,
                        images=pil_images,
                        return_tensors="pt"
                    ).to(self.model.device)
                else:
                    inputs = self.processor(
                        text=prompt,
                        return_tensors="pt"
                    ).to(self.model.device)

                max_new_tokens = self.config.get('max_new_tokens', 2048)
                temperature = self.config.get('temperature', 0.1)
                top_p = self.config.get('top_p', 0.9)
                top_k = self.config.get('top_k', 40)

                tokenizer = getattr(self.processor, 'tokenizer', self.processor)
                pad_token_id = getattr(tokenizer, 'eos_token_id', None) or getattr(tokenizer, 'pad_token_id', None)

                gen_kwargs = {
                    'max_new_tokens': max_new_tokens,
                    'temperature': temperature,
                    'top_p': top_p,
                    'top_k': top_k,
                    'do_sample': True if temperature > 0 else False,
                }
                if pad_token_id is not None:
                    gen_kwargs['pad_token_id'] = pad_token_id

                with torch.no_grad():
                    outputs = self.model.generate(**inputs, **gen_kwargs)

                generated_text = self.processor.decode(outputs[0], skip_special_tokens=True)

                if generated_text.startswith(prompt):
                    generated_text = generated_text[len(prompt):].strip()

                logger.info(f"Generated response: {len(generated_text)} characters")
                return generated_text, images

        except Exception as e:
            logger.error(f"Error during HuggingFace model generation: {str(e)}", exc_info=True)
            return f"An error occurred while processing the request with {self.hf_model_name}: {str(e)}", images

    def handle_direct_chat(
        self,
        query: str,
        conversation_context: str = ""
    ) -> Tuple[str, List[str]]:
        """
        Handles direct chat using HuggingFace models (no images).
        """
        logger.info(f"--- HuggingFaceHandler: Handling direct chat using {self.model_choice} ({self.hf_model_name}) ---")

        # Build the prompt
        prompt = self._build_prompt(query, conversation_context, "", False)

        try:
            # Prepare inputs (text-only)
            inputs = self.processor(
                text=prompt,
                return_tensors="pt"
            ).to(self.model.device)

            # Generate response
            max_new_tokens = self.config.get('max_new_tokens', 2048)
            temperature = self.config.get('temperature', 0.1)
            top_p = self.config.get('top_p', 0.9)
            top_k = self.config.get('top_k', 40)

            # Get tokenizer (processor might be tokenizer or have tokenizer attribute)
            tokenizer = getattr(self.processor, 'tokenizer', self.processor)
            pad_token_id = getattr(tokenizer, 'eos_token_id', None) or getattr(tokenizer, 'pad_token_id', None)

            gen_kwargs = {
                'max_new_tokens': max_new_tokens,
                'temperature': temperature,
                'top_p': top_p,
                'top_k': top_k,
                'do_sample': True if temperature > 0 else False,
            }
            if pad_token_id is not None:
                gen_kwargs['pad_token_id'] = pad_token_id

            with torch.no_grad():
                outputs = self.model.generate(**inputs, **gen_kwargs)

            # Decode the response
            generated_text = self.processor.decode(outputs[0], skip_special_tokens=True)

            # Remove the prompt from the generated text
            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt):].strip()

            return generated_text, []

        except Exception as e:
            logger.error(f"Error during HuggingFace direct chat: {str(e)}", exc_info=True)
            return f"An error occurred during direct chat with {self.hf_model_name}: {str(e)}", []

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
        Generate a streaming response using HuggingFace models.

        Yields chunks in the format:
        - {'type': 'response', 'content': '...'}
        """
        logger.info(f"--- HuggingFaceHandler: Starting streaming response for {self.hf_model_name} ---")

        try:
            # === PaddleOCR-VL: Uses PaddleOCR pipeline API ===
            if "paddleocr" in self.hf_model_name.lower():
                logger.info("Using PaddleOCR-VL pipeline API")

                if not images or len(images) == 0:
                    yield {'type': 'response', 'content': "PaddleOCR-VL requires document images. Please upload an image first."}
                    return

                # Get absolute image path
                image_file = get_image_file_path(images[0], make_absolute=True, verify_exists=True)
                if not image_file:
                    yield {'type': 'response', 'content': f"Could not find image file: {images[0]}"}
                    return

                # PaddleOCR pipeline ocr method
                try:
                    result = self.model.ocr(image_file, cls=True)

                    # Format the OCR results as markdown
                    output_lines = []
                    if result and len(result) > 0:
                        for line in result[0]:
                            if line:
                                # line format: [[[x1,y1], [x2,y2], [x3,y3], [x4,y4]], (text, confidence)]
                                text = line[1][0]
                                confidence = line[1][1]
                                output_lines.append(text)

                    output = '\n'.join(output_lines) if output_lines else "No text detected"
                    yield {'type': 'response', 'content': output}
                    logger.info(f"PaddleOCR completed")
                except Exception as e:
                    logger.error(f"PaddleOCR processing failed: {e}", exc_info=True)
                    yield {'type': 'response', 'content': f"PaddleOCR processing failed: {str(e)}"}
                return

            # === POINTS-Reader: Uses chat() method ===
            elif "points-reader" in self.hf_model_name.lower():
                logger.info("Using POINTS-Reader chat() API")

                if not images or len(images) == 0:
                    yield {'type': 'response', 'content': "POINTS-Reader requires document images. Please upload an image first."}
                    return

                # Get absolute image path
                image_file = get_image_file_path(images[0], make_absolute=True, verify_exists=True)
                if not image_file:
                    yield {'type': 'response', 'content': f"Could not find image file: {images[0]}"}
                    return

                # POINTS-Reader uses a specific prompt format
                prompt = (
                    'Please extract all the text from the image with the following requirements:\n'
                    '1. Return tables in HTML format.\n'
                    '2. Return all other text in Markdown format.'
                )
                if query and query.strip():
                    prompt = query  # Use user's custom query if provided

                # Build content list for POINTS-Reader
                content = [
                    dict(type='image', image=image_file),
                    dict(type='text', text=prompt)
                ]
                messages = [{'role': 'user', 'content': content}]

                # Generation config
                generation_config = {
                    'max_new_tokens': self.config.get('max_new_tokens', 2048),
                    'repetition_penalty': 1.05,
                    'temperature': self.config.get('temperature', 0.7),
                    'top_p': self.config.get('top_p', 0.8),
                    'top_k': self.config.get('top_k', 20),
                    'do_sample': True
                }

                # processor is a tuple of (tokenizer, image_processor) for POINTS-Reader
                tokenizer, image_processor = self.processor

                # POINTS-Reader doesn't support streaming, return result all at once
                response = self.model.chat(messages, tokenizer, image_processor, generation_config)
                yield {'type': 'response', 'content': response}
                logger.info(f"POINTS-Reader completed")
                return

            # === Kosmos-2.5: Uses generate() with <md> prompt ===
            elif "kosmos-2.5" in self.hf_model_name.lower():
                logger.info("Using Kosmos-2.5 generate() API")

                if not images or len(images) == 0:
                    yield {'type': 'response', 'content': "Kosmos-2.5 requires document images. Please upload an image first."}
                    return

                # Prepare image
                pil_images = self._prepare_images(images)
                if len(pil_images) == 0:
                    yield {'type': 'response', 'content': "Failed to load image"}
                    return

                # Kosmos-2.5 uses <md> token for markdown generation
                prompt = "<md>"

                # Process inputs
                inputs = self.processor(
                    text=prompt,
                    images=pil_images[0],
                    return_tensors="pt"
                )

                # Move inputs to device and cast to model's dtype
                inputs = {k: v.to(self.model.device, dtype=torch.bfloat16 if v.dtype == torch.float32 else v.dtype)
                         for k, v in inputs.items()}

                # Generate (no streaming support for Kosmos-2.5)
                max_new_tokens = self.config.get('max_new_tokens', 2048)
                with torch.no_grad():
                    generated_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens
                    )

                # Decode
                generated_text = self.processor.batch_decode(
                    generated_ids,
                    skip_special_tokens=True
                )[0]

                yield {'type': 'response', 'content': generated_text}
                logger.info(f"Kosmos-2.5 completed")
                return

            # === Qwen3-VL and Nanonets-OCR2 (Qwen2.5-VL based) and olmOCR: Uses apply_chat_template() ===
            elif "qwen3-vl" in self.hf_model_name.lower() or "nanonets" in self.hf_model_name.lower() or "olmocr" in self.hf_model_name.lower():
                logger.info(f"Using Qwen-VL apply_chat_template() API for {self.hf_model_name}")

                if not images or len(images) == 0:
                    yield {'type': 'response', 'content': "This model requires images. Please upload an image first."}
                    return

                # Get absolute image path
                image_file = get_image_file_path(images[0], make_absolute=True, verify_exists=True)
                if not image_file:
                    yield {'type': 'response', 'content': f"Could not find image file: {images[0]}"}
                    return

                # Build messages in Qwen3-VL format
                user_prompt = query if query else "Describe this image in detail."
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image_file},
                            {"type": "text", "text": user_prompt},
                        ],
                    }
                ]

                # Apply chat template
                inputs = self.processor.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt"
                )
                inputs = inputs.to(self.model.device)

                # Get generation parameters
                max_new_tokens = self.config.get('max_new_tokens', 2048)

                # Generate with streaming
                from transformers import TextIteratorStreamer
                tokenizer = self.processor.tokenizer
                streamer = TextIteratorStreamer(
                    tokenizer,
                    skip_prompt=True,
                    skip_special_tokens=True
                )

                generation_kwargs = dict(
                    **inputs,
                    streamer=streamer,
                    max_new_tokens=max_new_tokens
                )

                # Run generation in a separate thread
                thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
                thread.start()

                # Stream the output
                for text_chunk in streamer:
                    if text_chunk:
                        yield {'type': 'response', 'content': text_chunk}

                thread.join()
                logger.info(f"Qwen3-VL streaming completed")
                return

            # === Standard/Generic HuggingFace model handling ===
            else:
                logger.info("Using standard HuggingFace generate() API")
                from transformers import TextIteratorStreamer

                # Prepare images
                pil_images = self._prepare_images(images) if images else []
                logger.info(f"Prepared {len(pil_images)} images for streaming")

                # Get OCR context if needed
                ocr_context_str = self.prepare_ocr_context(
                    image_paths=images,
                    direct_ocr_results=direct_ocr_results,
                    session_id=session_id,
                    use_ocr=use_ocr
                )

                # Build the prompt
                prompt = self._build_prompt(query, conversation_context, ocr_context_str, len(pil_images) > 0)

                # Prepare inputs for the model
                if pil_images:
                    inputs = self.processor(
                        text=prompt,
                        images=pil_images,
                        return_tensors="pt"
                    ).to(self.model.device)
                else:
                    inputs = self.processor(
                        text=prompt,
                        return_tensors="pt"
                    ).to(self.model.device)

                # Get generation parameters
                max_new_tokens = self.config.get('max_new_tokens', 2048)
                temperature = self.config.get('temperature', 0.1)
                top_p = self.config.get('top_p', 0.9)
                top_k = self.config.get('top_k', 40)

                # Create streamer
                tokenizer = getattr(self.processor, 'tokenizer', self.processor)
                streamer = TextIteratorStreamer(
                    tokenizer,
                    skip_prompt=True,
                    skip_special_tokens=True
                )

                # Get pad token id
                pad_token_id = getattr(tokenizer, 'eos_token_id', None)
                if pad_token_id is None:
                    pad_token_id = getattr(tokenizer, 'pad_token_id', None)

                # Generation kwargs
                generation_kwargs = dict(
                    **inputs,
                    streamer=streamer,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    do_sample=True if temperature > 0 else False,
                )

                if pad_token_id is not None:
                    generation_kwargs['pad_token_id'] = pad_token_id

                # Run generation in a separate thread
                thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
                thread.start()

                # Stream the output
                for text_chunk in streamer:
                    if text_chunk:
                        yield {'type': 'response', 'content': text_chunk}

                thread.join()
                logger.info(f"Streaming completed for {self.hf_model_name}")

        except Exception as e:
            logger.error(f"Error during HuggingFace streaming: {str(e)}", exc_info=True)
            yield {'type': 'response', 'content': f"An error occurred during streaming with {self.hf_model_name}: {str(e)}"}
