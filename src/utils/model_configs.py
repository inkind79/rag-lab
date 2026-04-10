"""
Model configuration management.

This module provides utilities for managing model-specific configurations
across different providers (Ollama, HuggingFace) and model types.
"""
import copy # Keep copy for deepcopying defaults if needed elsewhere
import logging
# Removed os and json imports as file operations are removed
from typing import Dict, Any, Optional, List, Tuple

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Path to the model configurations file (REMOVED - No longer used for loading/saving)
# MODEL_CONFIGS_PATH = "config/model_configs.json"

# Default configurations for all models and providers
DEFAULT_MODEL_CONFIGS = {
    "default": {
        "temperature": 0.1,
        "max_tokens": 1024,
    },
    "ollama": {
        "temperature": 0.1,
        "num_predict": 2048,
        "num_ctx": 8192,
        "repeat_penalty": 1.1,
        "top_p": 0.9,
        "top_k": 40,
        "models": {
            "llama-vision": {
                "temperature": 0.1,
                "description": "Llama 3.2 Vision 11B",
                "supports_vision": True,
                "num_predict": 2048,
                "num_ctx": 8192,
                "repeat_penalty": 1.1,
                "top_p": 0.9,
                "top_k": 40
            },
            "llama3.2-vision:11b-instruct-q8_0": {
                "temperature": 0.1,
                "description": "Meta - Llama 3.2 Vision 11B (Q8)",
                "supports_vision": True,
                "num_predict": 2048,
                "num_ctx": 8192,
                "repeat_penalty": 1.1,
                "top_p": 0.9,
                "top_k": 40
            },
            "llama3.2-vision:11b-instruct-fp16": {
                "temperature": 0.1,
                "description": "Meta - Llama 3.2 Vision 11B (FP16)",
                "supports_vision": True,
                "num_predict": 2048,
                "num_ctx": 8192,
                "repeat_penalty": 1.1,
                "top_p": 0.9,
                "top_k": 40
            },
            "minicpm-vision": {
                "temperature": 0.1,
                "description": "MiniCPM Vision 7B",
                "supports_vision": True,
                "num_predict": 2048,
                "num_ctx": 8192,
                "repeat_penalty": 1.1,
                "top_p": 0.9,
                "top_k": 40
            },
            "granite3.2-vision": {
                "temperature": 0.1,
                "description": "Granite 3.2 Vision 8B",
                "supports_vision": True,
                "num_predict": 2048,
                "num_ctx": 8192,
                "repeat_penalty": 1.1,
                "top_p": 0.9,
                "top_k": 40
            },
            "gemma-vision": {
                "temperature": 0.1,
                "description": "Gemma 3 Vision (12B)",
                "supports_vision": True,
                "num_predict": 2048,
                "num_ctx": 8192,
                "repeat_penalty": 1.1,
                "top_p": 0.9,
                "top_k": 40
            },
            "gemma-vision-27b": {
                "temperature": 0.1,
                "description": "Gemma 3 Vision (27B)",
                "supports_vision": True,
                "num_predict": 2048,
                "num_ctx": 4096,
                "repeat_penalty": 1.1,
                "top_p": 0.9,
                "top_k": 40
            },
            "gemma3:27b": {
                "temperature": 0.1,
                "description": "Google - Gemma 3 Vision 27B (Q4)",
                "supports_vision": True,
                "num_predict": 2048,
                "num_ctx": 4096,
                "repeat_penalty": 1.1,
                "top_p": 0.9,
                "top_k": 40
            },
            "phi4": {
                "temperature": 0.1,
                "description": "Phi-4 Plus 14B (Text-only)",
                "supports_vision": False,
                "num_predict": 2048,
                "num_ctx": 8192,
                "repeat_penalty": 1.1,
                "top_p": 0.9,
                "top_k": 40
            },
            "phi4-mini": {
                "temperature": 0.1,
                "description": "Phi-4 Mini 3.8B (Text-only)",
                "supports_vision": False,
                "num_predict": 2048,
                "num_ctx": 8192,
                "repeat_penalty": 1.1,
                "top_p": 0.9,
                "top_k": 40
            },
            "olmo2": {
                "temperature": 0.1,
                "description": "OLMo 2:13B (Text-only)",
                "supports_vision": False,
                "num_predict": 2048,
                "num_ctx": 8192,
                "repeat_penalty": 1.1,
                "top_p": 0.9,
                "top_k": 40
            },
            "llama4:scout": {
                "temperature": 0.1,
                "description": "Meta - Llama 4 Scout 109B (Q4)",
                "supports_vision": True,
                "num_predict": 1024,
                "repeat_penalty": 1.1,
                "top_p": 0.9,
                "top_k": 40,
                "num_ctx": 2048
            },
            "llama4:17b-scout-16e-instruct-q8_0": {
                "temperature": 0.1,
                "description": "Meta - Llama 4 Scout 109B (Q8)",
                "supports_vision": True,
                "num_predict": 1024,
                "repeat_penalty": 1.1,
                "top_p": 0.9,
                "top_k": 40,
                "num_ctx": 2048
            },
            "llama4:17b-scout-16e-instruct-fp16": {
                "temperature": 0.1,
                "description": "Meta - Llama 4 Scout 109B (FP16)",
                "supports_vision": True,
                "num_predict": 1024,
                "repeat_penalty": 1.1,
                "top_p": 0.9,
                "top_k": 40,
                "num_ctx": 2048
            },
            "llama4:maverick": {
                "temperature": 0.1,
                "description": "Meta - Llama 4 Maverick 402B (Q4)",
                "supports_vision": True,
                "num_predict": 1024,
                "repeat_penalty": 1.1,
                "top_p": 0.9,
                "top_k": 40,
                "num_ctx": 2048
            },
            "llama4:17b-maverick-128e-instruct-q8_0": {
                "temperature": 0.1,
                "description": "Meta - Llama 4 Maverick 402B (Q8)",
                "supports_vision": True,
                "num_predict": 1024,
                "repeat_penalty": 1.1,
                "top_p": 0.9,
                "top_k": 40,
                "num_ctx": 2048
            },
            "llama4:17b-maverick-128e-instruct-fp16": {
                "temperature": 0.1,
                "description": "Meta - Llama 4 Maverick 402B (FP16)",
                "supports_vision": True,
                "num_predict": 1024,
                "repeat_penalty": 1.1,
                "top_p": 0.9,
                "top_k": 40,
                "num_ctx": 2048
            },
            "qwen2.5vl:7b": {
                "temperature": 0.1,
                "description": "Alibaba - Qwen 2.5 Vision 7B (Q4)",
                "supports_vision": True,
                "num_predict": 4096,
                "repeat_penalty": 1.1,
                "top_p": 0.9,
                "top_k": 40,
                "num_ctx": 8192
            },
            "qwen2.5vl:7b-q8_0": {
                "temperature": 0.1,
                "description": "Alibaba - Qwen 2.5 Vision 7B (Q8)",
                "supports_vision": True,
                "num_predict": 4096,
                "repeat_penalty": 1.1,
                "top_p": 0.9,
                "top_k": 40,
                "num_ctx": 8192
            },
            "qwen2.5vl:7b-fp16": {
                "temperature": 0.1,
                "description": "Alibaba - Qwen 2.5 Vision 7B (FP16)",
                "supports_vision": True,
                "num_predict": 4096,
                "repeat_penalty": 1.1,
                "top_p": 0.9,
                "top_k": 40,
                "num_ctx": 8192
            },
            "qwen2.5vl:32b": {
                "temperature": 0.1,
                "description": "Alibaba - Qwen 2.5 Vision 32B (Q4)",
                "supports_vision": True,
                "num_predict": 4096,
                "repeat_penalty": 1.1,
                "top_p": 0.9,
                "top_k": 40,
                "num_ctx": 4096
            },
            "qwen2.5vl:32b-q8_0": {
                "temperature": 0.1,
                "description": "Alibaba - Qwen 2.5 Vision 32B (Q8)",
                "supports_vision": True,
                "num_predict": 4096,
                "repeat_penalty": 1.1,
                "top_p": 0.9,
                "top_k": 40,
                "num_ctx": 4096
            },
            "qwen2.5vl:32b-fp16": {
                "temperature": 0.1,
                "description": "Alibaba - Qwen 2.5 Vision 32B (FP16)",
                "supports_vision": True,
                "num_predict": 4096,
                "repeat_penalty": 1.1,
                "top_p": 0.9,
                "top_k": 40,
                "num_ctx": 4096
            },
            "qwen2.5vl:72b": {
                "temperature": 0.1,
                "description": "Alibaba - Qwen 2.5 Vision 72B (Q4)",
                "supports_vision": True,
                "num_predict": 4096,
                "repeat_penalty": 1.1,
                "top_p": 0.9,
                "top_k": 40,
                "num_ctx": 2048
            },
            "qwen2.5vl:72b-q8_0": {
                "temperature": 0.1,
                "description": "Alibaba - Qwen 2.5 Vision 72B (Q8)",
                "supports_vision": True,
                "num_predict": 4096,
                "repeat_penalty": 1.1,
                "top_p": 0.9,
                "top_k": 40,
                "num_ctx": 2048
            },
            "qwen2.5vl:72b-fp16": {
                "temperature": 0.1,
                "description": "Alibaba - Qwen 2.5 Vision 72B (FP16)",
                "supports_vision": True,
                "num_predict": 4096,
                "repeat_penalty": 1.1,
                "top_p": 0.9,
                "top_k": 40,
                "num_ctx": 2048
            },
            "gemma3:27b-it-q8_0": {
                "temperature": 0.1,
                "description": "Google - Gemma 3 Vision 27B (Q8)",
                "supports_vision": True,
                "num_predict": 2048,
                "repeat_penalty": 1.1,
                "top_p": 0.9,
                "top_k": 40,
                "num_ctx": 4096
            },
            "gemma3:27b-it-fp16": {
                "temperature": 0.1,
                "description": "Google - Gemma 3 Vision 27B (FP16)",
                "supports_vision": True,
                "num_predict": 2048,
                "repeat_penalty": 1.1,
                "top_p": 0.9,
                "top_k": 40,
                "num_ctx": 4096
            },
            "gemma3n:e4b-it-fp16": {
                "temperature": 0.1,
                "description": "Google - Gemma 3n E4B (FP16)",
                "supports_vision": True,
                "num_predict": 2048,
                "repeat_penalty": 1.1,
                "top_p": 0.9,
                "top_k": 40,
                "num_ctx": 8192
            },
            "mistral-small3.2:latest": {
                "temperature": 0.1,
                "description": "Mistral - Small 3.2 24B (Q4)",
                "supports_vision": True,
                "num_predict": 2048,
                "repeat_penalty": 1.1,
                "top_p": 0.9,
                "top_k": 40,
                "num_ctx": 4096
            },
            "mistral-small3.2:24b-instruct-2506-q8_0": {
                "temperature": 0.1,
                "description": "Mistral - Small 3.2 24B (Q8)",
                "supports_vision": True,
                "num_predict": 2048,
                "repeat_penalty": 1.1,
                "top_p": 0.9,
                "top_k": 40,
                "num_ctx": 4096
            },
            "mistral-small3.2:24b-instruct-2506-fp16": {
                "temperature": 0.1,
                "description": "Mistral - Small 3.2 24B (FP16)",
                "supports_vision": True,
                "num_predict": 2048,
                "repeat_penalty": 1.1,
                "top_p": 0.9,
                "top_k": 40,
                "num_ctx": 4096
            },
            "exaone-deep:32b": {
                "temperature": 0.1,
                "description": "LG - EXAONE Deep 32B (Q4)",
                "supports_vision": False,
                "num_predict": 2048,
                "repeat_penalty": 1.1,
                "top_p": 0.9,
                "top_k": 40,
                "num_ctx": 4096
            },
            "qwq": {
                "temperature": 0.1,
                "description": "Alibaba - QWQ 32B (Q4)",
                "supports_vision": False,
                "num_predict": 8192,
                "repeat_penalty": 1.1,
                "top_p": 0.9,
                "top_k": 40,
                "num_ctx": 8192
            },
            "qwen3:30b-a3b": {
                "temperature": 0.1,
                "description": "Alibaba - Qwen3 30B (Q4)",
                "supports_vision": False,
                "num_predict": 8192,
                "repeat_penalty": 1.1,
                "top_p": 0.9,
                "top_k": 40,
                "num_ctx": 8192
            },
            "qwen3:30b-a3b-q8_0": {
                "temperature": 0.1,
                "description": "Alibaba - Qwen3 30B (Q8)",
                "supports_vision": False,
                "num_predict": 8192,
                "repeat_penalty": 1.1,
                "top_p": 0.9,
                "top_k": 40,
                "num_ctx": 8192
            },
            "qwen3:30b-a3b-fp16": {
                "temperature": 0.1,
                "description": "Alibaba - Qwen3 30B (FP16)",
                "supports_vision": False,
                "num_predict": 8192,
                "repeat_penalty": 1.1,
                "top_p": 0.9,
                "top_k": 40,
                "num_ctx": 8192
            },
            "devstral:24b": {
                "temperature": 0.1,
                "description": "Mistral - Codestral 24B (Q4)",
                "supports_vision": False,
                "num_predict": 4096,
                "repeat_penalty": 1.1,
                "top_p": 0.9,
                "top_k": 40,
                "num_ctx": 4096
            },
            "devstral:24b-small-2505-q8_0": {
                "temperature": 0.1,
                "description": "Mistral - Codestral 24B (Q8)",
                "supports_vision": False,
                "num_predict": 4096,
                "repeat_penalty": 1.1,
                "top_p": 0.9,
                "top_k": 40,
                "num_ctx": 4096
            },
            "devstral:24b-small-2505-fp16": {
                "temperature": 0.1,
                "description": "Mistral - Codestral 24B (FP16)",
                "supports_vision": False,
                "num_predict": 4096,
                "repeat_penalty": 1.1,
                "top_p": 0.9,
                "top_k": 40,
                "num_ctx": 4096
            },
            "deepcoder:14b-preview-q8_0": {
                "temperature": 0.1,
                "description": "Agentica - DeepCoder 14B (Q8)",
                "supports_vision": False,
                "num_predict": 2048,
                "num_ctx": 8192,
                "repeat_penalty": 1.1,
                "top_p": 0.9,
                "top_k": 40
            },
            "deepcoder:14b-preview-fp16": {
                "temperature": 0.1,
                "description": "Agentica - DeepCoder 14B (FP16)",
                "supports_vision": False,
                "num_predict": 2048,
                "num_ctx": 8192,
                "repeat_penalty": 1.1,
                "top_p": 0.9,
                "top_k": 40
            },
            "granite3.3": {
                "temperature": 0.1,
                "description": "IBM - Granite 3.3 8B (Q4)",
                "supports_vision": False,
                "num_ctx": 8192,
                "num_predict": 8192,
                "repeat_penalty": 1.1,
                "top_p": 0.9,
                "top_k": 40
            },
            "granite3.3:2b": {
                "temperature": 0.1,
                "description": "IBM - Granite 3.3 2B (Q4)",
                "supports_vision": False,
                "num_ctx": 8192,
                "num_predict": 8192,
                "repeat_penalty": 1.1,
                "top_p": 0.9,
                "top_k": 40
            },
            "gemma3:12b": {
                "temperature": 0.1,
                "description": "Google - Gemma 3 Vision 12B (Q4)",
                "supports_vision": True,
                "num_predict": 2048,
                "num_ctx": 8192,
                "repeat_penalty": 1.1,
                "top_p": 0.9,
                "top_k": 40
            },
            "gemma3:12b-it-q8_0": {
                "temperature": 0.1,
                "description": "Google - Gemma 3 Vision 12B (Q8)",
                "supports_vision": True,
                "num_predict": 2048,
                "num_ctx": 8192,
                "repeat_penalty": 1.1,
                "top_p": 0.9,
                "top_k": 40
            },
            "gemma3:12b-it-fp16": {
                "temperature": 0.1,
                "description": "Google - Gemma 3 Vision 12B (FP16)",
                "supports_vision": True,
                "num_predict": 2048,
                "num_ctx": 8192,
                "repeat_penalty": 1.1,
                "top_p": 0.9,
                "top_k": 40
            },
            "minicpm-v": {
                "temperature": 0.5,
                "description": "MiniCPM Vision",
                "supports_vision": True,
                "num_predict": 2048,
                "num_ctx": 8192,
                "repeat_penalty": 1.1,
                "top_p": 0.95,
                "top_k": 40
            },
            "phi4:14b-q8_0": {
                "temperature": 0.1,
                "description": "Microsoft - Phi-4 Plus 14B (Q8)",
                "supports_vision": False,
                "num_predict": 2048,
                "num_ctx": 8192,
                "repeat_penalty": 1.1,
                "top_p": 0.9,
                "top_k": 40
            },
            "phi4:14b-fp16": {
                "temperature": 0.1,
                "description": "Microsoft - Phi-4 Plus 14B (FP16)",
                "supports_vision": False,
                "num_predict": 2048,
                "num_ctx": 8192,
                "repeat_penalty": 1.1,
                "top_p": 0.9,
                "top_k": 40
            },
            "phi4-reasoning:plus": {
                "temperature": 0.1,
                "description": "Microsoft - Phi-4 Plus Reasoning (Q4)",
                "supports_vision": False,
                "num_ctx": 8192,
                "num_predict": 8192,
                "repeat_penalty": 1.1,
                "top_p": 0.9,
                "top_k": 40
            },
            "phi4-mini-reasoning": {
                "temperature": 0.1,
                "description": "Microsoft - Phi-4 Mini Reasoning 3.8B (Q4)",
                "supports_vision": False,
                "num_ctx": 8192,
                "num_predict": 8192,
                "repeat_penalty": 1.1,
                "top_p": 0.9,
                "top_k": 40
            },
            "phi4-mini-reasoning:latest": {
                "temperature": 0.1,
                "description": "Microsoft - Phi-4 Mini Reasoning (Latest)",
                "supports_vision": False,
                "num_ctx": 8192,
                "num_predict": 8192,
                "repeat_penalty": 1.1,
                "top_p": 0.9,
                "top_k": 40
            },
            "phi4-mini-reasoning:3.8b-fp16": {
                "temperature": 0.1,
                "description": "Microsoft - Phi-4 Mini Reasoning 3.8B (FP16)",
                "supports_vision": False,
                "num_ctx": 8192,
                "num_predict": 8192,
                "repeat_penalty": 1.1,
                "top_p": 0.9,
                "top_k": 40
            },
            "llama3.2-vision": {
                "temperature": 0.1,
                "description": "Meta - Llama 3.2 Vision 11B (Q4)",
                "supports_vision": True,
                "num_predict": 2048,
                "repeat_penalty": 1.1,
                "top_p": 0.9,
                "top_k": 40,
                "num_ctx": 8192
            },
            "deepseek-r1:8b-0528-qwen3-q4_K_M": {
                "temperature": 0.1,
                "description": "DeepSeek - R1 8B (Q4)",
                "supports_vision": False,
                "num_predict": 8192,
                "repeat_penalty": 1.1,
                "top_p": 0.9,
                "top_k": 40,
                "num_ctx": 8192
            },
            "deepseek-r1:8b-0528-qwen3-q8_0": {
                "temperature": 0.1,
                "description": "DeepSeek - R1 8B (Q8)",
                "supports_vision": False,
                "num_predict": 8192,
                "repeat_penalty": 1.1,
                "top_p": 0.9,
                "top_k": 40,
                "num_ctx": 8192
            },
            "deepseek-r1:8b-0528-qwen3-fp16": {
                "temperature": 0.1,
                "description": "DeepSeek - R1 8B (FP16)",
                "supports_vision": False,
                "num_predict": 8192,
                "repeat_penalty": 1.1,
                "top_p": 0.9,
                "top_k": 40,
                "num_ctx": 8192
            },
            "magistral:24b": {
                "temperature": 0.1,
                "description": "Mistral - Magistral 24B (Q4) - [Text Only]",
                "supports_vision": False,
                "num_predict": 16384,
                "repeat_penalty": 1.1,
                "top_p": 0.9,
                "top_k": 40,
                "num_ctx": 16384
            },
            "magistral:24b-small-2506-q8_0": {
                "temperature": 0.1,
                "description": "Mistral - Magistral 24B (Q8) - [Text Only]",
                "supports_vision": False,
                "num_predict": 16384,
                "repeat_penalty": 1.1,
                "top_p": 0.9,
                "top_k": 40,
                "num_ctx": 16384
            },
            "magistral:24b-small-2506-fp16": {
                "temperature": 0.1,
                "description": "Mistral - Magistral 24B (FP16) - [Text Only]",
                "supports_vision": False,
                "num_predict": 16384,
                "repeat_penalty": 1.1,
                "top_p": 0.9,
                "top_k": 40,
                "num_ctx": 16384
            }
        }
    },
    "huggingface": {
        "temperature": 0.1,
        "max_new_tokens": 2048,
        "top_p": 0.9,
        "top_k": 40,
        "models": {
            "tencent/POINTS-Reader": {
                "temperature": 0.7,
                "description": "Tencent - POINTS-Reader (Document Extraction with Structured Output)",
                "supports_vision": True,
                "max_new_tokens": 2048,
                "top_p": 0.8,
                "top_k": 20,
                "repetition_penalty": 1.05,
                "do_sample": True
            },
            "microsoft/kosmos-2.5": {
                "temperature": 0.1,
                "description": "Microsoft - Kosmos-2.5 (OCR Specialist with Markdown Conversion)",
                "supports_vision": True,
                "max_new_tokens": 2048,
                "top_p": 0.9,
                "top_k": 40
            },
            "Qwen/Qwen3-VL-8B-Instruct": {
                "temperature": 0.1,
                "description": "Alibaba - Qwen3-VL 8B Instruct (Vision-Language Model)",
                "supports_vision": True,
                "max_new_tokens": 2048,
                "top_p": 0.9,
                "top_k": 40
            },
            "Qwen/Qwen3-VL-4B-Instruct": {
                "temperature": 0.1,
                "description": "Alibaba - Qwen3-VL 4B Instruct (Vision-Language Model)",
                "supports_vision": True,
                "max_new_tokens": 2048,
                "top_p": 0.9,
                "top_k": 40
            },
            "Qwen/Qwen3-VL-8B-Instruct-FP8": {
                "temperature": 0.1,
                "description": "Alibaba - Qwen3-VL 8B Instruct FP8 (Vision-Language Model)",
                "supports_vision": True,
                "max_new_tokens": 2048,
                "top_p": 0.9,
                "top_k": 40
            },
            "Qwen/Qwen3-VL-4B-Instruct-FP8": {
                "temperature": 0.1,
                "description": "Alibaba - Qwen3-VL 4B Instruct FP8 (Vision-Language Model)",
                "supports_vision": True,
                "max_new_tokens": 2048,
                "top_p": 0.9,
                "top_k": 40
            },
            "nanonets/Nanonets-OCR2-3B": {
                "temperature": 0.1,
                "description": "Nanonets - OCR2 3B (Advanced Document OCR with Markdown/LaTeX)",
                "supports_vision": True,
                "max_new_tokens": 2048,
                "top_p": 0.9,
                "top_k": 40,
                "repetition_penalty": 1.0
            },
            "PaddlePaddle/PaddleOCR-VL": {
                "temperature": 0.1,
                "description": "PaddlePaddle - OCR-VL 1B (Multilingual Document Parser)",
                "supports_vision": True,
                "max_new_tokens": 2048,
                "top_p": 0.9,
                "top_k": 40
            }
        }
    }
}

# Map from model choice to provider and API model name
MODEL_MAPPING = {
    # Retrieval models
    "colqwen2.5-multilingual": ("retrieval", "tsystems/colqwen2.5-3b-multilingual-v1.0-merged"),
    "colqwen3.5-4b": ("retrieval", "athrael-soju/colqwen3.5-4.5B-v3"),
    "colnomic-multimodal-7b": ("retrieval", "nomic-ai/colnomic-embed-multimodal-7b"),
    "colnomic-multimodal-3b": ("retrieval", "nomic-ai/colnomic-embed-multimodal-3b"),
    "colsmol-500m": ("retrieval", "vidore/colSmol-500M"),

    # HuggingFace models
    "huggingface-points-reader": ("huggingface", "tencent/POINTS-Reader"),
    "huggingface-kosmos25": ("huggingface", "microsoft/kosmos-2.5"),
    "huggingface-qwen3vl": ("huggingface", "Qwen/Qwen3-VL-8B-Instruct"),
    "huggingface-qwen3vl-4b": ("huggingface", "Qwen/Qwen3-VL-4B-Instruct"),
    "huggingface-qwen3vl-fp8": ("huggingface", "Qwen/Qwen3-VL-8B-Instruct-FP8"),
    "huggingface-qwen3vl-4b-fp8": ("huggingface", "Qwen/Qwen3-VL-4B-Instruct-FP8"),
    "huggingface-nanonets-ocr2": ("huggingface", "nanonets/Nanonets-OCR2-3B"),
    "huggingface-paddleocr-vl": ("huggingface", "PaddlePaddle/PaddleOCR-VL"),
    "huggingface-olmocr": ("huggingface", "allenai/olmOCR-2-7B-1025"),

    # Ollama models
    "ollama-llama-vision": ("ollama", "llama3.2-vision"),
    "ollama-llama-vision-q8": ("ollama", "llama3.2-vision:11b-instruct-q8_0"),
    "ollama-llama-vision-fp16": ("ollama", "llama3.2-vision:11b-instruct-fp16"),
    "ollama-minicpm-vision": ("ollama", "minicpm-v"),
    "ollama-granite-vision": ("ollama", "granite3.2-vision"),
    "ollama-gemma-vision": ("ollama", "gemma3:12b"),
    "ollama-gemma-vision-12b-q8": ("ollama", "gemma3:12b-it-q8_0"),
    "ollama-gemma-vision-12b-fp16": ("ollama", "gemma3:12b-it-fp16"),
    "ollama-gemma-vision-27b": ("ollama", "gemma3:27b"),
    "ollama-gemma-vision-27b-q8": ("ollama", "gemma3:27b-it-q8_0"),
    "ollama-gemma-vision-27b-fp16": ("ollama", "gemma3:27b-it-fp16"),
    "ollama-gemma3n-vision-fp16": ("ollama", "gemma3n:e4b-it-fp16"),
    "ollama-phi4": ("ollama", "phi4-reasoning:plus"),
    "ollama-phi4-q8": ("ollama", "phi4:14b-q8_0"),
    "ollama-phi4-fp16": ("ollama", "phi4:14b-fp16"),
    "ollama-phi4-mini": ("ollama", "phi4-mini-reasoning:latest"),
    "ollama-phi4-mini-fp16": ("ollama", "phi4-mini-reasoning:3.8b-fp16"),
    "ollama-phi4-mini-reasoning": ("ollama", "phi4-mini-reasoning"),
    "ollama-phi4-mini-reasoning:3.8b-fp16": ("ollama", "phi4-mini-reasoning:3.8b-fp16"),
    "ollama-olmo2": ("ollama", "olmo2"),
    "ollama-exaone-deep:32b": ("ollama", "exaone-deep:32b"),
    "ollama-qwq": ("ollama", "qwq"),
    "ollama-qwen3": ("ollama", "qwen3:30b-a3b"),
    "ollama-qwen3-q8": ("ollama", "qwen3:30b-a3b-q8_0"),
    "ollama-qwen3-fp16": ("ollama", "qwen3:30b-a3b-fp16"),
    "ollama-mistral-small32": ("ollama", "mistral-small3.2:latest"),
    "ollama-mistral-small32-24b": ("ollama", "mistral-small3.2:24b-instruct-2506-q8_0"),
    "ollama-mistral-small32-24b-fp16": ("ollama", "mistral-small3.2:24b-instruct-2506-fp16"),
    # Agentica DeepCoder models
    "ollama-deepcoder:14b-preview-q8_0": ("ollama", "deepcoder:14b-preview-q8_0"),
    "ollama-deepcoder:14b-preview-fp16": ("ollama", "deepcoder:14b-preview-fp16"),
    "ollama-granite3.3": ("ollama", "granite3.3"),
    "ollama-granite3.3:2b": ("ollama", "granite3.3:2b"),
    "ollama-llama3.2-vision": ("ollama", "llama3.2-vision"),
    "ollama-phi4": ("ollama", "phi4-reasoning:plus"),
    "ollama-phi4:14b-q8": ("ollama", "phi4:14b:q8_0"),
    "ollama-phi4:14b-fp16": ("ollama", "phi4:14b:fp16"),
    "ollama-phi4-mini-reasoning:latest": ("ollama", "phi4-mini-reasoning:latest"),
    "ollama-phi4-reasoning:plus": ("ollama", "phi4-reasoning:plus"),
    "ollama-phi4-reasoning:14b-plus-q8_0": ("ollama", "phi4-reasoning:14b-plus-q8_0"),
    "ollama-phi4-reasoning:14b-plus-fp16": ("ollama", "phi4-reasoning:14b-plus-fp16"),
    # Llama4 Scout models
    "ollama-llama4-scout": ("ollama", "llama4:scout"),
    "ollama-llama4-scout-q8": ("ollama", "llama4:17b-scout-16e-instruct-q8_0"),
    "ollama-llama4-scout-fp16": ("ollama", "llama4:17b-scout-16e-instruct-fp16"),
    # Llama4 Maverick models
    "ollama-llama4-maverick": ("ollama", "llama4:maverick"),
    "ollama-llama4-maverick-q8": ("ollama", "llama4:17b-maverick-128e-instruct-q8_0"),
    "ollama-llama4-maverick-fp16": ("ollama", "llama4:17b-maverick-128e-instruct-fp16"),
    # Qwen 2.5 Vision models
    "ollama-qwen2.5vl": ("ollama", "qwen2.5vl:7b"),
    "ollama-qwen2.5vl-q8": ("ollama", "qwen2.5vl:7b-q8_0"),
    "ollama-qwen2.5vl-fp16": ("ollama", "qwen2.5vl:7b-fp16"),
    "ollama-qwen2.5vl-32b": ("ollama", "qwen2.5vl:32b"),
    "ollama-qwen2.5vl-32b-q8": ("ollama", "qwen2.5vl:32b-q8_0"),
    "ollama-qwen2.5vl-32b-fp16": ("ollama", "qwen2.5vl:32b-fp16"),
    "ollama-qwen2.5vl-72b": ("ollama", "qwen2.5vl:72b"),
    "ollama-qwen2.5vl-72b-q8": ("ollama", "qwen2.5vl:72b-q8_0"),
    "ollama-qwen2.5vl-72b-fp16": ("ollama", "qwen2.5vl:72b-fp16"),

    # Mistral Codestral models
    "ollama-devstral": ("ollama", "devstral:24b"),
    "ollama-devstral-q8": ("ollama", "devstral:24b-small-2505-q8_0"),
    "ollama-devstral-fp16": ("ollama", "devstral:24b-small-2505-fp16"),
    # DeepSeek R1 models
    "ollama-deepseek-r1-q4": ("ollama", "deepseek-r1:8b-0528-qwen3-q4_K_M"),
    "ollama-deepseek-r1-q8": ("ollama", "deepseek-r1:8b-0528-qwen3-q8_0"),
    "ollama-deepseek-r1-fp16": ("ollama", "deepseek-r1:8b-0528-qwen3-fp16"),
    # Mistral Magistral models
    "ollama-magistral": ("ollama", "magistral:24b"),
    "ollama-magistral-q8": ("ollama", "magistral:24b-small-2506-q8_0"),
    "ollama-magistral-fp16": ("ollama", "magistral:24b-small-2506-fp16"),
    # Ollama cloud-proxied models
    "ollama-glm5.1": ("ollama", "glm-5.1:cloud"),
    "ollama-minimax-m2.7": ("ollama", "minimax-m2.7:cloud"),
    # GPT-OSS local model
    "ollama-gpt-oss": ("ollama", "gpt-oss:20b"),

}

# --- REMOVED FUNCTIONS ---
# The following functions are removed as configuration loading/saving/merging
# is now handled by the session manager and settings routes based on user defaults
# and session files.
# - load_model_configs()
# - save_model_configs()
# - update_model_config() -> This name is kept by a deprecated route, but the function is gone.
# - update_provider_config()
# - get_model_config()

# --- KEPT FUNCTIONS ---
# These functions provide static mappings or definitions useful elsewhere.

def get_configurable_params(provider: str) -> List[Dict[str, Any]]:
    """
    Get list of configurable parameters for a provider.

    Args:
        provider: Provider name (ollama, huggingface)

    Returns:
        List of parameter objects with name, description, type, min, max, etc.
    """
    # Common parameters
    common_params = [
        {
            "name": "temperature",
            "description": "Controls randomness. Higher values (e.g., 0.8) make output more random, lower values (e.g., 0.2) make it more deterministic.",
            "type": "float",
            "min": 0.0,
            "max": 1.0,
            "step": 0.05,
            "default": 0.1
        },
        {
            "name": "max_tokens",
            "description": "Maximum number of tokens to generate in the response.",
            "type": "int",
            "min": 100,
            "max": 4096,
            "step": 100,
            "default": 1024
        }
    ]

    # Provider-specific parameters
    provider_params = {
        "ollama": [
            {
                "name": "num_ctx",
                "description": "Maximum context window size in tokens. Smaller models use 8192, medium models use 4096, large models use 2048.",
                "type": "int",
                "min": 1024,
                "max": 8192,
                "step": 1024,
                "default": 8192
            },
            {
                "name": "num_predict",
                "description": "Maximum number of tokens to predict when generating text. Most models use 2048, Llama 4 Scout uses 1024.",
                "type": "int",
                "min": 128,
                "max": 2048,
                "step": 64,
                "default": 2048
            },
            {
                "name": "repeat_penalty",
                "description": "Penalty for repeated tokens. Higher values prevent repetition.",
                "type": "float",
                "min": 0.1,
                "max": 2.0,
                "step": 0.1,
                "default": 1.1
            },
            {
                "name": "top_p",
                "description": "Controls diversity via nucleus sampling.",
                "type": "float",
                "min": 0.0,
                "max": 1.0,
                "step": 0.05,
                "default": 0.9
            },
            {
                "name": "top_k",
                "description": "Only consider the top k tokens for each generation step.",
                "type": "int",
                "min": 1,
                "max": 100,
                "step": 1,
                "default": 40
            }
        ]
    }

    # Return common parameters + provider-specific parameters
    if provider in provider_params:
        return common_params + provider_params[provider]
    else:
        return common_params

def get_display_name(model_choice: str) -> str:
    """
    Get a human-readable display name for a model choice.

    Args:
        model_choice: Model choice identifier (e.g., 'ollama-llama-vision')

    Returns:
        Human-readable display name
    """
    # Mapping of model choices to display names
    display_names = {
        "huggingface-points-reader": "Tencent - POINTS-Reader (FP16)",
        "huggingface-kosmos25": "Microsoft - Kosmos-2.5 (OCR Specialist)",
        "huggingface-qwen3vl": "Alibaba - Qwen3-VL 8B Instruct (BF16)",
        "huggingface-qwen3vl-4b": "Alibaba - Qwen3-VL 4B Instruct (BF16)",
        "huggingface-qwen3vl-fp8": "Alibaba - Qwen3-VL 8B Instruct (FP8)",
        "huggingface-qwen3vl-4b-fp8": "Alibaba - Qwen3-VL 4B Instruct (FP8)",
        "huggingface-nanonets-ocr2": "Nanonets - OCR2 3B (BF16)",
        "huggingface-paddleocr-vl": "PaddlePaddle - OCR-VL 1B (BF16)",
        "huggingface-olmocr": "AI2 - olmOCR 7B (BF16)",
        "colqwen2.5-multilingual": "T-Systems - ColQwen2.5 3B Multilingual",
        "tsystems/colqwen2.5-3b-multilingual-v1.0": "T-Systems - ColQwen2.5 3B Multilingual",
        "tsystems/colqwen2.5-3b-multilingual-v1.0-merged": "T-Systems - ColQwen2.5 3B Multilingual",
        "colqwen3.5-4b": "ColQwen3.5 4.5B (ViDoRe v3 #4)",
        "athrael-soju/colqwen3.5-4.5B-v3": "ColQwen3.5 4.5B (ViDoRe v3 #4)",
        "colsmol-500m": "ColSmol 500M (Lightweight)",
        "vidore/colSmol-500M": "ColSmol 500M (Lightweight)",
        "colnomic-multimodal-7b": "Nomic AI - ColNomic Embed Multimodal 7B",
        "nomic-ai/colnomic-embed-multimodal-7b": "Nomic AI - ColNomic Embed Multimodal 7B",
        "colnomic-multimodal-3b": "Nomic AI - ColNomic Embed Multimodal 3B",
        "nomic-ai/colnomic-embed-multimodal-3b": "Nomic AI - ColNomic Embed Multimodal 3B",
        "ollama-llama-vision": "Meta - Llama 3.2 Vision 11B (Q4)",
        "ollama-llama-vision-q8": "Meta - Llama 3.2 Vision 11B (Q8)",
        "ollama-llama-vision-fp16": "Meta - Llama 3.2 Vision 11B (FP16)",
        "ollama-minicpm-vision": "MiniCPM Vision",
        "ollama-granite-vision": "Granite 3.2 Vision",
        "ollama-gemma-vision": "Google - Gemma 3 Vision 12B (Q4)",
        "ollama-gemma-vision-12b-q8": "Google - Gemma 3 Vision 12B (Q8)",
        "ollama-gemma-vision-12b-fp16": "Google - Gemma 3 Vision 12B (FP16)",
        "ollama-gemma-vision-27b": "Google - Gemma 3 Vision 27B (Q4)",
        "ollama-gemma-vision-27b-q8": "Google - Gemma 3 Vision 27B (Q8)",
        "ollama-gemma3n-vision-fp16": "Google - Gemma 3n E4B (FP16)",
        "ollama-phi4": "Microsoft - Phi-4 Plus 14B (Q4)",
        "ollama-phi4-q8": "Microsoft - Phi-4 Plus 14B (Q8)",
        "ollama-phi4-fp16": "Microsoft - Phi-4 Plus 14B (FP16)",
        "ollama-phi4-mini": "Microsoft - Phi-4 Mini Reasoning 3.8B (Q4)",
        "ollama-phi4-mini-fp16": "Microsoft - Phi-4 Mini Reasoning 3.8B (FP16)",
        "ollama-phi4-mini-reasoning": "Microsoft - Phi-4 Mini Reasoning 3.8B (Q4)",
        "ollama-phi4-mini-reasoning:3.8b-fp16": "Microsoft - Phi-4 Mini Reasoning 3.8B (FP16)",
        "ollama-olmo2": "OLMo 2:13B (Text-only)",
        "ollama-exaone-deep:32b": "LG - EXAONE Deep 32B (Q4)",
        "ollama-qwq": "Alibaba - QWQ 32B (Q4)",
        "ollama-qwen3": "Alibaba - Qwen3 30B (Q4)",
        "ollama-qwen3-q8": "Alibaba - Qwen3 30B (Q8)",
        "ollama-qwen3-fp16": "Alibaba - Qwen3 30B (FP16)",
        "ollama-mistral-small32": "Mistral - Small 3.2 24B (Q4)",
        "ollama-mistral-small32-24b": "Mistral - Small 3.2 24B (Q8)",
        "ollama-mistral-small32-24b-fp16": "Mistral - Small 3.2 24B (FP16)",
        # Agentica DeepCoder models
        "ollama-deepcoder:14b-preview-q8_0": "Agentica - DeepCoder 14B (Q8)",
        "ollama-deepcoder:14b-preview-fp16": "Agentica - DeepCoder 14B (FP16)",
        "ollama-granite3.3": "IBM - Granite 3.3 8B (Q4)",
        "ollama-granite3.3:2b": "IBM - Granite 3.3 2B (Q4)",
        "ollama-llama3.2-vision": "Meta - Llama 3.2 Vision 90B (Q4)",
        "ollama-phi4": "Microsoft - Phi-4 Plus 14B (Q4)",
        "ollama-phi4:14b-q8": "Microsoft - Phi-4 Plus 14B (Q8)",
        "ollama-phi4:14b-fp16": "Microsoft - Phi-4 Plus 14B (FP16)",
        "ollama-phi4-mini-reasoning:latest": "Microsoft - Phi-4 Mini Reasoning (Latest)",
        "ollama-phi4-reasoning:plus": "Microsoft - Phi-4 Reasoning Plus",
        "ollama-phi4-reasoning:14b-plus-q8_0": "Microsoft - Phi-4 Reasoning Plus 14B (Q8)",
        "ollama-phi4-reasoning:14b-plus-fp16": "Microsoft - Phi-4 Reasoning Plus 14B (FP16)",
        # Llama4 Scout models
        "ollama-llama4-scout": "Meta - Llama 4 Scout 109B (Q4)",
        "ollama-llama4-scout-q8": "Meta - Llama 4 Scout 109B (Q8)",
        "ollama-llama4-scout-fp16": "Meta - Llama 4 Scout 109B (FP16)",
        # Llama4 Maverick models
        "ollama-llama4-maverick": "Meta - Llama 4 Maverick 402B (Q4)",
        "ollama-llama4-maverick-q8": "Meta - Llama 4 Maverick 402B (Q8)",
        "ollama-llama4-maverick-fp16": "Meta - Llama 4 Maverick 402B (FP16)",
        # Qwen 2.5 Vision models
        "ollama-qwen2.5vl": "Alibaba - Qwen 2.5 Vision 7B (Q4)",
        "ollama-qwen2.5vl-q8": "Alibaba - Qwen 2.5 Vision 7B (Q8)",
        "ollama-qwen2.5vl-fp16": "Alibaba - Qwen 2.5 Vision 7B (FP16)",
        "ollama-qwen2.5vl-32b": "Alibaba - Qwen 2.5 Vision 32B (Q4)",
        "ollama-qwen2.5vl-32b-q8": "Alibaba - Qwen 2.5 Vision 32B (Q8)",
        "ollama-qwen2.5vl-32b-fp16": "Alibaba - Qwen 2.5 Vision 32B (FP16)",
        "ollama-qwen2.5vl-72b": "Alibaba - Qwen 2.5 Vision 72B (Q4)",
        "ollama-qwen2.5vl-72b-q8": "Alibaba - Qwen 2.5 Vision 72B (Q8)",
        "ollama-qwen2.5vl-72b-fp16": "Alibaba - Qwen 2.5 Vision 72B (FP16)",
        # Mistral Codestral models
        "ollama-devstral": "Mistral - Codestral 24B (Q4) - [Text Only]",
        "ollama-devstral-q8": "Mistral - Codestral 24B (Q8) - [Text Only]",
        "ollama-devstral-fp16": "Mistral - Codestral 24B (FP16) - [Text Only]",
        # DeepSeek R1 models
        "ollama-deepseek-r1-q4": "DeepSeek - R1 8B (Q4) - [Text Only]",
        "ollama-deepseek-r1-q8": "DeepSeek - R1 8B (Q8) - [Text Only]",
        "ollama-deepseek-r1-fp16": "DeepSeek - R1 8B (FP16) - [Text Only]",
        # Mistral Magistral models
        "ollama-magistral": "Mistral - Magistral 24B (Q4) - [Text Only]",
        "ollama-magistral-q8": "Mistral - Magistral 24B (Q8) - [Text Only]",
        "ollama-magistral-fp16": "Mistral - Magistral 24B (FP16) - [Text Only]",
        # Ollama cloud-proxied models
        "ollama-glm5.1": "Zhipu - GLM 5.1 (Cloud)",
        "ollama-minimax-m2.7": "MiniMax - M2.7 (Cloud)",
        # GPT-OSS
        "ollama-gpt-oss": "GPT-OSS 20B (MXFP4)",
    }

    return display_names.get(model_choice, model_choice)


def get_model_config(model_id):
    """
    Get the configuration for a specific model.

    Args:
        model_id: The model identifier (e.g., 'ollama-qwen2.5vl-32b-fp16')

    Returns:
        dict: The model configuration, or empty dict if not found
    """
    # First check if there's a specific config for this model
    if model_id in DEFAULT_MODEL_CONFIGS:
        return DEFAULT_MODEL_CONFIGS[model_id].copy()

    # If not, check if the model exists in mapping and return provider defaults
    if model_id in MODEL_MAPPING:
        provider, model_name = MODEL_MAPPING[model_id]

        # Return provider-specific defaults
        if provider == "ollama":
            return {
                "temperature": 0.1,
                "top_p": 0.9,
                "top_k": 40,
                "num_predict": 2048,
                "num_ctx": 8192,
                "repeat_penalty": 1.1,
                "num_gpu": -1
            }

    # Model not found
    return {}
