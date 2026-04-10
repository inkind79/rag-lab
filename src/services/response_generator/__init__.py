"""
Response generator service for handling query responses.

This service is responsible for generating responses to user queries
using the appropriate model and context.
"""
from .generator import generate_streaming_response

__all__ = ['generate_streaming_response']
