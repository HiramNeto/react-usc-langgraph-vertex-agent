"""
LLM Provider helpers for the ReAct USC Agent.

This module provides convenience functions for creating LangChain chat models
for various providers, plus protocol definitions for type checking.

Currently supported providers:
- Vertex AI (Google Cloud) - requires langchain-google-genai

Example:
    >>> from react_usc.providers import make_chat_vertex_ai, ChatModelProtocol
    >>> model = make_chat_vertex_ai(
    ...     model="gemini-1.5-pro",
    ...     project="my-gcp-project",
    ...     location="us-central1",
    ... )
    >>> isinstance(model, ChatModelProtocol)  # True
"""
from __future__ import annotations

from .base import ChatModelProtocol, StructuredOutputModelProtocol
from .vertex import make_chat_vertex_ai

__all__ = [
    "ChatModelProtocol",
    "StructuredOutputModelProtocol",
    "make_chat_vertex_ai",
]
