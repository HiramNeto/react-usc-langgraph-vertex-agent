"""
Base protocols and types for LLM providers.

This module defines the interfaces that LLM provider implementations should follow.
Using protocols allows for duck-typing while still providing type hints.
"""
from __future__ import annotations

from typing import Any, List, Protocol, runtime_checkable


@runtime_checkable
class ChatModelProtocol(Protocol):
    """
    Protocol that chat models must implement.
    
    This defines the minimal interface required for a chat model to be used
    with the ReAct USC Agent. Models should support the `invoke` method
    that takes a list of messages and returns a response.
    
    For structured output support, models should also implement
    `with_structured_output()`.
    
    Example:
        >>> def check_model(model: ChatModelProtocol) -> bool:
        ...     return isinstance(model, ChatModelProtocol)
        
        >>> # LangChain models typically satisfy this protocol
        >>> from langchain_google_genai import ChatGoogleGenerativeAI
        >>> model = ChatGoogleGenerativeAI(model="gemini-1.5-pro")
        >>> check_model(model)  # True
    """
    
    def invoke(self, messages: List[Any]) -> Any:
        """
        Invoke the model with a list of messages.
        
        Args:
            messages: List of message objects (typically HumanMessage, AIMessage, etc.)
        
        Returns:
            Model response (typically has a `content` attribute)
        """
        ...


@runtime_checkable
class StructuredOutputModelProtocol(ChatModelProtocol, Protocol):
    """
    Protocol for models that support structured output.
    
    Extends ChatModelProtocol with the ability to configure the model
    to return structured output matching a schema.
    """
    
    def with_structured_output(self, schema: Any) -> "ChatModelProtocol":
        """
        Configure the model for structured output.
        
        Args:
            schema: Pydantic model or JSON schema defining the output structure
        
        Returns:
            A configured model that will return structured output
        """
        ...


__all__ = [
    "ChatModelProtocol",
    "StructuredOutputModelProtocol",
]
