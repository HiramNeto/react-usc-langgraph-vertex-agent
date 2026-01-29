"""
LangChain model factory for Google Generative AI (Gemini) via Vertex AI.

This module provides a helper function to create ChatGoogleGenerativeAI instances
configured for the Vertex AI backend using Google's default credential flow (ADC).

Prerequisites:
    1. Install: pip install react-usc[vertex]
    2. Authenticate: gcloud auth application-default login

Example:
    >>> from react_usc.providers import make_chat_vertex_ai
    >>> model = make_chat_vertex_ai(
    ...     model="gemini-1.5-pro",
    ...     project="my-gcp-project",
    ...     location="us-central1",
    ... )
"""
from __future__ import annotations

import os
from typing import Any, Optional


def make_chat_vertex_ai(
    *,
    model: str,
    location: Optional[str] = None,
    project: Optional[str] = None,
) -> Any:
    """
    Create a LangChain ChatGoogleGenerativeAI model instance configured for Vertex AI.
    
    Args:
        model: The Gemini model name (e.g., "gemini-1.5-pro", "gemini-2.5-flash")
        location: GCP region (e.g., "us-central1"). If not provided, uses default.
        project: GCP project ID. If not provided, uses default from credentials.
    
    Returns:
        A ChatGoogleGenerativeAI instance (with vertexai=True) ready for use with the agent.
    
    Raises:
        RuntimeError: If langchain-google-genai is not installed.
    
    Example:
        >>> model = make_chat_vertex_ai(model="gemini-1.5-pro")
    """
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI  # type: ignore
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(
            "Missing langchain-google-genai. Install with: "
            "`pip install react-usc[vertex]` or "
            "`pip install langchain-google-genai`"
        ) from e

    # The new ChatGoogleGenerativeAI uses GOOGLE_CLOUD_LOCATION env var for location.
    # Set it if the caller provides a location parameter.
    if location:
        os.environ.setdefault("GOOGLE_CLOUD_LOCATION", location)

    return ChatGoogleGenerativeAI(
        model=model,
        project=project,
        vertexai=True,
    )
