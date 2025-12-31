from __future__ import annotations

"""
LangChain model factory for Vertex AI Gemini.

This uses Google's default credential flow (ADC). To use gcloud:
  gcloud auth application-default login

Then LangChain's ChatVertexAI can pick up credentials without API keys.
"""

from typing import Any, Optional


def make_chat_vertex_ai(*, model: str, location: Optional[str] = None, project: Optional[str] = None) -> Any:
    try:
        from langchain_google_vertexai import ChatVertexAI  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Missing langchain-google-vertexai. Install with: `python -m pip install -r requirements.txt`"
        ) from e

    kwargs = {"model": model}
    if location:
        kwargs["location"] = location
    if project:
        kwargs["project"] = project
    return ChatVertexAI(**kwargs)


