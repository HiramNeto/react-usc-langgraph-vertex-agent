from __future__ import annotations

from typing import Any, Dict, cast


def invoke_chat_text(model: Any, *, system: str, user: str) -> str:
    """
    Invoke a LangChain chat model using proper message objects so "system" content is
    actually treated as system instructions.
    """
    # Lazy import to avoid importing langchain at module import time.
    from langchain_core.messages import HumanMessage, SystemMessage  # type: ignore

    out = model.invoke([SystemMessage(content=system), HumanMessage(content=user)])
    content = getattr(out, "content", None)
    return content if isinstance(content, str) else cast(str, out)


def invoke_chat_structured_obj(model: Any, *, system: str, user: str, schema: Any) -> Dict[str, Any]:
    """
    Best-effort wrapper around LangChain structured output.

    Returns a plain dict, or raises to allow the caller to fall back to the legacy JSON parsing path.
    """
    # Lazy import to avoid importing langchain at module import time.
    from langchain_core.messages import HumanMessage, SystemMessage  # type: ignore

    if not hasattr(model, "with_structured_output"):
        raise TypeError("Model does not support with_structured_output")

    runnable = model.with_structured_output(schema)  # type: ignore[attr-defined]
    out = runnable.invoke([SystemMessage(content=system), HumanMessage(content=user)])

    if isinstance(out, dict):
        return cast(Dict[str, Any], out)
    # Pydantic v2
    if hasattr(out, "model_dump"):
        return cast(Dict[str, Any], out.model_dump())
    # Pydantic v1
    if hasattr(out, "dict"):
        return cast(Dict[str, Any], out.dict())

    raise TypeError(f"Unsupported structured output type: {type(out).__name__}")


def json_loads_object(text: str) -> Dict[str, Any]:
    import json

    cleaned = text.strip()
    if not cleaned:
        raise ValueError("Empty model output (expected JSON object).")

    # Handle markdown fences like:
    #   ```json
    #   {...}
    #   ```
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if len(lines) >= 3 and lines[0].startswith("```") and lines[-1].strip() == "```":
            cleaned = "\n".join(lines[1:-1]).strip()
            # If the first line is a language tag (e.g. "json"), drop it.
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:].lstrip()

    # If the model included extra text, extract the first JSON object substring.
    if not cleaned.startswith("{"):
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start >= 0 and end > start:
            cleaned = cleaned[start : end + 1].strip()

    data = json.loads(cleaned)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object, got {type(data).__name__}")
    return data


