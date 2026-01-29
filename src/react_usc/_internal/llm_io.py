"""
LLM invocation utilities for the ReAct USC Agent.

This module provides helpers for invoking LangChain chat models
with proper retry logic, JSON parsing, and unified fallback handling.
"""
from __future__ import annotations

import json
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, cast

from ..config import RetryConfig
from ..exceptions import JSONParseError


def _execute_with_retry(
    func: Any, retry_config: Optional[RetryConfig], *args: Any, **kwargs: Any
) -> Any:
    """
    Execute a function with exponential backoff retry logic.
    """
    if not retry_config or retry_config.max_retries <= 0:
        return func(*args, **kwargs)

    retries = 0
    while True:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if retries >= retry_config.max_retries:
                raise e
            
            wait_time = retry_config.backoff_seconds * (2 ** retries)
            time.sleep(wait_time)
            retries += 1


def invoke_chat_text(model: Any, *, system: str, user: str, retry_config: Optional[RetryConfig] = None) -> str:
    """
    Invoke a LangChain chat model using proper message objects so "system" content is
    actually treated as system instructions.
    """
    # Lazy import to avoid importing langchain at module import time.
    from langchain_core.messages import HumanMessage, SystemMessage  # type: ignore

    def _call():
        out = model.invoke([SystemMessage(content=system), HumanMessage(content=user)])
        content = getattr(out, "content", None)
        return content if isinstance(content, str) else cast(str, out)

    return _execute_with_retry(_call, retry_config)


def invoke_chat_structured_obj(
    model: Any, *, system: str, user: str, schema: Any, retry_config: Optional[RetryConfig] = None
) -> Dict[str, Any]:
    """
    Best-effort wrapper around LangChain structured output.

    Returns a plain dict, or raises to allow the caller to fall back to the legacy JSON parsing path.
    """
    # Lazy import to avoid importing langchain at module import time.
    from langchain_core.messages import HumanMessage, SystemMessage  # type: ignore

    if not hasattr(model, "with_structured_output"):
        raise TypeError("Model does not support with_structured_output")

    runnable = model.with_structured_output(schema)  # type: ignore[attr-defined]

    def _call():
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

    return _execute_with_retry(_call, retry_config)


def json_loads_object(text: str) -> Dict[str, Any]:
    """
    Parse a JSON object from text, handling common LLM output quirks.
    """
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


def _build_repair_prompt(
    original_user: str,
    error_reason: str,
    schema: Dict[str, Any],
) -> str:
    """
    Build a repair prompt with error context for reprompting.
    
    Args:
        original_user: The original user prompt
        error_reason: Description of why the previous output was invalid
        schema: The JSON schema the output should conform to
    
    Returns:
        A new user prompt with error context appended
    """
    schema_hint = json.dumps(schema, indent=2)
    return "\n\n".join([
        original_user,
        "---",
        "YOUR PREVIOUS RESPONSE WAS INVALID:",
        error_reason,
        "",
        "REQUIRED SCHEMA:",
        f"```json\n{schema_hint}\n```",
        "",
        "Please return ONLY valid JSON that matches the schema above. Do not include markdown fences or extra text.",
    ])


def _truncate_for_log(text: str, max_chars: int = 500) -> str:
    """Truncate text for logging purposes."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."


def invoke_with_unified_fallback(
    model: Any,
    *,
    system: str,
    user: str,
    schema: Dict[str, Any],
    normalizer: Callable[[Dict[str, Any]], Dict[str, Any]],
    validator: Callable[[Dict[str, Any]], Tuple[Any, List[str]]],
    salvage_fn: Optional[Callable[[str], Optional[Dict[str, Any]]]] = None,
    retry_config: Optional[RetryConfig] = None,
    use_structured_output: bool = True,
    accept_non_json_final: bool = False,
    max_reprompts: int = 1,
    phase: str = "LLM",
    logger: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Unified LLM invocation with full fallback chain.
    
    This function provides a consistent fallback strategy for all LLM calls:
    1. Try structured output (if enabled)
    2. Fall back to text parsing + json_loads_object
    3. Try salvage function (if provided and accept_non_json_final is True)
    4. Reprompt with error context (if max_reprompts > 0)
    5. Raise JSONParseError if all attempts fail
    
    Args:
        model: LangChain chat model
        system: System prompt
        user: User prompt
        schema: JSON schema for structured output and reprompt context
        normalizer: Function to normalize the raw output dict
        validator: Function that validates and returns (result, errors)
        salvage_fn: Optional function to salvage non-JSON output
        retry_config: Retry configuration for LLM calls
        use_structured_output: Whether to try structured output first
        accept_non_json_final: Whether to try salvage on parse failure
        max_reprompts: Number of repair attempts on invalid output
        phase: Phase name for logging (e.g., "Reasoner", "Judge")
        logger: Optional logger instance
    
    Returns:
        Validated and normalized dict from LLM output
    
    Raises:
        JSONParseError: If all fallback attempts fail
    """
    raw_text: Optional[str] = None
    last_error: Optional[str] = None
    
    # Step 1: Try structured output if enabled
    if use_structured_output:
        if logger:
            logger.structured_output_attempt(phase)
        
        try:
            obj = invoke_chat_structured_obj(
                model,
                system=system,
                user=user,
                schema=schema,
                retry_config=retry_config,
            )
            
            # Validate structured output before accepting
            normalized = normalizer(obj)
            result, errors = validator(normalized)
            
            if result is not None and not errors:
                if logger:
                    logger.structured_output_success(phase)
                return normalized
            
            # Structured output parsed but failed validation - will try text fallback
            last_error = f"Structured output validation failed: {errors}"
            if logger:
                logger.structured_output_fallback(phase, ValueError(last_error))
                
        except Exception as e:
            last_error = f"Structured output failed: {type(e).__name__}: {e}"
            if logger:
                logger.structured_output_fallback(phase, e)
    
    # Step 2: Fall back to text parsing
    try:
        raw_text = invoke_chat_text(
            model,
            system=system,
            user=user,
            retry_config=retry_config,
        )
        
        obj = json_loads_object(raw_text)
        normalized = normalizer(obj)
        result, errors = validator(normalized)
        
        if result is not None and not errors:
            return normalized
        
        # Parsed JSON but failed validation
        last_error = f"Validation failed: {errors}"
        
    except Exception as parse_e:
        last_error = f"JSON parse failed: {type(parse_e).__name__}: {parse_e}"
        if logger:
            logger.parse_error(
                phase,
                parse_e,
                _truncate_for_log(raw_text or ""),
            )
        
        # Step 3: Try salvage if enabled and we have raw text
        if accept_non_json_final and salvage_fn and raw_text:
            salvaged = salvage_fn(raw_text)
            if salvaged:
                if logger:
                    logger.warning(
                        f"Salvaged FINAL answer from non-JSON {phase.lower()} output",
                    )
                normalized = normalizer(salvaged)
                result, errors = validator(normalized)
                if result is not None and not errors:
                    return normalized
                last_error = f"Salvage validation failed: {errors}"
    
    # Step 4: Reprompt if we have attempts remaining
    if max_reprompts > 0 and last_error:
        if logger:
            logger.trace(f"{phase} reprompting due to: {last_error}")
        
        repair_user = _build_repair_prompt(
            original_user=user,
            error_reason=last_error,
            schema=schema,
        )
        
        return invoke_with_unified_fallback(
            model,
            system=system,
            user=repair_user,
            schema=schema,
            normalizer=normalizer,
            validator=validator,
            salvage_fn=salvage_fn,
            retry_config=retry_config,
            use_structured_output=use_structured_output,
            accept_non_json_final=accept_non_json_final,
            max_reprompts=max_reprompts - 1,
            phase=phase,
            logger=logger,
        )
    
    # Step 5: All attempts exhausted - raise error
    raise JSONParseError(
        f"Failed to get valid {phase} output after all fallback attempts",
        raw_output=raw_text or "",
        original_error=ValueError(last_error) if last_error else None,
    )
