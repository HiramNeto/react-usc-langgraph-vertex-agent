"""
Utility functions for the ReAct USC Agent.

This module provides common utilities used across the codebase:
- String truncation
- State summary building
- JSON serialization helpers
- Text processing utilities
"""
from __future__ import annotations

import json
import re
from typing import Any, Sequence

from ..types import AgentConstants


def truncate(s: str, max_chars: int) -> str:
    """
    Truncate a string to a maximum length with a suffix indicator.
    
    Args:
        s: String to truncate
        max_chars: Maximum length (0 or negative = no truncation)
    
    Returns:
        Truncated string with "... [truncated N chars]" suffix if truncated
    
    Example:
        >>> truncate("Hello World", 8)
        'Hel... [truncated 11 chars]'
    """
    if max_chars <= 0 or len(s) <= max_chars:
        return s
    
    # Reserve space for the suffix
    suffix_space = AgentConstants.TRUNCATION_SUFFIX_CHARS
    content_len = max(0, max_chars - suffix_space)
    
    return f"{s[:content_len]}... [truncated {len(s)} chars]"


def build_state_summary(
    *,
    observations: Sequence[str],
    step_index: int,
    max_steps: int,
) -> str:
    """
    Build a formatted state summary for prompts.
    
    Creates a compact, stable text block representing the current
    agent state for use in reasoner and judge prompts.
    
    Args:
        observations: List of tool observation strings
        step_index: Current step number (1-indexed)
        max_steps: Maximum allowed steps
    
    Returns:
        Formatted state summary string
    
    Example:
        >>> build_state_summary(
        ...     observations=["calculator => 4", "search => info"],
        ...     step_index=2,
        ...     max_steps=10,
        ... )
        'step: 2/10\\nobservations (most recent last):\\n- calculator => 4\\n- search => info'
    """
    # Limit observations to prevent context bloat
    max_obs = AgentConstants.MAX_OBSERVATIONS_IN_SUMMARY
    recent_obs = observations[-max_obs:] if observations else []
    
    if recent_obs:
        obs_lines = "\n".join([f"- {o}" for o in recent_obs])
    else:
        obs_lines = "- (none)"
    
    lines = [
        f"step: {step_index}/{max_steps}",
        "observations (most recent last):",
        obs_lines,
    ]
    
    return "\n".join(lines)


def safe_json_dumps(obj: Any) -> str:
    """
    Safely serialize an object to JSON string.
    
    Falls back to repr() if JSON serialization fails.
    
    Args:
        obj: Object to serialize
    
    Returns:
        JSON string or repr() fallback
    
    Example:
        >>> safe_json_dumps({"key": "value"})
        '{"key": "value"}'
        >>> safe_json_dumps(lambda x: x)  # Not JSON-serializable
        '<function ...>'
    """
    try:
        return json.dumps(obj, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError, OverflowError):
        return repr(obj)


def simple_word_hits(query: str, key: str) -> int:
    """
    Count word overlap between a query and a key.
    
    Useful for basic text matching in search tools.
    Only considers words of 3+ characters.
    
    Args:
        query: Search query string
        key: Key to match against
    
    Returns:
        Number of matching words
    
    Example:
        >>> simple_word_hits("react agent", "react-based agent system")
        2
    """
    # Extract words (3+ characters) from query
    q_tokens = {
        t for t in re.findall(r"[a-z]+", query.lower())
        if len(t) >= 3
    }
    
    # Extract all words from key
    k_tokens = set(re.findall(r"[a-z]+", key.lower()))
    
    return len(q_tokens & k_tokens)


def format_error(error: Exception) -> str:
    """
    Format an exception as a readable string.
    
    Args:
        error: Exception to format
    
    Returns:
        Formatted error string: "ExceptionType: message"
    """
    return f"{type(error).__name__}: {error}"


def is_json_like(text: str) -> bool:
    """
    Check if a string looks like it might be JSON.
    
    Quick heuristic check without actually parsing.
    
    Args:
        text: String to check
    
    Returns:
        True if the string appears to be JSON
    """
    stripped = text.strip()
    return (
        stripped.startswith("{") and stripped.endswith("}")
    ) or (
        stripped.startswith("[") and stripped.endswith("]")
    )


def extract_json_block(text: str) -> str:
    """
    Extract JSON from text that may contain markdown fences or extra content.
    
    Handles:
    - Markdown code fences (```json ... ```)
    - Leading/trailing text around JSON
    
    Args:
        text: Text potentially containing JSON
    
    Returns:
        Extracted JSON string (may still fail to parse)
    """
    cleaned = text.strip()
    
    # Handle markdown fences
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            # Remove first and last lines
            inner = "\n".join(lines[1:-1]).strip()
            # Remove language tag if present
            if inner.lower().startswith("json"):
                inner = inner[4:].lstrip()
            return inner
    
    # Try to extract JSON object from text
    if not cleaned.startswith("{"):
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start >= 0 and end > start:
            return cleaned[start:end + 1].strip()
    
    return cleaned
