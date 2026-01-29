"""
Validation utilities for the ReAct USC Agent.

This module provides lightweight validation for:
- JSON schema validation (subset)
- ReasonerDecision validation
- JudgeDecision validation
- ReflectionDecision validation
"""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Tuple, cast

from ..decisions import JudgeDecision, ReasonerDecision
from ..types import DecisionType

# Type alias for reflection verdict
ReflectionVerdict = Literal["RETRY", "WAIT", "ABORT"]


def _type_matches(value: Any, expected_type: str) -> bool:
    """Check if a value matches the expected JSON schema type."""
    if expected_type == "string":
        return isinstance(value, str)
    if expected_type == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected_type == "boolean":
        return isinstance(value, bool)
    if expected_type == "object":
        return isinstance(value, dict)
    if expected_type == "array":
        return isinstance(value, list)
    if expected_type == "null":
        return value is None
    return True  # Unknown type: accept to keep validator lightweight.


def validate_json_obj(obj: Any, schema: Dict[str, Any]) -> List[str]:
    """
    Lightweight JSON schema validator for objects used in tool args.

    Supported subset:
      - type: "object"
      - required: [..]
      - properties: { key: {type: ...} }
    """
    errors: List[str] = []
    if schema.get("type") == "object":
        if not isinstance(obj, dict):
            return [f"Expected object, got {type(obj).__name__}"]

        required = schema.get("required", [])
        for key in required:
            if key not in obj:
                errors.append(f"Missing required key: {key}")

        props = schema.get("properties", {})
        for key, prop_schema in props.items():
            if key not in obj:
                continue
            expected_type = prop_schema.get("type")
            if isinstance(expected_type, str) and not _type_matches(obj[key], expected_type):
                errors.append(
                    f"Key '{key}' expected type {expected_type}, got {type(obj[key]).__name__}"
                )
    return errors


def validate_reasoner_decision_dict(d: Any) -> Tuple[Optional[ReasonerDecision], List[str]]:
    """
    Validate a dict and convert to ReasonerDecision if valid.
    
    Returns:
        Tuple of (decision, errors). If errors is non-empty, decision is None.
    """
    errors: List[str] = []
    if not isinstance(d, dict):
        return None, ["ReasonerDecision must be an object"]
    dt = d.get("decision_type")
    if dt not in ("TOOL_CALL", "FINAL"):
        errors.append("decision_type must be TOOL_CALL or FINAL")
    brief = d.get("brief_rationale")
    if not isinstance(brief, str) or not brief.strip():
        errors.append("brief_rationale must be a non-empty string")

    tool_name = d.get("tool_name")
    tool_args = d.get("tool_args")
    final_answer = d.get("final_answer")

    if dt == "TOOL_CALL":
        if not isinstance(tool_name, str) or not tool_name:
            errors.append("tool_name must be a non-empty string for TOOL_CALL")
        if tool_args is not None and not isinstance(tool_args, dict):
            errors.append("tool_args must be an object if provided")
        if final_answer is not None:
            errors.append("final_answer must be null for TOOL_CALL")
    elif dt == "FINAL":
        if not isinstance(final_answer, str) or not final_answer.strip():
            errors.append("final_answer must be a non-empty string for FINAL")
        if tool_name is not None or tool_args is not None:
            errors.append("tool_name/tool_args must be null for FINAL")

    expected_signal = d.get("expected_signal")
    if expected_signal is not None and not isinstance(expected_signal, str):
        errors.append("expected_signal must be a string or null")

    if errors:
        return None, errors

    return (
        ReasonerDecision(
            decision_type=cast(DecisionType, dt),
            tool_name=cast(Optional[str], tool_name),
            tool_args=cast(Optional[Dict[str, Any]], tool_args),
            final_answer=cast(Optional[str], final_answer),
            brief_rationale=cast(str, brief),
            expected_signal=cast(Optional[str], expected_signal),
        ),
        [],
    )


def validate_judge_decision_dict(d: Any) -> Tuple[Optional[JudgeDecision], List[str]]:
    """
    Validate a dict and convert to JudgeDecision if valid.
    
    Returns:
        Tuple of (decision, errors). If errors is non-empty, decision is None.
    """
    errors: List[str] = []
    if not isinstance(d, dict):
        return None, ["JudgeDecision must be an object"]
    dt = d.get("decision_type")
    if dt not in ("TOOL_CALL", "FINAL"):
        errors.append("decision_type must be TOOL_CALL or FINAL")
    justification = d.get("justification")
    if not isinstance(justification, str) or not justification.strip():
        errors.append("justification must be a non-empty string")

    selected_index = d.get("selected_index")
    if selected_index is not None and not isinstance(selected_index, int):
        errors.append("selected_index must be an integer or null")

    tool_name = d.get("tool_name")
    tool_args = d.get("tool_args")
    final_answer = d.get("final_answer")

    if dt == "TOOL_CALL":
        if not isinstance(tool_name, str) or not tool_name.strip():
            errors.append("tool_name must be a non-empty string for TOOL_CALL")
        if tool_args is not None and not isinstance(tool_args, dict):
            errors.append("tool_args must be object or null for TOOL_CALL")
        if final_answer is not None:
            errors.append("final_answer must be null for TOOL_CALL")
    elif dt == "FINAL":
        if not isinstance(final_answer, str) or not final_answer.strip():
            errors.append("final_answer must be non-empty string for FINAL")
        if tool_name is not None or tool_args is not None:
            errors.append("tool_name/tool_args must be null for FINAL")

    if errors:
        return None, errors

    return (
        JudgeDecision(
            decision_type=cast(DecisionType, dt),
            selected_index=cast(Optional[int], selected_index),
            tool_name=cast(Optional[str], tool_name),
            tool_args=cast(Optional[Dict[str, Any]], tool_args),
            final_answer=cast(Optional[str], final_answer),
            justification=cast(str, justification),
        ),
        [],
    )


def validate_reflection_decision_dict(d: Any) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    """
    Validate a dict for reflection decision and return validated dict if valid.
    
    Unlike the other validators, this returns a validated dict (not a dataclass)
    to avoid circular imports with plugins.py which defines ReflectionResult.
    
    Returns:
        Tuple of (validated_dict, errors). If errors is non-empty, validated_dict is None.
    """
    errors: List[str] = []
    if not isinstance(d, dict):
        return None, ["ReflectionDecision must be an object"]
    
    # Validate verdict
    verdict = d.get("verdict")
    if verdict not in ("RETRY", "WAIT", "ABORT"):
        errors.append("verdict must be RETRY, WAIT, or ABORT")
    
    # Validate analysis
    analysis = d.get("analysis")
    if not isinstance(analysis, str) or not analysis.strip():
        errors.append("analysis must be a non-empty string")
    
    # Validate retry_args for RETRY verdict
    retry_args = d.get("retry_args")
    if verdict == "RETRY":
        if retry_args is not None and not isinstance(retry_args, dict):
            errors.append("retry_args must be an object or null for RETRY")
    
    # Validate abort_suggestion for ABORT verdict
    abort_suggestion = d.get("abort_suggestion")
    if abort_suggestion is not None and not isinstance(abort_suggestion, str):
        errors.append("abort_suggestion must be a string or null")
    
    if errors:
        return None, errors
    
    return (
        {
            "verdict": cast(str, verdict),
            "analysis": cast(str, analysis),
            "retry_args": cast(Optional[Dict[str, Any]], retry_args),
            "abort_suggestion": cast(Optional[str], abort_suggestion),
        },
        [],
    )
