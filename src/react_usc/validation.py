from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, cast

from .models import DecisionType, JudgeDecision, ReasonerDecision


def _type_matches(value: Any, expected_type: str) -> bool:
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
        if tool_name is not None and not isinstance(tool_name, str):
            errors.append("tool_name must be string or null")
        if tool_args is not None and not isinstance(tool_args, dict):
            errors.append("tool_args must be object or null")
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


