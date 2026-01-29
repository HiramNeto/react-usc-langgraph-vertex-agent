"""
Structured-output schemas for LangChain `with_structured_output(...)`.

We intentionally use plain JSON schema dicts (instead of Pydantic models) to avoid
provider-specific incompatibilities. In particular, Vertex/Gemini logs noisy warnings
for the JSON-schema key `additionalProperties` (commonly produced by Pydantic),
even when it's harmless.

Cross-field constraints (e.g. tool_name required for TOOL_CALL) are enforced by our
validators in `validation.py`, and we also do a "sanity validation" pass in the agent
before accepting structured-output results (to trigger fallback to text parsing when
structured outputs omit required tool args).
"""
from __future__ import annotations

from typing import Any, Dict, List, Sequence


def _get_tool_args_options(tool_schemas: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Helper to build the `anyOf` options for tool_args.
    """
    options = []
    for s in tool_schemas:
        # We assume s is a valid JSON schema for the arguments.
        # We copy it to avoid mutating the original.
        ts = s.copy()
        # Add a title if missing, as some providers prefer named schemas in anyOf
        if "title" not in ts:
            ts["title"] = "tool_arguments"
        options.append(ts)
    
    # Also allow empty object (for FINAL decisions where args are null/empty)
    options.append({"type": "object", "properties": {}, "title": "empty_args"})
    return options


def get_reasoner_decision_schema(tool_schemas: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Build a dynamic Reasoner schema where `tool_args` uses `anyOf`.
    """
    return {
        "title": "ReasonerDecision",
        "description": "Single next-step decision from the REASONER: either call one tool with JSON args, or return a final answer.",
        "type": "object",
        "required": ["decision_type", "brief_rationale"],
        "properties": {
            "decision_type": {"type": "string", "enum": ["TOOL_CALL", "FINAL"]},
            "tool_name": {"type": "string"},
            "tool_args": {
                "anyOf": _get_tool_args_options(tool_schemas)
            },
            "final_answer": {"type": "string"},
            "brief_rationale": {"type": "string"},
            "expected_signal": {"type": "string"},
        },
    }


def get_judge_decision_schema(tool_schemas: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Build a dynamic Judge schema where `tool_args` uses `anyOf`.
    """
    return {
        "title": "JudgeDecision",
        "description": "Decision from the JUDGE: select/synthesize one candidate decision, either call one tool or return a final answer.",
        "type": "object",
        "required": ["decision_type", "justification"],
        "properties": {
            "decision_type": {"type": "string", "enum": ["TOOL_CALL", "FINAL"]},
            "selected_index": {"type": "integer"},
            "tool_name": {"type": "string"},
            "tool_args": {
                 "anyOf": _get_tool_args_options(tool_schemas)
            },
            "final_answer": {"type": "string"},
            "justification": {"type": "string"},
        },
    }


def get_reflection_decision_schema(tool_input_schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build a dynamic Reflection schema where `retry_args` matches the failed tool's schema.
    
    Args:
        tool_input_schema: The JSON schema for the tool that failed
    
    Returns:
        JSON schema for ReflectionDecision
    """
    # Copy the tool schema to avoid mutation
    retry_args_schema = tool_input_schema.copy() if tool_input_schema else {"type": "object", "properties": {}}
    if "title" not in retry_args_schema:
        retry_args_schema["title"] = "retry_arguments"
    
    return {
        "title": "ReflectionDecision",
        "description": "Decision from the reflection model: analyze a failed tool call and decide whether to retry, wait, or abort.",
        "type": "object",
        "required": ["analysis", "verdict"],
        "properties": {
            "analysis": {
                "type": "string",
                "description": "Analysis of why the tool call failed and what can be done about it.",
            },
            "verdict": {
                "type": "string",
                "enum": ["RETRY", "WAIT", "ABORT"],
                "description": "RETRY with new args, WAIT and retry same args, or ABORT the retry loop.",
            },
            "retry_args": retry_args_schema,
            "abort_suggestion": {
                "type": "string",
                "description": "Suggestion message if verdict is ABORT.",
            },
        },
    }


# Static fallbacks for when no tools are provided
REASONER_DECISION_SCHEMA = get_reasoner_decision_schema([])
JUDGE_DECISION_SCHEMA = get_judge_decision_schema([])
REFLECTION_DECISION_SCHEMA = get_reflection_decision_schema({})
