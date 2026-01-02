from __future__ import annotations

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


def get_reflection_decision_schema(tool_schemas: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Build a dynamic Reflection schema for retry logic.
    """
    return {
        "title": "ReflectionDecision",
        "description": "Decision from the REFLECTION model: either retry with fixed args or abort.",
        "type": "object",
        "required": ["analysis", "verdict"],
        "properties": {
            "analysis": {"type": "string", "description": "Brief thought process debugging the error."},
            "verdict": {"type": "string", "enum": ["RETRY", "WAIT", "ABORT"]},
            "retry_args": {
                "anyOf": _get_tool_args_options(tool_schemas),
                "description": "Corrected arguments if verdict is RETRY. Ignored if verdict is WAIT or ABORT."
            },
            "abort_suggestion": {"type": "string", "description": "Explanation for the agent why this tool is wrong if verdict is ABORT."}
        },
    }


# Keep static fallbacks
REASONER_DECISION_SCHEMA = get_reasoner_decision_schema([])
JUDGE_DECISION_SCHEMA = get_judge_decision_schema([])
REFLECTION_DECISION_SCHEMA = get_reflection_decision_schema([])
