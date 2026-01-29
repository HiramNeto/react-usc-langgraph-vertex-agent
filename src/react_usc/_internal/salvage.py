"""
Non-JSON salvage utilities for the ReAct USC Agent.

This module provides shared functions for extracting structured decisions
from non-JSON LLM output when the model ignores JSON formatting constraints.

These are best-effort fallback mechanisms and should only be used when
proper JSON parsing fails.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional


def salvage_non_json_final_answer(
    text: str,
    *,
    extra_fields: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Best-effort extraction of FINAL answer when model ignores JSON constraints.
    
    Handles pseudo-structured text like:
        decision_type: FINAL
        final_answer: "..."
        justification: ...
    
    Args:
        text: Raw LLM output text
        extra_fields: Additional fields to include in the result dict
            (e.g., {"brief_rationale": "...", "expected_signal": None} for reasoner,
             or {"selected_index": None, "justification": "..."} for judge)
    
    Returns:
        Dict suitable for validation, or None if salvage not possible
    
    Example:
        >>> text = 'decision_type: FINAL\\nfinal_answer: "42"'
        >>> salvage_non_json_final_answer(text, extra_fields={"justification": "computed"})
        {'decision_type': 'FINAL', 'tool_name': None, 'tool_args': None, 
         'final_answer': '42', 'justification': 'computed'}
    """
    raw = (text or "").strip()
    if not raw:
        return None
    
    # Don't salvage tool calls - these require precise arguments
    if "TOOL_CALL" in raw or "tool_name" in raw:
        return None
    if "final_answer" not in raw:
        return None
    
    lines = raw.splitlines()
    start_idx = None
    for i, ln in enumerate(lines):
        if ln.strip().lower().startswith("final_answer:"):
            start_idx = i
            break
    
    if start_idx is None:
        return None
    
    first = lines[start_idx]
    after = first.split(":", 1)[1].strip() if ":" in first else ""
    
    # Single-line value (possibly quoted)
    if after:
        v = after.strip()
        # Remove surrounding quotes if present
        if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
            v = v[1:-1]
        # Handle escaped newlines
        v = v.replace("\\n", "\n")
        final_answer = v.strip()
    else:
        # Multiline value - collect lines until next field
        buf: List[str] = []
        for ln in lines[start_idx + 1:]:
            s = ln.rstrip("\n")
            # Stop at next field (identifier followed by colon)
            if s.strip() and s.split(":", 1)[0].strip().isidentifier() and ":" in s:
                break
            buf.append(s)
        
        final_answer = "\n".join(buf).strip()
        if not final_answer:
            return None
    
    # Build result dict
    result: Dict[str, Any] = {
        "decision_type": "FINAL",
        "tool_name": None,
        "tool_args": None,
        "final_answer": final_answer,
    }
    
    # Add extra fields (e.g., justification for judge, brief_rationale for reasoner)
    if extra_fields:
        result.update(extra_fields)
    
    return result


def salvage_reasoner_final(text: str) -> Optional[Dict[str, Any]]:
    """
    Salvage a FINAL answer from non-JSON reasoner output.
    
    Args:
        text: Raw LLM output text
    
    Returns:
        Dict suitable for validate_reasoner_decision_dict, or None
    """
    return salvage_non_json_final_answer(
        text,
        extra_fields={
            "brief_rationale": "Non-JSON reasoner output; salvaged final_answer.",
            "expected_signal": None,
        },
    )


def salvage_judge_final(text: str) -> Optional[Dict[str, Any]]:
    """
    Salvage a FINAL answer from non-JSON judge output.
    
    Args:
        text: Raw LLM output text
    
    Returns:
        Dict suitable for validate_judge_decision_dict, or None
    """
    return salvage_non_json_final_answer(
        text,
        extra_fields={
            "selected_index": None,
            "justification": "Non-JSON judge output; salvaged final_answer.",
        },
    )


def salvage_reflection_final(text: str) -> Optional[Dict[str, Any]]:
    """
    Salvage a reflection decision from non-JSON output.
    
    Handles pseudo-structured text like:
        verdict: ABORT
        abort_suggestion: The API key is invalid
        analysis: ...
    
    or:
        verdict: RETRY
        retry_args: {"key": "value"}
    
    Args:
        text: Raw LLM output text
    
    Returns:
        Dict suitable for validate_reflection_decision_dict, or None
    """
    import json
    
    raw = (text or "").strip()
    if not raw:
        return None
    
    result: Dict[str, Any] = {}
    lines = raw.splitlines()
    
    # Look for verdict
    verdict = None
    for ln in lines:
        ln_lower = ln.strip().lower()
        if ln_lower.startswith("verdict:"):
            v = ln.split(":", 1)[1].strip().upper()
            if v in ("RETRY", "WAIT", "ABORT"):
                verdict = v
                break
    
    if not verdict:
        # Try to infer verdict from content
        raw_upper = raw.upper()
        if "ABORT" in raw_upper:
            verdict = "ABORT"
        elif "WAIT" in raw_upper:
            verdict = "WAIT"
        elif "RETRY" in raw_upper:
            verdict = "RETRY"
        else:
            return None
    
    result["verdict"] = verdict
    
    # Extract analysis if present
    for ln in lines:
        if ln.strip().lower().startswith("analysis:"):
            analysis = ln.split(":", 1)[1].strip() if ":" in ln else ""
            if analysis:
                if (analysis.startswith('"') and analysis.endswith('"')) or \
                   (analysis.startswith("'") and analysis.endswith("'")):
                    analysis = analysis[1:-1]
                result["analysis"] = analysis
            break
    
    # Ensure analysis is present
    if "analysis" not in result:
        result["analysis"] = "Salvaged from non-JSON reflection output."
    
    # Extract abort_suggestion if present
    for ln in lines:
        if ln.strip().lower().startswith("abort_suggestion:"):
            suggestion = ln.split(":", 1)[1].strip() if ":" in ln else ""
            if suggestion:
                # Remove quotes if present
                if (suggestion.startswith('"') and suggestion.endswith('"')) or \
                   (suggestion.startswith("'") and suggestion.endswith("'")):
                    suggestion = suggestion[1:-1]
                result["abort_suggestion"] = suggestion
            break
    
    # For RETRY, try to extract retry_args
    if verdict == "RETRY":
        for ln in lines:
            if ln.strip().lower().startswith("retry_args:"):
                args_str = ln.split(":", 1)[1].strip() if ":" in ln else ""
                if args_str:
                    try:
                        args = json.loads(args_str)
                        if isinstance(args, dict):
                            result["retry_args"] = args
                    except Exception:
                        pass
                break
        
        # Also look for embedded JSON object anywhere in the text
        if "retry_args" not in result:
            # Find JSON object in text
            start = raw.find("{")
            end = raw.rfind("}")
            if start >= 0 and end > start:
                try:
                    args = json.loads(raw[start:end + 1])
                    if isinstance(args, dict):
                        result["retry_args"] = args
                except Exception:
                    pass
    
    return result


__all__ = [
    "salvage_non_json_final_answer",
    "salvage_reasoner_final",
    "salvage_judge_final",
    "salvage_reflection_final",
]
