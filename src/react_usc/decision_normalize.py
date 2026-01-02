from __future__ import annotations

from typing import Any, Dict


def normalize_reasoner_decision_obj(obj: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize common model deviations so validation can succeed:
      - decision_type: "TOOL" -> "TOOL_CALL"
      - missing brief_rationale: fill with a minimal placeholder
    """
    dt = obj.get("decision_type")
    if isinstance(dt, str):
        dt_norm = dt.strip().upper().replace(" ", "_")
        if dt_norm in {"TOOL", "TOOLCALL", "TOOL_CALL"}:
            obj["decision_type"] = "TOOL_CALL"
        elif dt_norm in {"FINAL", "ANSWER"}:
            obj["decision_type"] = "FINAL"
    if "brief_rationale" not in obj or not isinstance(obj.get("brief_rationale"), str) or not obj.get("brief_rationale"):
        obj["brief_rationale"] = ""
    # Replace placeholder rationales with something readable.
    br = obj.get("brief_rationale")
    if isinstance(br, str) and br.strip().upper() in {"", "N/A", "NA", "NONE"}:
        if obj.get("decision_type") == "TOOL_CALL":
            tool = obj.get("tool_name") if isinstance(obj.get("tool_name"), str) else "a tool"
            obj["brief_rationale"] = f"Use {tool} to gather the missing information/result needed to proceed."
        else:
            obj["brief_rationale"] = "We have enough information from observations to answer now."
    # Some models emit FINAL answers as numbers/objects. Our schema expects a string.
    if obj.get("decision_type") == "FINAL":
        fa = obj.get("final_answer")
        if fa is not None and not isinstance(fa, str):
            obj["final_answer"] = str(fa)
    return obj


def normalize_judge_decision_obj(obj: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize common model deviations so validation can succeed:
      - decision_type: "TOOL" -> "TOOL_CALL"
      - tool_name aliases: "search.run" -> "simple_search"
      - nested shape: {"decision": {...}} -> flatten into top-level JudgeDecision
    """
    # Some models nest the actual decision under a "decision" key and include extra commentary.
    nested = obj.get("decision")
    if isinstance(nested, dict):
        flattened: Dict[str, Any] = {}
        # Keep selection + justification if provided at top-level.
        if "selected_index" in obj:
            flattened["selected_index"] = obj.get("selected_index")
        if "justification" in obj:
            flattened["justification"] = obj.get("justification")
        # Merge nested decision fields.
        flattened.update(nested)
        obj = flattened

    dt = obj.get("decision_type")
    if isinstance(dt, str):
        dt_norm = dt.strip().upper().replace(" ", "_")
        if dt_norm in {"TOOL", "TOOLCALL", "TOOL_CALL"}:
            obj["decision_type"] = "TOOL_CALL"
        elif dt_norm in {"FINAL", "ANSWER"}:
            obj["decision_type"] = "FINAL"

    tool_name = obj.get("tool_name")
    if tool_name == "search.run":
        obj["tool_name"] = "simple_search"

    # JudgeDecision requires a non-empty justification; fill a placeholder if missing.
    just = obj.get("justification")
    if not isinstance(just, str) or just.strip().upper() in {"", "N/A", "NA", "NONE"}:
        # Some models put "brief_rationale" instead of "justification".
        br = obj.get("brief_rationale")
        if isinstance(br, str) and br.strip():
            obj["justification"] = br.strip()
        else:
            # Deterministic fallback justification.
            if obj.get("decision_type") == "TOOL_CALL":
                tn = obj.get("tool_name") if isinstance(obj.get("tool_name"), str) else "a tool"
                obj["justification"] = f"Select {tn} because it is the most direct next action to reduce uncertainty."
            else:
                obj["justification"] = "Select FINAL because the observations are sufficient to answer."

    return obj


