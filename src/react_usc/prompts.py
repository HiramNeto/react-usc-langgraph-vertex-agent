from __future__ import annotations

import json
from typing import Any, Dict, Sequence, Tuple

from .models import AgentConfig, ReasonerDecision, ToolSpec


def reasoner_decision_schema() -> Dict[str, Any]:
    # JSON-schema-like; providers may enforce this in "JSON mode".
    return {
        "type": "object",
        "required": ["decision_type", "brief_rationale"],
        "properties": {
            "decision_type": {"type": "string"},
            "tool_name": {"type": ["string", "null"]},  # informational
            "tool_args": {"type": ["object", "null"]},
            "final_answer": {"type": ["string", "null"]},
            "brief_rationale": {"type": "string"},
            "expected_signal": {"type": ["string", "null"]},
        },
    }


def judge_decision_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "required": ["decision_type", "justification"],
        "properties": {
            "decision_type": {"type": "string"},
            "selected_index": {"type": ["integer", "null"]},
            "tool_name": {"type": ["string", "null"]},
            "tool_args": {"type": ["object", "null"]},
            "final_answer": {"type": ["string", "null"]},
            "justification": {"type": "string"},
        },
    }


def build_tools_block(tools: Sequence[ToolSpec]) -> str:
    parts = []
    for t in tools:
        parts.append(
            "\n".join(
                [
                    f"- name: {t.name}",
                    f"  description: {t.description}",
                    f"  input_schema: {json.dumps(t.input_schema, ensure_ascii=False)}",
                ]
            )
        )
    return "\n".join(parts)


def build_state_summary(*, observations: Sequence[str], step_index: int, max_steps: int) -> str:
    obs_lines = "\n".join([f"- {o}" for o in observations[-10:]]) if observations else "- (none)"
    return "\n".join(
        [
            f"step: {step_index}/{max_steps}",
            "observations (most recent last):",
            obs_lines,
        ]
    )


def reasoner_decision_to_json(d: ReasonerDecision) -> Dict[str, Any]:
    return {
        "decision_type": d.decision_type,
        "tool_name": d.tool_name,
        "tool_args": d.tool_args,
        "final_answer": d.final_answer,
        "brief_rationale": d.brief_rationale,
        "expected_signal": d.expected_signal,
    }


def build_reasoner_prompt(
    *,
    system_prompt: str,
    user_query: str,
    state_summary: str,
    tools: Sequence[ToolSpec],
    path_id: int,
) -> Tuple[str, str]:
    system = (
        "You are a REASONER model inside a ReAct-style agent.\n"
        "Follow the agent system instructions, then decide the single best next action.\n"
        "Return ONLY a JSON object matching the ReasonerDecision schema.\n"
        "Never include extra keys.\n"
    )
    user = "\n".join(
        [
            "REASONER INSTRUCTIONS:",
            system_prompt.strip(),
            "",
            f"PATH_ID: {path_id}",
            "",
            "ORIGINAL_USER_QUERY:",
            user_query.strip(),
            "",
            "CURRENT_STATE_SUMMARY:",
            state_summary,
            "",
            "AVAILABLE_TOOLS:",
            build_tools_block(tools),
            "",
            "OUTPUT_FORMAT:",
            "Return ONLY a JSON object that matches ReasonerDecision with either:",
            '- decision_type="TOOL_CALL" and tool_name/tool_args set, final_answer null; OR',
            '- decision_type="FINAL" and final_answer set, tool_name/tool_args null.',
            "Do NOT wrap the JSON in markdown fences (no ```json).",
            "brief_rationale is REQUIRED: write 1-2 short sentences explaining why this is the best next step.",
            "Do NOT use placeholders like 'N/A'.",
            "",
            "JSON_ONLY:",
        ]
    )
    return system, user


def build_judge_prompt(
    *,
    user_query: str,
    state_summary: str,
    candidates: Sequence[ReasonerDecision],
    config: AgentConfig,
) -> Tuple[str, str]:
    system = (
        "You are the JUDGE model for a Universal Self-Consistency (USC) agent.\n"
        "You must pick the single best next decision from multiple candidates, or synthesize one.\n"
        "Return ONLY a JSON object matching the JudgeDecision schema.\n"
    )

    candidates_json = [reasoner_decision_to_json(c) for c in candidates]
    user = "\n".join(
        [
            "JUDGE INSTRUCTIONS:",
            "",
            # Required: MUST include original user query in judge prompt context.
            "ORIGINAL_USER_QUERY:",
            user_query.strip(),
            "",
            "CURRENT_STATE_SUMMARY:",
            state_summary,
            "",
            f"SELECTION_STRATEGY: {config.selection_strategy}",
            f"ALLOW_TOOL_SYNTHESIS: {str(config.allow_tool_synthesis).lower()}",
            "",
            "CANDIDATES:",
            json.dumps(candidates_json, ensure_ascii=False),
            "",
            "RUBRIC (score high on these):",
            "- query alignment",
            "- consistency with observations",
            "- tool appropriateness/minimality",
            "- safety/policy compliance (basic)",
            "- expected value for reducing uncertainty",
            "",
            "DECISION_RULES:",
            '- If selection_strategy="select_one": set selected_index to the chosen candidate index and copy its decision.',
            '- If selection_strategy="synthesize_one": selected_index must be null; you may synthesize a better single decision.',
            "- If allow_tool_synthesis=false: do not invent a tool call that is not present among candidates.",
            "",
            "OUTPUT_FORMAT:",
            "Return ONLY a JSON object matching JudgeDecision.",
            "Do NOT wrap the JSON in markdown fences (no ```json).",
            "Do NOT nest the decision under a 'decision' key. Do NOT include extra keys like 'comment'.",
            "justification is REQUIRED: write 1-2 short sentences explaining why this is the best single next step.",
            "Do NOT use placeholders like 'N/A'.",
            "",
            "JSON_ONLY:",
        ]
    )
    return system, user


def build_repair_prompt(
    *,
    user_query: str,
    tool: ToolSpec,
    existing_args: Dict[str, Any],
    validation_errors: Sequence[str],
) -> Tuple[str, str, Dict[str, Any]]:
    system = (
        "You are a tool-argument repair helper.\n"
        "Return ONLY a JSON object of tool arguments matching the tool schema.\n"
    )
    user = "\n".join(
        [
            "REPAIR TOOL ARGS",
            "",
            "ORIGINAL_USER_QUERY:",
            user_query.strip(),
            "",
            "TOOL_SCHEMA:",
            json.dumps(tool.input_schema, ensure_ascii=False),
            "",
            "VALIDATION_ERRORS:",
            json.dumps(list(validation_errors), ensure_ascii=False),
            "",
            "EXISTING_ARGS_JSON:",
            json.dumps(existing_args, ensure_ascii=False),
            "",
            "JSON_ONLY:",
        ]
    )
    return system, user, tool.input_schema


