from __future__ import annotations

import json
from typing import Any, Dict, Sequence, Tuple

from .models import AgentConfig, ReasonerDecision, ToolSpec

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

def _tool_name_list(tools: Sequence[ToolSpec]) -> str:
    return ", ".join([t.name for t in tools])

def _tool_examples_block(tools: Sequence[ToolSpec]) -> str:
    """
    Provide short, concrete examples so models reliably include required tool args.
    """
    examples: list[str] = []
    for t in tools:
        # Minimal hand-written examples for our demo tools.
        if t.name == "calculator":
            examples.append('{"decision_type":"TOOL_CALL","tool_name":"calculator","tool_args":{"expression":"2+2*10"},"final_answer":null,"brief_rationale":"Compute the expression to avoid arithmetic mistakes.","expected_signal":null}')
        elif t.name == "simple_search":
            examples.append('{"decision_type":"TOOL_CALL","tool_name":"simple_search","tool_args":{"query":"What is ReAct and self-consistency?"},"final_answer":null,"brief_rationale":"Search the KB for a precise definition.","expected_signal":null}')
    if not examples:
        return ""
    return "\n".join(["EXAMPLES (copy these shapes):"] + [f"- {ex}" for ex in examples])


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
            f"tool_name MUST be one of: {_tool_name_list(tools)}",
            "If decision_type is TOOL_CALL, tool_args MUST include ALL required keys from that tool's input_schema. Do not return empty {}.",
            "Do NOT wrap the JSON in markdown fences (no ```json).",
            "brief_rationale is REQUIRED: write 1-2 short sentences explaining why this is the best next step.",
            "Do NOT use placeholders like 'N/A'.",
            "",
            _tool_examples_block(tools),
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
    tools: Sequence[ToolSpec],
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
            "AVAILABLE_TOOLS:",
            build_tools_block(tools),
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
            f"tool_name MUST be one of: {_tool_name_list(tools)}",
            "If decision_type is TOOL_CALL, tool_args MUST include ALL required keys from that tool's input_schema. Do not return empty {}.",
            "",
            "OUTPUT_FORMAT:",
            "Return ONLY a JSON object matching JudgeDecision.",
            "Do NOT wrap the JSON in markdown fences (no ```json).",
            "Do NOT nest the decision under a 'decision' key. Do NOT include extra keys like 'comment'.",
            "justification is REQUIRED: write 1-2 short sentences explaining why this is the best single next step.",
            "Do NOT use placeholders like 'N/A'.",
            "",
            _tool_examples_block(tools),
            "",
            "JSON_ONLY:",
        ]
    )
    return system, user

def build_reflection_prompt(
    *,
    user_query: str,
    tool_name: str,
    tool_args: Dict[str, Any],
    error: str,
    tools: Sequence[ToolSpec],
) -> Tuple[str, str]:
    system = (
        "You are a Tool Usage Expert debugging a failed tool call.\n"
        "Analyze the error and decide whether to RETRY with corrected args, WAIT for transient errors, or ABORT if the tool is inappropriate.\n"
        "Return ONLY a JSON object matching the ReflectionDecision schema.\n"
    )

    user = "\n".join(
        [
            "REFLECTION INSTRUCTIONS:",
            "A tool execution failed. Your goal is to fix it if possible, or advise the agent to stop if the tool is wrong.",
            "",
            "ORIGINAL USER QUERY:",
            user_query.strip(),
            "",
            "FAILED TOOL CALL:",
            f"Tool: {tool_name}",
            f"Args: {json.dumps(tool_args, ensure_ascii=False)}",
            f"Error: {error}",
            "",
            "AVAILABLE TOOLS:",
            build_tools_block(tools),
            "",
            "DECISION RULES:",
            "1. RETRY: If the error is a syntax error, invalid argument format, or hallucinated argument, and the tool IS appropriate for the query -> Generate corrected 'retry_args'.",
            "2. WAIT: If the error looks transient (e.g. network timeout, rate limit, server error 5xx, connection reset) and arguments look correct -> Select WAIT to pause and retry with the SAME arguments.",
            "3. ABORT: If the tool itself is not capable of handling the query (e.g. using calculator for search), or if you cannot fix it -> Provide an 'abort_suggestion' explaining why and what tool might be better.",
            "",
            "OUTPUT_FORMAT:",
            "Return ONLY a JSON object matching ReflectionDecision.",
            "Do NOT wrap the JSON in markdown fences (no ```json).",
            "If verdict is RETRY, 'retry_args' must be valid JSON matching the tool's schema.",
            "If verdict is WAIT or ABORT, 'retry_args' should be null/omitted.",
            "",
            "JSON_ONLY:",
        ]
    )
    return system, user
