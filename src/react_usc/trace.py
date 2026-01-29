"""
Trace output for the ReAct USC Agent.

This module provides formatted trace output for debugging agent execution.
Trace output is printed to stdout for easy visibility during development.

When trace mode is enabled, you'll see:
- Reasoner candidates (valid and invalid)
- Judge decisions
- Tool calls and results

Note: This module uses print() for trace output because trace is meant
for developer-facing console output, distinct from structured logging
which is for machine-readable logs.
"""
from __future__ import annotations

import sys
from typing import Sequence

from ._internal.utils import safe_json_dumps, truncate
from .decisions import JudgeDecision, ReasonerDecision


# ANSI color codes for terminal output
class _Colors:
    """ANSI color codes for terminal formatting."""
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"
    
    @classmethod
    def supports_color(cls) -> bool:
        """Check if the terminal supports color output."""
        return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


def _colorize(text: str, color: str) -> str:
    """Apply color to text if terminal supports it."""
    if _Colors.supports_color():
        return f"{color}{text}{_Colors.RESET}"
    return text


def _format_decision_type(decision_type: str) -> str:
    """Format decision type with appropriate color."""
    if decision_type == "TOOL_CALL":
        return _colorize("TOOL_CALL", _Colors.CYAN)
    return _colorize("FINAL", _Colors.GREEN)


def trace_candidates(
    *,
    step: int,
    k: int,
    valid: Sequence[ReasonerDecision],
    invalid: Sequence[str],
) -> None:
    """
    Print formatted trace output for reasoner candidates.
    
    Shows both valid and invalid candidates to help debug
    why certain decisions were rejected.
    
    Args:
        step: Current step number
        k: Total number of reasoners (K paths)
        valid: List of valid ReasonerDecision objects
        invalid: List of error strings for invalid candidates
    """
    header = _colorize(f"\n{'='*60}", _Colors.DIM)
    step_label = _colorize(f"Step {step}", _Colors.BOLD)
    
    print(f"{header}")
    print(f"{step_label}: Reasoner candidates (K={k})")
    print(_colorize("-" * 60, _Colors.DIM))
    
    # Show invalid candidates first (usually indicates problems)
    if invalid:
        print(_colorize("  Invalid candidates:", _Colors.YELLOW))
        max_show = 8
        for i, reason in enumerate(invalid[:max_show]):
            print(f"   {_colorize('✗', _Colors.RED)} {truncate(reason, 260)}")
        if len(invalid) > max_show:
            remaining = len(invalid) - max_show
            print(_colorize(f"   ... ({remaining} more invalid)", _Colors.DIM))
    
    # Show valid candidates
    if not valid:
        print(_colorize("  Valid candidates: (none)", _Colors.RED))
        return
    
    print(_colorize("  Valid candidates:", _Colors.GREEN))
    for i, candidate in enumerate(valid):
        _print_candidate(i, candidate)


def _print_candidate(index: int, candidate: ReasonerDecision) -> None:
    """Print a single candidate decision."""
    idx_str = _colorize(f"[{index}]", _Colors.BOLD)
    
    if candidate.is_tool_call:
        decision_str = _format_decision_type("TOOL_CALL")
        tool_str = _colorize(candidate.tool_name or "?", _Colors.CYAN)
        args_str = truncate(safe_json_dumps(candidate.tool_args), 140)
        rationale = truncate(candidate.brief_rationale, 120)
        
        print(f"   {idx_str} {decision_str} tool={tool_str}")
        print(f"       args={args_str}")
        print(f"       rationale: {_colorize(rationale, _Colors.DIM)}")
    else:
        decision_str = _format_decision_type("FINAL")
        rationale = truncate(candidate.brief_rationale, 120)
        
        print(f"   {idx_str} {decision_str}")
        print(f"       rationale: {_colorize(rationale, _Colors.DIM)}")


def trace_judge(*, step: int, decision: JudgeDecision) -> None:
    """
    Print formatted trace output for a judge decision.
    
    Shows the selected action and justification.
    
    Args:
        step: Current step number
        decision: The JudgeDecision object
    """
    step_label = _colorize(f"Step {step}", _Colors.BOLD)
    
    if decision.is_final:
        decision_str = _format_decision_type("FINAL")
        selected = f"(selected_index={decision.selected_index})" if decision.selected_index is not None else ""
        justification = truncate(decision.justification, 220)
        
        print(f"{step_label}: judge => {decision_str} {selected}")
        print(f"  justification: {_colorize(justification, _Colors.DIM)}")
        
        if decision.final_answer:
            answer_preview = truncate(decision.final_answer, 300)
            print(f"  answer: {_colorize(answer_preview, _Colors.GREEN)}")
    else:
        decision_str = _format_decision_type("TOOL_CALL")
        tool_str = _colorize(decision.tool_name or "?", _Colors.CYAN)
        selected = f"(selected_index={decision.selected_index})" if decision.selected_index is not None else ""
        justification = truncate(decision.justification, 220)
        
        print(f"{step_label}: judge => {decision_str} {tool_str} {selected}")
        print(f"  justification: {_colorize(justification, _Colors.DIM)}")
    
    print(_colorize("-" * 60, _Colors.DIM))


def trace_tool_call(*, tool_name: str, args: str) -> None:
    """
    Print trace output for a tool call.
    
    Args:
        tool_name: Name of the tool being called
        args: JSON string of tool arguments
    """
    tool_str = _colorize(tool_name, _Colors.CYAN)
    print(f"  Tool call: {tool_str} args={truncate(args, 220)}")


def trace_tool_result(*, tool_name: str, result: str) -> None:
    """
    Print trace output for a tool result.
    
    Args:
        tool_name: Name of the tool
        result: JSON string of tool result
    """
    tool_str = _colorize(tool_name, _Colors.CYAN)
    print(f"  Tool result: {tool_str} => {truncate(result, 400)}")


def trace_tool_exception(*, tool_name: str, error: str) -> None:
    """
    Print trace output for a tool exception.
    
    Args:
        tool_name: Name of the tool
        error: Error message
    """
    tool_str = _colorize(tool_name, _Colors.CYAN)
    error_str = _colorize(truncate(error, 400), _Colors.RED)
    print(f"  Tool exception: {tool_str} => {error_str}")


def trace_message(message: str, *, level: str = "info") -> None:
    """
    Print a general trace message.
    
    Args:
        message: Message to print
        level: Message level ("info", "warning", "error")
    """
    if level == "warning":
        prefix = _colorize("[WARN]", _Colors.YELLOW)
    elif level == "error":
        prefix = _colorize("[ERROR]", _Colors.RED)
    else:
        prefix = _colorize("[INFO]", _Colors.BLUE)
    
    print(f"  {prefix} {message}")
