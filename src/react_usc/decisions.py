"""
Decision dataclasses for the ReAct USC Agent.

This module contains:
- ReasonerDecision: Output from reasoner models
- JudgeDecision: Output from the judge model

All decision classes are frozen dataclasses with helper methods.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from .types import DecisionType


# =============================================================================
# Reasoner Decision
# =============================================================================

@dataclass(frozen=True)
class ReasonerDecision:
    """
    Decision output from a reasoner model.
    
    A reasoner can either:
    - Request a TOOL_CALL with tool_name and tool_args
    - Return a FINAL answer with final_answer
    
    Attributes:
        decision_type: "TOOL_CALL" or "FINAL"
        tool_name: Name of tool to call (required for TOOL_CALL)
        tool_args: Arguments for the tool (required for TOOL_CALL)
        final_answer: The final response (required for FINAL)
        brief_rationale: Explanation for this decision
        expected_signal: What the reasoner expects to learn from the tool
    
    Example:
        >>> decision = ReasonerDecision(
        ...     decision_type="TOOL_CALL",
        ...     tool_name="calculator",
        ...     tool_args={"expression": "2+2"},
        ...     final_answer=None,
        ...     brief_rationale="Need to compute the sum",
        ... )
        >>> decision.is_tool_call
        True
    """
    decision_type: DecisionType
    tool_name: Optional[str]
    tool_args: Optional[Dict[str, Any]]
    final_answer: Optional[str]
    brief_rationale: str
    expected_signal: Optional[str] = None
    
    @property
    def is_tool_call(self) -> bool:
        """Check if this is a tool call decision."""
        return self.decision_type == "TOOL_CALL"
    
    @property
    def is_final(self) -> bool:
        """Check if this is a final answer decision."""
        return self.decision_type == "FINAL"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "decision_type": self.decision_type,
            "tool_name": self.tool_name,
            "tool_args": self.tool_args,
            "final_answer": self.final_answer,
            "brief_rationale": self.brief_rationale,
            "expected_signal": self.expected_signal,
        }


# =============================================================================
# Judge Decision
# =============================================================================

@dataclass(frozen=True)
class JudgeDecision:
    """
    Decision output from the judge model.
    
    The judge selects or synthesizes the best next action from
    reasoner candidates.
    
    Attributes:
        decision_type: "TOOL_CALL" or "FINAL"
        selected_index: Index of selected candidate (None if synthesized)
        tool_name: Name of tool to call (required for TOOL_CALL)
        tool_args: Arguments for the tool (required for TOOL_CALL)
        final_answer: The final response (required for FINAL)
        justification: Explanation for this selection
    
    Example:
        >>> decision = JudgeDecision.create_final(
        ...     answer="The answer is 4",
        ...     justification="All candidates agree on this result",
        ... )
        >>> decision.is_final
        True
    """
    decision_type: DecisionType
    selected_index: Optional[int]
    tool_name: Optional[str]
    tool_args: Optional[Dict[str, Any]]
    final_answer: Optional[str]
    justification: str
    
    @property
    def is_tool_call(self) -> bool:
        """Check if this is a tool call decision."""
        return self.decision_type == "TOOL_CALL"
    
    @property
    def is_final(self) -> bool:
        """Check if this is a final answer decision."""
        return self.decision_type == "FINAL"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "decision_type": self.decision_type,
            "selected_index": self.selected_index,
            "tool_name": self.tool_name,
            "tool_args": self.tool_args,
            "final_answer": self.final_answer,
            "justification": self.justification,
        }
    
    @classmethod
    def create_final(cls, answer: str, justification: str) -> "JudgeDecision":
        """
        Create a FINAL decision with the given answer.
        
        Args:
            answer: The final answer text
            justification: Explanation for this decision
        
        Returns:
            A JudgeDecision with decision_type="FINAL"
        """
        return cls(
            decision_type="FINAL",
            selected_index=None,
            tool_name=None,
            tool_args=None,
            final_answer=answer,
            justification=justification,
        )
    
    @classmethod
    def create_tool_call(
        cls,
        tool_name: str,
        tool_args: Dict[str, Any],
        justification: str,
        selected_index: Optional[int] = None,
    ) -> "JudgeDecision":
        """
        Create a TOOL_CALL decision.
        
        Args:
            tool_name: Name of the tool to call
            tool_args: Arguments for the tool
            justification: Explanation for this decision
            selected_index: Index of the selected candidate (if applicable)
        
        Returns:
            A JudgeDecision with decision_type="TOOL_CALL"
        """
        return cls(
            decision_type="TOOL_CALL",
            selected_index=selected_index,
            tool_name=tool_name,
            tool_args=tool_args,
            final_answer=None,
            justification=justification,
        )


__all__ = [
    "ReasonerDecision",
    "JudgeDecision",
]
