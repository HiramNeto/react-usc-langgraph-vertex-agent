"""
Type definitions and constants for the ReAct USC Agent.

This module contains:
- Type aliases for common types used throughout the library
- AgentConstants class with centralized configuration constants
- ToolSpec dataclass for defining tools

All types are designed to be immutable and thread-safe.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Literal, Union


# =============================================================================
# Type Aliases
# =============================================================================

JSONValue = Union[None, bool, int, float, str, List["JSONValue"], Dict[str, "JSONValue"]]
"""Recursive type alias for JSON-compatible values."""

DecisionType = Literal["TOOL_CALL", "FINAL"]
"""Type of decision: either a tool call or a final answer."""

SelectionStrategy = Literal["select_one", "synthesize_one"]
"""Strategy for the judge to select from candidates."""


# =============================================================================
# Constants
# =============================================================================

class AgentConstants:
    """
    Centralized constants for the agent.
    
    These replace magic numbers scattered throughout the codebase,
    making behavior clearer and easier to tune.
    
    Example:
        >>> from react_usc import AgentConstants
        >>> AgentConstants.DEFAULT_TIMEOUT_SECONDS
        20.0
    """
    
    # Maximum number of concurrent reasoner threads
    MAX_REASONER_THREADS: int = 32
    
    # Number of recent observations to include in state summary
    MAX_OBSERVATIONS_IN_SUMMARY: int = 10
    
    # Default timeout for parallel reasoner calls (seconds)
    DEFAULT_TIMEOUT_SECONDS: float = 20.0
    
    # Default maximum characters for tool result truncation
    DEFAULT_TOOL_RESULT_MAX_CHARS: int = 4000
    
    # Truncation suffix format
    TRUNCATION_SUFFIX_CHARS: int = 24  # Length reserved for " [truncated N chars]"
    
    # Default retry configuration
    DEFAULT_MAX_RETRIES: int = 2
    DEFAULT_BACKOFF_SECONDS: float = 1.0
    
    # Minimum/maximum values for configuration validation
    MIN_K_PATHS: int = 1
    MAX_K_PATHS: int = 100
    MIN_MAX_STEPS: int = 1
    MAX_MAX_STEPS: int = 50
    MIN_TEMPERATURE: float = 0.0
    MAX_TEMPERATURE: float = 2.0


# =============================================================================
# Tool Specification
# =============================================================================

@dataclass(frozen=True)
class ToolSpec:
    """
    Specification for a tool available to the agent.
    
    Tools are the primary way for agents to interact with external systems,
    perform calculations, or retrieve information.
    
    Attributes:
        name: Unique identifier for the tool
        description: Human-readable description of what the tool does
        input_schema: JSON Schema defining the expected input format
        func: Callable that executes the tool (receives args dict, returns any)
    
    Example:
        >>> def my_func(args: dict) -> str:
        ...     return f"Hello, {args['name']}!"
        >>> tool = ToolSpec(
        ...     name="greeter",
        ...     description="Greets a person by name",
        ...     input_schema={
        ...         "type": "object",
        ...         "required": ["name"],
        ...         "properties": {"name": {"type": "string"}},
        ...     },
        ...     func=my_func,
        ... )
    """
    name: str
    description: str
    input_schema: Dict[str, Any]
    func: Callable[[Dict[str, Any]], Any]
    
    def __post_init__(self) -> None:
        """Validate tool specification."""
        if not self.name or not self.name.strip():
            raise ValueError("Tool name cannot be empty")
        if not self.description or not self.description.strip():
            raise ValueError(f"Tool '{self.name}' must have a description")
        if not isinstance(self.input_schema, dict):
            raise ValueError(f"Tool '{self.name}' input_schema must be a dict")
        if not callable(self.func):
            raise ValueError(f"Tool '{self.name}' func must be callable")


__all__ = [
    "JSONValue",
    "DecisionType",
    "SelectionStrategy",
    "AgentConstants",
    "ToolSpec",
]
