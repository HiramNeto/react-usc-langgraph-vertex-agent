"""
Custom exception hierarchy for the ReAct USC Agent.

This module provides a clear exception hierarchy that enables:
- Specific error handling for different failure modes
- Consistent error recovery strategies
- Clear error messages for debugging

Exception Hierarchy:
    USCAgentError (base)
    ├── ConfigurationError - Invalid configuration
    ├── LLMError (base for LLM-related errors)
    │   ├── StructuredOutputError - Structured output failed
    │   ├── JSONParseError - Failed to parse LLM output as JSON
    │   └── LLMTimeoutError - LLM call timed out
    ├── ValidationError (base for validation errors)
    │   ├── DecisionValidationError - Invalid reasoner/judge decision
    │   └── ToolArgsValidationError - Invalid tool arguments
    ├── ToolError (base for tool-related errors)
    │   ├── UnknownToolError - Tool not found in registry
    │   ├── ToolExecutionError - Tool execution failed
    │   └── ToolReflectionError - Reflection mechanism failed
    └── AgentLoopError - Agent loop failed (max steps, etc.)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


class USCAgentError(Exception):
    """Base exception for all USC Agent errors."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}
    
    def __str__(self) -> str:
        if self.details:
            detail_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            return f"{self.message} ({detail_str})"
        return self.message


# =============================================================================
# Configuration Errors
# =============================================================================


class ConfigurationError(USCAgentError):
    """Raised when agent configuration is invalid."""
    pass


# =============================================================================
# LLM-Related Errors
# =============================================================================


class LLMError(USCAgentError):
    """Base exception for LLM-related errors."""
    pass


class StructuredOutputError(LLMError):
    """Raised when structured output parsing fails."""
    
    def __init__(
        self,
        message: str,
        phase: str,
        original_error: Optional[Exception] = None,
        **details: Any,
    ) -> None:
        super().__init__(message, {"phase": phase, **details})
        self.phase = phase
        self.original_error = original_error


class JSONParseError(LLMError):
    """Raised when LLM output cannot be parsed as JSON."""
    
    def __init__(
        self,
        message: str,
        raw_output: str,
        original_error: Optional[Exception] = None,
        **details: Any,
    ) -> None:
        # Truncate raw output for error message
        preview = raw_output[:200] + "..." if len(raw_output) > 200 else raw_output
        super().__init__(message, {"output_preview": preview, **details})
        self.raw_output = raw_output
        self.original_error = original_error


class LLMTimeoutError(LLMError):
    """Raised when LLM call times out."""
    
    def __init__(
        self,
        message: str,
        timeout_seconds: float,
        phase: str,
        **details: Any,
    ) -> None:
        super().__init__(message, {"timeout_seconds": timeout_seconds, "phase": phase, **details})
        self.timeout_seconds = timeout_seconds
        self.phase = phase


# =============================================================================
# Validation Errors
# =============================================================================


class ValidationError(USCAgentError):
    """Base exception for validation errors."""
    
    def __init__(
        self,
        message: str,
        errors: Optional[List[str]] = None,
        **details: Any,
    ) -> None:
        super().__init__(message, {"errors": errors or [], **details})
        self.errors = errors or []


class DecisionValidationError(ValidationError):
    """Raised when a reasoner or judge decision is invalid."""
    
    def __init__(
        self,
        message: str,
        decision_type: str,  # "reasoner" or "judge"
        raw_decision: Dict[str, Any],
        errors: List[str],
        **details: Any,
    ) -> None:
        super().__init__(message, errors, decision_type=decision_type, **details)
        self.decision_type = decision_type
        self.raw_decision = raw_decision


class ToolArgsValidationError(ValidationError):
    """Raised when tool arguments fail validation."""
    
    def __init__(
        self,
        message: str,
        tool_name: str,
        tool_args: Dict[str, Any],
        errors: List[str],
        **details: Any,
    ) -> None:
        super().__init__(message, errors, tool_name=tool_name, **details)
        self.tool_name = tool_name
        self.tool_args = tool_args


# =============================================================================
# Tool-Related Errors
# =============================================================================


class ToolError(USCAgentError):
    """Base exception for tool-related errors."""
    pass


class UnknownToolError(ToolError):
    """Raised when a requested tool is not found in the registry."""
    
    def __init__(
        self,
        tool_name: str,
        available_tools: Optional[List[str]] = None,
        **details: Any,
    ) -> None:
        message = f"Unknown tool: '{tool_name}'"
        if available_tools:
            message += f". Available tools: {', '.join(available_tools)}"
        super().__init__(message, {"tool_name": tool_name, **details})
        self.tool_name = tool_name
        self.available_tools = available_tools


class ToolExecutionError(ToolError):
    """Raised when tool execution fails."""
    
    def __init__(
        self,
        message: str,
        tool_name: str,
        tool_args: Dict[str, Any],
        original_error: Optional[Exception] = None,
        **details: Any,
    ) -> None:
        super().__init__(message, {"tool_name": tool_name, **details})
        self.tool_name = tool_name
        self.tool_args = tool_args
        self.original_error = original_error


class ToolReflectionError(ToolError):
    """Raised when the reflection mechanism fails."""
    
    def __init__(
        self,
        message: str,
        tool_name: str,
        original_error: Optional[Exception] = None,
        suggestion: Optional[str] = None,
        **details: Any,
    ) -> None:
        super().__init__(message, {"tool_name": tool_name, "suggestion": suggestion, **details})
        self.tool_name = tool_name
        self.original_error = original_error
        self.suggestion = suggestion


# =============================================================================
# Agent Loop Errors
# =============================================================================


class AgentLoopError(USCAgentError):
    """Raised when the agent loop encounters an unrecoverable error."""
    pass


class MaxStepsExceededError(AgentLoopError):
    """Raised when the agent exceeds maximum allowed steps."""
    
    def __init__(
        self,
        max_steps: int,
        current_step: int,
        **details: Any,
    ) -> None:
        message = f"Agent exceeded maximum steps ({current_step}/{max_steps})"
        super().__init__(message, {"max_steps": max_steps, "current_step": current_step, **details})
        self.max_steps = max_steps
        self.current_step = current_step


class NoValidCandidatesError(AgentLoopError):
    """Raised when no valid reasoner candidates are produced."""
    
    def __init__(
        self,
        step: int,
        k_paths: int,
        invalid_reasons: Optional[List[str]] = None,
        **details: Any,
    ) -> None:
        message = f"No valid candidates from {k_paths} reasoners at step {step}"
        super().__init__(
            message,
            {"step": step, "k_paths": k_paths, "invalid_reasons": invalid_reasons or [], **details},
        )
        self.step = step
        self.k_paths = k_paths
        self.invalid_reasons = invalid_reasons or []


# =============================================================================
# Result Types (for operations that can fail gracefully)
# =============================================================================


@dataclass(frozen=True)
class Result:
    """
    A Result type for operations that can fail gracefully.
    
    This provides an alternative to exceptions for expected failures,
    allowing callers to decide how to handle errors.
    
    Usage:
        result = parse_decision(raw)
        if result.is_ok:
            decision = result.value
        else:
            handle_error(result.error)
    """
    value: Any = None
    error: Optional[str] = None
    error_details: Optional[Dict[str, Any]] = None
    
    @property
    def is_ok(self) -> bool:
        """Check if the result is successful."""
        return self.error is None
    
    @property
    def is_error(self) -> bool:
        """Check if the result is an error."""
        return self.error is not None
    
    @classmethod
    def ok(cls, value: Any) -> "Result":
        """Create a successful result."""
        return cls(value=value)
    
    @classmethod
    def fail(cls, error: str, **details: Any) -> "Result":
        """Create a failed result."""
        return cls(error=error, error_details=details if details else None)
    
    def unwrap(self) -> Any:
        """
        Get the value or raise an exception if error.
        
        Raises:
            USCAgentError: If the result is an error
        """
        if self.is_error:
            raise USCAgentError(self.error or "Unknown error", self.error_details)
        return self.value
    
    def unwrap_or(self, default: Any) -> Any:
        """Get the value or return a default if error."""
        return self.value if self.is_ok else default
