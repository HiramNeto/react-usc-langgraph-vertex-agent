"""
Configuration classes for the ReAct USC Agent.

This module contains:
- ModelConfig: Configuration for LLM models
- RetryConfig: Configuration for retry behavior
- AgentConfig: Complete agent configuration

All configuration classes are frozen dataclasses with validation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .types import AgentConstants, SelectionStrategy


# =============================================================================
# Model Configuration
# =============================================================================

@dataclass(frozen=True)
class ModelConfig:
    """
    Configuration for an LLM model.
    
    Attributes:
        name: Model identifier (e.g., "gemini-1.5-pro", "gpt-4")
        temperature: Sampling temperature (0.0 = deterministic, higher = more random)
        max_tokens: Maximum output tokens (None = provider default)
    
    Example:
        >>> config = ModelConfig(name="gemini-1.5-pro", temperature=0.7)
        >>> config.name
        'gemini-1.5-pro'
    """
    name: str
    temperature: float
    max_tokens: Optional[int] = None
    
    def __post_init__(self) -> None:
        """Validate model configuration."""
        if not self.name or not self.name.strip():
            raise ValueError("Model name cannot be empty")
        if not AgentConstants.MIN_TEMPERATURE <= self.temperature <= AgentConstants.MAX_TEMPERATURE:
            raise ValueError(
                f"Temperature must be between {AgentConstants.MIN_TEMPERATURE} and "
                f"{AgentConstants.MAX_TEMPERATURE}, got {self.temperature}"
            )
        if self.max_tokens is not None and self.max_tokens <= 0:
            raise ValueError(f"max_tokens must be positive, got {self.max_tokens}")


# =============================================================================
# Retry Configuration
# =============================================================================

@dataclass(frozen=True)
class RetryConfig:
    """
    Configuration for retry behavior with exponential backoff.
    
    Attributes:
        max_retries: Maximum number of retry attempts (0 = no retries)
        backoff_seconds: Base delay between retries (doubled each attempt)
    
    Example:
        >>> config = RetryConfig.default()
        >>> config.max_retries
        2
    """
    max_retries: int
    backoff_seconds: float
    
    def __post_init__(self) -> None:
        """Validate retry configuration."""
        if self.max_retries < 0:
            raise ValueError(f"max_retries cannot be negative, got {self.max_retries}")
        if self.backoff_seconds < 0:
            raise ValueError(f"backoff_seconds cannot be negative, got {self.backoff_seconds}")
    
    @classmethod
    def default(cls) -> "RetryConfig":
        """Create default retry configuration."""
        return cls(
            max_retries=AgentConstants.DEFAULT_MAX_RETRIES,
            backoff_seconds=AgentConstants.DEFAULT_BACKOFF_SECONDS,
        )
    
    @classmethod
    def none(cls) -> "RetryConfig":
        """Create configuration with no retries."""
        return cls(max_retries=0, backoff_seconds=0.0)


# =============================================================================
# Agent Configuration
# =============================================================================

@dataclass(frozen=True)
class AgentConfig:
    """
    Complete configuration for the ReAct USC Agent.
    
    This is the main configuration object that controls agent behavior.
    All fields are validated on creation.
    
    Attributes:
        system_prompt: Base system prompt for the agent
        k_paths: Number of parallel reasoning paths (K in USC)
        max_steps: Maximum ReAct loop iterations
        reasoner_model: Model configuration for reasoners
        judge_model: Model configuration for the judge
        selection_strategy: How the judge selects from candidates
        allow_tool_synthesis: Whether judge can create new tool calls
        llm_retry: Retry configuration for LLM calls
        trace: Enable detailed trace output
        tool_result_max_chars: Max chars for tool result truncation (0 = no limit)
        truncate_agent_observations: Whether to truncate observations sent to LLM
        timeout_seconds: Timeout for parallel reasoner calls
        use_structured_output: Use LangChain structured output when available
        log_structured_output: Log structured output attempts/results
        accept_non_json_final: Salvage FINAL answer from non-JSON output
        max_reprompts: Number of repair attempts when LLM output is invalid (0 = disable)
    
    Example:
        >>> config = AgentConfig.default()
        >>> config.k_paths
        3
    """
    system_prompt: str
    k_paths: int
    max_steps: int
    reasoner_model: ModelConfig
    judge_model: ModelConfig
    selection_strategy: SelectionStrategy
    allow_tool_synthesis: bool
    llm_retry: RetryConfig
    trace: bool
    tool_result_max_chars: int
    truncate_agent_observations: bool = False
    timeout_seconds: float = AgentConstants.DEFAULT_TIMEOUT_SECONDS
    use_structured_output: bool = True
    log_structured_output: bool = False
    accept_non_json_final: bool = False
    max_reprompts: int = 1
    
    def __post_init__(self) -> None:
        """Validate agent configuration."""
        if not AgentConstants.MIN_K_PATHS <= self.k_paths <= AgentConstants.MAX_K_PATHS:
            raise ValueError(
                f"k_paths must be between {AgentConstants.MIN_K_PATHS} and "
                f"{AgentConstants.MAX_K_PATHS}, got {self.k_paths}"
            )
        if not AgentConstants.MIN_MAX_STEPS <= self.max_steps <= AgentConstants.MAX_MAX_STEPS:
            raise ValueError(
                f"max_steps must be between {AgentConstants.MIN_MAX_STEPS} and "
                f"{AgentConstants.MAX_MAX_STEPS}, got {self.max_steps}"
            )
        if self.timeout_seconds <= 0:
            raise ValueError(f"timeout_seconds must be positive, got {self.timeout_seconds}")
        if self.tool_result_max_chars < 0:
            raise ValueError(
                f"tool_result_max_chars cannot be negative, got {self.tool_result_max_chars}"
            )
        if self.selection_strategy not in ("select_one", "synthesize_one"):
            raise ValueError(
                f"selection_strategy must be 'select_one' or 'synthesize_one', "
                f"got {self.selection_strategy!r}"
            )
        if self.max_reprompts < 0:
            raise ValueError(f"max_reprompts cannot be negative, got {self.max_reprompts}")
    
    @classmethod
    def default(
        cls,
        *,
        system_prompt: str = "You are a helpful AI assistant.",
        k_paths: int = 3,
        max_steps: int = 10,
        model_name: str = "gemini-1.5-pro",
    ) -> "AgentConfig":
        """
        Create a default agent configuration.
        
        Useful for quick setup and testing.
        
        Args:
            system_prompt: Base system prompt
            k_paths: Number of parallel reasoning paths
            max_steps: Maximum loop iterations
            model_name: Model name for both reasoner and judge
        
        Returns:
            AgentConfig with sensible defaults
        """
        model_config = ModelConfig(name=model_name, temperature=0.7)
        return cls(
            system_prompt=system_prompt,
            k_paths=k_paths,
            max_steps=max_steps,
            reasoner_model=model_config,
            judge_model=model_config,
            selection_strategy="select_one",
            allow_tool_synthesis=False,
            llm_retry=RetryConfig.default(),
            trace=False,
            tool_result_max_chars=AgentConstants.DEFAULT_TOOL_RESULT_MAX_CHARS,
            truncate_agent_observations=False,
        )


__all__ = [
    "ModelConfig",
    "RetryConfig",
    "AgentConfig",
]
