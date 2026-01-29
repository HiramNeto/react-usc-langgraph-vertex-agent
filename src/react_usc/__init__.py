"""
ReAct + Universal Self-Consistency (USC) Agent.

This package provides a LangGraph-based ReAct agent that implements
Universal Self-Consistency (USC) for improved decision-making.

Key Features:
    - Parallel reasoning with K paths (USC)
    - Judge-based decision selection/synthesis
    - Tool execution with validation
    - Optional retry/reflection for error recovery
    - Comprehensive logging and tracing

Quick Start:
    ```python
    from react_usc import (
        LangGraphReActUSCAgent,
        LangGraphModels,
        AgentConfig,
        ToolSpec,
    )
    
    # Define your tools using ToolSpec
    my_tool = ToolSpec(
        name="my_tool",
        description="Description of what this tool does",
        input_schema={
            "type": "object",
            "required": ["query"],
            "properties": {"query": {"type": "string"}},
        },
        func=lambda args: f"Result for: {args['query']}",
    )
    
    # Create models (using your LangChain chat models)
    models = LangGraphModels(reasoner=chat_model, judge=chat_model)
    
    # Create configuration
    config = AgentConfig.default()
    
    # Create agent
    agent = LangGraphReActUSCAgent(
        models=models,
        tools=[my_tool],
        config=config,
    )
    
    # Run
    answer = agent.run("Your query here")
    ```

For example tool implementations, see the `examples/tools/` directory.

Package Structure:
    - agent: Main agent class and LangGraph integration
    - types: Type aliases, constants, and ToolSpec
    - config: Configuration classes (AgentConfig, ModelConfig, RetryConfig)
    - decisions: Decision classes (ReasonerDecision, JudgeDecision)
    - executors: Executor classes for reasoner/judge/tool operations
    - plugins: Plugin system (e.g., retry with reflection)
    - logging: Centralized logging configuration
    - exceptions: Custom exception hierarchy
    - tools: Tool registry class
    - trace: Debug trace output
    - providers/: LLM provider helpers (optional)
    - integrations/: External integrations like A2A (optional)
    - _internal/: Private implementation details (do not import directly)
"""
from __future__ import annotations

# =============================================================================
# Core Agent
# =============================================================================

from .agent import LangGraphModels, LangGraphReActUSCAgent

# =============================================================================
# Types and Configuration
# =============================================================================

# Prefer importing from specific modules, but also re-export from models
# for backward compatibility
from .types import (
    AgentConstants,
    DecisionType,
    JSONValue,
    SelectionStrategy,
    ToolSpec,
)

from .config import (
    AgentConfig,
    ModelConfig,
    RetryConfig,
)

from .decisions import (
    JudgeDecision,
    ReasonerDecision,
)

# =============================================================================
# Tools
# =============================================================================

from .tools import ToolRegistry

# =============================================================================
# Executors (for advanced usage)
# =============================================================================

from .executors import (
    JudgeExecutor,
    ReasonerExecutor,
    ReasonerResult,
    ToolExecutor,
    ToolResult,
)

# =============================================================================
# Plugins
# =============================================================================

from .plugins import ReflectAndRetryToolPlugin, ReflectionResult

# =============================================================================
# Logging
# =============================================================================

from .logging import (
    AgentLogger,
    LogContext,
    LoggingConfig,
    configure_logging,
    generate_trace_id,
    get_logger,
)

# =============================================================================
# Exceptions
# =============================================================================

from .exceptions import (
    AgentLoopError,
    ConfigurationError,
    DecisionValidationError,
    JSONParseError,
    LLMError,
    LLMTimeoutError,
    MaxStepsExceededError,
    NoValidCandidatesError,
    Result,
    StructuredOutputError,
    ToolArgsValidationError,
    ToolError,
    ToolExecutionError,
    ToolReflectionError,
    UnknownToolError,
    USCAgentError,
    ValidationError,
)

# =============================================================================
# Optional: Providers (only if dependencies installed)
# =============================================================================

try:
    from .providers import make_chat_vertex_ai
except ImportError:
    # langchain-google-genai not installed
    pass

# =============================================================================
# Optional: Integrations (only if dependencies installed)
# =============================================================================

try:
    from .integrations import (
        A2AAgentWrapper,
        AgentCapability,
        AgentCard,
        TaskInput,
        TaskOutput,
        create_a2a_app,
    )
except ImportError:
    # fastapi/uvicorn not installed
    pass

# =============================================================================
# Version
# =============================================================================

__version__ = "0.4.0"

# =============================================================================
# Public API
# =============================================================================

__all__ = [
    # Core
    "LangGraphReActUSCAgent",
    "LangGraphModels",
    # Types
    "AgentConstants",
    "DecisionType",
    "JSONValue",
    "SelectionStrategy",
    "ToolSpec",
    # Configuration
    "AgentConfig",
    "ModelConfig",
    "RetryConfig",
    # Decisions
    "JudgeDecision",
    "ReasonerDecision",
    # Tools
    "ToolRegistry",
    # Executors
    "JudgeExecutor",
    "ReasonerExecutor",
    "ReasonerResult",
    "ToolExecutor",
    "ToolResult",
    # Plugins
    "ReflectAndRetryToolPlugin",
    "ReflectionResult",
    # Logging
    "AgentLogger",
    "LogContext",
    "LoggingConfig",
    "configure_logging",
    "generate_trace_id",
    "get_logger",
    # Exceptions
    "USCAgentError",
    "ConfigurationError",
    "LLMError",
    "StructuredOutputError",
    "JSONParseError",
    "LLMTimeoutError",
    "ValidationError",
    "DecisionValidationError",
    "ToolArgsValidationError",
    "ToolError",
    "UnknownToolError",
    "ToolExecutionError",
    "ToolReflectionError",
    "AgentLoopError",
    "MaxStepsExceededError",
    "NoValidCandidatesError",
    "Result",
    # Optional: Providers (added to __all__ if available)
    # "make_chat_vertex_ai",
    # Optional: Integrations (added to __all__ if available)
    # "A2AAgentWrapper",
    # "AgentCapability",
    # "AgentCard",
    # "TaskInput",
    # "TaskOutput",
    # "create_a2a_app",
]

# Dynamically add optional exports
if "make_chat_vertex_ai" in dir():
    __all__.append("make_chat_vertex_ai")

if "A2AAgentWrapper" in dir():
    __all__.extend([
        "A2AAgentWrapper",
        "AgentCapability",
        "AgentCard",
        "TaskInput",
        "TaskOutput",
        "create_a2a_app",
    ])
