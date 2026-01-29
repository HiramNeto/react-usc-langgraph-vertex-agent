"""
Shared fixtures for the ReAct USC Agent test suite.

This module provides common test fixtures that can be used across all test modules.

Fixtures:
- test_tool: Simple test tool
- test_tool_registry: Registry with test tool
- mock_model: Mocked LangChain model
- test_config: Basic AgentConfig
- test_config_with_salvage: AgentConfig with non-JSON salvage
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from react_usc import (
    AgentConfig,
    ModelConfig,
    RetryConfig,
    ToolRegistry,
    ToolSpec,
)


@pytest.fixture
def test_tool() -> ToolSpec:
    """Create a simple test tool."""
    return ToolSpec(
        name="test_tool",
        description="A simple test tool",
        input_schema={
            "type": "object",
            "required": ["query"],
            "properties": {
                "query": {"type": "string"},
            },
        },
        func=lambda args: f"Result for: {args.get('query', '')}",
    )


@pytest.fixture
def test_tool_registry(test_tool: ToolSpec) -> ToolRegistry:
    """Create a tool registry with the test tool."""
    return ToolRegistry([test_tool])


@pytest.fixture
def mock_model() -> MagicMock:
    """Create a mock LangChain model."""
    model = MagicMock()
    model.invoke = MagicMock(return_value=MagicMock(content="{}"))
    return model


@pytest.fixture
def test_config() -> AgentConfig:
    """Create a minimal AgentConfig for testing."""
    return AgentConfig(
        system_prompt="You are a test agent.",
        k_paths=3,
        max_steps=5,
        reasoner_model=ModelConfig(name="test-model", temperature=0.7),
        judge_model=ModelConfig(name="test-model", temperature=0.7),
        selection_strategy="select_one",
        allow_tool_synthesis=False,
        llm_retry=RetryConfig.none(),
        trace=False,
        tool_result_max_chars=1000,
        accept_non_json_final=False,
        use_structured_output=True,
    )


@pytest.fixture
def test_config_with_salvage() -> AgentConfig:
    """Create an AgentConfig with non-JSON salvage enabled."""
    return AgentConfig(
        system_prompt="You are a test agent.",
        k_paths=3,
        max_steps=5,
        reasoner_model=ModelConfig(name="test-model", temperature=0.7),
        judge_model=ModelConfig(name="test-model", temperature=0.7),
        selection_strategy="select_one",
        allow_tool_synthesis=False,
        llm_retry=RetryConfig.none(),
        trace=False,
        tool_result_max_chars=1000,
        accept_non_json_final=True,
        use_structured_output=True,
    )


# =============================================================================
# Helper functions (for use in test modules)
# =============================================================================


def make_test_config(
    accept_non_json_final: bool = False,
    use_structured_output: bool = True,
) -> AgentConfig:
    """Create a minimal AgentConfig for testing with custom options."""
    return AgentConfig(
        system_prompt="You are a test agent.",
        k_paths=3,
        max_steps=5,
        reasoner_model=ModelConfig(name="test-model", temperature=0.7),
        judge_model=ModelConfig(name="test-model", temperature=0.7),
        selection_strategy="select_one",
        allow_tool_synthesis=False,
        llm_retry=RetryConfig.none(),
        trace=False,
        tool_result_max_chars=1000,
        accept_non_json_final=accept_non_json_final,
        use_structured_output=use_structured_output,
    )


def make_test_tool() -> ToolSpec:
    """Create a simple test tool."""
    return ToolSpec(
        name="test_tool",
        description="A simple test tool",
        input_schema={
            "type": "object",
            "required": ["query"],
            "properties": {
                "query": {"type": "string"},
            },
        },
        func=lambda args: f"Result for: {args.get('query', '')}",
    )


def make_mock_model() -> MagicMock:
    """Create a mock LangChain model."""
    model = MagicMock()
    model.invoke = MagicMock(return_value=MagicMock(content="{}"))
    return model


# =============================================================================
# Additional pytest fixtures
# =============================================================================





@pytest.fixture
def test_config_no_structured_output() -> AgentConfig:
    """Create an AgentConfig with structured output disabled."""
    return AgentConfig(
        system_prompt="You are a test agent.",
        k_paths=2,
        max_steps=3,
        reasoner_model=ModelConfig(name="test-model", temperature=0.7),
        judge_model=ModelConfig(name="test-model", temperature=0.7),
        selection_strategy="select_one",
        allow_tool_synthesis=False,
        llm_retry=RetryConfig.none(),
        trace=False,
        tool_result_max_chars=1000,
        accept_non_json_final=False,
        use_structured_output=False,
    )


@pytest.fixture
def test_config_with_trace() -> AgentConfig:
    """Create an AgentConfig with trace enabled."""
    return AgentConfig(
        system_prompt="You are a test agent.",
        k_paths=3,
        max_steps=5,
        reasoner_model=ModelConfig(name="test-model", temperature=0.7),
        judge_model=ModelConfig(name="test-model", temperature=0.7),
        selection_strategy="select_one",
        allow_tool_synthesis=False,
        llm_retry=RetryConfig.none(),
        trace=True,
        tool_result_max_chars=1000,
        accept_non_json_final=False,
        use_structured_output=True,
    )


