"""
Shared fixtures for the ReAct USC Agent test suite.

This module provides common test fixtures that can be used across all test modules.

Fixtures:
- test_tool: Simple test tool
- test_tool_registry: Registry with test tool
- mock_model: Mocked LangChain model
- test_config: Basic AgentConfig
- test_config_with_salvage: AgentConfig with non-JSON salvage
- mock_reasoner_model: Configurable reasoner model mock
- mock_judge_model: Configurable judge model mock
- sample_tools: Collection of test tools
- valid_input_schema: Reusable JSON schema
"""
from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Optional
from unittest.mock import MagicMock

import pytest

from react_usc import (
    AgentConfig,
    JudgeDecision,
    LangGraphModels,
    ModelConfig,
    ReasonerDecision,
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
def valid_input_schema() -> Dict[str, Any]:
    """Create a reusable valid JSON schema for tool inputs."""
    return {
        "type": "object",
        "required": ["query"],
        "properties": {
            "query": {"type": "string"},
            "limit": {"type": "integer"},
        },
    }


@pytest.fixture
def mock_reasoner_model() -> MagicMock:
    """
    Create a mock reasoner model with configurable responses.
    
    Use model.invoke.return_value to configure the response.
    """
    model = MagicMock()
    # Default: return a valid FINAL decision
    default_response = json.dumps({
        "decision_type": "FINAL",
        "tool_name": None,
        "tool_args": None,
        "final_answer": "Default answer",
        "brief_rationale": "Default rationale",
    })
    model.invoke = MagicMock(return_value=MagicMock(content=default_response))
    return model


@pytest.fixture
def mock_judge_model() -> MagicMock:
    """
    Create a mock judge model with configurable responses.
    
    Use model.invoke.return_value to configure the response.
    """
    model = MagicMock()
    # Default: return a valid FINAL decision
    default_response = json.dumps({
        "decision_type": "FINAL",
        "selected_index": 0,
        "tool_name": None,
        "tool_args": None,
        "final_answer": "Default answer",
        "justification": "Default justification",
    })
    model.invoke = MagicMock(return_value=MagicMock(content=default_response))
    return model


@pytest.fixture
def sample_tools(valid_input_schema: Dict[str, Any]) -> List[ToolSpec]:
    """
    Create a collection of test tools with different behaviors.
    
    Includes:
    - calculator: Returns calculation results
    - search: Returns search results
    - failing_tool: Always raises an exception
    """
    calculator = ToolSpec(
        name="calculator",
        description="Performs arithmetic calculations",
        input_schema={
            "type": "object",
            "required": ["expression"],
            "properties": {
                "expression": {"type": "string"},
            },
        },
        func=lambda args: f"Result: {args.get('expression', '?')} = 42",
    )
    
    search = ToolSpec(
        name="search",
        description="Searches for information",
        input_schema=valid_input_schema,
        func=lambda args: f"Found results for: {args.get('query', '')}",
    )
    
    def failing_func(args: Dict[str, Any]) -> str:
        raise ValueError("This tool always fails")
    
    failing_tool = ToolSpec(
        name="failing_tool",
        description="A tool that always fails for testing error handling",
        input_schema={"type": "object"},
        func=failing_func,
    )
    
    return [calculator, search, failing_tool]


@pytest.fixture
def sample_tool_registry(sample_tools: List[ToolSpec]) -> ToolRegistry:
    """Create a tool registry with sample tools."""
    return ToolRegistry(sample_tools)


@pytest.fixture
def mock_lang_graph_models(
    mock_reasoner_model: MagicMock,
    mock_judge_model: MagicMock,
) -> LangGraphModels:
    """Create LangGraphModels with mock reasoner and judge."""
    return LangGraphModels(
        reasoner=mock_reasoner_model,
        judge=mock_judge_model,
    )


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


# =============================================================================
# Decision factory helpers
# =============================================================================


def make_final_reasoner_decision(
    answer: str = "Test answer",
    rationale: str = "Test rationale",
) -> ReasonerDecision:
    """Create a FINAL ReasonerDecision."""
    return ReasonerDecision(
        decision_type="FINAL",
        tool_name=None,
        tool_args=None,
        final_answer=answer,
        brief_rationale=rationale,
    )


def make_tool_call_reasoner_decision(
    tool_name: str = "test_tool",
    tool_args: Optional[Dict[str, Any]] = None,
    rationale: str = "Need to use tool",
) -> ReasonerDecision:
    """Create a TOOL_CALL ReasonerDecision."""
    return ReasonerDecision(
        decision_type="TOOL_CALL",
        tool_name=tool_name,
        tool_args=tool_args or {"query": "test"},
        final_answer=None,
        brief_rationale=rationale,
    )


def make_final_judge_decision(
    answer: str = "Test answer",
    justification: str = "Test justification",
    selected_index: Optional[int] = None,
) -> JudgeDecision:
    """Create a FINAL JudgeDecision."""
    return JudgeDecision.create_final(
        answer=answer,
        justification=justification,
    )


def make_tool_call_judge_decision(
    tool_name: str = "test_tool",
    tool_args: Optional[Dict[str, Any]] = None,
    justification: str = "Execute tool",
    selected_index: Optional[int] = None,
) -> JudgeDecision:
    """Create a TOOL_CALL JudgeDecision."""
    return JudgeDecision.create_tool_call(
        tool_name=tool_name,
        tool_args=tool_args or {"query": "test"},
        justification=justification,
        selected_index=selected_index,
    )
