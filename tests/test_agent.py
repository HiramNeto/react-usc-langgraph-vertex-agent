"""
Tests for agent classes (LangGraphModels, LangGraphReActUSCAgent).

These tests cover:
- LangGraphModels validation
- LangGraphReActUSCAgent initialization
- Agent properties
- Agent run() method with mocked executors
"""
from __future__ import annotations

import json
import unittest
from unittest.mock import MagicMock, patch

from react_usc import (
    AgentConfig,
    JudgeDecision,
    LangGraphModels,
    LangGraphReActUSCAgent,
    ModelConfig,
    ReasonerDecision,
    RetryConfig,
    ToolRegistry,
    ToolSpec,
)


# =============================================================================
# Test: LangGraphModels
# =============================================================================


class TestLangGraphModels(unittest.TestCase):
    """Test LangGraphModels validation."""

    def test_valid_models(self):
        """Test creating LangGraphModels with valid models."""
        reasoner = MagicMock()
        reasoner.invoke = MagicMock(return_value=MagicMock(content="{}"))
        
        judge = MagicMock()
        judge.invoke = MagicMock(return_value=MagicMock(content="{}"))
        
        models = LangGraphModels(reasoner=reasoner, judge=judge)
        
        self.assertEqual(models.reasoner, reasoner)
        self.assertEqual(models.judge, judge)

    def test_missing_invoke_on_reasoner(self):
        """Test that missing invoke on reasoner raises ValueError."""
        reasoner = MagicMock(spec=[])  # No invoke method
        judge = MagicMock()
        judge.invoke = MagicMock()
        
        with self.assertRaises(ValueError) as ctx:
            LangGraphModels(reasoner=reasoner, judge=judge)
        
        self.assertIn("reasoner", str(ctx.exception).lower())
        self.assertIn("invoke", str(ctx.exception).lower())

    def test_missing_invoke_on_judge(self):
        """Test that missing invoke on judge raises ValueError."""
        reasoner = MagicMock()
        reasoner.invoke = MagicMock()
        judge = MagicMock(spec=[])  # No invoke method
        
        with self.assertRaises(ValueError) as ctx:
            LangGraphModels(reasoner=reasoner, judge=judge)
        
        self.assertIn("judge", str(ctx.exception).lower())
        self.assertIn("invoke", str(ctx.exception).lower())

    def test_same_model_for_both(self):
        """Test using same model for both reasoner and judge."""
        model = MagicMock()
        model.invoke = MagicMock()
        
        models = LangGraphModels(reasoner=model, judge=model)
        
        self.assertIs(models.reasoner, models.judge)

    def test_frozen_dataclass(self):
        """Test that LangGraphModels is immutable."""
        model = MagicMock()
        model.invoke = MagicMock()
        models = LangGraphModels(reasoner=model, judge=model)
        
        with self.assertRaises(AttributeError):
            models.reasoner = MagicMock()  # type: ignore


# =============================================================================
# Test: LangGraphReActUSCAgent Initialization
# =============================================================================


class TestLangGraphReActUSCAgentInit(unittest.TestCase):
    """Test LangGraphReActUSCAgent initialization."""

    def setUp(self):
        """Create test fixtures."""
        self.mock_model = MagicMock()
        self.mock_model.invoke = MagicMock(return_value=MagicMock(content="{}"))
        
        self.models = LangGraphModels(
            reasoner=self.mock_model,
            judge=self.mock_model,
        )
        
        self.tool = ToolSpec(
            name="test_tool",
            description="A test tool",
            input_schema={
                "type": "object",
                "required": ["query"],
                "properties": {"query": {"type": "string"}},
            },
            func=lambda args: f"Result: {args.get('query', '')}",
        )
        
        self.config = AgentConfig(
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
        )

    def test_init_with_valid_config(self):
        """Test agent initialization with valid configuration."""
        agent = LangGraphReActUSCAgent(
            models=self.models,
            tools=[self.tool],
            config=self.config,
        )
        
        self.assertIsNotNone(agent)
        self.assertEqual(agent.config, self.config)

    def test_config_property(self):
        """Test that config property returns correct config."""
        agent = LangGraphReActUSCAgent(
            models=self.models,
            tools=[self.tool],
            config=self.config,
        )
        
        self.assertEqual(agent.config.k_paths, 3)
        self.assertEqual(agent.config.max_steps, 5)
        self.assertEqual(agent.config.system_prompt, "You are a test agent.")

    def test_tool_names_property(self):
        """Test that tool_names property returns tool names."""
        tool2 = ToolSpec(
            name="another_tool",
            description="Another test tool",
            input_schema={"type": "object"},
            func=lambda args: "result",
        )
        
        agent = LangGraphReActUSCAgent(
            models=self.models,
            tools=[self.tool, tool2],
            config=self.config,
        )
        
        tool_names = agent.tool_names
        
        self.assertEqual(len(tool_names), 2)
        self.assertIn("test_tool", tool_names)
        self.assertIn("another_tool", tool_names)

    def test_init_with_empty_tools(self):
        """Test agent initialization with empty tools list."""
        agent = LangGraphReActUSCAgent(
            models=self.models,
            tools=[],
            config=self.config,
        )
        
        self.assertEqual(agent.tool_names, [])

    def test_init_with_plugins(self):
        """Test agent initialization with plugins."""
        mock_plugin = MagicMock()
        
        agent = LangGraphReActUSCAgent(
            models=self.models,
            tools=[self.tool],
            config=self.config,
            plugins=[mock_plugin],
        )
        
        self.assertIsNotNone(agent)


# =============================================================================
# Test: LangGraphReActUSCAgent.run()
# =============================================================================


class TestLangGraphReActUSCAgentRun(unittest.TestCase):
    """Test LangGraphReActUSCAgent.run() method."""

    def setUp(self):
        """Create test fixtures."""
        self.mock_model = MagicMock()
        self.mock_model.invoke = MagicMock(return_value=MagicMock(content="{}"))
        
        self.models = LangGraphModels(
            reasoner=self.mock_model,
            judge=self.mock_model,
        )
        
        self.tool = ToolSpec(
            name="test_tool",
            description="A test tool",
            input_schema={
                "type": "object",
                "required": ["query"],
                "properties": {"query": {"type": "string"}},
            },
            func=lambda args: f"Result: {args.get('query', '')}",
        )
        
        self.config = AgentConfig(
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
            use_structured_output=False,
        )

    @patch('react_usc._internal.llm_io.invoke_chat_text')
    def test_run_returns_final_answer(self, mock_invoke):
        """Test that run returns final answer directly."""
        # Configure mock to return FINAL decision
        reasoner_response = json.dumps({
            "decision_type": "FINAL",
            "tool_name": None,
            "tool_args": None,
            "final_answer": "The answer is 42",
            "brief_rationale": "Computed directly",
        })
        
        judge_response = json.dumps({
            "decision_type": "FINAL",
            "selected_index": 0,
            "tool_name": None,
            "tool_args": None,
            "final_answer": "The answer is 42",
            "justification": "Selected the correct answer",
        })
        
        mock_invoke.side_effect = [
            reasoner_response,  # Reasoner 1
            reasoner_response,  # Reasoner 2
            judge_response,     # Judge
        ]
        
        agent = LangGraphReActUSCAgent(
            models=self.models,
            tools=[self.tool],
            config=self.config,
        )
        
        result = agent.run("What is 2 + 2?")
        
        self.assertEqual(result, "The answer is 42")

    @patch('react_usc._internal.llm_io.invoke_chat_text')
    def test_run_executes_tool_then_final(self, mock_invoke):
        """Test that run executes tool call then returns final."""
        # First round: TOOL_CALL
        reasoner_tool_call = json.dumps({
            "decision_type": "TOOL_CALL",
            "tool_name": "test_tool",
            "tool_args": {"query": "search"},
            "final_answer": None,
            "brief_rationale": "Need to search",
        })
        
        judge_tool_call = json.dumps({
            "decision_type": "TOOL_CALL",
            "selected_index": 0,
            "tool_name": "test_tool",
            "tool_args": {"query": "search"},
            "final_answer": None,
            "justification": "Execute search",
        })
        
        # Second round: FINAL
        reasoner_final = json.dumps({
            "decision_type": "FINAL",
            "tool_name": None,
            "tool_args": None,
            "final_answer": "Found the answer",
            "brief_rationale": "Based on search results",
        })
        
        judge_final = json.dumps({
            "decision_type": "FINAL",
            "selected_index": 0,
            "tool_name": None,
            "tool_args": None,
            "final_answer": "Found the answer",
            "justification": "Correct answer from search",
        })
        
        mock_invoke.side_effect = [
            reasoner_tool_call, reasoner_tool_call,  # Step 1 reasoners
            judge_tool_call,                          # Step 1 judge
            reasoner_final, reasoner_final,           # Step 2 reasoners
            judge_final,                              # Step 2 judge
        ]
        
        agent = LangGraphReActUSCAgent(
            models=self.models,
            tools=[self.tool],
            config=self.config,
        )
        
        result = agent.run("Search for something")
        
        self.assertEqual(result, "Found the answer")

    @patch('react_usc._internal.llm_io.invoke_chat_text')
    def test_run_handles_max_steps(self, mock_invoke):
        """Test that run handles max_steps gracefully."""
        # Always return TOOL_CALL to force max steps
        tool_call = json.dumps({
            "decision_type": "TOOL_CALL",
            "tool_name": "test_tool",
            "tool_args": {"query": "test"},
            "final_answer": None,
            "brief_rationale": "Need more info",
        })
        
        judge_tool = json.dumps({
            "decision_type": "TOOL_CALL",
            "selected_index": 0,
            "tool_name": "test_tool",
            "tool_args": {"query": "test"},
            "final_answer": None,
            "justification": "Continue searching",
        })
        
        best_effort = json.dumps({
            "decision_type": "FINAL",
            "final_answer": "Best effort answer",
            "justification": "Step limit reached",
        })
        
        # For k_paths=2, max_steps=3: need 2*3 reasoner calls + 3 judge calls + 1 best effort
        mock_invoke.side_effect = [
            tool_call, tool_call, judge_tool,     # Step 1
            tool_call, tool_call, judge_tool,     # Step 2  
            tool_call, tool_call, judge_tool,     # Step 3
            best_effort,                          # Best effort final
        ]
        
        agent = LangGraphReActUSCAgent(
            models=self.models,
            tools=[self.tool],
            config=self.config,
        )
        
        result = agent.run("Keep searching forever")
        
        # Should return some answer (best effort or step limit message)
        self.assertIsNotNone(result)
        self.assertIsInstance(result, str)

    @patch('react_usc._internal.llm_io.invoke_chat_text')
    def test_run_handles_no_final_answer(self, mock_invoke):
        """Test run when no final answer is produced."""
        # Return invalid JSON to cause failures
        mock_invoke.return_value = "This is not valid JSON"
        
        agent = LangGraphReActUSCAgent(
            models=self.models,
            tools=[self.tool],
            config=self.config,
        )
        
        result = agent.run("Query that fails")
        
        # Should return some fallback message
        self.assertIsNotNone(result)
        self.assertIsInstance(result, str)


# =============================================================================
# Test: Agent with Different Configurations
# =============================================================================


class TestAgentConfigurations(unittest.TestCase):
    """Test agent with various configurations."""

    def setUp(self):
        self.mock_model = MagicMock()
        self.mock_model.invoke = MagicMock(return_value=MagicMock(content="{}"))
        self.models = LangGraphModels(reasoner=self.mock_model, judge=self.mock_model)
        
        self.tool = ToolSpec(
            name="tool",
            description="Test",
            input_schema={"type": "object"},
            func=lambda args: "result",
        )

    def test_agent_with_trace_enabled(self):
        """Test agent with trace enabled."""
        config = AgentConfig(
            system_prompt="Test",
            k_paths=2,
            max_steps=3,
            reasoner_model=ModelConfig(name="test", temperature=0.7),
            judge_model=ModelConfig(name="test", temperature=0.7),
            selection_strategy="select_one",
            allow_tool_synthesis=False,
            llm_retry=RetryConfig.none(),
            trace=True,  # Trace enabled
            tool_result_max_chars=1000,
        )
        
        agent = LangGraphReActUSCAgent(
            models=self.models,
            tools=[self.tool],
            config=config,
        )
        
        self.assertTrue(agent.config.trace)

    def test_agent_with_synthesize_strategy(self):
        """Test agent with synthesize_one strategy."""
        config = AgentConfig(
            system_prompt="Test",
            k_paths=3,
            max_steps=5,
            reasoner_model=ModelConfig(name="test", temperature=0.7),
            judge_model=ModelConfig(name="test", temperature=0.7),
            selection_strategy="synthesize_one",
            allow_tool_synthesis=True,
            llm_retry=RetryConfig.none(),
            trace=False,
            tool_result_max_chars=1000,
        )
        
        agent = LangGraphReActUSCAgent(
            models=self.models,
            tools=[self.tool],
            config=config,
        )
        
        self.assertEqual(agent.config.selection_strategy, "synthesize_one")
        self.assertTrue(agent.config.allow_tool_synthesis)

    def test_agent_with_retry_config(self):
        """Test agent with retry configuration."""
        config = AgentConfig(
            system_prompt="Test",
            k_paths=2,
            max_steps=3,
            reasoner_model=ModelConfig(name="test", temperature=0.7),
            judge_model=ModelConfig(name="test", temperature=0.7),
            selection_strategy="select_one",
            allow_tool_synthesis=False,
            llm_retry=RetryConfig(max_retries=3, backoff_seconds=2.0),
            trace=False,
            tool_result_max_chars=1000,
        )
        
        agent = LangGraphReActUSCAgent(
            models=self.models,
            tools=[self.tool],
            config=config,
        )
        
        self.assertEqual(agent.config.llm_retry.max_retries, 3)
        self.assertEqual(agent.config.llm_retry.backoff_seconds, 2.0)


if __name__ == "__main__":
    unittest.main()
