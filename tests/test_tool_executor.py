"""
Tests for ToolExecutor and ToolResult.

These tests cover:
- Successful tool execution
- Unknown tool handling
- Tool execution errors
- ToolResult properties and formatting
- Retry plugin integration
"""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from react_usc import (
    AgentConfig,
    ModelConfig,
    RetryConfig,
    ToolRegistry,
    ToolSpec,
)
from react_usc.executors import ToolExecutor, ToolResult


# =============================================================================
# Test: ToolResult
# =============================================================================


class TestToolResult(unittest.TestCase):
    """Test ToolResult dataclass."""

    def test_successful_result(self):
        """Test creating a successful ToolResult."""
        result = ToolResult(
            tool_name="calculator",
            output={"value": 42},
        )
        
        self.assertEqual(result.tool_name, "calculator")
        self.assertEqual(result.output, {"value": 42})
        self.assertIsNone(result.error)
        self.assertFalse(result.is_reflection_abort)
        self.assertTrue(result.is_success)

    def test_error_result(self):
        """Test creating an error ToolResult."""
        result = ToolResult(
            tool_name="api_client",
            output=None,
            error="Connection timeout",
        )
        
        self.assertIsNone(result.output)
        self.assertEqual(result.error, "Connection timeout")
        self.assertFalse(result.is_success)

    def test_reflection_abort_result(self):
        """Test creating a reflection abort ToolResult."""
        result = ToolResult(
            tool_name="api_client",
            output="Reflection Error: API key invalid",
            is_reflection_abort=True,
        )
        
        self.assertTrue(result.is_reflection_abort)
        self.assertFalse(result.is_success)

    def test_to_observation_success(self):
        """Test to_observation for successful result."""
        result = ToolResult(
            tool_name="calculator",
            output=42,
        )
        
        observation = result.to_observation()
        
        self.assertIn("calculator", observation)
        self.assertIn("42", observation)
        self.assertIn("=>", observation)

    def test_to_observation_error(self):
        """Test to_observation for error result."""
        result = ToolResult(
            tool_name="api_client",
            output=None,
            error="Network error",
        )
        
        observation = result.to_observation()
        
        self.assertIn("api_client", observation)
        self.assertIn("tool_exception", observation)
        self.assertIn("Network error", observation)

    def test_to_observation_reflection_abort(self):
        """Test to_observation for reflection abort."""
        result = ToolResult(
            tool_name="api_client",
            output="Reflection Error: Cannot recover",
            is_reflection_abort=True,
        )
        
        observation = result.to_observation()
        
        self.assertIn("Reflection Error", observation)

    def test_to_observation_with_truncation(self):
        """Test to_observation with truncation enabled."""
        long_output = "x" * 1000
        result = ToolResult(
            tool_name="search",
            output=long_output,
        )
        
        observation = result.to_observation(max_chars=100, truncate_enabled=True)
        
        # Should be truncated
        self.assertLess(len(observation), 200)

    def test_to_observation_without_truncation(self):
        """Test to_observation without truncation."""
        long_output = "x" * 1000
        result = ToolResult(
            tool_name="search",
            output=long_output,
        )
        
        observation = result.to_observation(max_chars=0, truncate_enabled=False)
        
        # Should not be truncated
        self.assertIn(long_output, observation)

    def test_to_observation_dict_output(self):
        """Test to_observation with dict output."""
        result = ToolResult(
            tool_name="api",
            output={"status": "ok", "data": [1, 2, 3]},
        )
        
        observation = result.to_observation()
        
        self.assertIn("status", observation)
        self.assertIn("ok", observation)

    def test_frozen_dataclass(self):
        """Test that ToolResult is immutable."""
        result = ToolResult(tool_name="test", output="value")
        
        with self.assertRaises(AttributeError):
            result.output = "new value"  # type: ignore


# =============================================================================
# Test: ToolExecutor
# =============================================================================


class TestToolExecutor(unittest.TestCase):
    """Test ToolExecutor functionality."""

    def setUp(self):
        """Create test fixtures."""
        self.valid_schema = {
            "type": "object",
            "required": ["query"],
            "properties": {"query": {"type": "string"}},
        }
        
        self.tool = ToolSpec(
            name="test_tool",
            description="A test tool",
            input_schema=self.valid_schema,
            func=lambda args: f"Result for: {args.get('query', '')}",
        )
        
        self.failing_tool = ToolSpec(
            name="failing_tool",
            description="A tool that always fails",
            input_schema=self.valid_schema,
            func=self._raise_error,
        )
        
        self.registry = ToolRegistry([self.tool, self.failing_tool])
        
        self.config = AgentConfig(
            system_prompt="test",
            k_paths=3,
            max_steps=10,
            reasoner_model=ModelConfig(name="test", temperature=0.7),
            judge_model=ModelConfig(name="test", temperature=0.7),
            selection_strategy="select_one",
            allow_tool_synthesis=False,
            llm_retry=RetryConfig.none(),
            trace=False,
            tool_result_max_chars=4000,
        )

    @staticmethod
    def _raise_error(args):
        """Helper that raises an error."""
        raise ValueError("Tool execution failed intentionally")

    def test_execute_successful(self):
        """Test successful tool execution."""
        executor = ToolExecutor(
            tools=self.registry,
            config=self.config,
        )
        
        result = executor.execute(
            tool_name="test_tool",
            tool_args={"query": "hello"},
            user_query="Test query",
        )
        
        self.assertTrue(result.is_success)
        self.assertEqual(result.tool_name, "test_tool")
        self.assertIn("hello", result.output)
        self.assertIsNone(result.error)

    def test_execute_unknown_tool(self):
        """Test execution of unknown tool returns error result."""
        executor = ToolExecutor(
            tools=self.registry,
            config=self.config,
        )
        
        result = executor.execute(
            tool_name="nonexistent_tool",
            tool_args={},
            user_query="Test query",
        )
        
        self.assertFalse(result.is_success)
        self.assertIn("Unknown tool", result.error)

    def test_execute_invalid_args(self):
        """Test execution with invalid arguments."""
        executor = ToolExecutor(
            tools=self.registry,
            config=self.config,
        )
        
        # Missing required 'query' field
        result = executor.execute(
            tool_name="test_tool",
            tool_args={"wrong_field": "value"},
            user_query="Test query",
        )
        
        self.assertFalse(result.is_success)
        self.assertIn("Invalid args", result.error)

    def test_execute_tool_exception(self):
        """Test handling of tool execution exception."""
        executor = ToolExecutor(
            tools=self.registry,
            config=self.config,
        )
        
        result = executor.execute(
            tool_name="failing_tool",
            tool_args={"query": "test"},
            user_query="Test query",
        )
        
        self.assertFalse(result.is_success)
        self.assertIn("ValueError", result.error)
        self.assertIn("intentionally", result.error)

    def test_execute_with_empty_args(self):
        """Test execution with empty args when tool requires args."""
        executor = ToolExecutor(
            tools=self.registry,
            config=self.config,
        )
        
        result = executor.execute(
            tool_name="test_tool",
            tool_args={},
            user_query="Test query",
        )
        
        # Should fail validation since 'query' is required
        self.assertFalse(result.is_success)

    def test_execute_with_retry_plugin(self):
        """Test execution with retry plugin."""
        mock_plugin = MagicMock()
        mock_plugin.run.return_value = "Plugin handled result"
        
        executor = ToolExecutor(
            tools=self.registry,
            config=self.config,
            retry_plugin=mock_plugin,
        )
        
        result = executor.execute(
            tool_name="test_tool",
            tool_args={"query": "test"},
            user_query="Test query",
        )
        
        # Plugin should be called
        mock_plugin.run.assert_called_once()
        self.assertTrue(result.is_success)
        self.assertEqual(result.output, "Plugin handled result")

    def test_execute_reflection_abort_signal(self):
        """Test handling of reflection abort signal from plugin."""
        mock_plugin = MagicMock()
        mock_plugin.run.return_value = "Reflection Error: Cannot fix this"
        
        executor = ToolExecutor(
            tools=self.registry,
            config=self.config,
            retry_plugin=mock_plugin,
        )
        
        result = executor.execute(
            tool_name="test_tool",
            tool_args={"query": "test"},
            user_query="Test query",
        )
        
        self.assertTrue(result.is_reflection_abort)
        self.assertFalse(result.is_success)
        self.assertIn("Reflection Error", result.output)

    def test_execute_logs_tool_call(self):
        """Test that tool execution logs the call."""
        mock_logger = MagicMock()
        
        executor = ToolExecutor(
            tools=self.registry,
            config=self.config,
            logger=mock_logger,
        )
        
        executor.execute(
            tool_name="test_tool",
            tool_args={"query": "test"},
            user_query="Test query",
        )
        
        mock_logger.tool_call.assert_called()

    def test_execute_none_tool_args(self):
        """Test execution with None tool_args."""
        tool_no_args = ToolSpec(
            name="no_args_tool",
            description="Tool with no required args",
            input_schema={"type": "object"},
            func=lambda args: "no args needed",
        )
        
        registry = ToolRegistry([tool_no_args])
        executor = ToolExecutor(tools=registry, config=self.config)
        
        result = executor.execute(
            tool_name="no_args_tool",
            tool_args={},
            user_query="Test query",
        )
        
        self.assertTrue(result.is_success)


# =============================================================================
# Test: ToolExecutor with Various Output Types
# =============================================================================


class TestToolExecutorOutputTypes(unittest.TestCase):
    """Test ToolExecutor with various tool output types."""

    def setUp(self):
        self.config = AgentConfig(
            system_prompt="test",
            k_paths=3,
            max_steps=10,
            reasoner_model=ModelConfig(name="test", temperature=0.7),
            judge_model=ModelConfig(name="test", temperature=0.7),
            selection_strategy="select_one",
            allow_tool_synthesis=False,
            llm_retry=RetryConfig.none(),
            trace=False,
            tool_result_max_chars=4000,
        )
        self.schema = {"type": "object"}

    def test_string_output(self):
        """Test tool returning a string."""
        tool = ToolSpec(
            name="string_tool",
            description="Returns string",
            input_schema=self.schema,
            func=lambda args: "Hello, world!",
        )
        executor = ToolExecutor(tools=ToolRegistry([tool]), config=self.config)
        
        result = executor.execute(
            tool_name="string_tool",
            tool_args={},
            user_query="test",
        )
        
        self.assertTrue(result.is_success)
        self.assertEqual(result.output, "Hello, world!")

    def test_dict_output(self):
        """Test tool returning a dict."""
        tool = ToolSpec(
            name="dict_tool",
            description="Returns dict",
            input_schema=self.schema,
            func=lambda args: {"key": "value", "count": 42},
        )
        executor = ToolExecutor(tools=ToolRegistry([tool]), config=self.config)
        
        result = executor.execute(
            tool_name="dict_tool",
            tool_args={},
            user_query="test",
        )
        
        self.assertTrue(result.is_success)
        self.assertEqual(result.output["key"], "value")
        self.assertEqual(result.output["count"], 42)

    def test_list_output(self):
        """Test tool returning a list."""
        tool = ToolSpec(
            name="list_tool",
            description="Returns list",
            input_schema=self.schema,
            func=lambda args: [1, 2, 3, 4, 5],
        )
        executor = ToolExecutor(tools=ToolRegistry([tool]), config=self.config)
        
        result = executor.execute(
            tool_name="list_tool",
            tool_args={},
            user_query="test",
        )
        
        self.assertTrue(result.is_success)
        self.assertEqual(result.output, [1, 2, 3, 4, 5])

    def test_none_output(self):
        """Test tool returning None."""
        tool = ToolSpec(
            name="none_tool",
            description="Returns None",
            input_schema=self.schema,
            func=lambda args: None,
        )
        executor = ToolExecutor(tools=ToolRegistry([tool]), config=self.config)
        
        result = executor.execute(
            tool_name="none_tool",
            tool_args={},
            user_query="test",
        )
        
        self.assertTrue(result.is_success)
        self.assertIsNone(result.output)

    def test_integer_output(self):
        """Test tool returning an integer."""
        tool = ToolSpec(
            name="int_tool",
            description="Returns int",
            input_schema=self.schema,
            func=lambda args: 42,
        )
        executor = ToolExecutor(tools=ToolRegistry([tool]), config=self.config)
        
        result = executor.execute(
            tool_name="int_tool",
            tool_args={},
            user_query="test",
        )
        
        self.assertTrue(result.is_success)
        self.assertEqual(result.output, 42)


if __name__ == "__main__":
    unittest.main()
