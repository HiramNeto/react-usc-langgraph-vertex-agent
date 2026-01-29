"""
Tests for type definitions (ToolSpec, AgentConstants).

These tests cover:
- ToolSpec validation (name, description, schema, func)
- ToolSpec immutability
- AgentConstants values verification
"""
from __future__ import annotations

import unittest
from dataclasses import FrozenInstanceError

from react_usc import ToolSpec
from react_usc.types import AgentConstants


# =============================================================================
# Test: ToolSpec
# =============================================================================


class TestToolSpec(unittest.TestCase):
    """Test ToolSpec validation and behavior."""

    def setUp(self):
        """Create a valid input schema for tests."""
        self.valid_schema = {
            "type": "object",
            "required": ["query"],
            "properties": {
                "query": {"type": "string"},
            },
        }
        self.valid_func = lambda args: f"Result: {args.get('query', '')}"

    def test_valid_tool_spec(self):
        """Test creating a valid ToolSpec."""
        tool = ToolSpec(
            name="search",
            description="Search for information",
            input_schema=self.valid_schema,
            func=self.valid_func,
        )
        
        self.assertEqual(tool.name, "search")
        self.assertEqual(tool.description, "Search for information")
        self.assertEqual(tool.input_schema, self.valid_schema)
        self.assertTrue(callable(tool.func))

    def test_empty_name_rejected(self):
        """Test that empty tool name raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            ToolSpec(
                name="",
                description="Test tool",
                input_schema=self.valid_schema,
                func=self.valid_func,
            )
        
        self.assertIn("name", str(ctx.exception).lower())
        self.assertIn("empty", str(ctx.exception).lower())

    def test_whitespace_name_rejected(self):
        """Test that whitespace-only tool name raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            ToolSpec(
                name="   \t\n",
                description="Test tool",
                input_schema=self.valid_schema,
                func=self.valid_func,
            )
        
        self.assertIn("name", str(ctx.exception).lower())
        self.assertIn("empty", str(ctx.exception).lower())

    def test_empty_description_rejected(self):
        """Test that empty description raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            ToolSpec(
                name="test_tool",
                description="",
                input_schema=self.valid_schema,
                func=self.valid_func,
            )
        
        self.assertIn("description", str(ctx.exception).lower())

    def test_whitespace_description_rejected(self):
        """Test that whitespace-only description raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            ToolSpec(
                name="test_tool",
                description="   \n\t",
                input_schema=self.valid_schema,
                func=self.valid_func,
            )
        
        self.assertIn("description", str(ctx.exception).lower())

    def test_non_dict_input_schema_rejected(self):
        """Test that non-dict input_schema raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            ToolSpec(
                name="test_tool",
                description="Test tool",
                input_schema="not a dict",  # type: ignore
                func=self.valid_func,
            )
        
        self.assertIn("input_schema", str(ctx.exception).lower())
        self.assertIn("dict", str(ctx.exception).lower())

    def test_list_input_schema_rejected(self):
        """Test that list input_schema raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            ToolSpec(
                name="test_tool",
                description="Test tool",
                input_schema=[],  # type: ignore
                func=self.valid_func,
            )
        
        self.assertIn("input_schema", str(ctx.exception).lower())

    def test_none_input_schema_rejected(self):
        """Test that None input_schema raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            ToolSpec(
                name="test_tool",
                description="Test tool",
                input_schema=None,  # type: ignore
                func=self.valid_func,
            )
        
        self.assertIn("input_schema", str(ctx.exception).lower())

    def test_non_callable_func_rejected(self):
        """Test that non-callable func raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            ToolSpec(
                name="test_tool",
                description="Test tool",
                input_schema=self.valid_schema,
                func="not callable",  # type: ignore
            )
        
        self.assertIn("func", str(ctx.exception).lower())
        self.assertIn("callable", str(ctx.exception).lower())

    def test_none_func_rejected(self):
        """Test that None func raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            ToolSpec(
                name="test_tool",
                description="Test tool",
                input_schema=self.valid_schema,
                func=None,  # type: ignore
            )
        
        self.assertIn("func", str(ctx.exception).lower())

    def test_frozen_dataclass(self):
        """Test that ToolSpec is immutable (frozen)."""
        tool = ToolSpec(
            name="test_tool",
            description="Test tool",
            input_schema=self.valid_schema,
            func=self.valid_func,
        )
        
        # Frozen dataclass raises FrozenInstanceError or AttributeError
        with self.assertRaises((FrozenInstanceError, AttributeError)):
            tool.name = "new_name"  # type: ignore

    def test_func_is_callable(self):
        """Test that the stored func can be called."""
        tool = ToolSpec(
            name="test_tool",
            description="Test tool",
            input_schema=self.valid_schema,
            func=self.valid_func,
        )
        
        result = tool.func({"query": "hello"})
        self.assertEqual(result, "Result: hello")

    def test_empty_input_schema_valid(self):
        """Test that empty dict input_schema is valid."""
        tool = ToolSpec(
            name="no_args_tool",
            description="Tool with no arguments",
            input_schema={},
            func=lambda args: "no args",
        )
        
        self.assertEqual(tool.input_schema, {})

    def test_class_method_as_func(self):
        """Test that bound methods work as func."""
        class Handler:
            def process(self, args):
                return f"Processed: {args}"
        
        handler = Handler()
        tool = ToolSpec(
            name="method_tool",
            description="Tool with method",
            input_schema=self.valid_schema,
            func=handler.process,
        )
        
        result = tool.func({"query": "test"})
        self.assertIn("Processed", result)


# =============================================================================
# Test: AgentConstants
# =============================================================================


class TestAgentConstants(unittest.TestCase):
    """Test AgentConstants values and accessibility."""

    def test_max_reasoner_threads_value(self):
        """Test MAX_REASONER_THREADS has expected value."""
        self.assertEqual(AgentConstants.MAX_REASONER_THREADS, 32)
        self.assertIsInstance(AgentConstants.MAX_REASONER_THREADS, int)

    def test_max_observations_in_summary_value(self):
        """Test MAX_OBSERVATIONS_IN_SUMMARY has expected value."""
        self.assertEqual(AgentConstants.MAX_OBSERVATIONS_IN_SUMMARY, 10)
        self.assertIsInstance(AgentConstants.MAX_OBSERVATIONS_IN_SUMMARY, int)

    def test_default_timeout_seconds_value(self):
        """Test DEFAULT_TIMEOUT_SECONDS has expected value."""
        self.assertEqual(AgentConstants.DEFAULT_TIMEOUT_SECONDS, 20.0)
        self.assertIsInstance(AgentConstants.DEFAULT_TIMEOUT_SECONDS, float)

    def test_default_tool_result_max_chars_value(self):
        """Test DEFAULT_TOOL_RESULT_MAX_CHARS has expected value."""
        self.assertEqual(AgentConstants.DEFAULT_TOOL_RESULT_MAX_CHARS, 4000)
        self.assertIsInstance(AgentConstants.DEFAULT_TOOL_RESULT_MAX_CHARS, int)

    def test_truncation_suffix_chars_value(self):
        """Test TRUNCATION_SUFFIX_CHARS has expected value."""
        self.assertEqual(AgentConstants.TRUNCATION_SUFFIX_CHARS, 24)
        self.assertIsInstance(AgentConstants.TRUNCATION_SUFFIX_CHARS, int)

    def test_default_retry_config_values(self):
        """Test default retry configuration values."""
        self.assertEqual(AgentConstants.DEFAULT_MAX_RETRIES, 2)
        self.assertEqual(AgentConstants.DEFAULT_BACKOFF_SECONDS, 1.0)

    def test_k_paths_bounds(self):
        """Test k_paths boundary values."""
        self.assertEqual(AgentConstants.MIN_K_PATHS, 1)
        self.assertEqual(AgentConstants.MAX_K_PATHS, 100)
        self.assertLess(AgentConstants.MIN_K_PATHS, AgentConstants.MAX_K_PATHS)

    def test_max_steps_bounds(self):
        """Test max_steps boundary values."""
        self.assertEqual(AgentConstants.MIN_MAX_STEPS, 1)
        self.assertEqual(AgentConstants.MAX_MAX_STEPS, 50)
        self.assertLess(AgentConstants.MIN_MAX_STEPS, AgentConstants.MAX_MAX_STEPS)

    def test_temperature_bounds(self):
        """Test temperature boundary values."""
        self.assertEqual(AgentConstants.MIN_TEMPERATURE, 0.0)
        self.assertEqual(AgentConstants.MAX_TEMPERATURE, 2.0)
        self.assertLess(AgentConstants.MIN_TEMPERATURE, AgentConstants.MAX_TEMPERATURE)

    def test_all_constants_accessible(self):
        """Test that all expected constants are accessible."""
        constants = [
            "MAX_REASONER_THREADS",
            "MAX_OBSERVATIONS_IN_SUMMARY",
            "DEFAULT_TIMEOUT_SECONDS",
            "DEFAULT_TOOL_RESULT_MAX_CHARS",
            "TRUNCATION_SUFFIX_CHARS",
            "DEFAULT_MAX_RETRIES",
            "DEFAULT_BACKOFF_SECONDS",
            "MIN_K_PATHS",
            "MAX_K_PATHS",
            "MIN_MAX_STEPS",
            "MAX_MAX_STEPS",
            "MIN_TEMPERATURE",
            "MAX_TEMPERATURE",
        ]
        
        for const_name in constants:
            self.assertTrue(
                hasattr(AgentConstants, const_name),
                f"AgentConstants missing constant: {const_name}",
            )


if __name__ == "__main__":
    unittest.main()
