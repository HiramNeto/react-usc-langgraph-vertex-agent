"""
Tests for plugin classes (ReflectAndRetryToolPlugin).

These tests cover:
- Non-JSON salvage logic in reflection plugin
- ReflectAndRetryToolPlugin initialization
- ReflectAndRetryToolPlugin.run() retry logic
- Reflection verdict handling (RETRY, WAIT, ABORT)
- Exponential backoff calculation
"""
from __future__ import annotations

import json
import unittest
from unittest.mock import MagicMock, patch

from react_usc import ReflectAndRetryToolPlugin, ToolSpec
from react_usc.plugins import ReflectionResult

from tests.conftest import make_mock_model


# =============================================================================
# Test: salvage_reflection_final (moved from plugin to _internal/salvage.py)
# =============================================================================


class TestReflectionSalvageNonJson(unittest.TestCase):
    """Test the non-JSON salvage logic for reflection decisions."""

    def test_salvage_abort_verdict(self):
        """Test salvaging ABORT verdict."""
        from react_usc._internal.salvage import salvage_reflection_final
        
        text = '''verdict: ABORT
abort_suggestion: The API key is invalid and cannot be fixed.
analysis: The 403 error indicates permanent authentication failure.'''

        result = salvage_reflection_final(text)

        self.assertIsNotNone(result)
        self.assertEqual(result["verdict"], "ABORT")
        self.assertIn("API key", result.get("abort_suggestion", ""))
        self.assertIn("authentication", result.get("analysis", ""))

    def test_salvage_wait_verdict(self):
        """Test salvaging WAIT verdict."""
        from react_usc._internal.salvage import salvage_reflection_final
        
        text = '''verdict: WAIT
analysis: The 503 error is transient, we should wait and retry.'''

        result = salvage_reflection_final(text)

        self.assertIsNotNone(result)
        self.assertEqual(result["verdict"], "WAIT")

    def test_salvage_retry_verdict_with_args(self):
        """Test salvaging RETRY verdict with JSON args."""
        from react_usc._internal.salvage import salvage_reflection_final
        
        text = '''verdict: RETRY
retry_args: {"query": "fixed_value", "limit": 10}
analysis: Added the missing limit parameter.'''

        result = salvage_reflection_final(text)

        self.assertIsNotNone(result)
        self.assertEqual(result["verdict"], "RETRY")
        self.assertIsNotNone(result.get("retry_args"))
        self.assertEqual(result["retry_args"]["query"], "fixed_value")
        self.assertEqual(result["retry_args"]["limit"], 10)

    def test_salvage_retry_with_embedded_json(self):
        """Test salvaging RETRY with JSON embedded in text."""
        from react_usc._internal.salvage import salvage_reflection_final
        
        text = '''The error indicates missing parameters.
verdict: RETRY
Here are the corrected args: {"endpoint": "/api/users", "method": "GET"}
Please try again.'''

        result = salvage_reflection_final(text)

        self.assertIsNotNone(result)
        self.assertEqual(result["verdict"], "RETRY")
        # Should find the embedded JSON
        self.assertIsNotNone(result.get("retry_args"))
        self.assertEqual(result["retry_args"]["endpoint"], "/api/users")

    def test_infer_abort_from_content(self):
        """Test inferring ABORT verdict from content."""
        from react_usc._internal.salvage import salvage_reflection_final
        
        text = '''This operation cannot succeed.
The user should ABORT and try a different approach.'''

        result = salvage_reflection_final(text)

        self.assertIsNotNone(result)
        self.assertEqual(result["verdict"], "ABORT")

    def test_infer_wait_from_content(self):
        """Test inferring WAIT verdict from content."""
        from react_usc._internal.salvage import salvage_reflection_final
        
        text = '''The service is temporarily unavailable.
We should WAIT for the service to recover.'''

        result = salvage_reflection_final(text)

        self.assertIsNotNone(result)
        self.assertEqual(result["verdict"], "WAIT")

    def test_no_salvage_for_empty_text(self):
        """Test that empty text returns None."""
        from react_usc._internal.salvage import salvage_reflection_final
        
        result = salvage_reflection_final("")
        self.assertIsNone(result)

        result = salvage_reflection_final("   ")
        self.assertIsNone(result)

    def test_no_salvage_without_verdict(self):
        """Test that text without any verdict indication returns None."""
        from react_usc._internal.salvage import salvage_reflection_final
        
        text = '''This is just some random text
with no decision information.'''

        result = salvage_reflection_final(text)

        self.assertIsNone(result)

    def test_salvage_with_quoted_values(self):
        """Test salvaging with quoted field values."""
        from react_usc._internal.salvage import salvage_reflection_final
        
        text = '''verdict: ABORT
abort_suggestion: "Use a different endpoint"'''

        result = salvage_reflection_final(text)

        self.assertIsNotNone(result)
        self.assertEqual(result["abort_suggestion"], "Use a different endpoint")


# =============================================================================
# Test: ReflectionResult
# =============================================================================


class TestReflectionResult(unittest.TestCase):
    """Test ReflectionResult dataclass."""

    def test_should_retry_true(self):
        """Test should_retry returns True when conditions met."""
        result = ReflectionResult(
            verdict="RETRY",
            retry_args={"key": "value"},
        )
        
        self.assertTrue(result.should_retry)
        self.assertFalse(result.should_wait)
        self.assertFalse(result.should_abort)

    def test_should_retry_false_without_args(self):
        """Test should_retry returns False when no retry_args."""
        result = ReflectionResult(verdict="RETRY")
        
        self.assertFalse(result.should_retry)

    def test_should_wait_true(self):
        """Test should_wait returns True for WAIT verdict."""
        result = ReflectionResult(verdict="WAIT")
        
        self.assertTrue(result.should_wait)
        self.assertFalse(result.should_retry)
        self.assertFalse(result.should_abort)

    def test_should_abort_true(self):
        """Test should_abort returns True for ABORT verdict."""
        result = ReflectionResult(
            verdict="ABORT",
            abort_suggestion="Cannot recover",
        )
        
        self.assertTrue(result.should_abort)
        self.assertFalse(result.should_retry)
        self.assertFalse(result.should_wait)

    def test_frozen_dataclass(self):
        """Test that ReflectionResult is immutable."""
        result = ReflectionResult(verdict="ABORT")
        
        with self.assertRaises(AttributeError):
            result.verdict = "RETRY"  # type: ignore


# =============================================================================
# Test: ReflectAndRetryToolPlugin Initialization
# =============================================================================


class TestReflectAndRetryToolPluginInit(unittest.TestCase):
    """Test ReflectAndRetryToolPlugin initialization."""

    def test_valid_initialization(self):
        """Test valid plugin initialization."""
        model = make_mock_model()
        
        plugin = ReflectAndRetryToolPlugin(
            model=model,
            max_retries=3,
            backoff_seconds=1.0,
        )
        
        self.assertEqual(plugin.max_retries, 3)
        self.assertEqual(plugin.backoff_seconds, 1.0)

    def test_negative_max_retries_rejected(self):
        """Test that negative max_retries raises ValueError."""
        model = make_mock_model()
        
        with self.assertRaises(ValueError) as ctx:
            ReflectAndRetryToolPlugin(
                model=model,
                max_retries=-1,
            )
        
        self.assertIn("max_retries", str(ctx.exception).lower())

    def test_negative_backoff_rejected(self):
        """Test that negative backoff_seconds raises ValueError."""
        model = make_mock_model()
        
        with self.assertRaises(ValueError) as ctx:
            ReflectAndRetryToolPlugin(
                model=model,
                backoff_seconds=-0.5,
            )
        
        self.assertIn("backoff_seconds", str(ctx.exception).lower())

    def test_zero_retries_valid(self):
        """Test that zero max_retries is valid (no retries)."""
        model = make_mock_model()
        
        plugin = ReflectAndRetryToolPlugin(
            model=model,
            max_retries=0,
        )
        
        self.assertEqual(plugin.max_retries, 0)


# =============================================================================
# Test: ReflectAndRetryToolPlugin.run()
# =============================================================================


class TestReflectAndRetryToolPluginRun(unittest.TestCase):
    """Test ReflectAndRetryToolPlugin.run() method."""

    def setUp(self):
        self.model = make_mock_model()
        self.tool_schema = {
            "type": "object",
            "required": ["query"],
            "properties": {"query": {"type": "string"}},
        }
        self.tool_spec = ToolSpec(
            name="test_tool",
            description="A test tool",
            input_schema=self.tool_schema,
            func=lambda args: f"Result: {args.get('query', '')}",
        )

    def test_run_successful_first_try(self):
        """Test run succeeds on first try."""
        plugin = ReflectAndRetryToolPlugin(
            model=self.model,
            max_retries=3,
        )
        
        result = plugin.run(
            tool_name="test_tool",
            tool_args={"query": "test"},
            tool_func=lambda args: f"Result: {args['query']}",
            all_tools=[self.tool_spec],
            user_query="Test query",
            tool_input_schema=self.tool_schema,
        )
        
        self.assertEqual(result, "Result: test")

    def test_run_exhausted_retries_raises(self):
        """Test run raises after exhausting retries."""
        def failing_func(args):
            raise ValueError("Always fails")
        
        # Make reflection return RETRY to continue loop
        self.model.invoke.return_value = MagicMock(
            content=json.dumps({
                "verdict": "RETRY",
                "retry_args": {"query": "new_value"},
            })
        )
        
        plugin = ReflectAndRetryToolPlugin(
            model=self.model,
            max_retries=2,
        )
        
        with self.assertRaises(ValueError) as ctx:
            plugin.run(
                tool_name="test_tool",
                tool_args={"query": "test"},
                tool_func=failing_func,
                all_tools=[self.tool_spec],
                user_query="Test query",
                tool_input_schema=self.tool_schema,
            )
        
        self.assertIn("Always fails", str(ctx.exception))

    @patch('react_usc._internal.llm_io.invoke_chat_text')
    def test_run_abort_returns_message(self, mock_invoke):
        """Test run returns abort message on ABORT verdict."""
        mock_invoke.return_value = json.dumps({
            "verdict": "ABORT",
            "abort_suggestion": "Cannot recover from this error",
        })
        
        call_count = [0]
        
        def failing_then_succeeds(args):
            call_count[0] += 1
            if call_count[0] == 1:
                raise ValueError("First call fails")
            return "Success"
        
        plugin = ReflectAndRetryToolPlugin(
            model=self.model,
            max_retries=3,
        )
        
        result = plugin.run(
            tool_name="test_tool",
            tool_args={"query": "test"},
            tool_func=failing_then_succeeds,
            all_tools=[self.tool_spec],
            user_query="Test query",
            tool_input_schema=self.tool_schema,
        )
        
        self.assertIn("Reflection Error", result)
        self.assertIn("Cannot recover", result)

    @patch('react_usc._internal.llm_io.invoke_chat_text')
    @patch('react_usc.plugins.time.sleep')
    def test_run_wait_applies_backoff(self, mock_sleep, mock_invoke):
        """Test run applies backoff on WAIT verdict."""
        call_count = [0]
        
        def response_generator(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return json.dumps({"verdict": "WAIT"})
            return json.dumps({"verdict": "ABORT", "abort_suggestion": "Stop"})
        
        mock_invoke.side_effect = response_generator
        
        call_attempt = [0]
        
        def failing_func(args):
            call_attempt[0] += 1
            raise ValueError(f"Fail {call_attempt[0]}")
        
        plugin = ReflectAndRetryToolPlugin(
            model=self.model,
            max_retries=3,
            backoff_seconds=1.0,
        )
        
        result = plugin.run(
            tool_name="test_tool",
            tool_args={"query": "test"},
            tool_func=failing_func,
            all_tools=[self.tool_spec],
            user_query="Test query",
            tool_input_schema=self.tool_schema,
        )
        
        # sleep should be called once for WAIT
        mock_sleep.assert_called()

    @patch('react_usc._internal.llm_io.invoke_chat_text')
    def test_run_retry_with_new_args(self, mock_invoke):
        """Test run retries with new args from reflection."""
        mock_invoke.return_value = json.dumps({
            "verdict": "RETRY",
            "retry_args": {"query": "fixed_query"},
        })
        
        call_count = [0]
        
        def check_args_func(args):
            call_count[0] += 1
            if call_count[0] == 1:
                raise ValueError("Need fixed query")
            return f"Success: {args['query']}"
        
        plugin = ReflectAndRetryToolPlugin(
            model=self.model,
            max_retries=3,
        )
        
        result = plugin.run(
            tool_name="test_tool",
            tool_args={"query": "original"},
            tool_func=check_args_func,
            all_tools=[self.tool_spec],
            user_query="Test query",
            tool_input_schema=self.tool_schema,
        )
        
        # Second call should succeed with fixed args
        self.assertEqual(result, "Success: fixed_query")

    def test_calculate_backoff(self):
        """Test exponential backoff calculation."""
        plugin = ReflectAndRetryToolPlugin(
            model=self.model,
            backoff_seconds=1.0,
        )
        
        # Attempt 0: 1.0 * 2^0 = 1.0
        self.assertEqual(plugin._calculate_backoff(0), 1.0)
        # Attempt 1: 1.0 * 2^1 = 2.0
        self.assertEqual(plugin._calculate_backoff(1), 2.0)
        # Attempt 2: 1.0 * 2^2 = 4.0
        self.assertEqual(plugin._calculate_backoff(2), 4.0)

    def test_create_abort_message(self):
        """Test abort message creation."""
        plugin = ReflectAndRetryToolPlugin(
            model=self.model,
        )
        
        message = plugin._create_abort_message("api_client", "Invalid credentials")
        
        self.assertIn("Reflection Error", message)
        self.assertIn("api_client", message)
        self.assertIn("Invalid credentials", message)


if __name__ == "__main__":
    unittest.main()
