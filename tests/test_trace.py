"""
Tests for trace output functions.

These tests cover:
- trace_candidates() formatting
- trace_judge() formatting
- trace_tool_call() formatting
- trace_tool_result() formatting
- trace_tool_exception() formatting
- trace_message() with levels
- Color support detection
"""
from __future__ import annotations

import unittest
from io import StringIO
from unittest.mock import patch

from react_usc import JudgeDecision, ReasonerDecision
from react_usc.trace import (
    _Colors,
    _colorize,
    trace_candidates,
    trace_judge,
    trace_message,
    trace_tool_call,
    trace_tool_exception,
    trace_tool_result,
)


# =============================================================================
# Test: Color Utilities
# =============================================================================


class TestColors(unittest.TestCase):
    """Test color utility functions."""

    def test_color_constants_exist(self):
        """Test that color constants are defined."""
        self.assertIsNotNone(_Colors.HEADER)
        self.assertIsNotNone(_Colors.BLUE)
        self.assertIsNotNone(_Colors.CYAN)
        self.assertIsNotNone(_Colors.GREEN)
        self.assertIsNotNone(_Colors.YELLOW)
        self.assertIsNotNone(_Colors.RED)
        self.assertIsNotNone(_Colors.BOLD)
        self.assertIsNotNone(_Colors.DIM)
        self.assertIsNotNone(_Colors.RESET)

    def test_colorize_with_color_support(self):
        """Test colorize when color is supported."""
        with patch.object(_Colors, 'supports_color', return_value=True):
            result = _colorize("test", _Colors.RED)
            
            self.assertIn(_Colors.RED, result)
            self.assertIn(_Colors.RESET, result)
            self.assertIn("test", result)

    def test_colorize_without_color_support(self):
        """Test colorize when color is not supported."""
        with patch.object(_Colors, 'supports_color', return_value=False):
            result = _colorize("test", _Colors.RED)
            
            # Should return plain text without ANSI codes
            self.assertEqual(result, "test")


# =============================================================================
# Test: trace_candidates
# =============================================================================


class TestTraceCandidates(unittest.TestCase):
    """Test trace_candidates function."""

    @patch('sys.stdout', new_callable=StringIO)
    def test_trace_candidates_with_valid(self, mock_stdout):
        """Test trace_candidates with valid candidates."""
        valid = [
            ReasonerDecision(
                decision_type="FINAL",
                tool_name=None,
                tool_args=None,
                final_answer="The answer is 42",
                brief_rationale="Computed from query",
            ),
        ]
        
        trace_candidates(step=1, k=3, valid=valid, invalid=[])
        
        output = mock_stdout.getvalue()
        self.assertIn("Step 1", output)
        self.assertIn("K=3", output)
        self.assertIn("Valid candidates", output)

    @patch('sys.stdout', new_callable=StringIO)
    def test_trace_candidates_with_tool_call(self, mock_stdout):
        """Test trace_candidates with TOOL_CALL candidate."""
        valid = [
            ReasonerDecision(
                decision_type="TOOL_CALL",
                tool_name="calculator",
                tool_args={"expression": "2+2"},
                final_answer=None,
                brief_rationale="Need to compute",
            ),
        ]
        
        trace_candidates(step=2, k=5, valid=valid, invalid=[])
        
        output = mock_stdout.getvalue()
        self.assertIn("Step 2", output)
        self.assertIn("calculator", output)

    @patch('sys.stdout', new_callable=StringIO)
    def test_trace_candidates_with_invalid(self, mock_stdout):
        """Test trace_candidates with invalid candidates."""
        invalid = ["Parse error: invalid JSON", "Timeout after 20s"]
        
        trace_candidates(step=1, k=3, valid=[], invalid=invalid)
        
        output = mock_stdout.getvalue()
        self.assertIn("Invalid candidates", output)
        self.assertIn("Parse error", output)

    @patch('sys.stdout', new_callable=StringIO)
    def test_trace_candidates_no_valid(self, mock_stdout):
        """Test trace_candidates with no valid candidates."""
        trace_candidates(step=1, k=3, valid=[], invalid=[])
        
        output = mock_stdout.getvalue()
        self.assertIn("none", output.lower())

    @patch('sys.stdout', new_callable=StringIO)
    def test_trace_candidates_truncates_long_invalid(self, mock_stdout):
        """Test that invalid reasons are truncated."""
        invalid = [f"Error {i}: " + "x" * 500 for i in range(10)]
        
        trace_candidates(step=1, k=10, valid=[], invalid=invalid)
        
        output = mock_stdout.getvalue()
        # Should show truncation indicator
        self.assertIn("more invalid", output)


# =============================================================================
# Test: trace_judge
# =============================================================================


class TestTraceJudge(unittest.TestCase):
    """Test trace_judge function."""

    @patch('sys.stdout', new_callable=StringIO)
    def test_trace_judge_final(self, mock_stdout):
        """Test trace_judge with FINAL decision."""
        decision = JudgeDecision.create_final(
            answer="The answer is 42",
            justification="Selected best candidate",
        )
        
        trace_judge(step=1, decision=decision)
        
        output = mock_stdout.getvalue()
        self.assertIn("Step 1", output)
        self.assertIn("FINAL", output)
        self.assertIn("42", output)

    @patch('sys.stdout', new_callable=StringIO)
    def test_trace_judge_tool_call(self, mock_stdout):
        """Test trace_judge with TOOL_CALL decision."""
        decision = JudgeDecision.create_tool_call(
            tool_name="search",
            tool_args={"query": "test"},
            justification="Need more information",
        )
        
        trace_judge(step=2, decision=decision)
        
        output = mock_stdout.getvalue()
        self.assertIn("Step 2", output)
        self.assertIn("TOOL_CALL", output)
        self.assertIn("search", output)

    @patch('sys.stdout', new_callable=StringIO)
    def test_trace_judge_with_selected_index(self, mock_stdout):
        """Test trace_judge shows selected_index."""
        decision = JudgeDecision.create_tool_call(
            tool_name="calc",
            tool_args={},
            justification="Best choice",
            selected_index=2,
        )
        
        trace_judge(step=3, decision=decision)
        
        output = mock_stdout.getvalue()
        self.assertIn("selected_index=2", output)


# =============================================================================
# Test: trace_tool_call
# =============================================================================


class TestTraceToolCall(unittest.TestCase):
    """Test trace_tool_call function."""

    @patch('sys.stdout', new_callable=StringIO)
    def test_trace_tool_call(self, mock_stdout):
        """Test trace_tool_call output."""
        trace_tool_call(tool_name="calculator", args='{"x": 1}')
        
        output = mock_stdout.getvalue()
        self.assertIn("Tool call", output)
        self.assertIn("calculator", output)

    @patch('sys.stdout', new_callable=StringIO)
    def test_trace_tool_call_truncates_long_args(self, mock_stdout):
        """Test that long args are truncated."""
        long_args = '{"data": "' + "x" * 500 + '"}'
        
        trace_tool_call(tool_name="api", args=long_args)
        
        output = mock_stdout.getvalue()
        # Output should be shorter than the original args
        self.assertLess(len(output), len(long_args))


# =============================================================================
# Test: trace_tool_result
# =============================================================================


class TestTraceToolResult(unittest.TestCase):
    """Test trace_tool_result function."""

    @patch('sys.stdout', new_callable=StringIO)
    def test_trace_tool_result(self, mock_stdout):
        """Test trace_tool_result output."""
        trace_tool_result(tool_name="search", result='{"count": 5}')
        
        output = mock_stdout.getvalue()
        self.assertIn("Tool result", output)
        self.assertIn("search", output)
        self.assertIn("count", output)

    @patch('sys.stdout', new_callable=StringIO)
    def test_trace_tool_result_truncates(self, mock_stdout):
        """Test that long results are truncated."""
        long_result = '{"data": "' + "x" * 1000 + '"}'
        
        trace_tool_result(tool_name="api", result=long_result)
        
        output = mock_stdout.getvalue()
        # Output should be shorter
        self.assertLess(len(output), len(long_result))


# =============================================================================
# Test: trace_tool_exception
# =============================================================================


class TestTraceToolException(unittest.TestCase):
    """Test trace_tool_exception function."""

    @patch('sys.stdout', new_callable=StringIO)
    def test_trace_tool_exception(self, mock_stdout):
        """Test trace_tool_exception output."""
        trace_tool_exception(
            tool_name="api_client",
            error="ConnectionError: Network unreachable",
        )
        
        output = mock_stdout.getvalue()
        self.assertIn("Tool exception", output)
        self.assertIn("api_client", output)
        self.assertIn("ConnectionError", output)

    @patch('sys.stdout', new_callable=StringIO)
    def test_trace_tool_exception_truncates(self, mock_stdout):
        """Test that long errors are truncated."""
        long_error = "Error: " + "x" * 1000
        
        trace_tool_exception(tool_name="api", error=long_error)
        
        output = mock_stdout.getvalue()
        # Output should be shorter
        self.assertLess(len(output), len(long_error))


# =============================================================================
# Test: trace_message
# =============================================================================


class TestTraceMessage(unittest.TestCase):
    """Test trace_message function."""

    @patch('sys.stdout', new_callable=StringIO)
    def test_trace_message_info(self, mock_stdout):
        """Test trace_message with info level."""
        trace_message("Test info message", level="info")
        
        output = mock_stdout.getvalue()
        self.assertIn("INFO", output)
        self.assertIn("Test info message", output)

    @patch('sys.stdout', new_callable=StringIO)
    def test_trace_message_warning(self, mock_stdout):
        """Test trace_message with warning level."""
        trace_message("Test warning message", level="warning")
        
        output = mock_stdout.getvalue()
        self.assertIn("WARN", output)
        self.assertIn("Test warning message", output)

    @patch('sys.stdout', new_callable=StringIO)
    def test_trace_message_error(self, mock_stdout):
        """Test trace_message with error level."""
        trace_message("Test error message", level="error")
        
        output = mock_stdout.getvalue()
        self.assertIn("ERROR", output)
        self.assertIn("Test error message", output)

    @patch('sys.stdout', new_callable=StringIO)
    def test_trace_message_default_level(self, mock_stdout):
        """Test trace_message with default level."""
        trace_message("Default level message")
        
        output = mock_stdout.getvalue()
        self.assertIn("INFO", output)


if __name__ == "__main__":
    unittest.main()
