"""
Tests for exception hierarchy and Result type.

These tests cover:
- Base USCAgentError behavior
- All exception subclasses
- Exception message formatting with details
- Result type success/error states
- Result.unwrap() and unwrap_or() methods
"""
from __future__ import annotations

import unittest

from react_usc.exceptions import (
    USCAgentError,
    ConfigurationError,
    LLMError,
    StructuredOutputError,
    JSONParseError,
    LLMTimeoutError,
    ValidationError,
    DecisionValidationError,
    ToolArgsValidationError,
    ToolError,
    UnknownToolError,
    ToolExecutionError,
    ToolReflectionError,
    AgentLoopError,
    MaxStepsExceededError,
    NoValidCandidatesError,
    Result,
)


# =============================================================================
# Test: Base USCAgentError
# =============================================================================


class TestUSCAgentError(unittest.TestCase):
    """Test base USCAgentError behavior."""

    def test_simple_message(self):
        """Test exception with just a message."""
        error = USCAgentError("Something went wrong")
        
        self.assertEqual(error.message, "Something went wrong")
        self.assertEqual(error.details, {})
        self.assertEqual(str(error), "Something went wrong")

    def test_message_with_details(self):
        """Test exception with message and details."""
        error = USCAgentError(
            "Operation failed",
            details={"code": 500, "reason": "timeout"},
        )
        
        self.assertEqual(error.message, "Operation failed")
        self.assertEqual(error.details["code"], 500)
        self.assertEqual(error.details["reason"], "timeout")
        
        error_str = str(error)
        self.assertIn("Operation failed", error_str)
        self.assertIn("code=500", error_str)
        self.assertIn("reason=timeout", error_str)

    def test_is_exception(self):
        """Test that USCAgentError is a proper Exception."""
        error = USCAgentError("test")
        
        self.assertIsInstance(error, Exception)
        
        with self.assertRaises(USCAgentError):
            raise error


# =============================================================================
# Test: Configuration Errors
# =============================================================================


class TestConfigurationError(unittest.TestCase):
    """Test ConfigurationError."""

    def test_inheritance(self):
        """Test ConfigurationError inherits from USCAgentError."""
        error = ConfigurationError("Invalid config")
        
        self.assertIsInstance(error, USCAgentError)
        self.assertIsInstance(error, Exception)

    def test_message(self):
        """Test ConfigurationError message."""
        error = ConfigurationError("k_paths must be positive")
        
        self.assertEqual(str(error), "k_paths must be positive")


# =============================================================================
# Test: LLM-Related Errors
# =============================================================================


class TestLLMError(unittest.TestCase):
    """Test LLMError base class."""

    def test_inheritance(self):
        """Test LLMError inherits from USCAgentError."""
        error = LLMError("LLM failed")
        
        self.assertIsInstance(error, USCAgentError)


class TestStructuredOutputError(unittest.TestCase):
    """Test StructuredOutputError."""

    def test_basic_creation(self):
        """Test creating StructuredOutputError."""
        error = StructuredOutputError(
            "Structured output failed",
            phase="reasoner",
        )
        
        self.assertEqual(error.phase, "reasoner")
        self.assertIsNone(error.original_error)
        self.assertIn("phase=reasoner", str(error))

    def test_with_original_error(self):
        """Test with original_error."""
        original = ValueError("Parse error")
        error = StructuredOutputError(
            "Structured output failed",
            phase="judge",
            original_error=original,
        )
        
        self.assertEqual(error.original_error, original)

    def test_with_extra_details(self):
        """Test with extra details."""
        error = StructuredOutputError(
            "Failed",
            phase="reasoner",
            model="gemini",
            attempt=3,
        )
        
        error_str = str(error)
        self.assertIn("model=gemini", error_str)
        self.assertIn("attempt=3", error_str)


class TestJSONParseError(unittest.TestCase):
    """Test JSONParseError."""

    def test_basic_creation(self):
        """Test creating JSONParseError."""
        error = JSONParseError(
            "Failed to parse JSON",
            raw_output='{"incomplete": ',
        )
        
        self.assertEqual(error.raw_output, '{"incomplete": ')
        self.assertIsNone(error.original_error)

    def test_truncates_long_output(self):
        """Test that long raw_output is truncated in str."""
        long_output = "x" * 500
        error = JSONParseError("Parse failed", raw_output=long_output)
        
        error_str = str(error)
        # Should truncate to ~200 chars + "..."
        self.assertIn("...", error_str)

    def test_with_original_error(self):
        """Test with original JSON decode error."""
        import json
        try:
            json.loads("{invalid}")
        except json.JSONDecodeError as e:
            error = JSONParseError(
                "JSON parse failed",
                raw_output="{invalid}",
                original_error=e,
            )
            self.assertIsNotNone(error.original_error)


class TestLLMTimeoutError(unittest.TestCase):
    """Test LLMTimeoutError."""

    def test_basic_creation(self):
        """Test creating LLMTimeoutError."""
        error = LLMTimeoutError(
            "LLM call timed out",
            timeout_seconds=30.0,
            phase="reasoner",
        )
        
        self.assertEqual(error.timeout_seconds, 30.0)
        self.assertEqual(error.phase, "reasoner")
        
        error_str = str(error)
        self.assertIn("timeout_seconds=30.0", error_str)
        self.assertIn("phase=reasoner", error_str)


# =============================================================================
# Test: Validation Errors
# =============================================================================


class TestValidationError(unittest.TestCase):
    """Test ValidationError base class."""

    def test_basic_creation(self):
        """Test creating ValidationError."""
        error = ValidationError(
            "Validation failed",
            errors=["Missing field: name", "Invalid type for age"],
        )
        
        self.assertEqual(len(error.errors), 2)
        self.assertIn("Missing field: name", error.errors)

    def test_empty_errors_list(self):
        """Test with empty errors list."""
        error = ValidationError("Validation failed")
        
        self.assertEqual(error.errors, [])


class TestDecisionValidationError(unittest.TestCase):
    """Test DecisionValidationError."""

    def test_basic_creation(self):
        """Test creating DecisionValidationError."""
        raw_decision = {"decision_type": "INVALID"}
        error = DecisionValidationError(
            "Invalid decision",
            decision_type="reasoner",
            raw_decision=raw_decision,
            errors=["Unknown decision_type: INVALID"],
        )
        
        self.assertEqual(error.decision_type, "reasoner")
        self.assertEqual(error.raw_decision, raw_decision)
        self.assertEqual(len(error.errors), 1)

    def test_str_includes_decision_type(self):
        """Test string representation includes decision_type."""
        error = DecisionValidationError(
            "Invalid",
            decision_type="judge",
            raw_decision={},
            errors=[],
        )
        
        self.assertIn("decision_type=judge", str(error))


class TestToolArgsValidationError(unittest.TestCase):
    """Test ToolArgsValidationError."""

    def test_basic_creation(self):
        """Test creating ToolArgsValidationError."""
        error = ToolArgsValidationError(
            "Invalid tool arguments",
            tool_name="calculator",
            tool_args={"expression": 123},
            errors=["expression must be a string"],
        )
        
        self.assertEqual(error.tool_name, "calculator")
        self.assertEqual(error.tool_args, {"expression": 123})
        self.assertEqual(len(error.errors), 1)


# =============================================================================
# Test: Tool-Related Errors
# =============================================================================


class TestToolError(unittest.TestCase):
    """Test ToolError base class."""

    def test_inheritance(self):
        """Test ToolError inherits from USCAgentError."""
        error = ToolError("Tool failed")
        
        self.assertIsInstance(error, USCAgentError)


class TestUnknownToolError(unittest.TestCase):
    """Test UnknownToolError."""

    def test_basic_creation(self):
        """Test creating UnknownToolError."""
        error = UnknownToolError("fake_tool")
        
        self.assertEqual(error.tool_name, "fake_tool")
        self.assertIsNone(error.available_tools)
        self.assertIn("fake_tool", str(error))

    def test_with_available_tools(self):
        """Test with available_tools list."""
        error = UnknownToolError(
            "fake_tool",
            available_tools=["calculator", "search", "api"],
        )
        
        self.assertEqual(error.available_tools, ["calculator", "search", "api"])
        
        error_str = str(error)
        self.assertIn("fake_tool", error_str)
        self.assertIn("calculator", error_str)
        self.assertIn("search", error_str)


class TestToolExecutionError(unittest.TestCase):
    """Test ToolExecutionError."""

    def test_basic_creation(self):
        """Test creating ToolExecutionError."""
        error = ToolExecutionError(
            "Tool execution failed",
            tool_name="api_client",
            tool_args={"endpoint": "/users"},
        )
        
        self.assertEqual(error.tool_name, "api_client")
        self.assertEqual(error.tool_args, {"endpoint": "/users"})
        self.assertIsNone(error.original_error)

    def test_with_original_error(self):
        """Test with original exception."""
        original = ConnectionError("Network unreachable")
        error = ToolExecutionError(
            "API call failed",
            tool_name="api_client",
            tool_args={},
            original_error=original,
        )
        
        self.assertEqual(error.original_error, original)


class TestToolReflectionError(unittest.TestCase):
    """Test ToolReflectionError."""

    def test_basic_creation(self):
        """Test creating ToolReflectionError."""
        error = ToolReflectionError(
            "Reflection failed",
            tool_name="calculator",
        )
        
        self.assertEqual(error.tool_name, "calculator")
        self.assertIsNone(error.suggestion)

    def test_with_suggestion(self):
        """Test with suggestion."""
        error = ToolReflectionError(
            "Cannot recover",
            tool_name="api_client",
            suggestion="Check API credentials",
        )
        
        self.assertEqual(error.suggestion, "Check API credentials")
        self.assertIn("suggestion=Check API credentials", str(error))


# =============================================================================
# Test: Agent Loop Errors
# =============================================================================


class TestAgentLoopError(unittest.TestCase):
    """Test AgentLoopError base class."""

    def test_inheritance(self):
        """Test AgentLoopError inherits from USCAgentError."""
        error = AgentLoopError("Loop failed")
        
        self.assertIsInstance(error, USCAgentError)


class TestMaxStepsExceededError(unittest.TestCase):
    """Test MaxStepsExceededError."""

    def test_basic_creation(self):
        """Test creating MaxStepsExceededError."""
        error = MaxStepsExceededError(max_steps=10, current_step=11)
        
        self.assertEqual(error.max_steps, 10)
        self.assertEqual(error.current_step, 11)
        
        error_str = str(error)
        self.assertIn("10", error_str)
        self.assertIn("11", error_str)


class TestNoValidCandidatesError(unittest.TestCase):
    """Test NoValidCandidatesError."""

    def test_basic_creation(self):
        """Test creating NoValidCandidatesError."""
        error = NoValidCandidatesError(step=3, k_paths=5)
        
        self.assertEqual(error.step, 3)
        self.assertEqual(error.k_paths, 5)
        self.assertEqual(error.invalid_reasons, [])

    def test_with_invalid_reasons(self):
        """Test with invalid_reasons list."""
        error = NoValidCandidatesError(
            step=2,
            k_paths=3,
            invalid_reasons=["Timeout", "Parse error", "Validation failed"],
        )
        
        self.assertEqual(len(error.invalid_reasons), 3)
        self.assertIn("Timeout", error.invalid_reasons)


# =============================================================================
# Test: Result Type
# =============================================================================


class TestResult(unittest.TestCase):
    """Test Result type for graceful error handling."""

    def test_ok_factory(self):
        """Test Result.ok() factory method."""
        result = Result.ok("success value")
        
        self.assertEqual(result.value, "success value")
        self.assertIsNone(result.error)
        self.assertTrue(result.is_ok)
        self.assertFalse(result.is_error)

    def test_fail_factory(self):
        """Test Result.fail() factory method."""
        result = Result.fail("Something went wrong")
        
        self.assertIsNone(result.value)
        self.assertEqual(result.error, "Something went wrong")
        self.assertFalse(result.is_ok)
        self.assertTrue(result.is_error)

    def test_fail_with_details(self):
        """Test Result.fail() with error details."""
        result = Result.fail("Failed", code=500, reason="timeout")
        
        self.assertEqual(result.error, "Failed")
        self.assertEqual(result.error_details["code"], 500)
        self.assertEqual(result.error_details["reason"], "timeout")

    def test_unwrap_success(self):
        """Test unwrap() on successful result."""
        result = Result.ok(42)
        
        value = result.unwrap()
        self.assertEqual(value, 42)

    def test_unwrap_error_raises(self):
        """Test unwrap() on error result raises exception."""
        result = Result.fail("Error occurred")
        
        with self.assertRaises(USCAgentError) as ctx:
            result.unwrap()
        
        self.assertIn("Error occurred", str(ctx.exception))

    def test_unwrap_or_success(self):
        """Test unwrap_or() on successful result."""
        result = Result.ok("real value")
        
        value = result.unwrap_or("default")
        self.assertEqual(value, "real value")

    def test_unwrap_or_error(self):
        """Test unwrap_or() on error result returns default."""
        result = Result.fail("Error")
        
        value = result.unwrap_or("default value")
        self.assertEqual(value, "default value")

    def test_unwrap_or_with_none_default(self):
        """Test unwrap_or() with None as default."""
        result = Result.fail("Error")
        
        value = result.unwrap_or(None)
        self.assertIsNone(value)

    def test_ok_with_none_value(self):
        """Test Result.ok() with None as value."""
        result = Result.ok(None)
        
        self.assertIsNone(result.value)
        self.assertTrue(result.is_ok)
        self.assertFalse(result.is_error)

    def test_ok_with_falsy_value(self):
        """Test Result.ok() with falsy values."""
        # Empty string
        result_str = Result.ok("")
        self.assertTrue(result_str.is_ok)
        self.assertEqual(result_str.unwrap(), "")
        
        # Zero
        result_zero = Result.ok(0)
        self.assertTrue(result_zero.is_ok)
        self.assertEqual(result_zero.unwrap(), 0)
        
        # Empty list
        result_list = Result.ok([])
        self.assertTrue(result_list.is_ok)
        self.assertEqual(result_list.unwrap(), [])

    def test_frozen_dataclass(self):
        """Test that Result is immutable (frozen)."""
        result = Result.ok("value")
        
        with self.assertRaises(AttributeError):
            result.value = "new value"  # type: ignore

    def test_error_details_none_when_not_provided(self):
        """Test that error_details is None when not provided."""
        result = Result.fail("Error without details")
        
        self.assertIsNone(result.error_details)


# =============================================================================
# Test: Exception Hierarchy
# =============================================================================


class TestExceptionHierarchy(unittest.TestCase):
    """Test the exception class hierarchy."""

    def test_all_errors_inherit_from_base(self):
        """Test all custom exceptions inherit from USCAgentError."""
        exception_classes = [
            ConfigurationError,
            LLMError,
            StructuredOutputError,
            JSONParseError,
            LLMTimeoutError,
            ValidationError,
            DecisionValidationError,
            ToolArgsValidationError,
            ToolError,
            UnknownToolError,
            ToolExecutionError,
            ToolReflectionError,
            AgentLoopError,
            MaxStepsExceededError,
            NoValidCandidatesError,
        ]
        
        for cls in exception_classes:
            self.assertTrue(
                issubclass(cls, USCAgentError),
                f"{cls.__name__} should inherit from USCAgentError",
            )

    def test_llm_errors_inherit_from_llm_error(self):
        """Test LLM-related exceptions inherit from LLMError."""
        llm_errors = [StructuredOutputError, JSONParseError, LLMTimeoutError]
        
        for cls in llm_errors:
            self.assertTrue(
                issubclass(cls, LLMError),
                f"{cls.__name__} should inherit from LLMError",
            )

    def test_validation_errors_inherit_from_validation_error(self):
        """Test validation exceptions inherit from ValidationError."""
        validation_errors = [DecisionValidationError, ToolArgsValidationError]
        
        for cls in validation_errors:
            self.assertTrue(
                issubclass(cls, ValidationError),
                f"{cls.__name__} should inherit from ValidationError",
            )

    def test_tool_errors_inherit_from_tool_error(self):
        """Test tool-related exceptions inherit from ToolError."""
        tool_errors = [UnknownToolError, ToolExecutionError, ToolReflectionError]
        
        for cls in tool_errors:
            self.assertTrue(
                issubclass(cls, ToolError),
                f"{cls.__name__} should inherit from ToolError",
            )

    def test_loop_errors_inherit_from_agent_loop_error(self):
        """Test loop exceptions inherit from AgentLoopError."""
        loop_errors = [MaxStepsExceededError, NoValidCandidatesError]
        
        for cls in loop_errors:
            self.assertTrue(
                issubclass(cls, AgentLoopError),
                f"{cls.__name__} should inherit from AgentLoopError",
            )


if __name__ == "__main__":
    unittest.main()
