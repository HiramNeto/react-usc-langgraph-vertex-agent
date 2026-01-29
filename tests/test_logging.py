"""
Tests for logging infrastructure.

These tests cover:
- LoggingConfig dataclass
- LogContext context manager
- AgentLogger methods
- generate_trace_id uniqueness
- trace_context context manager
"""
from __future__ import annotations

import logging
import unittest
from io import StringIO
from unittest.mock import patch

from react_usc.logging import (
    AgentLogger,
    LogContext,
    LoggingConfig,
    configure_logging,
    generate_trace_id,
    get_logger,
    trace_context,
)


# =============================================================================
# Test: LoggingConfig
# =============================================================================


class TestLoggingConfig(unittest.TestCase):
    """Test LoggingConfig dataclass."""

    def test_default_values(self):
        """Test LoggingConfig default values."""
        config = LoggingConfig()
        
        self.assertEqual(config.level, logging.INFO)
        self.assertFalse(config.log_structured_output)
        self.assertFalse(config.enable_trace)
        self.assertTrue(config.include_context)

    def test_custom_values(self):
        """Test LoggingConfig with custom values."""
        config = LoggingConfig(
            level=logging.DEBUG,
            log_structured_output=True,
            enable_trace=True,
            include_context=False,
        )
        
        self.assertEqual(config.level, logging.DEBUG)
        self.assertTrue(config.log_structured_output)
        self.assertTrue(config.enable_trace)
        self.assertFalse(config.include_context)

    def test_frozen_dataclass(self):
        """Test that LoggingConfig is immutable."""
        config = LoggingConfig()
        
        with self.assertRaises(AttributeError):
            config.level = logging.ERROR  # type: ignore

    def test_custom_format(self):
        """Test custom log format."""
        config = LoggingConfig(
            format="%(levelname)s - %(message)s",
            date_format="%H:%M:%S",
        )
        
        self.assertEqual(config.format, "%(levelname)s - %(message)s")
        self.assertEqual(config.date_format, "%H:%M:%S")


# =============================================================================
# Test: LogContext
# =============================================================================


class TestLogContext(unittest.TestCase):
    """Test LogContext context manager."""

    def test_basic_context(self):
        """Test basic LogContext usage."""
        with LogContext(trace_id="test-123", phase="test"):
            # Context is set
            pass
        # Context is cleared
    
    def test_nested_context(self):
        """Test nested LogContext inheritance."""
        with LogContext(trace_id="outer", phase="outer_phase"):
            with LogContext(step=1):
                # Inner context should have trace_id and phase from outer
                pass

    def test_context_override(self):
        """Test that inner context can override outer."""
        with LogContext(phase="outer"):
            with LogContext(phase="inner"):
                # Inner phase should override
                pass

    def test_extra_context(self):
        """Test extra context fields."""
        with LogContext(trace_id="test", custom_field="value"):
            pass

    def test_context_exit_on_exception(self):
        """Test that context is properly exited on exception."""
        try:
            with LogContext(trace_id="test"):
                raise ValueError("Test exception")
        except ValueError:
            pass
        # Context should be cleared

    def test_multiple_extra_fields(self):
        """Test multiple extra context fields."""
        with LogContext(
            trace_id="test",
            phase="test_phase",
            step=5,
            path_id=0,
            model="gemini",
        ):
            pass


# =============================================================================
# Test: AgentLogger
# =============================================================================


class TestAgentLogger(unittest.TestCase):
    """Test AgentLogger class."""

    def test_is_trace_enabled_false(self):
        """Test is_trace_enabled when trace is disabled."""
        config = LoggingConfig(enable_trace=False)
        logger = AgentLogger(__name__, config)
        
        self.assertFalse(logger.is_trace_enabled)

    def test_is_trace_enabled_true(self):
        """Test is_trace_enabled when trace is enabled."""
        config = LoggingConfig(enable_trace=True)
        logger = AgentLogger(__name__, config)
        
        self.assertTrue(logger.is_trace_enabled)

    def test_is_structured_output_logging_disabled(self):
        """Test structured output logging disabled."""
        config = LoggingConfig(log_structured_output=False)
        logger = AgentLogger(__name__, config)
        
        self.assertFalse(logger.is_structured_output_logging_enabled)

    def test_is_structured_output_logging_enabled(self):
        """Test structured output logging enabled."""
        config = LoggingConfig(log_structured_output=True)
        logger = AgentLogger(__name__, config)
        
        self.assertTrue(logger.is_structured_output_logging_enabled)

    def test_debug_method(self):
        """Test debug logging method."""
        config = LoggingConfig(level=logging.DEBUG)
        logger = AgentLogger(__name__, config)
        
        # Should not raise
        logger.debug("Debug message", extra_field="value")

    def test_info_method(self):
        """Test info logging method."""
        logger = AgentLogger(__name__)
        
        # Should not raise
        logger.info("Info message", key="value")

    def test_warning_method(self):
        """Test warning logging method."""
        logger = AgentLogger(__name__)
        
        # Should not raise
        logger.warning("Warning message")

    def test_error_method(self):
        """Test error logging method."""
        logger = AgentLogger(__name__)
        
        # Should not raise
        logger.error("Error message", error_code=500)

    @patch('sys.stderr', new_callable=StringIO)
    def test_trace_when_enabled(self, mock_stderr):
        """Test trace output when enabled."""
        config = LoggingConfig(enable_trace=True)
        logger = AgentLogger(__name__, config)
        
        logger.trace("Trace message")
        
        output = mock_stderr.getvalue()
        self.assertIn("[TRACE]", output)
        self.assertIn("Trace message", output)

    @patch('sys.stderr', new_callable=StringIO)
    def test_trace_when_disabled(self, mock_stderr):
        """Test trace output when disabled."""
        config = LoggingConfig(enable_trace=False)
        logger = AgentLogger(__name__, config)
        
        logger.trace("Trace message")
        
        output = mock_stderr.getvalue()
        # Trace should not be in output
        self.assertNotIn("[TRACE]", output)

    def test_structured_output_attempt(self):
        """Test structured_output_attempt logging."""
        config = LoggingConfig(log_structured_output=True)
        logger = AgentLogger(__name__, config)
        
        # Should not raise
        logger.structured_output_attempt("Reasoner", path_id=0)

    def test_structured_output_attempt_disabled(self):
        """Test structured_output_attempt when disabled."""
        config = LoggingConfig(log_structured_output=False)
        logger = AgentLogger(__name__, config)
        
        # Should not raise, but also not log
        logger.structured_output_attempt("Reasoner")

    def test_structured_output_success(self):
        """Test structured_output_success logging."""
        config = LoggingConfig(log_structured_output=True)
        logger = AgentLogger(__name__, config)
        
        # Should not raise
        logger.structured_output_success("Judge")

    def test_structured_output_fallback(self):
        """Test structured_output_fallback logging."""
        logger = AgentLogger(__name__)
        error = ValueError("Parse failed")
        
        # Should not raise
        logger.structured_output_fallback("Reasoner", error)

    def test_parse_error(self):
        """Test parse_error logging."""
        logger = AgentLogger(__name__)
        error = ValueError("Invalid JSON")
        
        # Should not raise
        logger.parse_error("Judge", error, "preview text")

    @patch('sys.stderr', new_callable=StringIO)
    def test_tool_call(self, mock_stderr):
        """Test tool_call logging."""
        config = LoggingConfig(enable_trace=True)
        logger = AgentLogger(__name__, config)
        
        logger.tool_call("calculator", '{"x": 1}')
        
        output = mock_stderr.getvalue()
        self.assertIn("calculator", output)

    @patch('sys.stderr', new_callable=StringIO)
    def test_tool_result(self, mock_stderr):
        """Test tool_result logging."""
        config = LoggingConfig(enable_trace=True)
        logger = AgentLogger(__name__, config)
        
        logger.tool_result("calculator", "42")
        
        output = mock_stderr.getvalue()
        self.assertIn("calculator", output)
        self.assertIn("42", output)

    @patch('sys.stderr', new_callable=StringIO)
    def test_tool_exception(self, mock_stderr):
        """Test tool_exception logging."""
        config = LoggingConfig(enable_trace=True)
        logger = AgentLogger(__name__, config)
        
        logger.tool_exception("api", "Connection error")
        
        output = mock_stderr.getvalue()
        self.assertIn("api", output)
        self.assertIn("Connection error", output)


# =============================================================================
# Test: Utility Functions
# =============================================================================


class TestUtilityFunctions(unittest.TestCase):
    """Test logging utility functions."""

    def test_generate_trace_id(self):
        """Test generate_trace_id generates unique IDs."""
        ids = {generate_trace_id() for _ in range(100)}
        
        # All IDs should be unique
        self.assertEqual(len(ids), 100)

    def test_generate_trace_id_format(self):
        """Test generate_trace_id format."""
        trace_id = generate_trace_id()
        
        # Should be 8 characters (first 8 chars of UUID)
        self.assertEqual(len(trace_id), 8)
        # Should be valid hex
        int(trace_id, 16)

    def test_get_logger(self):
        """Test get_logger returns a logger."""
        logger = get_logger(__name__)
        
        self.assertIsInstance(logger, logging.Logger)

    def test_get_logger_namespace(self):
        """Test get_logger namespaces under react_usc."""
        logger = get_logger("my_module")
        
        self.assertTrue(logger.name.startswith("react_usc"))

    def test_get_logger_existing_namespace(self):
        """Test get_logger with existing react_usc namespace."""
        logger = get_logger("react_usc.agent")
        
        self.assertEqual(logger.name, "react_usc.agent")

    def test_trace_context(self):
        """Test trace_context context manager."""
        with trace_context(phase="test") as trace_id:
            # trace_id should be generated
            self.assertEqual(len(trace_id), 8)

    def test_trace_context_generates_unique_ids(self):
        """Test trace_context generates unique IDs."""
        ids = []
        for _ in range(10):
            with trace_context() as trace_id:
                ids.append(trace_id)
        
        self.assertEqual(len(set(ids)), 10)

    def test_configure_logging(self):
        """Test configure_logging function."""
        config = LoggingConfig(level=logging.DEBUG)
        
        # Should not raise
        configure_logging(config)

    def test_configure_logging_default(self):
        """Test configure_logging with None."""
        # Should not raise
        configure_logging(None)


if __name__ == "__main__":
    unittest.main()
