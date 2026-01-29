"""
Centralized logging configuration for the ReAct USC Agent.

This module provides:
- Structured logging with contextual information
- Configurable log levels and formatters
- Agent-specific logger factory with trace ID support
- Clean separation between debug/trace output and production logging

Usage:
    from react_usc.logging import get_logger, LogContext

    logger = get_logger(__name__)
    
    with LogContext(trace_id="abc-123", phase="reasoner"):
        logger.info("Processing request", extra={"path_id": 0})
"""
from __future__ import annotations

import logging
import sys
import threading
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, Optional


# Thread-local storage for context
_context = threading.local()


@dataclass(frozen=True)
class LoggingConfig:
    """Configuration for agent logging behavior."""
    
    # Log level for the agent logger (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    level: int = logging.INFO
    
    # Enable structured output logging (attempts and results)
    log_structured_output: bool = False
    
    # Enable trace output (step-by-step agent execution details)
    enable_trace: bool = False
    
    # Format for log messages
    format: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    
    # Date format for log messages
    date_format: str = "%Y-%m-%d %H:%M:%S"
    
    # Include extra context fields in log output
    include_context: bool = True


# Default configuration (can be overridden)
_default_config = LoggingConfig()


def configure_logging(config: Optional[LoggingConfig] = None) -> None:
    """
    Configure the logging system for the agent.
    
    Should be called once at application startup.
    
    Args:
        config: Optional LoggingConfig. If None, uses defaults.
    """
    global _default_config
    if config is not None:
        _default_config = config
    
    # Configure the root logger for this package
    package_logger = logging.getLogger("react_usc")
    package_logger.setLevel(_default_config.level)
    
    # Remove existing handlers to avoid duplicates on reconfiguration
    package_logger.handlers.clear()
    
    # Create and configure handler
    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(_default_config.level)
    
    # Use custom formatter that includes context
    formatter = ContextAwareFormatter(
        fmt=_default_config.format,
        datefmt=_default_config.date_format,
        include_context=_default_config.include_context,
    )
    handler.setFormatter(formatter)
    
    package_logger.addHandler(handler)
    
    # Prevent propagation to root logger to avoid duplicate logs
    package_logger.propagate = False


class ContextAwareFormatter(logging.Formatter):
    """
    Custom formatter that appends context from LogContext and extra fields.
    """
    
    def __init__(
        self,
        fmt: Optional[str] = None,
        datefmt: Optional[str] = None,
        include_context: bool = True,
    ) -> None:
        super().__init__(fmt=fmt, datefmt=datefmt)
        self._include_context = include_context
    
    def format(self, record: logging.LogRecord) -> str:
        # Get base formatted message
        message = super().format(record)
        
        if not self._include_context:
            return message
        
        # Collect context fields
        context_parts: list[str] = []
        
        # Add thread-local context
        ctx = getattr(_context, "current", None)
        if ctx:
            if ctx.trace_id:
                context_parts.append(f"trace_id={ctx.trace_id}")
            if ctx.phase:
                context_parts.append(f"phase={ctx.phase}")
            if ctx.step is not None:
                context_parts.append(f"step={ctx.step}")
            for k, v in ctx.extra.items():
                context_parts.append(f"{k}={v}")
        
        # Add extra fields from the log record (excluding standard fields)
        standard_fields = {
            "name", "msg", "args", "created", "filename", "funcName",
            "levelname", "levelno", "lineno", "module", "msecs",
            "pathname", "process", "processName", "relativeCreated",
            "stack_info", "exc_info", "exc_text", "thread", "threadName",
            "message", "asctime",
        }
        for key, value in record.__dict__.items():
            if key not in standard_fields and not key.startswith("_"):
                context_parts.append(f"{key}={value}")
        
        if context_parts:
            message = f"{message} | {' '.join(context_parts)}"
        
        return message


@dataclass
class _LogContextData:
    """Internal storage for log context."""
    trace_id: Optional[str] = None
    phase: Optional[str] = None
    step: Optional[int] = None
    extra: Dict[str, Any] = field(default_factory=dict)


class LogContext:
    """
    Context manager for adding structured context to log messages.
    
    Context is thread-local and can be nested. Inner contexts inherit
    from outer contexts.
    
    Example:
        with LogContext(trace_id="req-123", phase="reasoner"):
            logger.info("Starting")  # Includes trace_id and phase
            with LogContext(step=1):
                logger.info("Step 1")  # Includes trace_id, phase, and step
    """
    
    def __init__(
        self,
        trace_id: Optional[str] = None,
        phase: Optional[str] = None,
        step: Optional[int] = None,
        **extra: Any,
    ) -> None:
        self._trace_id = trace_id
        self._phase = phase
        self._step = step
        self._extra = extra
        self._previous: Optional[_LogContextData] = None
    
    def __enter__(self) -> "LogContext":
        # Store previous context
        self._previous = getattr(_context, "current", None)
        
        # Build new context, inheriting from previous
        if self._previous:
            new_extra = {**self._previous.extra, **self._extra}
            _context.current = _LogContextData(
                trace_id=self._trace_id or self._previous.trace_id,
                phase=self._phase or self._previous.phase,
                step=self._step if self._step is not None else self._previous.step,
                extra=new_extra,
            )
        else:
            _context.current = _LogContextData(
                trace_id=self._trace_id,
                phase=self._phase,
                step=self._step,
                extra=self._extra,
            )
        
        return self
    
    def __exit__(self, _exc_type: Any, _exc_val: Any, _exc_tb: Any) -> None:
        _context.current = self._previous


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for the given module name.
    
    Loggers are namespaced under 'react_usc' to share configuration.
    
    Args:
        name: Module name (typically __name__)
    
    Returns:
        Configured logger instance
    """
    # Ensure the logger is under our package namespace
    if not name.startswith("react_usc"):
        name = f"react_usc.{name}"
    return logging.getLogger(name)


def generate_trace_id() -> str:
    """Generate a unique trace ID for request correlation."""
    return str(uuid.uuid4())[:8]


@contextmanager
def trace_context(
    phase: Optional[str] = None,
    step: Optional[int] = None,
    **extra: Any,
) -> Iterator[str]:
    """
    Convenience context manager that generates a trace ID and sets context.
    
    Yields the generated trace ID.
    
    Example:
        with trace_context(phase="agent_run") as trace_id:
            logger.info("Starting agent", extra={"query": user_query})
    """
    trace_id = generate_trace_id()
    with LogContext(trace_id=trace_id, phase=phase, step=step, **extra):
        yield trace_id


class AgentLogger:
    """
    High-level logger for agent operations with built-in context management.
    
    Provides semantic logging methods for common agent operations,
    ensuring consistent log messages and context.
    """
    
    def __init__(self, name: str, config: Optional[LoggingConfig] = None) -> None:
        self._logger = get_logger(name)
        self._config = config or _default_config
    
    @property
    def is_trace_enabled(self) -> bool:
        """Check if trace output is enabled."""
        return self._config.enable_trace
    
    @property
    def is_structured_output_logging_enabled(self) -> bool:
        """Check if structured output logging is enabled."""
        return self._config.log_structured_output
    
    def debug(self, msg: str, **extra: Any) -> None:
        """Log debug message."""
        self._logger.debug(msg, extra=extra)
    
    def info(self, msg: str, **extra: Any) -> None:
        """Log info message."""
        self._logger.info(msg, extra=extra)
    
    def warning(self, msg: str, **extra: Any) -> None:
        """Log warning message."""
        self._logger.warning(msg, extra=extra)
    
    def error(self, msg: str, **extra: Any) -> None:
        """Log error message."""
        self._logger.error(msg, extra=extra)
    
    def trace(self, msg: str, **extra: Any) -> None:
        """
        Log trace message (only if trace is enabled).
        
        Trace messages are printed directly to stdout for visibility
        when trace mode is enabled.
        """
        if self._config.enable_trace:
            # Print directly for immediate visibility (trace is for debugging)
            print(f"  [TRACE] {msg}", file=sys.stderr, flush=True)
    
    def structured_output_attempt(self, phase: str, **extra: Any) -> None:
        """Log structured output attempt (only if logging enabled)."""
        if self._config.log_structured_output:
            self._logger.info(f"{phase} structured output attempt", extra=extra)
    
    def structured_output_success(self, phase: str, **extra: Any) -> None:
        """Log structured output success (only if logging enabled)."""
        if self._config.log_structured_output:
            self._logger.info(f"{phase} structured output succeeded", extra=extra)
    
    def structured_output_fallback(
        self, phase: str, error: Exception, **extra: Any
    ) -> None:
        """Log structured output fallback to text parsing."""
        self._logger.warning(
            f"{phase} structured output failed; falling back to text JSON parsing",
            extra={"error": f"{type(error).__name__}: {error}", **extra},
        )
    
    def parse_error(self, phase: str, error: Exception, preview: str, **extra: Any) -> None:
        """Log JSON parse error."""
        self._logger.error(
            f"{phase} output JSON parse failed",
            extra={
                "error": f"{type(error).__name__}: {error}",
                "output_preview": preview,
                **extra,
            },
        )
    
    def tool_call(self, tool_name: str, args: str, **extra: Any) -> None:
        """Log tool call."""
        self.trace(f"Tool call: {tool_name} args={args}", **extra)
    
    def tool_result(self, tool_name: str, result: str, **extra: Any) -> None:
        """Log tool result."""
        self.trace(f"Tool result: {tool_name} => {result}", **extra)
    
    def tool_exception(self, tool_name: str, error: str, **extra: Any) -> None:
        """Log tool exception."""
        self.trace(f"Tool exception: {tool_name} => {error}", **extra)


# Initialize logging with defaults on module import
configure_logging()
