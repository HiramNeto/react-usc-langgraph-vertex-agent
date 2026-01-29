"""
Plugin system for the ReAct USC Agent.

This module provides plugins that extend agent behavior, particularly
for error recovery and retry logic.

Available Plugins:
    - ReflectAndRetryToolPlugin: Retry failed tool calls with LLM reflection

Design Principles:
    - Plugins are composable and optional
    - Each plugin has a single, focused responsibility
    - Plugins use proper logging (no print statements)
    - Error handling is explicit and recoverable
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Sequence

from ._internal.llm_io import invoke_with_unified_fallback
from ._internal.normalizers import normalize_reflection_decision_obj
from ._internal.prompts import build_reflection_prompt
from ._internal.salvage import salvage_reflection_final
from ._internal.schema import get_reflection_decision_schema
from ._internal.utils import safe_json_dumps
from ._internal.validation import validate_json_obj, validate_reflection_decision_dict
from .config import RetryConfig
from .logging import AgentLogger, LogContext, LoggingConfig
from .types import ToolSpec


@dataclass(frozen=True)
class ReflectionResult:
    """
    Result of a reflection decision.
    
    Attributes:
        verdict: One of "RETRY", "WAIT", or "ABORT"
        retry_args: New arguments if verdict is RETRY
        abort_suggestion: Suggestion message if verdict is ABORT
        analysis: The reflection model's analysis
    """
    verdict: str
    retry_args: Optional[Dict[str, Any]] = None
    abort_suggestion: Optional[str] = None
    analysis: Optional[str] = None
    
    @property
    def should_retry(self) -> bool:
        """Check if we should retry with new args."""
        return self.verdict == "RETRY" and self.retry_args is not None
    
    @property
    def should_wait(self) -> bool:
        """Check if we should wait and retry with same args."""
        return self.verdict == "WAIT"
    
    @property
    def should_abort(self) -> bool:
        """Check if we should abort the retry loop."""
        return self.verdict == "ABORT"


class ReflectAndRetryToolPlugin:
    """
    Plugin that implements retry logic with LLM-based reflection.
    
    When a tool execution fails, this plugin:
    1. Invokes a reflection model to analyze the error
    2. Based on the analysis, decides to:
       - RETRY: Try again with corrected arguments
       - WAIT: Wait and retry with same arguments (for transient errors)
       - ABORT: Stop retrying and return error to agent
    
    This enables the agent to recover from:
    - Invalid argument formats
    - Transient network/API errors
    - Temporary service unavailability
    
    Example:
        ```python
        plugin = ReflectAndRetryToolPlugin(
            model=reflection_model,
            max_retries=3,
            trace=True,
        )
        
        agent = LangGraphReActUSCAgent(
            models=models,
            tools=tools,
            config=config,
            plugins=[plugin],
        )
        ```
    
    Thread Safety:
        This plugin is stateless and thread-safe.
    """
    
    def __init__(
        self,
        model: Any,
        max_retries: int = 3,
        backoff_seconds: float = 1.0,
        trace: bool = False,
        llm_retry_config: Optional[RetryConfig] = None,
        logger: Optional[AgentLogger] = None,
        use_structured_output: bool = False,
        max_reprompts: int = 1,
    ) -> None:
        """
        Initialize the retry plugin.
        
        Args:
            model: LangChain chat model for reflection
            max_retries: Maximum number of retry attempts
            backoff_seconds: Base delay between retries (exponential backoff)
            trace: Enable trace logging
            llm_retry_config: Retry config for LLM calls
            logger: Optional custom logger
            use_structured_output: Whether to try structured output for reflection
                (default False - text mode often works better for nested retry_args)
            max_reprompts: Number of repair attempts for invalid reflection output
        """
        if max_retries < 0:
            raise ValueError(f"max_retries cannot be negative, got {max_retries}")
        if backoff_seconds < 0:
            raise ValueError(f"backoff_seconds cannot be negative, got {backoff_seconds}")
        if max_reprompts < 0:
            raise ValueError(f"max_reprompts cannot be negative, got {max_reprompts}")
        
        self._model = model
        self._max_retries = max_retries
        self._backoff_seconds = backoff_seconds
        self._llm_retry_config = llm_retry_config
        self._use_structured_output = use_structured_output
        self._max_reprompts = max_reprompts
        
        # Create logger with trace config
        logging_config = LoggingConfig(enable_trace=trace)
        self._logger = logger or AgentLogger(__name__, logging_config)
    
    def run(
        self,
        *,
        tool_name: str,
        tool_args: Dict[str, Any],
        tool_func: Callable[[Dict[str, Any]], Any],
        all_tools: Sequence[ToolSpec],
        user_query: str,
        tool_input_schema: Dict[str, Any],
    ) -> Any:
        """
        Execute a tool with retry loop and reflection logic.
        
        Args:
            tool_name: Name of the tool to execute
            tool_args: Initial arguments for the tool
            tool_func: The tool's execution function
            all_tools: All available tools (for reflection context)
            user_query: Original user query (for reflection context)
            tool_input_schema: JSON schema for tool arguments
        
        Returns:
            Tool execution result, or a reflection error message string
        
        Raises:
            Exception: Re-raises the last exception if all retries fail
        """
        current_args = tool_args
        last_error: Optional[Exception] = None
        
        for attempt in range(self._max_retries + 1):
            with LogContext(phase="tool_retry", tool_name=tool_name, attempt=attempt):
                try:
                    if attempt > 0:
                        self._logger.trace(
                            f"Retry {attempt}: Executing {tool_name}",
                            args=safe_json_dumps(current_args),
                        )
                    
                    # Try execution
                    return tool_func(current_args)
                    
                except Exception as e:
                    last_error = e
                    
                    # If exhausted retries, raise the last exception
                    if attempt == self._max_retries:
                        self._logger.warning(
                            f"Exhausted {self._max_retries} retries for {tool_name}",
                            final_error=str(e),
                        )
                        raise
                    
                    # Reflection step
                    self._logger.trace(f"Error caught: {e}. Reflecting...")
                    
                    reflection = self._reflect(
                        user_query=user_query,
                        tool_name=tool_name,
                        tool_args=current_args,
                        error=str(e),
                        tools=all_tools,
                        tool_input_schema=tool_input_schema,
                    )
                    
                    # Handle reflection decision
                    if reflection.should_abort:
                        suggestion = reflection.abort_suggestion or "Tool execution aborted by reflection."
                        self._logger.info(
                            f"Reflection ABORT for {tool_name}",
                            suggestion=suggestion,
                        )
                        return self._create_abort_message(tool_name, suggestion)
                    
                    if reflection.should_wait:
                        wait_time = self._calculate_backoff(attempt)
                        self._logger.trace(
                            f"Reflection WAIT: Sleeping {wait_time:.2f}s before retry"
                        )
                        time.sleep(wait_time)
                        continue
                    
                    if reflection.should_retry:
                        # Validate new args before using them
                        new_args = reflection.retry_args or {}
                        
                        self._logger.trace(
                            f"Reflection returned retry_args: {safe_json_dumps(new_args)}"
                        )
                        
                        # Merge with original args if reflection only provided partial args
                        # This handles the case where LLM only returns the "fixed" fields
                        merged_args = {**current_args, **new_args}
                        
                        arg_errors = validate_json_obj(merged_args, tool_input_schema)
                        
                        if arg_errors:
                            self._logger.warning(
                                f"Reflection produced invalid args even after merge: {arg_errors}",
                                tool_name=tool_name,
                            )
                            # Fall through to raise the original error
                            raise
                        
                        current_args = merged_args
                        self._logger.trace(
                            f"Reflection RETRY with merged args: {safe_json_dumps(current_args)}"
                        )
                        continue
                    
                    # Unknown verdict - raise the error
                    self._logger.warning(
                        f"Unknown reflection verdict: {reflection.verdict}",
                        tool_name=tool_name,
                    )
                    raise
        
        # Should not reach here, but safety net
        if last_error:
            raise last_error
        raise RuntimeError("Unexpected state in retry loop")
    
    def _reflect(
        self,
        *,
        user_query: str,
        tool_name: str,
        tool_args: Dict[str, Any],
        error: str,
        tools: Sequence[ToolSpec],
        tool_input_schema: Dict[str, Any],
    ) -> ReflectionResult:
        """
        Invoke the reflection model to analyze a tool failure.
        
        Uses the unified fallback function with optional structured output,
        text parsing, salvage, and reprompt support.
        
        Args:
            user_query: Original user query
            tool_name: Name of the failed tool
            tool_args: Arguments that were used
            error: Error message from the failure
            tools: All available tools
            tool_input_schema: JSON schema for the tool's arguments
        
        Returns:
            ReflectionResult with verdict and optional retry args
        """
        system, user = build_reflection_prompt(
            user_query=user_query,
            tool_name=tool_name,
            tool_args=tool_args,
            error=error,
            tools=tools,
        )
        
        # Build reflection schema with the specific tool's input schema
        reflection_schema = get_reflection_decision_schema(tool_input_schema)

        try:
            # Use unified fallback function
            # Note: use_structured_output defaults to False because structured output
            # often fails to populate nested retry_args correctly
            normalized = invoke_with_unified_fallback(
                self._model,
                system=system,
                user=user,
                schema=reflection_schema,
                normalizer=normalize_reflection_decision_obj,
                validator=validate_reflection_decision_dict,
                salvage_fn=salvage_reflection_final,
                retry_config=self._llm_retry_config,
                use_structured_output=self._use_structured_output,
                accept_non_json_final=True,  # Always try salvage for reflection
                max_reprompts=self._max_reprompts,
                phase="Reflection",
                logger=self._logger,
            )
            
            self._logger.trace(f"Reflection output: {safe_json_dumps(normalized)}")
            
            return ReflectionResult(
                verdict=normalized.get("verdict", "ABORT"),
                retry_args=normalized.get("retry_args"),
                abort_suggestion=normalized.get("abort_suggestion"),
                analysis=normalized.get("analysis"),
            )
            
        except Exception as e:
            self._logger.error(
                f"Reflection model call failed: {e}",
                tool_name=tool_name,
            )
            # If reflection fails, return ABORT to prevent infinite loops
            return ReflectionResult(
                verdict="ABORT",
                abort_suggestion=f"Reflection mechanism failed: {e}",
            )
    
    def _calculate_backoff(self, attempt: int) -> float:
        """Calculate exponential backoff time for a retry attempt."""
        return self._backoff_seconds * (2 ** attempt)
    
    def _create_abort_message(self, tool_name: str, suggestion: str) -> str:
        """Create the abort message that signals reflection abort to the agent."""
        return f"Reflection Error: The tool '{tool_name}' failed and reflection decided to abort. Suggestion: {suggestion}"
    
    # =========================================================================
    # Properties
    # =========================================================================
    
    @property
    def max_retries(self) -> int:
        """Get maximum retry count."""
        return self._max_retries
    
    @property
    def backoff_seconds(self) -> float:
        """Get base backoff time in seconds."""
        return self._backoff_seconds
