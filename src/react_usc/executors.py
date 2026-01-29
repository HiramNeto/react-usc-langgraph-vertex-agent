"""
Executor classes for the ReAct USC Agent.

This module provides focused, single-responsibility classes that handle
specific phases of the agent loop:

- ReasonerExecutor: Manages parallel reasoner invocation
- JudgeExecutor: Manages judge invocation and selection
- ToolExecutor: Manages tool execution with optional retry/reflection

These executors are composed by the main agent class, following the
composition over inheritance principle.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ._internal.llm_io import invoke_with_unified_fallback
from ._internal.normalizers import normalize_judge_decision_obj, normalize_reasoner_decision_obj
from ._internal.prompts import build_judge_prompt, build_reasoner_prompt
from ._internal.salvage import salvage_judge_final, salvage_reasoner_final
from ._internal.schema import get_judge_decision_schema, get_reasoner_decision_schema
from ._internal.utils import build_state_summary, safe_json_dumps, truncate
from ._internal.validation import validate_json_obj, validate_judge_decision_dict, validate_reasoner_decision_dict
from .config import AgentConfig
from .decisions import JudgeDecision, ReasonerDecision
from .logging import AgentLogger, LogContext
from .tools import ToolRegistry
from .types import AgentConstants, ToolSpec


@dataclass(frozen=True)
class ReasonerResult:
    """
    Result of a single reasoner invocation.
    
    Attributes:
        path_id: The reasoner path index
        decision: Validated decision (None if invalid)
        raw_output: Raw output dict from the model
        error: Error message if the call failed
    """
    path_id: int
    decision: Optional[ReasonerDecision]
    raw_output: Dict[str, Any]
    error: Optional[str] = None
    
    @property
    def is_valid(self) -> bool:
        """Check if this result has a valid decision."""
        return self.decision is not None


class ReasonerExecutor:
    """
    Executes parallel reasoner invocations with timeout handling.
    
    Responsibilities:
    - Build reasoner prompts
    - Invoke reasoners in parallel
    - Handle structured output with fallback to text parsing
    - Validate and normalize results
    - Handle timeouts gracefully
    
    This class is stateless and can be reused across multiple agent runs.
    """
    
    def __init__(
        self,
        model: Any,
        config: AgentConfig,
        tools: ToolRegistry,
        logger: Optional[AgentLogger] = None,
    ) -> None:
        """
        Initialize the reasoner executor.
        
        Args:
            model: LangChain chat model for reasoners
            config: Agent configuration
            tools: Tool registry for validation
            logger: Optional logger (creates default if not provided)
        """
        self._model = model
        self._config = config
        self._tools = tools
        self._logger = logger or AgentLogger(__name__)
    
    def execute(
        self,
        *,
        user_query: str,
        state_summary: str,
    ) -> Tuple[List[ReasonerDecision], List[str]]:
        """
        Execute K parallel reasoners and return validated candidates.
        
        Args:
            user_query: The original user query
            state_summary: Current state summary string
        
        Returns:
            Tuple of (valid_candidates, invalid_reasons)
        
        """
        tools = self._tools.all()
        tool_schemas = [t.input_schema for t in tools]
        reasoner_schema = get_reasoner_decision_schema(tool_schemas)
        
        results = self._execute_parallel(
            user_query=user_query,
            state_summary=state_summary,
            tools=tools,
            schema=reasoner_schema,
        )
        
        return self._validate_results(results)
    
    def _execute_parallel(
        self,
        *,
        user_query: str,
        state_summary: str,
        tools: List[ToolSpec],
        schema: Dict[str, Any],
    ) -> List[ReasonerResult]:
        """Execute K reasoners in parallel with timeout handling."""
        max_workers = min(AgentConstants.MAX_REASONER_THREADS, self._config.k_paths)
        results: List[ReasonerResult] = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    self._call_single_reasoner,
                    path_id=i,
                    user_query=user_query,
                    state_summary=state_summary,
                    tools=tools,
                    schema=schema,
                )
                for i in range(self._config.k_paths)
            ]
            
            done, not_done = wait(futures, timeout=self._config.timeout_seconds)
            
            # Collect completed results
            for future in done:
                try:
                    results.append(future.result(timeout=0))
                except Exception as e:
                    results.append(
                        ReasonerResult(
                            path_id=-1,
                            decision=None,
                            raw_output={},
                            error=f"Future exception: {type(e).__name__}: {e}",
                        )
                    )
            
            # Handle timed-out futures
            if not_done:
                self._logger.warning(
                    f"Reasoner timeout: {len(not_done)}/{len(futures)} candidates unfinished",
                    timeout_seconds=self._config.timeout_seconds,
                )
                for future in not_done:
                    future.cancel()
                    results.append(
                        ReasonerResult(
                            path_id=-1,
                            decision=None,
                            raw_output={},
                            error=f"Timed out after {self._config.timeout_seconds}s",
                        )
                    )
        
        return results
    
    def _call_single_reasoner(
        self,
        *,
        path_id: int,
        user_query: str,
        state_summary: str,
        tools: List[ToolSpec],
        schema: Dict[str, Any],
    ) -> ReasonerResult:
        """
        Invoke a single reasoner and return its result.
        
        Uses the unified fallback function with structured output, text parsing,
        salvage, and reprompt support.
        """
        with LogContext(phase="reasoner", path_id=path_id):
            system, user = build_reasoner_prompt(
                system_prompt=self._config.system_prompt,
                user_query=user_query,
                state_summary=state_summary,
                tools=tools,
                path_id=path_id,
            )
            
            try:
                # Use unified fallback function
                normalized = invoke_with_unified_fallback(
                    self._model,
                    system=system,
                    user=user,
                    schema=schema,
                    normalizer=normalize_reasoner_decision_obj,
                    validator=validate_reasoner_decision_dict,
                    salvage_fn=salvage_reasoner_final if self._config.accept_non_json_final else None,
                    retry_config=self._config.llm_retry,
                    use_structured_output=self._config.use_structured_output,
                    accept_non_json_final=self._config.accept_non_json_final,
                    max_reprompts=self._config.max_reprompts,
                    phase=f"Reasoner[{path_id}]",
                    logger=self._logger,
                )
                
                # The unified function already normalized and validated
                decision, errors = validate_reasoner_decision_dict(normalized)
                
                if not decision:
                    return ReasonerResult(
                        path_id=path_id,
                        decision=None,
                        raw_output=normalized,
                        error=f"Validation failed: {errors}",
                    )
                
                # Additional tool validation for TOOL_CALL
                if decision.is_tool_call:
                    tool_error = self._validate_tool_call(decision)
                    if tool_error:
                        return ReasonerResult(
                            path_id=path_id,
                            decision=None,
                            raw_output=normalized,
                            error=tool_error,
                        )
                
                return ReasonerResult(
                    path_id=path_id,
                    decision=decision,
                    raw_output=normalized,
                )
                
            except Exception as e:
                return self._create_fallback_result(path_id, e)
    
    def _validate_tool_call(self, decision: ReasonerDecision) -> Optional[str]:
        """Validate a TOOL_CALL decision against the tool registry."""
        tool_name = decision.tool_name or ""
        tool = self._tools.get(tool_name)
        
        if not tool:
            return f"Unknown tool: {tool_name!r}"
        
        args = decision.tool_args or {}
        arg_errors = validate_json_obj(args, tool.input_schema)
        if arg_errors:
            return f"Invalid tool args: {arg_errors}"
        
        return None
    
    def _validate_results(
        self, results: List[ReasonerResult]
    ) -> Tuple[List[ReasonerDecision], List[str]]:
        """Separate valid decisions from invalid ones."""
        valid: List[ReasonerDecision] = []
        invalid: List[str] = []
        
        for result in results:
            if result.is_valid and result.decision:
                valid.append(result.decision)
            else:
                error = result.error or "Unknown error"
                invalid.append(f"[{result.path_id}] {error}")
        
        return valid, invalid
    
    def _create_fallback_result(self, path_id: int, error: Exception) -> ReasonerResult:
        """Create a fallback result for failed reasoner calls."""
        return ReasonerResult(
            path_id=path_id,
            decision=None,
            raw_output={
                "decision_type": "FINAL",
                "tool_name": None,
                "tool_args": None,
                "final_answer": "Reasoner failed to produce a valid JSON decision.",
                "brief_rationale": f"Reasoner call failed: {type(error).__name__}: {error}",
                "expected_signal": None,
            },
            error=f"{type(error).__name__}: {error}",
        )


class JudgeExecutor:
    """
    Executes judge invocation to select/synthesize the best decision.
    
    Responsibilities:
    - Build judge prompts
    - Invoke judge model
    - Handle structured output with fallback
    - Validate judge decisions
    
    This class is stateless and can be reused across multiple agent runs.
    """
    
    def __init__(
        self,
        model: Any,
        config: AgentConfig,
        tools: ToolRegistry,
        logger: Optional[AgentLogger] = None,
    ) -> None:
        """
        Initialize the judge executor.
        
        Args:
            model: LangChain chat model for the judge
            config: Agent configuration
            tools: Tool registry for validation
            logger: Optional logger
        """
        self._model = model
        self._config = config
        self._tools = tools
        self._logger = logger or AgentLogger(__name__)
    
    def execute(
        self,
        *,
        user_query: str,
        state_summary: str,
        candidates: Sequence[ReasonerDecision],
    ) -> JudgeDecision:
        """
        Execute the judge to select the best candidate.
        
        Uses the unified fallback function with structured output, text parsing,
        salvage, and reprompt support.
        
        Args:
            user_query: Original user query
            state_summary: Current state summary
            candidates: Validated reasoner candidates
        
        Returns:
            JudgeDecision selecting the best action
        """
        with LogContext(phase="judge"):
            tools = self._tools.all()
            tool_schemas = [t.input_schema for t in tools]
            judge_schema = get_judge_decision_schema(tool_schemas)
            
            system, user = build_judge_prompt(
                user_query=user_query,
                state_summary=state_summary,
                candidates=list(candidates),
                tools=tools,
                config=self._config,
            )
            
            try:
                # Use unified fallback function (handles reprompting internally)
                normalized = invoke_with_unified_fallback(
                    self._model,
                    system=system,
                    user=user,
                    schema=judge_schema,
                    normalizer=normalize_judge_decision_obj,
                    validator=validate_judge_decision_dict,
                    salvage_fn=salvage_judge_final if self._config.accept_non_json_final else None,
                    retry_config=self._config.llm_retry,
                    use_structured_output=self._config.use_structured_output,
                    accept_non_json_final=self._config.accept_non_json_final,
                    max_reprompts=self._config.max_reprompts,
                    phase="Judge",
                    logger=self._logger,
                )
                
                # The unified function already normalized and validated
                decision, errors = validate_judge_decision_dict(normalized)
                
                if not decision:
                    self._logger.trace(
                        f"Judge invalid JSON after unified fallback: {truncate(safe_json_dumps(normalized), 800)}"
                    )
                    return self._fallback_from_candidates(
                        candidates,
                        reason=f"Invalid judge output: {errors}",
                    )
                
                # Additional validation for TOOL_CALL
                if decision.is_tool_call:
                    tool_error = self._validate_tool_call(decision)
                    if tool_error:
                        return self._fallback_from_candidates(
                            candidates,
                            reason=f"Invalid judge tool call: {tool_error}",
                        )
                
                return decision
                
            except Exception as e:
                self._logger.trace(f"Judge unified fallback failed: {type(e).__name__}: {e}")
                return self._fallback_from_candidates(
                    candidates,
                    reason=f"Judge call failed: {type(e).__name__}: {e}",
                )
    
    def execute_best_effort_final(
        self,
        *,
        user_query: str,
        observations: Sequence[str],
        max_steps: int,
    ) -> JudgeDecision:
        """
        Request a best-effort final answer when step limit is reached.
        
        Uses the unified fallback function with structured output, text parsing,
        salvage, and reprompt support.
        
        Args:
            user_query: Original user query
            observations: All observations so far
            max_steps: Maximum steps (for context)
        
        Returns:
            JudgeDecision with best-effort final answer
        """
        state_summary = build_state_summary(
            observations=observations,
            step_index=max_steps,
            max_steps=max_steps,
        )
        
        # Use empty tools schema since we're forcing a FINAL decision
        judge_schema = get_judge_decision_schema([])
        
        system = (
            "You are the JUDGE model for a Universal Self-Consistency (USC) agent.\n"
            "The agent has reached its step limit. Produce the best possible final answer.\n"
            "Return ONLY JSON.\n"
        )
        user = "\n".join([
            "ORIGINAL_USER_QUERY:",
            user_query.strip(),
            "",
            "CURRENT_STATE_SUMMARY:",
            state_summary,
            "",
            "Return a FINAL answer as JSON with keys: decision_type, final_answer, justification.",
        ])
        
        try:
            # Use unified fallback function
            normalized = invoke_with_unified_fallback(
                self._model,
                system=system,
                user=user,
                schema=judge_schema,
                normalizer=normalize_judge_decision_obj,
                validator=validate_judge_decision_dict,
                salvage_fn=salvage_judge_final if self._config.accept_non_json_final else None,
                retry_config=self._config.llm_retry,
                use_structured_output=self._config.use_structured_output,
                accept_non_json_final=self._config.accept_non_json_final,
                max_reprompts=self._config.max_reprompts,
                phase="BestEffortFinal",
                logger=self._logger,
            )
            
            decision, errors = validate_judge_decision_dict(normalized)
            if decision:
                return decision
            
            self._logger.debug(
                "Best-effort final validation failed after unified fallback",
                errors=errors,
            )
        except Exception as e:
            self._logger.debug(
                f"Best-effort final unified fallback failed: {type(e).__name__}: {e}"
            )
        
        return JudgeDecision.create_final(
            answer="Step limit exceeded; no valid final answer could be produced.",
            justification="Failed to parse judge output",
        )
    
    def _validate_tool_call(self, decision: JudgeDecision) -> Optional[str]:
        """Validate a TOOL_CALL decision against the tool registry."""
        tool_name = decision.tool_name or ""
        tool = self._tools.get(tool_name)
        
        if not tool:
            return f"Unknown tool: {tool_name!r}"
        
        args = decision.tool_args or {}
        arg_errors = validate_json_obj(args, tool.input_schema)
        if arg_errors:
            return f"Invalid tool args: {arg_errors}"
        
        return None

    def _fallback_from_candidates(
        self,
        candidates: Sequence[ReasonerDecision],
        *,
        reason: str,
    ) -> JudgeDecision:
        """Choose a safe fallback decision when judge output is invalid."""
        if not candidates:
            return JudgeDecision.create_final(
                answer="Judge failed to produce a valid JSON decision; stopping.",
                justification=reason,
            )

        # Prefer a FINAL candidate to avoid unnecessary tool calls.
        selected_index: Optional[int] = None
        for idx, candidate in enumerate(candidates):
            if candidate.is_final and candidate.final_answer:
                selected_index = idx
                chosen = candidate
                break
        else:
            selected_index = 0
            chosen = candidates[0]

        justification = f"Fallback selection because judge output was invalid. {reason}"
        if chosen.is_tool_call:
            return JudgeDecision.create_tool_call(
                tool_name=chosen.tool_name or "",
                tool_args=chosen.tool_args or {},
                justification=justification,
                selected_index=selected_index,
            )
        return JudgeDecision(
            decision_type="FINAL",
            selected_index=selected_index,
            tool_name=None,
            tool_args=None,
            final_answer=chosen.final_answer or "",
            justification=justification,
        )


@dataclass(frozen=True)
class ToolResult:
    """
    Result of a tool execution.
    
    Attributes:
        tool_name: Name of the executed tool
        output: Tool output (any type)
        error: Error message if execution failed
        is_reflection_abort: Whether reflection decided to abort
    """
    tool_name: str
    output: Any
    error: Optional[str] = None
    is_reflection_abort: bool = False
    
    @property
    def is_success(self) -> bool:
        """Check if execution was successful."""
        return self.error is None and not self.is_reflection_abort
    
    def to_observation(self, *, max_chars: int = 0, truncate_enabled: bool = False) -> str:
        """
        Format the result as an observation string.
        
        Args:
            max_chars: Maximum characters (0 = no limit)
            truncate_enabled: Whether truncation is enabled
        
        Returns:
            Formatted observation string
        """
        if self.is_reflection_abort:
            return self.output if isinstance(self.output, str) else str(self.output)
        
        if self.error:
            msg = f"{self.tool_name} => tool_exception: {self.error}"
        else:
            rendered = safe_json_dumps(self.output)
            if truncate_enabled and max_chars > 0:
                rendered = truncate(rendered, max_chars)
            msg = f"{self.tool_name} => {rendered}"
        
        return msg


class ToolExecutor:
    """
    Executes tools with optional retry/reflection support.
    
    Responsibilities:
    - Validate tool calls
    - Execute tool functions
    - Handle errors and format observations
    - Support retry plugin integration
    
    This class is stateless and can be reused across multiple agent runs.
    """
    
    def __init__(
        self,
        tools: ToolRegistry,
        config: AgentConfig,
        retry_plugin: Optional[Any] = None,
        logger: Optional[AgentLogger] = None,
    ) -> None:
        """
        Initialize the tool executor.
        
        Args:
            tools: Tool registry
            config: Agent configuration
            retry_plugin: Optional retry/reflection plugin
            logger: Optional logger
        """
        self._tools = tools
        self._config = config
        self._retry_plugin = retry_plugin
        self._logger = logger or AgentLogger(__name__)
    
    def execute(
        self,
        *,
        tool_name: str,
        tool_args: Dict[str, Any],
        user_query: str,
    ) -> ToolResult:
        """
        Execute a tool and return the result.
        
        Args:
            tool_name: Name of the tool to execute
            tool_args: Arguments for the tool
            user_query: Original user query (for reflection context)
        
        Returns:
            ToolResult with output or error
        """
        with LogContext(phase="tool", tool_name=tool_name):
            tool = self._tools.get(tool_name)
            
            if tool is None:
                return ToolResult(
                    tool_name=tool_name,
                    output=None,
                    error=f"Unknown tool: {tool_name!r}",
                )
            
            # Validate arguments
            args = tool_args or {}
            arg_errors = validate_json_obj(args, tool.input_schema)
            if arg_errors:
                error_msg = f"Invalid args: {arg_errors} args={safe_json_dumps(args)}"
                return ToolResult(
                    tool_name=tool_name,
                    output=None,
                    error=error_msg,
                )
            
            # Log the tool call
            self._logger.tool_call(
                tool_name,
                truncate(safe_json_dumps(args), 220),
            )
            
            try:
                if self._retry_plugin:
                    result = self._retry_plugin.run(
                        tool_name=tool_name,
                        tool_args=args,
                        tool_func=tool.func,
                        all_tools=self._tools.all(),
                        user_query=user_query,
                        tool_input_schema=tool.input_schema,
                    )
                else:
                    result = tool.func(args)
                
                # Check for reflection abort signal
                if isinstance(result, str) and result.startswith("Reflection Error:"):
                    self._logger.trace(f"Tool reflection abort: {result}")
                    return ToolResult(
                        tool_name=tool_name,
                        output=result,
                        is_reflection_abort=True,
                    )
                
                # Log successful result
                rendered = safe_json_dumps(result)
                if self._config.tool_result_max_chars > 0:
                    rendered_trace = truncate(rendered, self._config.tool_result_max_chars)
                else:
                    rendered_trace = rendered
                self._logger.tool_result(tool_name, rendered_trace)
                
                return ToolResult(tool_name=tool_name, output=result)
                
            except Exception as e:
                error_msg = f"{type(e).__name__}: {e}"
                if self._config.tool_result_max_chars > 0:
                    error_trace = truncate(error_msg, self._config.tool_result_max_chars)
                else:
                    error_trace = error_msg
                
                self._logger.tool_exception(tool_name, error_trace)
                
                return ToolResult(
                    tool_name=tool_name,
                    output=None,
                    error=error_msg,
                )
