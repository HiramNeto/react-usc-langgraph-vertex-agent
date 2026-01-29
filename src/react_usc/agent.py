"""
LangGraph-based ReAct agent with Universal Self-Consistency (USC).

This module provides the main agent class that orchestrates the ReAct loop
with K parallel reasoners, a judge for selection, and single tool execution.

Architecture:
    - Uses LangGraph for clean control-flow (state machine)
    - Uses LangChain for prompt + model invocation
    - Delegates to executor classes for specific operations
    - Supports optional retry/reflection plugin

Key Components:
    - LangGraphReActUSCAgent: Main agent class
    - LangGraphModels: Container for LangChain chat models
    - _State: Internal state representation for the graph

Usage:
    from react_usc import LangGraphReActUSCAgent, LangGraphModels, AgentConfig
    
    models = LangGraphModels(reasoner=chat_model, judge=chat_model)
    config = AgentConfig.default()
    
    agent = LangGraphReActUSCAgent(
        models=models,
        tools=[my_tool],
        config=config,
    )
    
    result = agent.run("What is 2 + 2?")
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Literal, Optional, Sequence, TypedDict

from .config import AgentConfig
from .decisions import JudgeDecision
from .executors import JudgeExecutor, ReasonerExecutor, ToolExecutor
from .logging import AgentLogger, LogContext, LoggingConfig, generate_trace_id
from .plugins import ReflectAndRetryToolPlugin
from .tools import ToolRegistry
from .trace import trace_candidates, trace_judge
from .types import ToolSpec
from ._internal.utils import build_state_summary


def _require_langchain() -> None:
    """Verify LangChain/LangGraph dependencies are available."""
    try:
        import langchain_core  # noqa: F401
        import langgraph  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            "LangChain/LangGraph dependencies not installed.\n"
            "Install with: `pip install react-usc` or `pip install langchain langgraph`"
        ) from e


class _State(TypedDict):
    """
    Internal state for the LangGraph state machine.
    
    This is passed between nodes and tracks the agent's progress.
    
    Attributes:
        user_query: The original user query
        observations: List of tool observations collected so far
        step: Current step number (1-indexed)
        judge: Last judge decision (used for routing)
        trace_id: Unique ID for this agent run (for logging correlation)
    """
    user_query: str
    observations: List[str]
    step: int
    judge: Optional[JudgeDecision]
    trace_id: str


@dataclass(frozen=True)
class LangGraphModels:
    """
    Container for LangChain chat models used by the agent.
    
    Both models should support `invoke()` with a list of messages.
    For structured output support, they should also support
    `with_structured_output()`.
    
    For Vertex AI, use `ChatGoogleGenerativeAI` (with vertexai=True) via ADC:
        - Run: `gcloud auth application-default login`
        - Set project/location via environment or constructor
    
    Attributes:
        reasoner: Model for reasoner invocations (called K times in parallel)
        judge: Model for judge invocations (called once per step)
    """
    reasoner: Any
    judge: Any
    
    def __post_init__(self) -> None:
        """Validate that models have the required interface."""
        if not hasattr(self.reasoner, "invoke"):
            raise ValueError("Reasoner model must have an 'invoke' method")
        if not hasattr(self.judge, "invoke"):
            raise ValueError("Judge model must have an 'invoke' method")


class LangGraphReActUSCAgent:
    """
    ReAct agent with Universal Self-Consistency (USC).
    
    This agent implements the USC pattern:
    1. Sample K parallel reasoning paths (reasoners)
    2. Judge selects or synthesizes the best next action
    3. Execute the single selected tool
    4. Repeat until FINAL answer or max steps
    
    The agent uses LangGraph for control flow, making the execution
    easy to visualize and debug.
    
    Key Features:
        - Parallel reasoner execution with timeout handling
        - Structured output with fallback to text parsing
        - Tool validation before execution
        - Optional retry/reflection plugin for error recovery
        - Comprehensive logging and tracing
    
    Thread Safety:
        The agent itself is NOT thread-safe for concurrent `run()` calls,
        but uses thread pools internally for parallel reasoner execution.
        Create separate agent instances for concurrent use.
    
    Example:
        ```python
        # Basic usage
        agent = LangGraphReActUSCAgent(
            models=models,
            tools=[calculator, search],
            config=config,
        )
        answer = agent.run("What is the capital of France?")
        
        # With retry plugin
        agent = LangGraphReActUSCAgent(
            models=models,
            tools=tools,
            config=config,
            plugins=[ReflectAndRetryToolPlugin(model=retry_model)],
        )
        ```
    """
    
    def __init__(
        self,
        *,
        models: LangGraphModels,
        tools: Sequence[ToolSpec],
        config: AgentConfig,
        plugins: Sequence[Any] = (),
        logger: Optional[AgentLogger] = None,
    ) -> None:
        """
        Initialize the agent.
        
        Args:
            models: LangChain models for reasoner and judge
            tools: Sequence of tool specifications
            config: Agent configuration
            plugins: Optional sequence of plugins (e.g., ReflectAndRetryToolPlugin)
            logger: Optional custom logger
        
        Raises:
            RuntimeError: If LangChain/LangGraph dependencies are not installed
            ValueError: If configuration is invalid
        """
        _require_langchain()
        
        self._tools = ToolRegistry(tools)
        self._config = config
        
        # Create logger with configuration
        logging_config = LoggingConfig(
            enable_trace=config.trace,
            log_structured_output=config.log_structured_output,
        )
        self._logger = logger or AgentLogger(__name__, logging_config)
        
        # Find retry plugin if present
        self._retry_plugin: Optional[ReflectAndRetryToolPlugin] = None
        for p in plugins:
            if isinstance(p, ReflectAndRetryToolPlugin):
                self._retry_plugin = p
                break
        
        # Initialize executors
        self._reasoner_executor = ReasonerExecutor(
            model=models.reasoner,
            config=config,
            tools=self._tools,
            logger=self._logger,
        )
        self._judge_executor = JudgeExecutor(
            model=models.judge,
            config=config,
            tools=self._tools,
            logger=self._logger,
        )
        self._tool_executor = ToolExecutor(
            tools=self._tools,
            config=config,
            retry_plugin=self._retry_plugin,
            logger=self._logger,
        )
        
        # Build LangGraph
        self._app = self._build_graph()
    
    def _build_graph(self) -> Any:
        """Build and compile the LangGraph state machine."""
        from langgraph.graph import END, START, StateGraph
        
        graph = StateGraph(_State)
        
        # Add nodes
        graph.add_node("reason_and_judge", self._node_reason_and_judge)
        graph.add_node("execute_tool", self._node_execute_tool)
        
        # Add edges
        graph.add_edge(START, "reason_and_judge")
        graph.add_conditional_edges(
            "reason_and_judge",
            self._route_after_judge,
            {"execute_tool": "execute_tool", "__end__": END},
        )
        graph.add_edge("execute_tool", "reason_and_judge")
        
        return graph.compile()
    
    def _route_after_judge(self, state: _State) -> Literal["execute_tool", "__end__"]:
        """
        Routing function: decide next node based on judge decision.
        
        Returns "execute_tool" for TOOL_CALL, "__end__" for FINAL.
        """
        judge = state.get("judge")
        if judge and judge.is_tool_call:
            return "execute_tool"
        return "__end__"
    
    def run(self, user_query: str) -> str:
        """
        Run the agent on a user query.
        
        Executes the ReAct loop until a FINAL answer is produced
        or max_steps is reached.
        
        Args:
            user_query: The user's question or task
        
        Returns:
            The final answer string
        
        Note:
            This method is NOT thread-safe. For concurrent usage,
            create separate agent instances.
        """
        trace_id = generate_trace_id()
        
        with LogContext(trace_id=trace_id, phase="agent_run"):
            self._logger.info(
                "Starting agent run",
                query_preview=user_query[:100] if len(user_query) > 100 else user_query,
                k_paths=self._config.k_paths,
                max_steps=self._config.max_steps,
            )
            
            initial_state: _State = {
                "user_query": user_query,
                "observations": [],
                "step": 0,
                "judge": None,
                "trace_id": trace_id,
            }
            
            final_state = self._app.invoke(initial_state)
            
            judge = final_state.get("judge")
            if judge and judge.is_final and judge.final_answer:
                self._logger.info(
                    "Agent completed with final answer",
                    steps=final_state.get("step", 0),
                )
                return judge.final_answer
            
            self._logger.warning("Agent completed without producing a final answer")
            return "No final answer produced."
    
    def _node_reason_and_judge(self, state: _State) -> _State:
        """
        Graph node: Execute reasoners and judge.
        
        This node:
        1. Checks if max_steps exceeded
        2. Executes K parallel reasoners
        3. Validates and traces candidates
        4. Invokes judge to select best action
        5. Returns updated state with judge decision
        """
        with LogContext(
            trace_id=state.get("trace_id", ""),
            phase="reason_and_judge",
            step=state["step"] + 1,
        ):
            user_query = state["user_query"]
            step = state["step"] + 1
            observations = state["observations"]
            
            # Check step limit
            if step > self._config.max_steps:
                self._logger.info("Step limit reached, producing best-effort final")
                judge = self._judge_executor.execute_best_effort_final(
                    user_query=user_query,
                    observations=observations,
                    max_steps=self._config.max_steps,
                )
                return {**state, "step": step, "judge": judge}
            
            # Build state summary
            state_summary = build_state_summary(
                observations=observations,
                step_index=step,
                max_steps=self._config.max_steps,
            )
            
            # Execute reasoners
            self._logger.debug(
                f"Executing {self._config.k_paths} parallel reasoners",
                step=step,
            )
            candidates, invalid = self._reasoner_executor.execute(
                user_query=user_query,
                state_summary=state_summary,
            )
            
            # Trace candidates if enabled
            if self._config.trace:
                trace_candidates(
                    step=step,
                    k=self._config.k_paths,
                    valid=candidates,
                    invalid=invalid,
                )
            
            # Handle case where all reasoners failed
            if not candidates:
                self._logger.warning(
                    f"All {self._config.k_paths} reasoners failed at step {step}",
                    invalid_reasons=invalid[:5],  # Log first 5 reasons
                )
                judge = self._handle_all_reasoners_failed(
                    user_query=user_query,
                    observations=observations,
                    step=step,
                    invalid_reasons=invalid,
                )
                return {**state, "step": step, "judge": judge}
            
            # Execute judge
            self._logger.debug("Executing judge", step=step, num_candidates=len(candidates))
            judge = self._judge_executor.execute(
                user_query=user_query,
                state_summary=state_summary,
                candidates=candidates,
            )
            
            # Trace judge decision if enabled
            if self._config.trace:
                trace_judge(step=step, decision=judge)
            
            self._logger.info(
                f"Step {step} complete",
                decision_type=judge.decision_type,
                tool_name=judge.tool_name if judge.is_tool_call else None,
            )
            
            return {**state, "step": step, "judge": judge}
    
    def _node_execute_tool(self, state: _State) -> _State:
        """
        Graph node: Execute the selected tool.
        
        This node:
        1. Gets tool info from judge decision
        2. Executes the tool via ToolExecutor
        3. Formats result as observation
        4. Returns updated state with new observation
        """
        judge = state.get("judge")
        if not judge or not judge.is_tool_call or not judge.tool_name:
            return state
        
        with LogContext(
            trace_id=state.get("trace_id", ""),
            phase="execute_tool",
            step=state["step"],
            tool_name=judge.tool_name,
        ):
            result = self._tool_executor.execute(
                tool_name=judge.tool_name,
                tool_args=judge.tool_args or {},
                user_query=state["user_query"],
            )
            
            # Format observation
            observation = result.to_observation(
                max_chars=self._config.tool_result_max_chars,
                truncate_enabled=self._config.truncate_agent_observations,
            )
            
            return {**state, "observations": state["observations"] + [observation]}
    
    def _handle_all_reasoners_failed(
        self,
        *,
        user_query: str,
        observations: Sequence[str],
        step: int,
        invalid_reasons: List[str],
    ) -> JudgeDecision:
        """
        Handle the edge case where all K reasoners fail to produce valid output.
        
        This can happen due to:
        - All LLM calls timing out
        - All outputs failing JSON parsing
        - All outputs failing validation
        
        Strategy:
        1. If we have observations, ask judge for best-effort final answer
        2. Otherwise, return a graceful failure message
        
        Args:
            user_query: Original user query
            observations: Observations collected so far
            step: Current step number
            invalid_reasons: List of reasons why reasoners failed
        
        Returns:
            JudgeDecision (always FINAL to gracefully exit the loop)
        """
        # If we have some observations, try to produce a best-effort answer
        if observations:
            self._logger.info(
                "Attempting best-effort final from existing observations",
                observation_count=len(observations),
            )
            return self._judge_executor.execute_best_effort_final(
                user_query=user_query,
                observations=observations,
                max_steps=self._config.max_steps,
            )
        
        # No observations - provide informative error
        error_summary = "; ".join(invalid_reasons[:3]) if invalid_reasons else "Unknown errors"
        return JudgeDecision.create_final(
            answer=(
                f"Unable to process your request. All {self._config.k_paths} reasoning "
                f"attempts failed at step {step}. This may be a temporary issue - please try again."
            ),
            justification=f"All reasoners failed: {error_summary}",
        )
    
    # =========================================================================
    # Public Properties
    # =========================================================================
    
    @property
    def config(self) -> AgentConfig:
        """Get the agent configuration (read-only)."""
        return self._config
    
    @property
    def tool_names(self) -> List[str]:
        """Get list of available tool names."""
        return [t.name for t in self._tools.all()]
