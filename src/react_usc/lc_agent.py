from __future__ import annotations

"""
Optional LangChain + LangGraph implementation.

Where each is used:
  - LangChain: prompt + model invocation + JSON parsing.
  - LangGraph: clean control-flow loop (reasoners fan-out -> judge -> single tool exec -> repeat).

Important: tools are NOT executed in parallel branches. We only execute the judged decision.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Sequence, TypedDict, cast

from .decision_normalize import normalize_judge_decision_obj, normalize_reasoner_decision_obj
from .llm_io import invoke_chat_structured_obj, invoke_chat_text, json_loads_object
from .models import AgentConfig, JudgeDecision, ReasonerDecision, ToolSpec
from .prompts import (
    build_judge_prompt,
    build_reasoner_prompt,
)
from .schema import JudgeDecisionStructured, ReasonerDecisionStructured
from .trace import trace_candidates, trace_judge
from .tools import ToolRegistry
from .utils import build_state_summary, safe_json_dumps, truncate
from .validation import (
    validate_judge_decision_dict,
    validate_json_obj,
    validate_reasoner_decision_dict,
)


def _require_langchain() -> None:
    try:
        import langchain_core  # noqa: F401
        import langgraph  # noqa: F401
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "LangChain/LangGraph dependencies not installed.\n"
            "Install with: `python -m pip install -r requirements.txt`"
        ) from e


class _State(TypedDict):
    user_query: str
    observations: List[str]
    step: int
    # last judge decision, used for routing
    judge: Optional[JudgeDecision]


@dataclass(frozen=True)
class LangGraphModels:
    """
    Bundles LangChain chat models (or any LC Runnable that can be invoked with a prompt string).

    For Vertex, you typically use ChatVertexAI with ADC:
      - `gcloud auth application-default login`
      - set project/location via environment or constructor
    """

    reasoner: Any
    judge: Any


class LangGraphReActUSCAgent:
    def __init__(
        self,
        *,
        models: LangGraphModels,
        tools: Sequence[ToolSpec],
        config: AgentConfig,
    ) -> None:
        _require_langchain()
        self._models = models
        self._tools = ToolRegistry(tools)
        self._config = config

        # Lazy imports after dependency check.
        from langgraph.graph import END, START, StateGraph  # type: ignore

        graph = StateGraph(_State)
        graph.add_node("reason_and_judge", self._node_reason_and_judge)
        graph.add_node("execute_tool", self._node_execute_tool)

        graph.add_edge(START, "reason_and_judge")

        def route(state: _State) -> Literal["execute_tool", "__end__"]:
            judge = state.get("judge")
            if judge and judge.decision_type == "TOOL_CALL":
                return "execute_tool"
            return "__end__"

        graph.add_conditional_edges("reason_and_judge", route, {"execute_tool": "execute_tool", "__end__": END})
        graph.add_edge("execute_tool", "reason_and_judge")

        self._app = graph.compile()

    def run(self, user_query: str) -> str:
        state: _State = {"user_query": user_query, "observations": [], "step": 0, "judge": None}
        final = self._app.invoke(state)
        judge = final.get("judge")
        if judge and judge.decision_type == "FINAL" and judge.final_answer:
            return judge.final_answer
        return "No final answer produced."

    # ---------------------------------------------------------------------
    # Nodes
    # ---------------------------------------------------------------------

    def _node_reason_and_judge(self, state: _State) -> _State:
        from concurrent.futures import ThreadPoolExecutor, wait

        user_query = state["user_query"]
        step = state["step"] + 1
        observations = state["observations"]

        # Step limit: ask judge for best-effort final (no more tools).
        if step > self._config.max_steps:
            final_answer = self._best_effort_final(user_query=user_query, observations=observations)
            return {**state, "step": step, "judge": final_answer}

        state_summary = build_state_summary(
            observations=observations, step_index=step, max_steps=self._config.max_steps
        )

        # --- K parallel reasoners (USC) ---
        tools = self._tools.all()

        def call_reasoner(path_id: int) -> Dict[str, Any]:
            system, user = build_reasoner_prompt(
                system_prompt=self._config.system_prompt,
                user_query=user_query,
                state_summary=state_summary,
                tools=tools,
                path_id=path_id,
            )
            try:
                if self._config.use_structured_output:
                    try:
                        obj = invoke_chat_structured_obj(
                            self._models.reasoner,
                            system=system,
                            user=user,
                            schema=ReasonerDecisionStructured,
                        )
                        return normalize_reasoner_decision_obj(obj)
                    except Exception:
                        # Fall back to the legacy JSON parse path (some backends don't support structured output).
                        pass

                raw_text = invoke_chat_text(self._models.reasoner, system=system, user=user)
                try:
                    obj = json_loads_object(raw_text)
                    return normalize_reasoner_decision_obj(obj)
                except Exception as parse_e:
                    if self._config.trace:
                        print(f"  Reasoner[{path_id}] non-JSON output preview: {truncate(raw_text, 400)}")
                    raise parse_e
            except Exception as e:
                # Return a VALID ReasonerDecision shape even on failures so validation remains predictable.
                return {
                    "decision_type": "FINAL",
                    "tool_name": None,
                    "tool_args": None,
                    "final_answer": "Reasoner failed to produce a valid JSON decision.",
                    "brief_rationale": f"Reasoner call failed: {type(e).__name__}: {e}",
                    "expected_signal": None,
                }

        raw_candidates: List[Dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=min(32, self._config.k_paths)) as ex:
            futures = [ex.submit(call_reasoner, i) for i in range(self._config.k_paths)]
            # Vertex requests can occasionally exceed small timeouts. Instead of crashing the graph,
            # we proceed with any completed candidates and mark unfinished ones as timeouts.
            done, not_done = wait(futures, timeout=self._config.timeout_seconds)

            for f in done:
                try:
                    raw_candidates.append(f.result(timeout=0))
                except Exception as e:
                    raw_candidates.append(
                        {
                            "decision_type": "FINAL",
                            "tool_name": None,
                            "tool_args": None,
                            "final_answer": "Reasoner failed to produce a valid JSON decision.",
                            "brief_rationale": f"Reasoner call failed: {type(e).__name__}: {e}",
                            "expected_signal": None,
                        }
                    )

            if not_done:
                for f in not_done:
                    f.cancel()
                if self._config.trace:
                    print(f"  Reasoner timeout: {len(not_done)}/{len(futures)} candidates unfinished after {self._config.timeout_seconds}s")
                for _ in not_done:
                    raw_candidates.append(
                        {
                            "decision_type": "FINAL",
                            "tool_name": None,
                            "tool_args": None,
                            "final_answer": "Reasoner timed out before producing a decision.",
                            "brief_rationale": f"Reasoner call timed out after {self._config.timeout_seconds}s.",
                            "expected_signal": None,
                        }
                    )

        candidates, invalid = self._validate_candidates(raw_candidates)
        if self._config.trace:
            trace_candidates(step=step, k=self._config.k_paths, valid=candidates, invalid=invalid)

        # --- Judge ---
        system, user = build_judge_prompt(
            user_query=user_query, state_summary=state_summary, candidates=candidates, config=self._config
        )
        try:
            if self._config.use_structured_output:
                try:
                    judge_raw = normalize_judge_decision_obj(
                        invoke_chat_structured_obj(
                            self._models.judge,
                            system=system,
                            user=user,
                            schema=JudgeDecisionStructured,
                        )
                    )
                except Exception:
                    # Fall back to the legacy JSON parse path (some backends don't support structured output).
                    judge_raw = {}
            else:
                judge_raw = {}

            if not judge_raw:
                judge_text = invoke_chat_text(self._models.judge, system=system, user=user)
                try:
                    judge_raw = normalize_judge_decision_obj(json_loads_object(judge_text))
                except Exception as parse_e:
                    if self._config.trace:
                        print(f"  Judge non-JSON output preview: {truncate(judge_text, 600)}")
                    raise parse_e

            judge, errors = validate_judge_decision_dict(judge_raw)
            if not judge:
                if self._config.trace:
                    print(f"  Judge invalid JSON (post-normalization): {truncate(safe_json_dumps(judge_raw), 800)}")
                judge = JudgeDecision(
                    decision_type="FINAL",
                    selected_index=None,
                    tool_name=None,
                    tool_args=None,
                    final_answer="Judge produced invalid output; cannot continue.",
                    justification=f"invalid judge output: {errors}",
                )
        except Exception as e:
            # Never crash the graph on a judge parsing failure; stop with a clear final message.
            judge = JudgeDecision(
                decision_type="FINAL",
                selected_index=None,
                tool_name=None,
                tool_args=None,
                final_answer="Judge failed to produce a valid JSON decision; stopping.",
                justification=f"Judge call failed: {type(e).__name__}: {e}",
            )

        if self._config.trace:
            trace_judge(step=step, decision=judge)

        return {**state, "step": step, "judge": judge}

    def _node_execute_tool(self, state: _State) -> _State:
        judge = state.get("judge")
        if not judge or judge.decision_type != "TOOL_CALL" or not judge.tool_name:
            return state

        tool = self._tools.get(judge.tool_name)
        if tool is None:
            obs = f"Tool error: unknown tool '{judge.tool_name}'"
            return {**state, "observations": state["observations"] + [obs]}

        args = judge.tool_args or {}
        arg_errors = validate_json_obj(args, tool.input_schema)
        if arg_errors:
            obs = f"{tool.name} => invalid_args: {arg_errors} args={safe_json_dumps(args)}"
            return {**state, "observations": state["observations"] + [obs]}

        if self._config.trace:
            print(f"  Tool call: {tool.name} args={truncate(safe_json_dumps(args), 220)}")
        try:
            result = tool.func(args)
            rendered = truncate(safe_json_dumps(result), self._config.tool_result_max_chars)
            if self._config.trace:
                print(f"  Tool result: {tool.name} => {rendered}")
            obs = f"{tool.name} => {rendered}"
        except Exception as e:
            msg = truncate(f"{type(e).__name__}: {e}", self._config.tool_result_max_chars)
            if self._config.trace:
                print(f"  Tool exception: {tool.name} => {msg}")
            obs = f"{tool.name} => tool_exception: {msg}"

        return {**state, "observations": state["observations"] + [obs]}

    # ---------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------

    def _validate_candidates(
        self, raw_candidates: Sequence[Dict[str, Any]]
    ) -> tuple[list[ReasonerDecision], list[str]]:
        valid: List[ReasonerDecision] = []
        invalid: List[str] = []

        for i, raw in enumerate(raw_candidates):
            cand, errors = validate_reasoner_decision_dict(raw)
            if not cand:
                invalid.append(f"[{i}] invalid decision: {errors}; raw={safe_json_dumps(raw)}")
                continue

            if cand.decision_type == "TOOL_CALL":
                tool_name = cand.tool_name or ""
                tool = self._tools.get(tool_name)
                if not tool:
                    invalid.append(f"[{i}] unknown tool '{tool_name}'")
                    continue
                args = cand.tool_args or {}
                arg_errors = validate_json_obj(args, tool.input_schema)
                if arg_errors:
                    invalid.append(f"[{i}] invalid tool args: {arg_errors}")
                    continue

            valid.append(cand)

        return valid, invalid

    def _best_effort_final(self, *, user_query: str, observations: Sequence[str]) -> JudgeDecision:
        state_summary = build_state_summary(
            observations=observations, step_index=self._config.max_steps, max_steps=self._config.max_steps
        )
        system = (
            "You are the JUDGE model for a Universal Self-Consistency (USC) agent.\n"
            "The agent has reached its step limit. Produce the best possible final answer.\n"
            "Return ONLY JSON.\n"
        )
        user = "\n".join(
            [
                "ORIGINAL_USER_QUERY:",
                user_query.strip(),
                "",
                "CURRENT_STATE_SUMMARY:",
                state_summary,
                "",
                "Return a FINAL answer as JSON with keys: decision_type, final_answer, justification.",
            ]
        )
        try:
            raw = json_loads_object(invoke_chat_text(self._models.judge, system=system, user=user))
            judge, _ = validate_judge_decision_dict(raw)
        except Exception:
            judge = None
        return judge or JudgeDecision(
            decision_type="FINAL",
            selected_index=None,
            tool_name=None,
            tool_args=None,
            final_answer="Step limit exceeded; no valid final answer could be produced.",
            justification="failed to parse judge output",
        )
