from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Literal, Optional, Union


JSONValue = Union[None, bool, int, float, str, List["JSONValue"], Dict[str, "JSONValue"]]

DecisionType = Literal["TOOL_CALL", "FINAL"]
SelectionStrategy = Literal["select_one", "synthesize_one"]


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_schema: Dict[str, Any]  # JSON-schema-like (minimal subset used for validation)
    func: Callable[[Dict[str, Any]], Any]


@dataclass(frozen=True)
class ModelConfig:
    name: str
    temperature: float
    # If None, the provider/model default is used (no explicit max token cap).
    max_tokens: Optional[int] = None


@dataclass(frozen=True)
class RetryConfig:
    max_retries: int
    backoff_seconds: float


@dataclass(frozen=True)
class AgentConfig:
    system_prompt: str
    k_paths: int
    max_steps: int
    reasoner_model: ModelConfig
    judge_model: ModelConfig
    selection_strategy: SelectionStrategy
    allow_tool_synthesis: bool
    retry: RetryConfig
    trace: bool
    tool_result_max_chars: int
    timeout_seconds: float = 20.0
    # If true, use LangChain structured output (`with_structured_output`) when possible.
    # The agent will fall back to text JSON parsing if the backend doesn't support it.
    use_structured_output: bool = True


@dataclass(frozen=True)
class ReasonerDecision:
    decision_type: DecisionType
    tool_name: Optional[str]
    tool_args: Optional[Dict[str, Any]]
    final_answer: Optional[str]
    brief_rationale: str
    expected_signal: Optional[str] = None


@dataclass(frozen=True)
class JudgeDecision:
    decision_type: DecisionType
    selected_index: Optional[int]
    tool_name: Optional[str]
    tool_args: Optional[Dict[str, Any]]
    final_answer: Optional[str]
    justification: str


