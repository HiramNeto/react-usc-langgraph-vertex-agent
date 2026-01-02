"""
Small demo runner for the ReAct + Universal Self-Consistency (USC) agent.

Run:
  python3 main.py
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

from src.react_usc.lc_agent import LangGraphModels, LangGraphReActUSCAgent
from src.react_usc.lc_vertex import make_chat_vertex_ai
from src.react_usc.models import AgentConfig, ModelConfig, RetryConfig
from src.react_usc.plugins import ReflectAndRetryToolPlugin
from src.react_usc.tools import make_calculator_tool, make_simple_search_tool
from src.react_usc.test_tools import make_flaky_tool


def _opt_int_env(name: str) -> int | None:
    val = os.getenv(name)
    if val is None:
        return None
    val = val.strip()
    if val == "":
        return None
    return int(val)


def main() -> None:
    # Auto-load .env from repo root (if present). Does not override existing env vars.
    load_dotenv(override=False)

    tools = [make_calculator_tool(), make_simple_search_tool(), make_flaky_tool()]
    # LangChain + LangGraph implementation only (Vertex AI Gemini via ADC).
    # Prereq:
    #   python -m pip install -r requirements.txt
    #   gcloud auth application-default login
    project_id = os.getenv("VERTEX_PROJECT_ID")
    location = os.getenv("VERTEX_LOCATION", "us-central1")
    default_vertex_model = os.getenv("VERTEX_MODEL", "gemini-2.0-flash-001")
    if not project_id:
        raise RuntimeError("VERTEX_PROJECT_ID is required (set it in your environment).")

    # Allow different underlying models for reasoner vs judge.
    # If REASONER_MODEL_NAME/JUDGE_MODEL_NAME are unset, fall back to VERTEX_MODEL.
    reasoner_vertex_model = os.getenv("REASONER_MODEL_NAME") or default_vertex_model
    judge_vertex_model = os.getenv("JUDGE_MODEL_NAME") or default_vertex_model

    reasoner_lc = make_chat_vertex_ai(model=reasoner_vertex_model, location=location, project=project_id)
    judge_lc = make_chat_vertex_ai(model=judge_vertex_model, location=location, project=project_id)

    config = AgentConfig(
        system_prompt=(
            "You are a helpful assistant.\n"
            "Use tools when they meaningfully reduce uncertainty or improve correctness.\n"
            "Prefer minimal tool use; do not call tools if you can answer directly.\n"
            "When calling a tool, make arguments valid and minimal.\n"
        ),
        k_paths=4,
        max_steps=6,
        reasoner_model=ModelConfig(
            name=reasoner_vertex_model,
            temperature=float(os.getenv("REASONER_TEMPERATURE", "0.7")),
            max_tokens=_opt_int_env("REASONER_MAX_TOKENS"),
        ),
        judge_model=ModelConfig(
            name=judge_vertex_model,
            temperature=float(os.getenv("JUDGE_TEMPERATURE", "0.0")),
            max_tokens=_opt_int_env("JUDGE_MAX_TOKENS"),
        ),
        selection_strategy=os.getenv("SELECTION_STRATEGY", "select_one"),  # or "synthesize_one"
        allow_tool_synthesis=os.getenv("ALLOW_TOOL_SYNTHESIS", "true").lower() == "true",
        retry=RetryConfig(max_retries=1, backoff_seconds=0.1),
        trace=os.getenv("TRACE", "true").lower() == "true",
        tool_result_max_chars=int(os.getenv("TOOL_RESULT_MAX_CHARS", "400")),
        timeout_seconds=float(os.getenv("LLM_TIMEOUT_SECONDS", "5.0")),
        use_structured_output=os.getenv("USE_STRUCTURED_OUTPUT", "true").lower() == "true",
    )

    reflection_plugin = ReflectAndRetryToolPlugin(
        model=reasoner_lc,
        max_retries=3,
        trace=config.trace,
        backoff_seconds=1.0,
    )

    agent = LangGraphReActUSCAgent(
        models=LangGraphModels(reasoner=reasoner_lc, judge=judge_lc),
        tools=tools,
        config=config,
        plugins=[reflection_plugin],
    )

    print("\n=== Demo 1: math ===")
    answer1 = agent.run("What is 2+2*10? Please compute it.")
    print("\nFINAL ANSWER:", answer1)

    print("\n=== Demo 2: search ===")
    answer2 = agent.run("Search: What is ReAct and how does self-consistency help?")
    print("\nFINAL ANSWER:", answer2)

    print("\n=== Demo 3: API Client - RETRY (Arg Fix) ===")
    # Expected: 400 Bad Request (missing param), Reflection sees it, retries with include_profile=true.
    answer3 = agent.run("Fetch details for user 123 using the api_client.")
    print("\nFINAL ANSWER:", answer3)

    print("\n=== Demo 4: API Client - WAIT (Transient 503) ===")
    # Expected: 503 Service Unavailable, Reflection chooses WAIT, retries same args, succeeds eventually.
    answer4 = agent.run("Sync data to the upstream service using POST /api/v1/sync/data.")
    print("\nFINAL ANSWER:", answer4)

    print("\n=== Demo 5: API Client - ABORT (Fatal 403) ===")
    # Expected: 403 Forbidden, Reflection sees it's a permission issue, chooses ABORT.
    answer5 = agent.run("Delete the system database using the api_client at /api/v1/admin/system.")
    print("\nFINAL ANSWER:", answer5)


if __name__ == "__main__":
    main()


