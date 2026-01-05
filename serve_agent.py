"""
A2A Server Runner.

This script demonstrates how to run the agent as an A2A-compliant server.
Requires:
  pip install fastapi uvicorn
"""

import os
import uvicorn
from dotenv import load_dotenv

from src.react_usc.lc_agent import LangGraphModels, LangGraphReActUSCAgent
from src.react_usc.lc_vertex import make_chat_vertex_ai
from src.react_usc.models import AgentConfig, ModelConfig, RetryConfig
from src.react_usc.plugins import ReflectAndRetryToolPlugin
from src.react_usc.tools import make_calculator_tool, make_simple_search_tool
from src.react_usc.test_tools import make_flaky_tool
from src.react_usc.a2a import A2AAgentWrapper, create_a2a_app


def _opt_int_env(name: str) -> int | None:
    val = os.getenv(name)
    if val is None:
        return None
    val = val.strip()
    if val == "":
        return None
    return int(val)


def create_agent() -> LangGraphReActUSCAgent:
    # Auto-load .env from repo root (if present)
    load_dotenv(override=False)

    tools = [make_calculator_tool(), make_simple_search_tool(), make_flaky_tool()]
    
    project_id = os.getenv("VERTEX_PROJECT_ID")
    location = os.getenv("VERTEX_LOCATION", "us-central1")
    default_vertex_model = os.getenv("VERTEX_MODEL", "gemini-2.0-flash-001")
    
    if not project_id:
        raise RuntimeError("VERTEX_PROJECT_ID is required (set it in your environment).")

    reasoner_vertex_model = os.getenv("REASONER_MODEL_NAME") or default_vertex_model
    judge_vertex_model = os.getenv("JUDGE_MODEL_NAME") or default_vertex_model

    reasoner_lc = make_chat_vertex_ai(model=reasoner_vertex_model, location=location, project=project_id)
    judge_lc = make_chat_vertex_ai(model=judge_vertex_model, location=location, project=project_id)

    config = AgentConfig(
        system_prompt=(
            "You are a helpful assistant.\n"
            "Use tools when they meaningfully reduce uncertainty or improve correctness.\n"
            "Prefer minimal tool use; do not call tools if you can answer directly.\n"
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
        selection_strategy=os.getenv("SELECTION_STRATEGY", "select_one"),
        allow_tool_synthesis=os.getenv("ALLOW_TOOL_SYNTHESIS", "true").lower() == "true",
        retry=RetryConfig(max_retries=1, backoff_seconds=0.1),
        trace=os.getenv("TRACE", "true").lower() == "true",
        # Agent reasoning is always unlimited, but can be set > 0 for terminal debugging limits
        tool_result_max_chars=int(os.getenv("TOOL_RESULT_MAX_CHARS", "0")),
        timeout_seconds=float(os.getenv("LLM_TIMEOUT_SECONDS", "5.0")),
        use_structured_output=os.getenv("USE_STRUCTURED_OUTPUT", "true").lower() == "true",
    )

    reflection_plugin = ReflectAndRetryToolPlugin(
        model=reasoner_lc,
        max_retries=3,
        trace=config.trace,
        backoff_seconds=1.0,
    )

    return LangGraphReActUSCAgent(
        models=LangGraphModels(reasoner=reasoner_lc, judge=judge_lc),
        tools=tools,
        config=config,
        plugins=[reflection_plugin],
    )


def main():
    agent = create_agent()
    wrapper = A2AAgentWrapper(
        agent=agent,
        agent_id="react-usc-demo",
        name="ReAct USC Demo Agent",
        description="A demo agent using Universal Self-Consistency."
    )
    
    app = create_a2a_app(wrapper)
    
    port = int(os.getenv("PORT", "8000"))
    print(f"Starting A2A server on http://0.0.0.0:{port}")
    print(f"Agent Card available at http://localhost:{port}/.well-known/a2a.json")
    
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()

