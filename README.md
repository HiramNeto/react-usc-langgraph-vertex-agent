### ReAct + Universal Self-Consistency (USC) Agent (LangChain + LangGraph, Vertex AI Gemini)

This project is a clean Python POC of a **ReAct-style agent loop** where each step samples **K parallel "reasoner" decisions** (Universal Self-Consistency), then a **judge model** picks (or synthesizes) the **single best next decision**, and **only that single decision is executed** as a tool call before continuing.

It is designed to run on **Vertex AI Gemini** using **GCP CLI authentication (ADC)** — no Gemini API keys.

---

### What you get

- **ReAct loop** with tool use and observations
- **USC fan-out**: K parallel candidate next steps per iteration
- **Judge selection**: choose the best single next step (or synthesize one)
- **Single tool execution**: *never* execute tools inside parallel branches
- **Structured decisions**: models are instructed to output **JSON-only** decisions
- **Resilience**: Optional "Reflect and Retry" plugin to recover from tool failures
- **A2A Support**: Optional wrapper to expose the agent via standard Agent-to-Agent protocols
- **Tracing & Logging**: Structured logging with context management and trace output

---

### How ReAct + USC is implemented (high-level)

At each step:

- **Reasoner fan-out (USC)**:
  - Build a ReAct context (system prompt, original user query, state summary, tool schemas).
  - Run **K** parallel reasoner model calls.
  - Each reasoner returns a **`ReasonerDecision` JSON object**:
    - `decision_type`: `"TOOL_CALL"` or `"FINAL"`
    - If tool call: `tool_name` + `tool_args`
    - If final: `final_answer`
    - A short `brief_rationale`

- **Judge**:
  - Judge prompt includes:
    - the **original user query** (always)
    - state summary (observations)
    - the **K validated candidate JSON decisions**
    - a rubric (alignment, consistency, tool minimality, etc.)
  - Judge returns a **`JudgeDecision` JSON object**:
    - either a `"FINAL"` answer
    - or a `"TOOL_CALL"` (one tool + args)

- **Act (single tool call)**:
  - Execute only the judged tool call.
  - Append the tool output as an **observation** (optionally truncated).
  - Loop again with updated state.

If the step limit is reached, the agent requests a **best-effort final answer**.

---

### How LangGraph is used (control flow)

LangGraph expresses the loop as a small state machine:

- **State** (`_State` in `src/react_usc/agent.py`):
  - `user_query`: original query (constant)
  - `observations`: list of tool results / errors
  - `step`: step counter
  - `judge`: last `JudgeDecision` (used for routing)
  - `trace_id`: unique ID for logging correlation

- **Nodes**:
  - `reason_and_judge`:
    - runs K parallel reasoners via `ReasonerExecutor`
    - validates candidates
    - runs the judge via `JudgeExecutor`
    - stores the judge decision in state
  - `execute_tool`:
    - executes **only** the judged tool call via `ToolExecutor`
    - appends an observation

- **Edges / routing**:
  - `START -> reason_and_judge`
  - If judge decides `TOOL_CALL` ⇒ `reason_and_judge -> execute_tool -> reason_and_judge`
  - If judge decides `FINAL` ⇒ `reason_and_judge -> END`

This keeps the loop readable: the "graph wiring" is separated from tool execution and prompt construction.

---

### Architecture

The codebase follows a **composition over inheritance** design with single-responsibility classes:

```
LangGraphReActUSCAgent (orchestrator)
├── ReasonerExecutor   - Parallel reasoner invocation
├── JudgeExecutor      - Judge invocation and selection
├── ToolExecutor       - Tool execution with optional retry
├── ToolRegistry       - Tool lookup and validation
└── AgentLogger        - Structured logging
```

---

### Installation

**From PyPI (when published):**

```bash
pip install react-usc
```

**With optional dependencies:**

```bash
# Vertex AI support
pip install react-usc[vertex]

# A2A server support
pip install react-usc[a2a]

# Development dependencies
pip install react-usc[dev]

# All optional dependencies
pip install react-usc[all]
```

**From source:**

```bash
git clone <repo-url>
cd react-usc
pip install -e .
# Or with extras:
pip install -e ".[all]"
```

---

### Project layout

```
├── pyproject.toml                    # Python packaging (dependencies, build config, tools)
├── pytest.ini                        # Test configuration
├── env.example                       # Environment variable template
├── tests/                            # Test suite
│   ├── conftest.py                   # Shared test fixtures
│   ├── test_decisions.py             # Decision class tests
│   ├── test_executors.py             # Executor tests
│   ├── test_plugins.py               # Plugin tests
│   └── test_validation.py            # Validation function tests
├── examples/                         # Demo code and example tools
│   ├── __init__.py                   # Package marker
│   ├── cli_demo.py                   # Demo runner (loads .env, builds tools/config, runs agent)
│   ├── a2a_server.py                 # A2A server runner (exposes agent via HTTP)
│   └── tools/                        # Example tool implementations
│       ├── __init__.py               # Re-exports all example tools
│       ├── calculator.py             # Safe arithmetic calculator tool
│       ├── search.py                 # Simple in-memory search tool
│       └── flaky_api.py              # Flaky API client for testing retry
└── src/react_usc/                    # Core library (importable package)
    ├── __init__.py                   # Public API exports
    ├── agent.py                      # LangGraphReActUSCAgent (main agent class)
    ├── config.py                     # Configuration classes (AgentConfig, ModelConfig, RetryConfig)
    ├── decisions.py                  # Decision dataclasses (ReasonerDecision, JudgeDecision)
    ├── types.py                      # Type aliases, constants, and ToolSpec
    ├── models.py                     # Re-exports for backward compatibility
    ├── executors.py                  # ReasonerExecutor, JudgeExecutor, ToolExecutor
    ├── exceptions.py                 # Custom exception hierarchy
    ├── logging.py                    # Centralized logging configuration
    ├── plugins.py                    # ReflectAndRetryToolPlugin (error recovery)
    ├── tools.py                      # Tool registry class
    ├── trace.py                      # Trace-print helpers (candidates + judge decision)
    ├── _internal/                    # Private implementation details (do not import directly)
    │   ├── __init__.py
    │   ├── llm_io.py                 # LangChain invocation + JSON parsing helpers
    │   ├── normalizers.py            # Normalizers for model output
    │   ├── prompts.py                # Reasoner/judge prompt builders
    │   ├── schema.py                 # Structured output schemas for LangChain
    │   ├── utils.py                  # Common utilities
    │   └── validation.py             # Lightweight decision + tool-arg validation
    ├── providers/                    # LLM provider helpers (optional)
    │   ├── __init__.py
    │   └── vertex.py                 # Helper to create LangChain ChatGoogleGenerativeAI model instances
    └── integrations/                 # Optional integrations
        ├── __init__.py
        └── a2a.py                    # A2A wrapper and FastAPI integration
```

---

### Prerequisites

- Python 3.10+ (recommended)
- Google Cloud CLI installed (`gcloud`)
- Access to Vertex AI and Gemini models in your GCP project

---

### Setup (recommended)

Create and activate a venv:

```bash
cd "<project-root>"
python3 -m venv .venv
source .venv/bin/activate
```

Install the package with Vertex AI support:

```bash
python -m pip install --upgrade pip
pip install -e ".[vertex]"
```

Authenticate via ADC (no API keys):

```bash
gcloud auth application-default login
```

---

### Configure with `.env` (auto-loaded)

The example scripts call `python-dotenv`'s `load_dotenv()`, so a root `.env` is loaded automatically.

Create `.env` by copying `env.example`:

```bash
cp env.example .env
```

Minimal required variables:

- `VERTEX_PROJECT_ID`
- `VERTEX_LOCATION` (optional, default is `us-central1`)
- `VERTEX_MODEL` (optional, fallback default is `gemini-2.0-flash-001`)

Example:

```bash
VERTEX_PROJECT_ID="my-project"
VERTEX_LOCATION="us-central1"
VERTEX_MODEL="gemini-2.5-flash"
```

Optional (use different models for reasoner vs judge):

- `REASONER_MODEL_NAME` (fallbacks to `VERTEX_MODEL`)
- `JUDGE_MODEL_NAME` (fallbacks to `VERTEX_MODEL`)

Example:

```bash
REASONER_MODEL_NAME="gemini-2.5-flash"
JUDGE_MODEL_NAME="gemini-2.5-pro"
```

---

### Configuration Reference

All configuration options with their defaults:

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `VERTEX_PROJECT_ID` | (required) | Your GCP project ID |
| `VERTEX_LOCATION` | `us-central1` | Vertex AI region |
| `VERTEX_MODEL` | `gemini-2.0-flash-001` | Default Gemini model |
| `REASONER_MODEL_NAME` | `VERTEX_MODEL` | Model for reasoners |
| `REASONER_TEMPERATURE` | `0.7` | Reasoner sampling temperature |
| `REASONER_MAX_TOKENS` | (provider default) | Max output tokens for reasoners |
| `JUDGE_MODEL_NAME` | `VERTEX_MODEL` | Model for the judge |
| `JUDGE_TEMPERATURE` | `0.0` | Judge sampling temperature |
| `JUDGE_MAX_TOKENS` | (provider default) | Max output tokens for judge |
| `SELECTION_STRATEGY` | `select_one` | `select_one` or `synthesize_one` |
| `ALLOW_TOOL_SYNTHESIS` | `true` | Allow judge to synthesize tool calls |
| `TRACE` | `true` | Enable trace output |
| `TOOL_RESULT_MAX_CHARS` | `400` | Max chars for tool result display |
| `TRUNCATE_AGENT_OBSERVATIONS` | `false` | Truncate observations sent to LLM |
| `LLM_TIMEOUT_SECONDS` | `30.0` | Timeout for parallel reasoner calls |
| `USE_STRUCTURED_OUTPUT` | `true` | Use LangChain structured output |

---

### Exception Hierarchy

The project uses a clear exception hierarchy for specific error handling:

```
USCAgentError (base)
├── ConfigurationError          # Invalid configuration
├── LLMError                    # Base for LLM-related errors
│   ├── StructuredOutputError   # Structured output parsing failed
│   ├── JSONParseError          # Failed to parse LLM output as JSON
│   └── LLMTimeoutError         # LLM call timed out
├── ValidationError             # Base for validation errors
│   ├── DecisionValidationError # Invalid reasoner/judge decision
│   └── ToolArgsValidationError # Invalid tool arguments
├── ToolError                   # Base for tool-related errors
│   ├── UnknownToolError        # Tool not found in registry
│   ├── ToolExecutionError      # Tool execution failed
│   └── ToolReflectionError     # Reflection mechanism failed
└── AgentLoopError              # Agent loop failed
    ├── MaxStepsExceededError   # Exceeded maximum steps
    └── NoValidCandidatesError  # No valid reasoner candidates
```

---

### Logging System

The project uses structured logging with contextual information:

```python
from react_usc import (
    configure_logging,
    LoggingConfig,
    LogContext,
    get_logger,
    AgentLogger,
)
import logging

# Configure logging at startup
configure_logging(LoggingConfig(
    level=logging.INFO,
    enable_trace=True,
    log_structured_output=True,
))

# Use context managers for structured context
with LogContext(trace_id="req-123", phase="reasoner", step=1):
    logger.info("Processing step", extra={"k_paths": 4})
```

Key features:
- Thread-local context via `LogContext`
- Automatic trace ID generation for request correlation
- Semantic logging methods in `AgentLogger`
- Configurable formatters with context inclusion

---

### Resilience: Reflect and Retry Plugin

The `ReflectAndRetryToolPlugin` (in `src/react_usc/plugins.py`) acts as a safety layer around tool execution. It intercepts errors and uses an LLM to decide on a recovery strategy:

1. **RETRY (Fix)**: If the error is due to bad arguments (e.g., missing keys), the model generates fixed arguments, and the tool is retried immediately.
2. **WAIT (Transient)**: If the error is transient (e.g., `503 Service Unavailable`, network timeout), the plugin waits (with exponential backoff) and retries.
3. **ABORT (Fold)**: If the error is fatal (e.g., `403 Forbidden`, wrong tool), the plugin aborts and returns a helpful error message to the agent's reasoning loop.

**Usage:**

```python
from react_usc import ReflectAndRetryToolPlugin, LangGraphReActUSCAgent

reflection_plugin = ReflectAndRetryToolPlugin(
    model=reasoner_model,       # Model used for reflection
    max_retries=3,              # Max retry attempts per tool call
    backoff_seconds=1.0,        # Base wait time for transient errors
    trace=True,                 # Log reflection steps
    llm_retry_config=config.llm_retry,  # Retry config for LLM calls
)

agent = LangGraphReActUSCAgent(
    ...,
    plugins=[reflection_plugin]
)
```

**Writing Tools for Reflection:**

To maximize the effectiveness of the reflection plugin, write tools that raise **descriptive exceptions**.

* **Good**: `raise ValueError("Missing required parameter 'user_id'.")` -> Model sees this and adds `user_id`.
* **Good**: `raise RuntimeError("503 Service Unavailable")` -> Model sees this and chooses `WAIT`.
* **Good**: `raise PermissionError("403 Forbidden: Missing scope 'admin'")` -> Model sees this and chooses `ABORT`.
* **Bad**: `raise Exception("Error")` -> Model has no context to fix it.

---

### Run

**Demo Mode:**

```bash
python -m examples.cli_demo
```

The demo runs 5 scenarios:
1. **Math calculation** - Using the calculator tool
2. **Search query** - Using the simple_search tool
3. **RETRY scenario** - API client with missing parameter (fixed by reflection)
4. **WAIT scenario** - API client with transient 503 errors (retried after backoff)
5. **ABORT scenario** - API client with fatal 403 forbidden (gracefully aborted)

You should see trace logs for each step:

- candidate list from K reasoners
- judge selection + short justification
- tool execution inputs/outputs
- reflection decisions (for error recovery)
- final answer

**A2A Server Mode:**

To expose the agent via an A2A-compliant HTTP API, install the optional server dependencies:

```bash
pip install fastapi uvicorn
```

Then run:

```bash
python -m examples.a2a_server
```

The agent card will be available at `http://localhost:8000/.well-known/a2a.json`.
You can post tasks to `http://localhost:8000/tasks`.

---

### Testing

Run the test suite with pytest:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/react_usc

# Run specific test file
pytest tests/test_executors.py -v
```

---

### Result (what you should see)

When everything is working, you should see an end-to-end trace like:

- **Step 1**: prints **K reasoner candidates**
- **Judge**: selects one decision (or synthesizes one)
- **Tool call**: executes **exactly one** tool with JSON args and prints the tool result
- **Next steps**: repeat until the judge returns `FINAL`

Example (more explicit):

```text
=== Demo 1: math ===

Step 1: reasoner candidates (K=4)
  Valid candidates:
   [0] TOOL_CALL tool=calculator args={"expression": "2+2*10"} | rationale=The user is asking for a mathematical computation. The calculator tool directly evaluates the expression.
   [1] TOOL_CALL tool=calculator args={"expression": "2+2*10"} | rationale=Using the calculator avoids arithmetic mistakes and produces a precise result.
   [2] TOOL_CALL tool=calculator args={"expression": "2+2*10"} | rationale=The best next step is to compute the value, and the calculator tool is designed for that.
   [3] TOOL_CALL tool=calculator args={"expression": "2+2*10"} | rationale=Compute the expression via the calculator tool, then answer with the observed result.
Step 1: judge => TOOL_CALL calculator (selected_index=0) because: All candidates agree on the same correct tool call; selecting one is sufficient and minimal.
  Tool call: calculator args={"expression": "2+2*10"}
  Tool result: calculator => 22

Step 2: reasoner candidates (K=4)
  Valid candidates:
   [0] FINAL | rationale=We already computed the expression using the calculator, so we can answer now.
   [1] FINAL | rationale=The observation contains the numeric result needed to respond to the user.
   [2] FINAL | rationale=No more tools are necessary; the tool output directly answers the question.
   [3] FINAL | rationale=Return the computed value with a short explanation.
Step 2: judge => FINAL (selected_index=2) because: The observation is sufficient and this candidate is concise and correct.

FINAL ANSWER: 2 + 2 * 10 = 22
```

Reflection example (RETRY scenario):

```text
=== Demo 3: API Client - RETRY (Arg Fix) ===

Step 1: judge => TOOL_CALL api_client ...
  Tool call: api_client args={"endpoint": "/api/v1/users/123", "method": "GET"}
  [TRACE] Error caught: 400 Bad Request: Missing required query parameter 'include_profile'. Reflecting...
  [TRACE] Reflection output: {"verdict": "RETRY", "retry_args": {"params": {"include_profile": true}}, ...}
  [TRACE] Reflection RETRY with merged args: {"endpoint": "/api/v1/users/123", "method": "GET", "params": {"include_profile": true}}
  Tool result: api_client => {"status": 200, "data": {"id": "123", "name": "Alice", "profile": "active"}}

FINAL ANSWER: User 123 is Alice with an active profile.
```

---

### Configuration knobs (agent behavior)

All knobs are read from environment and passed into `AgentConfig`:

- **USC paths**: `k_paths`
  - controls how many parallel reasoner candidates are generated per step
- **Loop bounds**: `max_steps`, `timeout_seconds`
  - step limit and how long to wait for parallel reasoners
- **Two-model setup**: `reasoner_model` and `judge_model`
  - `ModelConfig(name, temperature, max_tokens)`
  - `max_tokens` may be empty/unset to use provider default
- **Decision strategy**:
  - `selection_strategy`: `"select_one"` or `"synthesize_one"`
  - `allow_tool_synthesis`: whether the judge may propose a tool call not present among candidates
- **Retry configuration**:
  - `llm_retry`: `RetryConfig(max_retries, backoff_seconds)` for LLM call retries
- **Observation handling**:
  - `tool_result_max_chars`: truncates tool output in trace logs
  - `truncate_agent_observations`: whether to truncate observations sent to LLM
- **Trace/logging**:
  - `trace`: controls console trace output
  - `log_structured_output`: log structured output attempts and fallbacks
- **Output behavior**:
  - `use_structured_output`: use LangChain structured output when available
  - `accept_non_json_final`: salvage FINAL answer from non-JSON output (best-effort)

Most values can be set via `.env` using the keys in `env.example`.

---

### Tools: how tool calling works here

Tools are defined as `ToolSpec`:

- `name`
- `description`
- `input_schema` (JSON-schema-like subset: `type`, `required`, `properties`)
- `func(args: dict) -> Any`

Tool usage is entirely driven by **structured model output**:

1. Reasoners propose a `TOOL_CALL` decision with `tool_name` + `tool_args`.
2. The judge selects/synthesizes one decision.
3. The agent validates tool args (required keys + basic type checks).
4. The agent runs **exactly one** tool call and records its output as an observation.

Example tools included (in `examples/tools/`):

- `calculator`: safe arithmetic via AST parsing
- `simple_search`: tiny in-memory lookup for demo purposes
- `api_client` (flaky tool): simulates HTTP API with various failure modes for testing retry

---

### Adding a new tool (quick guide)

Create a `ToolSpec` in your own module:

```python
from react_usc import ToolSpec

def make_my_tool() -> ToolSpec:
    def my_func(args: dict) -> Any:
        # Your tool logic here
        return {"result": args.get("input")}
    
    return ToolSpec(
        name="my_tool",
        description="Does something useful",
        input_schema={
            "type": "object",
            "required": ["input"],
            "properties": {
                "input": {"type": "string", "description": "The input value"},
            },
        },
        func=my_func,
    )
```

Then pass it to your agent:

```python
from react_usc import LangGraphReActUSCAgent, LangGraphModels, AgentConfig

agent = LangGraphReActUSCAgent(
    models=LangGraphModels(reasoner=chat_model, judge=chat_model),
    tools=[make_my_tool()],
    config=AgentConfig.default(),
)
```

For reference implementations, see the `examples/tools/` directory.

Keep schemas minimal — the validator is intentionally lightweight.

---

### Troubleshooting

- **`VERTEX_PROJECT_ID is required`**
  - Add it to `.env` or export it in your shell.

- **Auth errors / 401 / permission denied**
  - Run `gcloud auth application-default login`
  - Ensure the account has access to Vertex AI in the project.

- **Dependency import errors (`langchain_google_genai` / `langgraph`)**
  - Run `pip install -e ".[vertex]"` inside your venv.

- **All reasoners timing out**
  - Increase `LLM_TIMEOUT_SECONDS` (Vertex AI can be slow sometimes)
  - Check your GCP quota limits

- **Structured output failures**
  - The agent automatically falls back to text JSON parsing
  - Enable `USC_LOG_STRUCTURED_OUTPUT=true` for debugging

---

### Notes on JSON-only outputs

This implementation **expects** reasoner and judge to return **JSON objects** (no markdown).
Prompts explicitly instruct "JSON ONLY", and the Vertex configuration in LangChain should be set so outputs are reliably parseable.

The agent handles JSON parsing failures gracefully:
1. **Structured output** is tried first (when `use_structured_output=true`)
2. **Text parsing** is used as fallback
3. **Non-JSON salvaging** can recover FINAL answers from malformed output (when `accept_non_json_final=true`)

---

### Version History

**v0.4.0** (Current)
- Major library reorganization for better modularity:
  - Created `pyproject.toml` for modern Python packaging with optional dependencies
  - Split `models.py` into `types.py`, `config.py`, and `decisions.py`
  - Created `_internal/` subpackage for private implementation details
  - Created `providers/` subpackage for LLM provider helpers (Vertex AI)
  - Created `integrations/` subpackage for optional integrations (A2A)
  - Renamed `lc_agent.py` to `agent.py` and `logging_config.py` to `logging.py`
- Optional dependencies: `pip install react-usc[vertex]`, `pip install react-usc[a2a]`
- Backward compatible: `models.py` re-exports all types for existing code

**v0.3.0**
- Refactored to library structure: core library in `src/react_usc/`, examples in `examples/`
- Moved example tools (calculator, search, flaky_api) to `examples/tools/`
- Moved demo scripts (cli_demo, a2a_server) to `examples/`
- Cleaned up public API to export only core components
- Example tools are no longer part of the public API (import from `examples.tools` instead)

**v0.2.0**
- Refactored to executor-based architecture (ReasonerExecutor, JudgeExecutor, ToolExecutor)
- Added custom exception hierarchy for specific error handling
- Implemented centralized logging with context management
- Added test tools for retry/reflection scenarios
- Improved configuration with validation and factory methods
- Added truncation control for agent observations
- Enhanced reflection plugin with non-JSON salvaging
- Added test suite with pytest

**v0.1.0**
- Initial release with ReAct + USC implementation
- Basic tool support (calculator, simple_search)
- Reflect and Retry plugin
- A2A server support
