### ReAct + Universal Self-Consistency (USC) Agent (LangChain + LangGraph, Vertex AI Gemini)

This project is a clean Python POC of a **ReAct-style agent loop** where each step samples **K parallel “reasoner” decisions** (Universal Self-Consistency), then a **judge model** picks (or synthesizes) the **single best next decision**, and **only that single decision is executed** as a tool call before continuing.

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
- **Tracing**: prints candidates, judge choice, tool I/O, and final answers

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
  - Append the tool output as an **observation** (truncated).
  - Loop again with updated state.

If the step limit is reached, the agent requests a **best-effort final answer**.

---

### How LangGraph is used (control flow)

LangGraph expresses the loop as a small state machine:

- **State** (`_State` in `src/react_usc/lc_agent.py`):
  - `user_query`: original query (constant)
  - `observations`: list of tool results / errors
  - `step`: step counter
  - `judge`: last `JudgeDecision` (used for routing)

- **Nodes**:
  - `reason_and_judge`:
    - runs K parallel reasoners
    - validates candidates
    - runs the judge
    - stores the judge decision in state
  - `execute_tool`:
    - executes **only** the judged tool call
    - appends an observation

- **Edges / routing**:
  - `START -> reason_and_judge`
  - If judge decides `TOOL_CALL` ⇒ `reason_and_judge -> execute_tool -> reason_and_judge`
  - If judge decides `FINAL` ⇒ `reason_and_judge -> END`

This keeps the loop readable: the “graph wiring” is separated from tool execution and prompt construction.

---

### Project layout

- `main.py`: demo runner (loads `.env`, builds tools/config, runs the agent)
- `serve_agent.py`: A2A server runner (exposes agent via HTTP)
- `src/react_usc/lc_agent.py`: **LangGraphReActUSCAgent** (USC fan-out + judge + single tool execution)
- `src/react_usc/a2a.py`: Optional A2A wrapper and FastAPI integration
- `src/react_usc/lc_vertex.py`: helper to create **LangChain ChatVertexAI** model instances
- `src/react_usc/models.py`: typed dataclasses (`AgentConfig`, `ModelConfig`, decisions, tools)
- `src/react_usc/prompts.py`: reasoner/judge prompt builders
- `src/react_usc/llm_io.py`: LangChain invocation helpers + robust JSON parsing helpers
- `src/react_usc/decision_normalize.py`: normalizers for model output (fix common deviations before validation)
- `src/react_usc/trace.py`: trace-print helpers (candidates + judge decision)
- `src/react_usc/schema.py`: structured output schemas (Pydantic) for LangChain `with_structured_output`
- `src/react_usc/validation.py`: lightweight decision + tool-arg validation
- `src/react_usc/tools.py`: tool registry + example tools (`calculator`, `simple_search`)
- `env.example`: env var template (copy to `.env`)

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

Install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Authenticate via ADC (no API keys):

```bash
gcloud auth application-default login
```

---

### Configure with `.env` (auto-loaded)

`main.py` calls `python-dotenv`’s `load_dotenv()`, so a root `.env` is loaded automatically.

Create `.env` by copying `env.example`:

```bash
cp env.example .env
```

Minimal required variables:

- `VERTEX_PROJECT_ID`
- `VERTEX_LOCATION` (optional, default in `main.py` is `us-central1`)
- `VERTEX_MODEL` (optional, fallback default in `main.py` is `gemini-2.0-flash-001`)

Example:

```bash
VERTEX_PROJECT_ID="my-project"
VERTEX_LOCATION="us-central1"
VERTEX_MODEL="gemini-2.0-flash-001"
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

### Resilience: Reflect and Retry Plugin

This project includes a powerful `ReflectAndRetryToolPlugin` (in `src/react_usc/plugins.py`) that acts as a safety layer around tool execution. It intercepts errors and uses an LLM to decide on a recovery strategy:

1.  **RETRY (Fix)**: If the error is due to bad arguments (e.g., missing keys), the model generates fixed arguments, and the tool is retried immediately.
2.  **WAIT (Transient)**: If the error is transient (e.g., `503 Service Unavailable`, network timeout), the plugin waits (with exponential backoff) and retries.
3.  **ABORT (Fold)**: If the error is fatal (e.g., `403 Forbidden`, wrong tool), the plugin aborts and returns a helpful error message to the agent's reasoning loop.

**Usage in `main.py`:**

```python
from src.react_usc.plugins import ReflectAndRetryToolPlugin

reflection_plugin = ReflectAndRetryToolPlugin(
    model=reasoner_model,  # Model used for reflection
    max_retries=3,         # Max retry attempts per tool call
    backoff_seconds=1.0,   # Base wait time for transient errors
    trace=True             # Log reflection steps
)

agent = LangGraphReActUSCAgent(
    ...,
    plugins=[reflection_plugin]
)
```

**Writing Tools for Reflection:**

To maximize the effectiveness of the reflection plugin, write tools that raise **descriptive exceptions**.

*   **Good**: `raise ValueError("Missing required parameter 'user_id'.")` -> Model sees this and adds `user_id`.
*   **Good**: `raise RuntimeError("503 Service Unavailable")` -> Model sees this and chooses `WAIT`.
*   **Good**: `raise PermissionError("403 Forbidden: Missing scope 'admin'")` -> Model sees this and chooses `ABORT`.
*   **Bad**: `raise Exception("Error")` -> Model has no context to fix it.

---

### Run

**Demo Mode:**

```bash
python main.py
```

You should see trace logs for each step:

- candidate list from K reasoners
- judge selection + short justification
- tool execution inputs/outputs
- final answer

**A2A Server Mode:**

To expose the agent via an A2A-compliant HTTP API, install the optional server dependencies:

```bash
pip install fastapi uvicorn
```

Then run:

```bash
python serve_agent.py
```

The agent card will be available at `http://localhost:8000/.well-known/a2a.json`.
You can post tasks to `http://localhost:8000/tasks`.

---

### Result (what you should see)

When everything is working, you should see an end-to-end trace like:

- **Step 1**: prints **K reasoner candidates**
- **Judge**: selects one decision (or synthesizes one)
- **Tool call**: executes **exactly one** tool with JSON args and prints the tool result
- **Next steps**: repeat until the judge returns `FINAL`

Example (more explicit):

```text
=== Demo: math ===

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

---

### Configuration knobs (agent behavior)

All knobs are read in `main.py` and passed into `AgentConfig`:

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
- **Trace/logging**:
  - `trace` controls console logging
  - `tool_result_max_chars` truncates tool output in observations/logs

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

Example tools included:

- `calculator`: safe arithmetic via AST parsing
- `simple_search`: tiny in-memory lookup for demo purposes

---

### Adding a new tool (quick guide)

In `src/react_usc/tools.py` (or a new module):

1. Create a `ToolSpec` with a small JSON schema.
2. Add it to the tool list in `main.py`.

Keep schemas minimal — the validator is intentionally lightweight.

---

### Troubleshooting

- **`VERTEX_PROJECT_ID is required`**
  - Add it to `.env` or export it in your shell.

- **Auth errors / 401 / permission denied**
  - Run `gcloud auth application-default login`
  - Ensure the account has access to Vertex AI in the project.

- **Dependency import errors (`langchain_google_vertexai` / `langgraph`)**
  - Run `python -m pip install -r requirements.txt` inside your venv.

---

### Notes on JSON-only outputs

This implementation **expects** reasoner and judge to return **JSON objects** (no markdown).
Prompts explicitly instruct “JSON ONLY”, and the Vertex configuration in LangChain should be set so outputs are reliably parseable.

If you see parsing errors, tighten prompts or add an output parser (LangChain) that retries once on invalid JSON.
