from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional, Sequence

from .llm_io import invoke_chat_structured_obj, invoke_chat_text, json_loads_object
from .models import ToolSpec
from .prompts import build_reflection_prompt
from .schema import REFLECTION_DECISION_SCHEMA
from .utils import safe_json_dumps
from .validation import validate_json_obj


class ReflectAndRetryToolPlugin:
    def __init__(self, model: Any, max_retries: int = 3, backoff_seconds: float = 1.0, trace: bool = False):
        self.model = model
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds
        self.trace = trace

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
        Execute tool with retry loop and reflection logic.
        """
        current_args = tool_args
        
        for attempt in range(self.max_retries + 1):
            try:
                if self.trace and attempt > 0:
                    print(f"    [Retry {attempt}] Executing {tool_name} with args={safe_json_dumps(current_args)}")
                
                # Try execution
                return tool_func(current_args)
            
            except Exception as e:
                # If we've exhausted retries, raise the last exception
                if attempt == self.max_retries:
                    if self.trace:
                        print(f"    [Retry] Exhausted {self.max_retries} retries. Raising exception: {e}")
                    raise e
                
                # Reflection Step
                if self.trace:
                    print(f"    [Retry] Error caught: {e}. Reflecting...")

                decision = self._reflect(
                    user_query=user_query,
                    tool_name=tool_name,
                    tool_args=current_args,
                    error=str(e),
                    tools=all_tools,
                )
                
                verdict = decision.get("verdict")

                if verdict == "ABORT":
                    suggestion = decision.get("abort_suggestion", "Tool execution aborted by reflection.")
                    if self.trace:
                         print(f"    [Retry] ABORT verdict. Suggestion: {suggestion}")
                    # Return a special string observation that the agent will see
                    return f"Reflection Error: The tool '{tool_name}' failed and reflection decided to abort. Suggestion: {suggestion}"

                if verdict == "WAIT":
                    # Exponential backoff: backoff * 2^(attempt)
                    wait_time = self.backoff_seconds * (2 ** attempt)
                    if self.trace:
                         print(f"    [Retry] WAIT verdict. Sleeping {wait_time:.2f}s before retrying same args...")
                    time.sleep(wait_time)
                    # continue loop with same current_args
                    continue

                if verdict == "RETRY":
                    new_args = decision.get("retry_args")
                    # Validate the new args against schema
                    arg_errors = validate_json_obj(new_args or {}, tool_input_schema)
                    if arg_errors:
                         if self.trace:
                            print(f"    [Retry] Reflection produced invalid args: {arg_errors}")
                         # If reflection failed to produce valid args, we might as well stop or try again?
                         # For now, let's treat it as a failed retry attempt and let the loop continue (or fail next execution)
                         pass 
                    else:
                        current_args = new_args
                        if self.trace:
                             print(f"    [Retry] Retrying with new args: {safe_json_dumps(current_args)}")
                        continue
                
                # If we get here (unknown verdict or other issue), just raise
                raise e

    def _reflect(
        self,
        *,
        user_query: str,
        tool_name: str,
        tool_args: Dict[str, Any],
        error: str,
        tools: Sequence[ToolSpec],
    ) -> Dict[str, Any]:
        system, user = build_reflection_prompt(
            user_query=user_query,
            tool_name=tool_name,
            tool_args=tool_args,
            error=error,
            tools=tools,
        )

        try:
            # Try structured output first
            # We assume the model supports it if passed in (similar to agent logic)
            # But here we might not know if the model supports it. 
            # Ideally we check config, but plugin is standalone.
            # We'll try structured, fallback to text.
            try:
                return invoke_chat_structured_obj(
                    self.model,
                    system=system,
                    user=user,
                    schema=REFLECTION_DECISION_SCHEMA,
                )
            except Exception:
                 # Fallback
                 text = invoke_chat_text(self.model, system=system, user=user)
                 return json_loads_object(text)
        except Exception as e:
            if self.trace:
                print(f"    [Retry] Reflection model call failed: {e}")
            # If reflection fails, return ABORT so we don't infinite loop blindly
            return {"verdict": "ABORT", "abort_suggestion": f"Reflection mechanism failed: {e}"}

