from __future__ import annotations

import json
import re
from typing import Any, Optional, Sequence


def truncate(s: str, max_chars: int) -> str:
    if max_chars <= 0 or len(s) <= max_chars:
        return s
    # Keep suffix info for debugging.
    return s[: max(0, max_chars - 24)] + f"... [truncated {len(s)} chars]"


def build_state_summary(*, observations: Sequence[str], step_index: int, max_steps: int) -> str:
    """
    Format the agent's current loop state as a compact, stable text block.
    Kept separate from prompt construction so it can be reused across prompting strategies.
    """
    obs_lines = "\n".join([f"- {o}" for o in observations[-10:]]) if observations else "- (none)"
    return "\n".join(
        [
            f"step: {step_index}/{max_steps}",
            "observations (most recent last):",
            obs_lines,
        ]
    )


def safe_json_dumps(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False, sort_keys=True)
    except Exception:
        return repr(obj)


def simple_word_hits(query: str, key: str) -> int:
    q_tokens = {t for t in re.findall(r"[a-z]+", query.lower()) if len(t) >= 3}
    k_tokens = set(re.findall(r"[a-z]+", key.lower()))
    return len(q_tokens & k_tokens)


