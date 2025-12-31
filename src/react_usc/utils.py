from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple


def truncate(s: str, max_chars: int) -> str:
    if len(s) <= max_chars:
        return s
    # Keep suffix info for debugging.
    return s[: max(0, max_chars - 24)] + f"... [truncated {len(s)} chars]"


def safe_json_dumps(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False, sort_keys=True)
    except Exception:
        return repr(obj)


def extract_between(text: str, start: str, end: str) -> str:
    s = text.find(start)
    if s < 0:
        return ""
    s += len(start)
    e = text.find(end, s)
    if e < 0:
        return text[s:].strip()
    return text[s:e].strip()


def extract_candidates_json(user: str) -> List[Dict[str, Any]]:
    # The judge prompt includes "CANDIDATES:\n<json>\n\nRUBRIC..."
    chunk = extract_between(user, "CANDIDATES:", "RUBRIC").strip()
    if not chunk:
        return []
    try:
        data = json.loads(chunk)
        if isinstance(data, list):
            return [c for c in data if isinstance(c, dict)]
    except Exception:
        return []
    return []


def last_observation_value(state_summary: str, *, tool_name: str) -> str:
    # State summary includes lines like "- calculator => ..."
    lines = [ln.strip() for ln in state_summary.splitlines()]
    matches = [ln for ln in lines if ln.startswith("-") and f"{tool_name} =>" in ln]
    if not matches:
        return "(no observation)"
    last = matches[-1]
    _, _, rhs = last.partition(f"{tool_name} =>")
    return rhs.strip()


def extract_expression_like(q: str) -> Optional[str]:
    """
    Very lightweight heuristic for the stub: find a substring that looks like arithmetic.
    """
    m = re.search(r"([-+*/().\s\d]{3,})", q)
    if not m:
        return None
    expr = m.group(1).strip()[:80]
    if not re.search(r"[\+\-\*/]", expr):
        return None
    return expr


def simple_word_hits(query: str, key: str) -> int:
    q_tokens = {t for t in re.findall(r"[a-z]+", query.lower()) if len(t) >= 3}
    k_tokens = set(re.findall(r"[a-z]+", key.lower()))
    return len(q_tokens & k_tokens)


