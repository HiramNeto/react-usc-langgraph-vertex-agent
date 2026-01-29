"""
Simple search tool example for the ReAct USC Agent.

This module provides an in-memory knowledge base search tool
that can be used as a tool by the agent.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple, cast

from react_usc import ToolSpec
from react_usc._internal.utils import simple_word_hits


def make_simple_search_tool() -> ToolSpec:
    """
    Create a tiny in-memory search tool (demo).
    
    This tool searches a small built-in knowledge base about
    AI agent concepts like ReAct, self-consistency, and tool calling.
    
    Returns:
        ToolSpec for the simple search tool
        
    Example:
        >>> tool = make_simple_search_tool()
        >>> tool.name
        'simple_search'
    """

    corpus: Dict[str, str] = {
        "react": "ReAct interleaves reasoning with tool use, producing observations that feed back into the loop.",
        "self-consistency": "Self-consistency samples multiple reasoning paths and picks the most consistent result.",
        "usc": "Universal Self-Consistency: sample K candidate next steps, then pick/synthesize ONE action to execute.",
        "tool calling": "Tool calling uses structured function invocation (name + JSON args) instead of parsing freeform text.",
    }

    def _search(args: Dict[str, Any]) -> Any:
        q = cast(str, args["query"]).lower()
        hits: List[Dict[str, str]] = []

        # Direct substring / token heuristics.
        for k, v in corpus.items():
            if k in q or any(tok in k for tok in q.split()):
                hits.append({"key": k, "value": v})

        if not hits:
            scored: List[Tuple[int, str]] = [(simple_word_hits(q, k), k) for k in corpus]
            scored.sort(reverse=True)
            for _, key in scored[:2]:
                hits.append({"key": key, "value": corpus[key]})

        return {"query": q, "matches": hits}

    return ToolSpec(
        name="simple_search",
        description="Search a tiny in-memory knowledge base and return matching snippets.",
        input_schema={
            "type": "object",
            "required": ["query"],
            "properties": {"query": {"type": "string"}},
        },
        func=_search,
    )
