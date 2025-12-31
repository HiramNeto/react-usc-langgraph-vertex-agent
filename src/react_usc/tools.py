from __future__ import annotations

import ast
import math
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple, cast

from .models import ToolSpec
from .utils import safe_json_dumps, simple_word_hits


class ToolRegistry:
    def __init__(self, tools: Sequence[ToolSpec]) -> None:
        self._tools: Dict[str, ToolSpec] = {t.name: t for t in tools}

    def get(self, name: str) -> Optional[ToolSpec]:
        return self._tools.get(name)

    def all(self) -> List[ToolSpec]:
        return list(self._tools.values())


class SafeCalculator:
    """
    Calculator using Python AST parsing with a very small allowed subset.

    Supported:
      - literals (int/float)
      - +, -, *, /, **, %, unary +/-

    No names, calls, attributes, subscripts, etc.
    """

    _allowed_binops = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod)
    _allowed_unaryops = (ast.UAdd, ast.USub)

    def eval(self, expression: str) -> float:
        node = ast.parse(expression, mode="eval")
        return float(self._eval_node(node.body))

    def _eval_node(self, node: ast.AST) -> float:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)

        if isinstance(node, ast.UnaryOp) and isinstance(node.op, self._allowed_unaryops):
            v = self._eval_node(node.operand)
            if isinstance(node.op, ast.UAdd):
                return +v
            if isinstance(node.op, ast.USub):
                return -v

        if isinstance(node, ast.BinOp) and isinstance(node.op, self._allowed_binops):
            left = self._eval_node(node.left)
            right = self._eval_node(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
            if isinstance(node.op, ast.Pow):
                return left**right
            if isinstance(node.op, ast.Mod):
                return left % right

        raise ValueError(f"Disallowed expression node: {type(node).__name__}")


def make_calculator_tool() -> ToolSpec:
    calc = SafeCalculator()

    def _calc(args: Dict[str, Any]) -> Any:
        expr = cast(str, args["expression"])
        value = calc.eval(expr)
        if math.isfinite(value) and abs(value - round(value)) < 1e-12:
            return int(round(value))
        return value

    return ToolSpec(
        name="calculator",
        description="Evaluate a simple arithmetic expression safely (+ - * / ** % and parentheses).",
        input_schema={
            "type": "object",
            "required": ["expression"],
            "properties": {"expression": {"type": "string"}},
        },
        func=_calc,
    )


def make_simple_search_tool() -> ToolSpec:
    """
    Tiny in-memory search tool (demo).
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


def tool_output_preview(value: Any, max_chars: int) -> str:
    return safe_json_dumps(value)[:max_chars]


