"""
Calculator tool example for the ReAct USC Agent.

This module provides a safe arithmetic expression evaluator
that can be used as a tool by the agent.
"""
from __future__ import annotations

import ast
import math
from typing import Any, Dict, cast

from react_usc import ToolSpec


class SafeCalculator:
    """
    Calculator using Python AST parsing with a very small allowed subset.

    Supported:
      - literals (int/float)
      - +, -, *, /, **, %, unary +/-

    No names, calls, attributes, subscripts, etc.
    
    Example:
        >>> calc = SafeCalculator()
        >>> calc.eval("2 + 2 * 10")
        22.0
        >>> calc.eval("(1 + 2) ** 3")
        27.0
    """

    _allowed_binops = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod)
    _allowed_unaryops = (ast.UAdd, ast.USub)

    def eval(self, expression: str) -> float:
        """
        Evaluate an arithmetic expression safely.
        
        Args:
            expression: String containing the arithmetic expression
            
        Returns:
            The computed result as a float
            
        Raises:
            ValueError: If the expression contains disallowed constructs
        """
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
    """
    Create a calculator tool for safe arithmetic evaluation.
    
    Returns:
        ToolSpec for the calculator tool
        
    Example:
        >>> tool = make_calculator_tool()
        >>> tool.name
        'calculator'
    """
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
