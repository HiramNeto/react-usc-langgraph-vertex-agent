"""
Example tool implementations for the ReAct USC Agent.

This module provides sample tools that demonstrate how to create
tools using the ToolSpec interface.

Available Tools:
    - make_calculator_tool(): Safe arithmetic expression evaluator
    - make_simple_search_tool(): In-memory knowledge base search
    - make_flaky_tool(): Simulated API client with failure modes (for testing)
"""
from .calculator import SafeCalculator, make_calculator_tool
from .search import make_simple_search_tool
from .flaky_api import make_flaky_tool

__all__ = [
    "SafeCalculator",
    "make_calculator_tool",
    "make_simple_search_tool",
    "make_flaky_tool",
]
