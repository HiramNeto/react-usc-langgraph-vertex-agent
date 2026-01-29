"""
Tool registry for the ReAct USC Agent.

This module provides the ToolRegistry class for managing available tools.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from .types import ToolSpec


class ToolRegistry:
    """
    Registry for managing available tools.
    
    Provides lookup and enumeration of tools by name.
    
    Example:
        >>> from react_usc import ToolSpec, ToolRegistry
        >>> my_tool = ToolSpec(name="my_tool", description="...", input_schema={...}, func=my_func)
        >>> registry = ToolRegistry([my_tool])
        >>> registry.get("my_tool")
        ToolSpec(name='my_tool', ...)
    """
    
    def __init__(self, tools: Sequence[ToolSpec]) -> None:
        """
        Initialize the registry with a sequence of tools.
        
        Args:
            tools: Sequence of ToolSpec instances to register
        """
        self._tools: Dict[str, ToolSpec] = {t.name: t for t in tools}

    def get(self, name: str) -> Optional[ToolSpec]:
        """
        Get a tool by name.
        
        Args:
            name: The tool name to look up
            
        Returns:
            The ToolSpec if found, None otherwise
        """
        return self._tools.get(name)

    def all(self) -> List[ToolSpec]:
        """
        Get all registered tools.
        
        Returns:
            List of all ToolSpec instances
        """
        return list(self._tools.values())
