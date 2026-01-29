"""
Tests for ToolRegistry.

These tests cover:
- Tool lookup by name
- Unknown tool handling
- Registry enumeration
- Empty registry handling
- Duplicate tool name behavior
"""
from __future__ import annotations

import unittest

from react_usc import ToolRegistry, ToolSpec


# =============================================================================
# Test: ToolRegistry
# =============================================================================


class TestToolRegistry(unittest.TestCase):
    """Test ToolRegistry functionality."""

    def setUp(self):
        """Create test tools."""
        self.valid_schema = {
            "type": "object",
            "required": ["query"],
            "properties": {"query": {"type": "string"}},
        }
        
        self.tool1 = ToolSpec(
            name="calculator",
            description="Performs calculations",
            input_schema=self.valid_schema,
            func=lambda args: f"calc: {args}",
        )
        
        self.tool2 = ToolSpec(
            name="search",
            description="Searches for information",
            input_schema=self.valid_schema,
            func=lambda args: f"search: {args}",
        )
        
        self.tool3 = ToolSpec(
            name="api_client",
            description="Makes API calls",
            input_schema={"type": "object"},
            func=lambda args: "api result",
        )

    def test_get_existing_tool(self):
        """Test getting an existing tool by name."""
        registry = ToolRegistry([self.tool1, self.tool2])
        
        tool = registry.get("calculator")
        
        self.assertIsNotNone(tool)
        self.assertEqual(tool.name, "calculator")
        self.assertEqual(tool.description, "Performs calculations")

    def test_get_unknown_tool_returns_none(self):
        """Test that getting an unknown tool returns None."""
        registry = ToolRegistry([self.tool1])
        
        tool = registry.get("nonexistent_tool")
        
        self.assertIsNone(tool)

    def test_get_with_empty_name(self):
        """Test getting tool with empty name returns None."""
        registry = ToolRegistry([self.tool1, self.tool2])
        
        tool = registry.get("")
        
        self.assertIsNone(tool)

    def test_all_returns_all_tools(self):
        """Test that all() returns all registered tools."""
        registry = ToolRegistry([self.tool1, self.tool2, self.tool3])
        
        tools = registry.all()
        
        self.assertEqual(len(tools), 3)
        tool_names = {t.name for t in tools}
        self.assertEqual(tool_names, {"calculator", "search", "api_client"})

    def test_all_returns_list(self):
        """Test that all() returns a list (not a view)."""
        registry = ToolRegistry([self.tool1])
        
        tools = registry.all()
        
        self.assertIsInstance(tools, list)

    def test_empty_registry(self):
        """Test behavior with an empty registry."""
        registry = ToolRegistry([])
        
        self.assertIsNone(registry.get("any_tool"))
        self.assertEqual(registry.all(), [])

    def test_single_tool_registry(self):
        """Test registry with a single tool."""
        registry = ToolRegistry([self.tool1])
        
        self.assertEqual(len(registry.all()), 1)
        self.assertIsNotNone(registry.get("calculator"))
        self.assertIsNone(registry.get("search"))

    def test_duplicate_tool_names_last_wins(self):
        """Test that duplicate tool names are handled (last one wins)."""
        duplicate_tool = ToolSpec(
            name="calculator",
            description="Different calculator",
            input_schema=self.valid_schema,
            func=lambda args: "duplicate",
        )
        
        registry = ToolRegistry([self.tool1, duplicate_tool])
        
        tool = registry.get("calculator")
        self.assertIsNotNone(tool)
        self.assertEqual(tool.description, "Different calculator")
        self.assertEqual(len(registry.all()), 1)

    def test_tool_func_is_callable_after_registration(self):
        """Test that tool functions can be called after registration."""
        registry = ToolRegistry([self.tool1])
        
        tool = registry.get("calculator")
        self.assertIsNotNone(tool)
        
        result = tool.func({"query": "test"})
        self.assertIn("calc", result)

    def test_tools_from_tuple(self):
        """Test registry can be initialized with a tuple."""
        registry = ToolRegistry((self.tool1, self.tool2))
        
        self.assertEqual(len(registry.all()), 2)

    def test_get_returns_same_tool_instance(self):
        """Test that get returns the same tool instance."""
        registry = ToolRegistry([self.tool1])
        
        tool1 = registry.get("calculator")
        tool2 = registry.get("calculator")
        
        self.assertIs(tool1, tool2)

    def test_case_sensitive_lookup(self):
        """Test that tool lookup is case-sensitive."""
        registry = ToolRegistry([self.tool1])
        
        self.assertIsNotNone(registry.get("calculator"))
        self.assertIsNone(registry.get("Calculator"))
        self.assertIsNone(registry.get("CALCULATOR"))


if __name__ == "__main__":
    unittest.main()
