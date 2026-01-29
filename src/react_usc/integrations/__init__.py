"""
Optional integrations for the ReAct USC Agent.

This module provides integrations with external systems and protocols.

Currently supported integrations:
- A2A (Agent-to-Agent) - requires fastapi and uvicorn

Example:
    >>> from react_usc.integrations import A2AAgentWrapper, create_a2a_app
    >>> wrapper = A2AAgentWrapper(agent=my_agent)
    >>> app = create_a2a_app(wrapper)
"""
from __future__ import annotations

from .a2a import (
    A2AAgentWrapper,
    AgentCapability,
    AgentCard,
    TaskInput,
    TaskOutput,
    create_a2a_app,
)

__all__ = [
    "A2AAgentWrapper",
    "AgentCapability",
    "AgentCard",
    "TaskInput",
    "TaskOutput",
    "create_a2a_app",
]
