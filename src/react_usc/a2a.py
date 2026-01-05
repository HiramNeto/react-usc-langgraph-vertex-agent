"""
Optional A2A (Agent-to-Agent) wrapper layer.

This module implements a basic wrapper to expose the LangGraphReActUSCAgent
via a standard interface compatible with A2A protocols (like Google's A2A).

It requires `fastapi` and `uvicorn` to run the server.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Literal

from pydantic import BaseModel, Field

from .lc_agent import LangGraphReActUSCAgent


# --- A2A Schemas (Simplified) ---

class AgentCapability(BaseModel):
    name: str
    description: str
    input_schema: Dict[str, Any]


class AgentCard(BaseModel):
    """
    Metadata about the agent for discovery.
    """
    id: str
    name: str
    description: str
    version: str = "1.0.0"
    capabilities: List[AgentCapability] = Field(default_factory=list)
    endpoints: Dict[str, str] = Field(default_factory=dict)


class TaskInput(BaseModel):
    """
    Standard input for a task.
    """
    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    input_text: str
    context: Dict[str, Any] = Field(default_factory=dict)


class TaskOutput(BaseModel):
    """
    Standard output for a task.
    """
    task_id: str
    status: Literal["completed", "failed", "processing"]
    output_text: Optional[str] = None
    artifacts: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    completed_at: Optional[str] = None


# --- Wrapper Class ---

class A2AAgentWrapper:
    """
    Wraps the LangGraphReActUSCAgent to provide A2A-compliant methods.
    """

    def __init__(
        self,
        agent: LangGraphReActUSCAgent,
        agent_id: str = "react-usc-agent",
        name: str = "Universal Self-Consistency Agent",
        description: str = "An agent that uses reasoning paths and a judge to solve complex queries.",
        base_url: str = "http://localhost:8000",
    ):
        self._agent = agent
        self.agent_id = agent_id
        self.name = name
        self.description = description
        self.base_url = base_url

    def get_agent_card(self) -> AgentCard:
        """
        Returns the Agent Card describing this agent.
        """
        # Expose the single main capability: answering queries
        cap = AgentCapability(
            name="query",
            description="Process a natural language query using reasoning and tools.",
            input_schema={
                "type": "object",
                "properties": {
                    "input_text": {"type": "string", "description": "The user query."}
                },
                "required": ["input_text"],
            },
        )
        
        return AgentCard(
            id=self.agent_id,
            name=self.name,
            description=self.description,
            capabilities=[cap],
            endpoints={
                "card": f"{self.base_url}/.well-known/a2a.json",
                "tasks": f"{self.base_url}/tasks",
            },
        )

    def execute_task(self, task_input: TaskInput) -> TaskOutput:
        """
        Synchronously executes a task.
        In a production A2A implementation, this might push to a queue and return a 'processing' status.
        Here we block for simplicity as the underlying agent is synchronous.
        """
        try:
            # Run the underlying agent
            result = self._agent.run(task_input.input_text)
            
            return TaskOutput(
                task_id=task_input.task_id,
                status="completed",
                output_text=result,
                completed_at=datetime.utcnow().isoformat(),
            )
        except Exception as e:
            return TaskOutput(
                task_id=task_input.task_id,
                status="failed",
                error=str(e),
                completed_at=datetime.utcnow().isoformat(),
            )


# --- FastAPI Integration ---

def create_a2a_app(wrapper: A2AAgentWrapper) -> Any:
    """
    Creates a FastAPI app to serve the agent via A2A protocols.
    Requires `fastapi` and `uvicorn`.
    """
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.responses import JSONResponse
    except ImportError:
        raise RuntimeError("FastAPI is required for A2A server. Install it with: pip install fastapi uvicorn")

    app = FastAPI(title=wrapper.name, description=wrapper.description, version="1.0.0")

    @app.get("/.well-known/a2a.json", response_model=AgentCard)
    async def get_agent_card():
        return wrapper.get_agent_card()

    @app.post("/tasks", response_model=TaskOutput)
    async def create_task(task: TaskInput):
        # Note: This is blocking. For high load, offload to a background task/worker.
        return wrapper.execute_task(task)

    @app.get("/health")
    async def health_check():
        return {"status": "ok"}

    return app

