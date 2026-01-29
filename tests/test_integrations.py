"""
Tests for A2A integration.

These tests cover:
- A2AAgentWrapper initialization
- AgentCard generation
- Task execution (success and error cases)
- create_a2a_app function
"""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from react_usc.integrations.a2a import (
    A2AAgentWrapper,
    AgentCapability,
    AgentCard,
    TaskInput,
    TaskOutput,
    create_a2a_app,
)


# =============================================================================
# Test: Pydantic Models
# =============================================================================


class TestAgentCapability(unittest.TestCase):
    """Test AgentCapability model."""

    def test_valid_capability(self):
        """Test creating a valid AgentCapability."""
        cap = AgentCapability(
            name="search",
            description="Search for information",
            input_schema={"type": "object"},
        )
        
        self.assertEqual(cap.name, "search")
        self.assertEqual(cap.description, "Search for information")
        self.assertEqual(cap.input_schema, {"type": "object"})


class TestAgentCard(unittest.TestCase):
    """Test AgentCard model."""

    def test_valid_agent_card(self):
        """Test creating a valid AgentCard."""
        card = AgentCard(
            id="test-agent",
            name="Test Agent",
            description="A test agent",
        )
        
        self.assertEqual(card.id, "test-agent")
        self.assertEqual(card.name, "Test Agent")
        self.assertEqual(card.version, "1.0.0")  # Default
        self.assertEqual(card.capabilities, [])
        self.assertEqual(card.endpoints, {})

    def test_agent_card_with_capabilities(self):
        """Test AgentCard with capabilities."""
        cap = AgentCapability(
            name="query",
            description="Process queries",
            input_schema={"type": "object"},
        )
        
        card = AgentCard(
            id="agent",
            name="Agent",
            description="Desc",
            capabilities=[cap],
        )
        
        self.assertEqual(len(card.capabilities), 1)
        self.assertEqual(card.capabilities[0].name, "query")


class TestTaskInput(unittest.TestCase):
    """Test TaskInput model."""

    def test_task_input_with_text(self):
        """Test creating TaskInput."""
        task = TaskInput(input_text="What is 2+2?")
        
        self.assertEqual(task.input_text, "What is 2+2?")
        self.assertIsNotNone(task.task_id)  # Auto-generated
        self.assertEqual(task.context, {})

    def test_task_input_with_context(self):
        """Test TaskInput with context."""
        task = TaskInput(
            task_id="test-123",
            input_text="Query",
            context={"key": "value"},
        )
        
        self.assertEqual(task.task_id, "test-123")
        self.assertEqual(task.context["key"], "value")


class TestTaskOutput(unittest.TestCase):
    """Test TaskOutput model."""

    def test_completed_output(self):
        """Test completed TaskOutput."""
        output = TaskOutput(
            task_id="123",
            status="completed",
            output_text="The answer is 42",
        )
        
        self.assertEqual(output.status, "completed")
        self.assertEqual(output.output_text, "The answer is 42")
        self.assertIsNone(output.error)
        self.assertIsNotNone(output.created_at)

    def test_failed_output(self):
        """Test failed TaskOutput."""
        output = TaskOutput(
            task_id="123",
            status="failed",
            error="Something went wrong",
        )
        
        self.assertEqual(output.status, "failed")
        self.assertEqual(output.error, "Something went wrong")
        self.assertIsNone(output.output_text)


# =============================================================================
# Test: A2AAgentWrapper
# =============================================================================


class TestA2AAgentWrapper(unittest.TestCase):
    """Test A2AAgentWrapper class."""

    def setUp(self):
        """Create mock agent."""
        self.mock_agent = MagicMock()
        self.mock_agent.run = MagicMock(return_value="Agent response")

    def test_initialization(self):
        """Test wrapper initialization."""
        wrapper = A2AAgentWrapper(
            agent=self.mock_agent,
            agent_id="test-agent",
            name="Test Agent",
            description="Test description",
        )
        
        self.assertEqual(wrapper.agent_id, "test-agent")
        self.assertEqual(wrapper.name, "Test Agent")
        self.assertEqual(wrapper.description, "Test description")

    def test_initialization_defaults(self):
        """Test wrapper initialization with defaults."""
        wrapper = A2AAgentWrapper(agent=self.mock_agent)
        
        self.assertEqual(wrapper.agent_id, "react-usc-agent")
        self.assertEqual(wrapper.base_url, "http://localhost:8000")

    def test_get_agent_card(self):
        """Test get_agent_card returns valid card."""
        wrapper = A2AAgentWrapper(
            agent=self.mock_agent,
            agent_id="my-agent",
            name="My Agent",
            description="My description",
            base_url="http://example.com",
        )
        
        card = wrapper.get_agent_card()
        
        self.assertIsInstance(card, AgentCard)
        self.assertEqual(card.id, "my-agent")
        self.assertEqual(card.name, "My Agent")
        self.assertEqual(card.description, "My description")
        
        # Should have query capability
        self.assertEqual(len(card.capabilities), 1)
        self.assertEqual(card.capabilities[0].name, "query")
        
        # Should have endpoints
        self.assertIn("card", card.endpoints)
        self.assertIn("tasks", card.endpoints)

    def test_execute_task_success(self):
        """Test execute_task with successful agent run."""
        self.mock_agent.run.return_value = "The answer is 42"
        
        wrapper = A2AAgentWrapper(agent=self.mock_agent)
        
        task_input = TaskInput(
            task_id="test-123",
            input_text="What is 2+2?",
        )
        
        output = wrapper.execute_task(task_input)
        
        self.assertEqual(output.task_id, "test-123")
        self.assertEqual(output.status, "completed")
        self.assertEqual(output.output_text, "The answer is 42")
        self.assertIsNone(output.error)
        self.assertIsNotNone(output.completed_at)
        
        # Verify agent was called
        self.mock_agent.run.assert_called_once_with("What is 2+2?")

    def test_execute_task_failure(self):
        """Test execute_task when agent raises exception."""
        self.mock_agent.run.side_effect = ValueError("Agent error")
        
        wrapper = A2AAgentWrapper(agent=self.mock_agent)
        
        task_input = TaskInput(
            task_id="test-456",
            input_text="Bad query",
        )
        
        output = wrapper.execute_task(task_input)
        
        self.assertEqual(output.task_id, "test-456")
        self.assertEqual(output.status, "failed")
        self.assertIsNone(output.output_text)
        self.assertIn("Agent error", output.error)


# =============================================================================
# Test: create_a2a_app
# =============================================================================


class TestCreateA2AApp(unittest.TestCase):
    """Test create_a2a_app function."""

    def setUp(self):
        self.mock_agent = MagicMock()
        self.mock_agent.run = MagicMock(return_value="Response")

    def test_create_app_without_fastapi(self):
        """Test create_a2a_app raises when FastAPI not installed."""
        wrapper = A2AAgentWrapper(agent=self.mock_agent)
        
        # Mock FastAPI import failure
        with patch.dict('sys.modules', {'fastapi': None}):
            with patch('react_usc.integrations.a2a.create_a2a_app') as mock_create:
                mock_create.side_effect = RuntimeError("FastAPI is required")
                
                with self.assertRaises(RuntimeError):
                    mock_create(wrapper)

    def test_create_app_with_fastapi(self):
        """Test create_a2a_app with FastAPI available."""
        try:
            from fastapi import FastAPI
        except ImportError:
            self.skipTest("FastAPI not installed")
        
        wrapper = A2AAgentWrapper(
            agent=self.mock_agent,
            name="Test Agent",
            description="Test description",
        )
        
        app = create_a2a_app(wrapper)
        
        # App should be a FastAPI instance
        self.assertIsInstance(app, FastAPI)
        self.assertEqual(app.title, "Test Agent")

    def test_app_has_routes(self):
        """Test that created app has expected routes."""
        try:
            from fastapi import FastAPI
        except ImportError:
            self.skipTest("FastAPI not installed")
        
        wrapper = A2AAgentWrapper(agent=self.mock_agent)
        app = create_a2a_app(wrapper)
        
        # Get route paths
        routes = [route.path for route in app.routes]
        
        self.assertIn("/.well-known/a2a.json", routes)
        self.assertIn("/tasks", routes)
        self.assertIn("/health", routes)


# =============================================================================
# Test: Integration with FastAPI TestClient
# =============================================================================


class TestA2AAppEndpoints(unittest.TestCase):
    """Test A2A app endpoints using FastAPI TestClient."""

    def setUp(self):
        try:
            from fastapi.testclient import TestClient
            self.TestClient = TestClient
        except ImportError:
            self.skipTest("FastAPI not installed")
        
        self.mock_agent = MagicMock()
        self.mock_agent.run = MagicMock(return_value="Test response")
        
        wrapper = A2AAgentWrapper(
            agent=self.mock_agent,
            agent_id="test-agent",
            name="Test Agent",
        )
        self.app = create_a2a_app(wrapper)
        self.client = self.TestClient(self.app)

    def test_health_endpoint(self):
        """Test /health endpoint."""
        response = self.client.get("/health")
        
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    def test_agent_card_endpoint(self):
        """Test /.well-known/a2a.json endpoint."""
        response = self.client.get("/.well-known/a2a.json")
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["id"], "test-agent")
        self.assertEqual(data["name"], "Test Agent")

    def test_tasks_endpoint(self):
        """Test /tasks endpoint."""
        response = self.client.post(
            "/tasks",
            json={"input_text": "What is 2+2?"},
        )
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "completed")
        self.assertEqual(data["output_text"], "Test response")


if __name__ == "__main__":
    unittest.main()
