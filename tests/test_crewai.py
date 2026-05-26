from __future__ import annotations

import uuid

import pytest
from unittest.mock import MagicMock

from memori.proxy.crewai import MemoriCrewAIAdapter


class DummyAgent:
    def __init__(self, role, step_callback=None):
        self.role = role
        self.step_callback = step_callback


class DummyTaskOutput:
    def __init__(self, agent, description, raw):
        self.agent = agent
        self.description = description
        self.raw = raw


class DummyCrew:
    def __init__(self, agents, task_callback=None):
        self.agents = agents
        self.task_callback = task_callback


@pytest.fixture
def mock_memory_client():
    mock_client = MagicMock()
    return mock_client


# ---------------------------------------------------------------------------
# Core setup
# ---------------------------------------------------------------------------

def test_setup_crew(mock_memory_client):
    adapter = MemoriCrewAIAdapter(memori_client=mock_memory_client)
    agent1 = DummyAgent(role="researcher")
    agent2 = DummyAgent(role="writer")
    crew = DummyCrew(agents=[agent1, agent2])

    adapter.setup_crew(crew, project_id="test_user", session_id="test_run")

    assert crew.task_callback is not None
    assert agent1.step_callback is not None
    assert agent2.step_callback is not None


# ---------------------------------------------------------------------------
# Task callback
# ---------------------------------------------------------------------------

def test_task_callback_invokes_memory_client(mock_memory_client):
    adapter = MemoriCrewAIAdapter(memori_client=mock_memory_client)

    task_callback_called = False

    def original_task_callback(output):
        nonlocal task_callback_called
        task_callback_called = True

    crew = DummyCrew(agents=[], task_callback=original_task_callback)
    adapter.setup_crew(crew, project_id="test_user", session_id="test_run")

    task_output = DummyTaskOutput(
        agent="researcher", description="Research AI", raw="Found facts"
    )
    crew.task_callback(task_output)

    assert task_callback_called
    mock_memory_client.capture_agent_turn.assert_called_once()
    call_args = mock_memory_client.capture_agent_turn.call_args[1]
    assert call_args["project_id"] == "test_user"
    assert call_args["session_id"] == "test_run"
    assert call_args["trace"]["event_type"] == "task_completion"
    assert call_args["trace"]["agent_role"] == "researcher"


# ---------------------------------------------------------------------------
# Step callback
# ---------------------------------------------------------------------------

def test_step_callback_invokes_memory_client(mock_memory_client):
    adapter = MemoriCrewAIAdapter(memori_client=mock_memory_client)

    step_callback_called = False

    def original_step_callback(step):
        nonlocal step_callback_called
        step_callback_called = True

    agent = DummyAgent(role="writer", step_callback=original_step_callback)
    crew = DummyCrew(agents=[agent])
    adapter.setup_crew(crew, project_id="test_user", session_id="test_run")

    agent.step_callback("Step output")

    assert step_callback_called
    mock_memory_client.capture_agent_turn.assert_called_once()
    call_args = mock_memory_client.capture_agent_turn.call_args[1]
    assert call_args["project_id"] == "test_user"
    assert call_args["session_id"] == "test_run"
    assert call_args["trace"]["event_type"] == "agent_step"
    assert call_args["trace"]["agent_role"] == "writer"


# ---------------------------------------------------------------------------
# Async rejection
# ---------------------------------------------------------------------------

def test_async_client_rejection():
    class AsyncDummyClient:
        pass

    with pytest.raises(ValueError, match="only supports synchronous clients"):
        MemoriCrewAIAdapter(memori_client=AsyncDummyClient())


# ---------------------------------------------------------------------------
# Malformed / edge-case payloads
# ---------------------------------------------------------------------------

def test_malformed_task_output(mock_memory_client):
    adapter = MemoriCrewAIAdapter(memori_client=mock_memory_client)
    crew = DummyCrew(agents=[], task_callback=None)
    adapter.setup_crew(crew, project_id="test_user")

    # Dict payload (malformed for typical crewai, but handled robustly)
    crew.task_callback(
        {"agent": "system", "description": "dict task", "raw": "dict raw"}
    )
    mock_memory_client.capture_agent_turn.assert_called_once()
    call_args = mock_memory_client.capture_agent_turn.call_args[1]
    assert call_args["trace"]["agent_role"] == "system"
    assert call_args["trace"]["task_description"] == "dict task"

    mock_memory_client.reset_mock()

    # Null payload
    crew.task_callback(None)
    mock_memory_client.capture_agent_turn.assert_not_called()


def test_task_callback_none_is_noop(mock_memory_client):
    """Passing None to task_callback should not call capture_agent_turn."""
    adapter = MemoriCrewAIAdapter(memori_client=mock_memory_client)
    crew = DummyCrew(agents=[], task_callback=None)
    adapter.setup_crew(crew, project_id="test_project")

    crew.task_callback(None)
    mock_memory_client.capture_agent_turn.assert_not_called()


def test_step_callback_none_is_noop(mock_memory_client):
    """Passing None to step_callback should not call capture_agent_turn."""
    adapter = MemoriCrewAIAdapter(memori_client=mock_memory_client)
    agent = DummyAgent(role="tester")
    crew = DummyCrew(agents=[agent])
    adapter.setup_crew(crew, project_id="test_project")

    agent.step_callback(None)
    mock_memory_client.capture_agent_turn.assert_not_called()


# ---------------------------------------------------------------------------
# Exception safety
# ---------------------------------------------------------------------------

def test_exception_propagation_safety(mock_memory_client, caplog):
    adapter = MemoriCrewAIAdapter(memori_client=mock_memory_client)
    mock_memory_client.capture_agent_turn.side_effect = Exception(
        "Simulated network failure"
    )

    crew = DummyCrew(agents=[], task_callback=None)
    adapter.setup_crew(crew, project_id="test_user")

    # This should NOT raise an exception despite the mock throwing one
    crew.task_callback(DummyTaskOutput("agent", "desc", "raw"))

    assert "MemoriCrewAIAdapter task_callback error" in caplog.text


def test_step_callback_exception_safety(mock_memory_client, caplog):
    """Step callback errors should be logged, not raised."""
    adapter = MemoriCrewAIAdapter(memori_client=mock_memory_client)
    mock_memory_client.capture_agent_turn.side_effect = Exception("boom")

    agent = DummyAgent(role="tester")
    crew = DummyCrew(agents=[agent])
    adapter.setup_crew(crew, project_id="test_project")

    agent.step_callback("some output")  # should not raise
    assert "MemoriCrewAIAdapter step_callback error" in caplog.text


def test_original_callbacks_called_on_error(mock_memory_client):
    """Original callbacks must still fire even when Memori errors."""
    adapter = MemoriCrewAIAdapter(memori_client=mock_memory_client)
    mock_memory_client.capture_agent_turn.side_effect = Exception("fail")

    called = False

    def original_cb(output):
        nonlocal called
        called = True

    crew = DummyCrew(agents=[], task_callback=original_cb)
    adapter.setup_crew(crew, project_id="test_project")
    crew.task_callback(DummyTaskOutput("agent", "desc", "raw"))

    assert called, "Original task_callback must be called even on Memori error"


# ---------------------------------------------------------------------------
# Auto-generated session_id
# ---------------------------------------------------------------------------

def test_setup_crew_generates_session_id(mock_memory_client):
    """session_id should be auto-generated as a valid UUID when not provided."""
    adapter = MemoriCrewAIAdapter(memori_client=mock_memory_client)
    crew = DummyCrew(agents=[], task_callback=None)
    adapter.setup_crew(crew, project_id="test_project")

    crew.task_callback(DummyTaskOutput("agent", "desc", "raw"))
    call_args = mock_memory_client.capture_agent_turn.call_args[1]
    # Should be a valid UUID — raises ValueError if not
    uuid.UUID(call_args["session_id"])


# ---------------------------------------------------------------------------
# Crew without agents attribute
# ---------------------------------------------------------------------------

def test_crew_without_agents_attribute(mock_memory_client):
    """Crew objects without .agents should not crash."""
    adapter = MemoriCrewAIAdapter(memori_client=mock_memory_client)

    class MinimalCrew:
        task_callback = None

    crew = MinimalCrew()
    result = adapter.setup_crew(crew, project_id="test_project")
    assert result is crew
    assert crew.task_callback is not None
