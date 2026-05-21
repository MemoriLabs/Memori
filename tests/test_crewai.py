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


def test_setup_crew(mock_memory_client):
    adapter = MemoriCrewAIAdapter(memori_client=mock_memory_client)
    agent1 = DummyAgent(role="researcher")
    agent2 = DummyAgent(role="writer")
    crew = DummyCrew(agents=[agent1, agent2])

    adapter.setup_crew(crew, user_id="test_user", run_id="test_run")

    assert crew.task_callback is not None
    assert agent1.step_callback is not None
    assert agent2.step_callback is not None


def test_task_callback_invokes_memory_client(mock_memory_client):
    adapter = MemoriCrewAIAdapter(memori_client=mock_memory_client)

    task_callback_called = False

    def original_task_callback(output):
        nonlocal task_callback_called
        task_callback_called = True

    crew = DummyCrew(agents=[], task_callback=original_task_callback)
    adapter.setup_crew(crew, user_id="test_user", run_id="test_run")

    task_output = DummyTaskOutput(agent="researcher", description="Research AI", raw="Found facts")
    crew.task_callback(task_output)

    assert task_callback_called
    mock_memory_client.add.assert_called_once()
    call_args = mock_memory_client.add.call_args[1]
    assert call_args["user_id"] == "test_user"
    assert call_args["agent_id"] == "researcher"
    assert call_args["run_id"] == "test_run"
    assert call_args["metadata"]["event_type"] == "task_completion"


def test_step_callback_invokes_memory_client(mock_memory_client):
    adapter = MemoriCrewAIAdapter(memori_client=mock_memory_client)

    step_callback_called = False

    def original_step_callback(step):
        nonlocal step_callback_called
        step_callback_called = True

    agent = DummyAgent(role="writer", step_callback=original_step_callback)
    crew = DummyCrew(agents=[agent])
    adapter.setup_crew(crew, user_id="test_user", run_id="test_run")

    agent.step_callback("Step output")

    assert step_callback_called
    mock_memory_client.add.assert_called_once()
    call_args = mock_memory_client.add.call_args[1]
    assert call_args["user_id"] == "test_user"
    assert call_args["agent_id"] == "writer"
    assert call_args["run_id"] == "test_run"
    assert call_args["metadata"]["event_type"] == "agent_step"


def test_async_client_rejection():
    class AsyncDummyClient:
        pass

    with pytest.raises(ValueError, match="only supports synchronous clients"):
        MemoriCrewAIAdapter(memori_client=AsyncDummyClient())


def test_malformed_task_output(mock_memory_client):
    adapter = MemoriCrewAIAdapter(memori_client=mock_memory_client)
    crew = DummyCrew(agents=[], task_callback=None)
    adapter.setup_crew(crew, user_id="test_user")

    # Dict payload (malformed for typical crewai, but handled robustly)
    crew.task_callback({"agent": "system", "description": "dict task", "raw": "dict raw"})
    mock_memory_client.add.assert_called_once()
    assert mock_memory_client.add.call_args[1]["agent_id"] == "system"
    assert mock_memory_client.add.call_args[1]["metadata"]["task_description"] == "dict task"

    mock_memory_client.reset_mock()

    # Null payload
    crew.task_callback(None)
    mock_memory_client.add.assert_not_called()


def test_exception_propagation_safety(mock_memory_client, caplog):
    adapter = MemoriCrewAIAdapter(memori_client=mock_memory_client)
    mock_memory_client.add.side_effect = Exception("Simulated network failure")

    crew = DummyCrew(agents=[], task_callback=None)
    adapter.setup_crew(crew, user_id="test_user")

    # This should NOT raise an exception despite the mock throwing one
    crew.task_callback(DummyTaskOutput("agent", "desc", "raw"))

    assert "MemoriCrewAIAdapter task_callback error" in caplog.text
