"""Tests for CrewAI memory integration."""

import pytest
from unittest.mock import MagicMock, patch

from memori._config import Config
from memori.llm.clients.crewai import CrewAIMemory


@pytest.fixture
def config():
    """Create a test Memori config."""
    return Config()


@pytest.fixture
def crewai_memory(config):
    """Create a CrewAIMemory client."""
    return CrewAIMemory(config)


@pytest.fixture
def mock_crew():
    """Create a mock CrewAI Crew."""
    crew = MagicMock()
    crew.__class__.__module__ = "crewai.crew"
    crew.__class__.__name__ = "Crew"
    crew.kickoff = MagicMock(return_value="result")
    crew.agents = [
        MagicMock(role="Agent1", goal="Goal1", backstory="Story1"),
        MagicMock(role="Agent2", goal="Goal2", backstory="Story2"),
    ]
    crew.tasks = [
        MagicMock(
            description="Task1",
            agent=MagicMock(role="Agent1"),
        ),
        MagicMock(
            description="Task2",
            agent=MagicMock(role="Agent2"),
        ),
    ]
    return crew


def test_crewai_memory_init(crewai_memory, config):
    """Test CrewAIMemory initialization."""
    assert crewai_memory.config is config
    assert crewai_memory._crewai_clients == {}
    assert isinstance(crewai_memory._crewai_clients, dict)


def test_crewai_register_valid_crew(crewai_memory, mock_crew):
    """Test registering a valid CrewAI crew."""
    result = crewai_memory.register(crew=mock_crew)
    assert result is crewai_memory
    assert hasattr(mock_crew, "_memori_installed")
    assert mock_crew._memori_installed is True
    assert id(mock_crew) in crewai_memory._crewai_clients


def test_crewai_register_none_crew(crewai_memory):
    """Test register fails with None crew."""
    with pytest.raises(RuntimeError, match="without crew"):
        crewai_memory.register(crew=None)


def test_crewai_register_invalid_crew(crewai_memory):
    """Test register fails with invalid crew type."""
    invalid_crew = MagicMock()
    invalid_crew.__class__.__module__ = "some_other_module"
    invalid_crew.__class__.__name__ = "NotCrew"

    with pytest.raises(RuntimeError, match="not instance of"):
        crewai_memory.register(crew=invalid_crew)


def test_crewai_is_crewai_crew_valid():
    """Test CrewAI crew detection."""
    crew = MagicMock()
    crew.__class__.__module__ = "crewai.crew"
    crew.__class__.__name__ = "Crew"

    assert CrewAIMemory._is_crewai_crew(crew) is True


def test_crewai_is_crewai_crew_invalid():
    """Test invalid crew detection."""
    not_crew = MagicMock()
    not_crew.__class__.__module__ = "other_module"
    not_crew.__class__.__name__ = "Other"

    assert CrewAIMemory._is_crewai_crew(not_crew) is False


def test_crewai_is_crewai_crew_exception():
    """Test crew detection with exception."""
    not_crew = None
    assert CrewAIMemory._is_crewai_crew(not_crew) is False


def test_crewai_extract_agents_info():
    """Test agent metadata extraction."""
    crew = MagicMock()
    crew.agents = [
        MagicMock(role="Researcher", goal="Find info", backstory="Bio1"),
        MagicMock(role="Writer", goal="Write", backstory="Bio2"),
    ]

    agents_info = CrewAIMemory._extract_agents_info(crew)

    assert len(agents_info) == 2
    assert agents_info[0]["role"] == "Researcher"
    assert agents_info[1]["role"] == "Writer"
    assert "goal" in agents_info[0]
    assert "backstory" in agents_info[0]


def test_crewai_extract_agents_info_empty():
    """Test agent extraction with empty crew."""
    crew = MagicMock()
    crew.agents = []

    agents_info = CrewAIMemory._extract_agents_info(crew)
    assert agents_info == []


def test_crewai_extract_agents_info_exception():
    """Test agent extraction with exception."""
    crew = MagicMock()
    crew.agents = None  # Will raise AttributeError

    agents_info = CrewAIMemory._extract_agents_info(crew)
    assert agents_info == []


def test_crewai_extract_tasks_info():
    """Test task metadata extraction."""
    crew = MagicMock()
    crew.tasks = [
        MagicMock(
            description="Research task",
            agent=MagicMock(role="Researcher"),
        ),
        MagicMock(
            description="Writing task",
            agent=MagicMock(role="Writer"),
        ),
    ]

    tasks_info = CrewAIMemory._extract_tasks_info(crew)

    assert len(tasks_info) == 2
    assert "Research task" in tasks_info[0]["description"]
    assert tasks_info[0]["assigned_to"] == "Researcher"
    assert tasks_info[1]["assigned_to"] == "Writer"


def test_crewai_extract_tasks_info_empty():
    """Test task extraction with empty crew."""
    crew = MagicMock()
    crew.tasks = []

    tasks_info = CrewAIMemory._extract_tasks_info(crew)
    assert tasks_info == []


def test_crewai_extract_tasks_info_no_agent():
    """Test task extraction when task has no agent."""
    crew = MagicMock()
    crew.tasks = [
        MagicMock(
            description="Task without agent",
            agent=None,
        ),
    ]

    tasks_info = CrewAIMemory._extract_tasks_info(crew)
    assert len(tasks_info) == 1
    assert tasks_info[0]["assigned_to"] == "unassigned"


def test_crewai_wrap_crew_execute(crewai_memory, mock_crew):
    """Test crew execution wrapping."""
    crewai_memory.register(crew=mock_crew)
    original_kickoff = mock_crew.kickoff

    # Call wrapped kickoff
    result = mock_crew.kickoff(inputs={"test": "input"})

    assert result == "result"


def test_crewai_capture_crew_execution(crewai_memory, mock_crew):
    """Test crew execution capture."""
    config = MagicMock()
    config.augmentation = MagicMock()
    config.augmentation.store = MagicMock()
    crewai_memory.config = config

    crewai_memory.register(crew=mock_crew)

    # Trigger execution
    mock_crew.kickoff()

    # Verify augmentation store was called (if defined)
    # Note: MagicMock will create the method if not present
    assert config.augmentation.store.called


def test_crewai_double_wrapping_prevented(crewai_memory, mock_crew):
    """Test that double-wrapping is prevented."""
    crewai_memory.register(crew=mock_crew)
    assert mock_crew._memori_installed is True

    # Register again
    crewai_memory.register(crew=mock_crew)

    # Should still only have one entry
    assert len(crewai_memory._crewai_clients) == 1


def test_crewai_multiple_crews(crewai_memory):
    """Test managing multiple CrewAI crews."""
    crew1 = MagicMock()
    crew1.__class__.__module__ = "crewai.crew"
    crew1.__class__.__name__ = "Crew"
    crew1.agents = []
    crew1.tasks = []
    crew1.kickoff = MagicMock(return_value="result1")

    crew2 = MagicMock()
    crew2.__class__.__module__ = "crewai.crew"
    crew2.__class__.__name__ = "Crew"
    crew2.agents = []
    crew2.tasks = []
    crew2.kickoff = MagicMock(return_value="result2")

    crewai_memory.register(crew=crew1)
    crewai_memory.register(crew=crew2)

    assert len(crewai_memory._crewai_clients) == 2
    assert id(crew1) in crewai_memory._crewai_clients
    assert id(crew2) in crewai_memory._crewai_clients


def test_crewai_capture_handles_missing_augmentation(crewai_memory, mock_crew):
    """Test capture handles missing augmentation gracefully."""
    crewai_memory.config.augmentation = None
    crewai_memory.register(crew=mock_crew)

    # Should not raise exception
    mock_crew.kickoff()


def test_crewai_capture_handles_store_exception(crewai_memory, mock_crew):
    """Test capture handles store exceptions gracefully."""
    config = MagicMock()
    config.augmentation = MagicMock()
    config.augmentation.store = MagicMock(side_effect=Exception("Store error"))
    crewai_memory.config = config

    crewai_memory.register(crew=mock_crew)

    # Should not raise exception
    mock_crew.kickoff()
