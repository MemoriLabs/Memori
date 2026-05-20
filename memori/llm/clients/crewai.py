r"""
 __  __                           _
|  \/  | ___ _ __ ___   ___  _ __(_)
| |\/| |/ _ \ '_ ` _ \ / _ \| '__| |
| |  | |  __| | | | | | (_) | |  | |
|_|  |_|\___|_| |_| |_|\___/|_|  |_|
                   perfectam memoriam
                        memorilabs.ai
"""

import logging
from typing import TYPE_CHECKING, Any

from memori.llm._base import BaseClient
from memori.llm._constants import CREWAI_FRAMEWORK_PROVIDER

if TYPE_CHECKING:
    from memori import Memori

logger = logging.getLogger(__name__)


class CrewAIMemory(BaseClient):
    """Memori integration for CrewAI agent framework.

    Enables CrewAI agents to seamlessly use Memori for long-term,
    structured memory persistence. Captures agent traces, decisions,
    and contextual information automatically.

    Example:
        ```python
        from crewai import Agent, Crew, Task
        from memori import Memori

        mem = Memori().attribution("team_123", "crewai_crew")
        agent = Agent(role="Researcher")
        crew = Crew(agents=[agent], tasks=[...])
        mem.crewai.register(crew=crew)
        result = crew.kickoff()
        # Memories automatically persisted and augmented
        ```

    Attributes:
        config: Memori configuration instance.
        _crewai_clients: Registry of wrapped CrewAI Crew instances.
    """

    def __init__(self, config: "Memori") -> None:
        """Initialize CrewAI memory client.

        Args:
            config: Memori configuration instance.
        """
        super().__init__(config)
        self._crewai_clients: dict[int, Any] = {}

    def register(self, crew: Any | None = None) -> "CrewAIMemory":
        """Register a CrewAI Crew instance for memory integration.

        Args:
            crew: CrewAI Crew instance to wrap with Memori.

        Returns:
            Self for method chaining.

        Raises:
            RuntimeError: If crew is None or not a CrewAI Crew.
        """
        if crew is None:
            raise RuntimeError("CrewAI::register called without crew")

        if not self._is_crewai_crew(crew):
            raise RuntimeError(
                "crew provided is not instance of crewai.crew.Crew"
            )

        if not hasattr(crew, "_memori_installed"):
            self._wrap_crew_execute(crew)
            crew._memori_installed = True

        self._crewai_clients[id(crew)] = crew
        return self

    def _wrap_crew_execute(self, crew: Any) -> None:
        """Wrap CrewAI Crew.kickoff to intercept and capture traces.

        Args:
            crew: CrewAI Crew instance to wrap.
        """
        original_kickoff = crew.kickoff

        def wrapped_kickoff(*args: Any, **kwargs: Any) -> Any:
            """Wrapped kickoff that captures agent execution traces."""
            result = original_kickoff(*args, **kwargs)
            self._capture_crew_execution(crew, result, kwargs)
            return result

        crew.kickoff = wrapped_kickoff

        # Wrap async kickoff if available
        if hasattr(crew, "kickoff_async"):
            original_kickoff_async = crew.kickoff_async

            async def wrapped_kickoff_async(
                *args: Any, **kwargs: Any
            ) -> Any:
                """Wrapped async kickoff for coroutine execution."""
                result = await original_kickoff_async(*args, **kwargs)
                self._capture_crew_execution(crew, result, kwargs)
                return result

            crew.kickoff_async = wrapped_kickoff_async

    def _capture_crew_execution(
        self, crew: Any, result: Any, context: dict[str, Any]
    ) -> None:
        """Capture and augment crew execution as structured memory.

        Args:
            crew: CrewAI Crew instance.
            result: Execution result from kickoff.
            context: Execution context (inputs, etc).
        """
        try:
            # Extract metadata from crew
            agents_info = self._extract_agents_info(crew)
            tasks_info = self._extract_tasks_info(crew)

            # Build memory object
            memory_data = {
                "type": "crew_execution",
                "crew_id": id(crew),
                "agents": agents_info,
                "tasks": tasks_info,
                "inputs": context.get("inputs", {}),
                "output": str(result)[:1000],
                "output_type": (
                    result.__class__.__name__ if result is not None else None
                ),
            }

            # Store via Memori augmentation
            if hasattr(self.config, "augmentation"):
                augmentation = getattr(self.config, "augmentation", None)
                if augmentation is not None and hasattr(
                    augmentation, "store"
                ):
                    augmentation.store(
                        memory_data,
                        provider=CREWAI_FRAMEWORK_PROVIDER,
                    )

        except Exception as e:
            logger.warning(
                f"Failed to capture CrewAI execution: {str(e)}"
            )

    @staticmethod
    def _extract_agents_info(crew: Any) -> list[dict[str, str]]:
        """Extract agent roles and descriptions.

        Args:
            crew: CrewAI Crew instance.

        Returns:
            List of agent metadata dicts.
        """
        agents_info = []
        try:
            for agent in crew.agents:
                agents_info.append(
                    {
                        "role": getattr(agent, "role", "unknown"),
                        "goal": getattr(agent, "goal", ""),
                        "backstory": getattr(agent, "backstory", "")[
                            :200
                        ],
                    }
                )
        except Exception as e:
            logger.debug(f"Failed to extract agents info: {e}")
        return agents_info

    @staticmethod
    def _extract_tasks_info(crew: Any) -> list[dict[str, str]]:
        """Extract task descriptions and assignments.

        Args:
            crew: CrewAI Crew instance.

        Returns:
            List of task metadata dicts.
        """
        tasks_info = []
        try:
            for task in crew.tasks:
                tasks_info.append(
                    {
                        "description": getattr(task, "description", "")[
                            :500
                        ],
                        "assigned_to": (
                            getattr(task.agent, "role", "unknown")
                            if hasattr(task, "agent")
                            else "unassigned"
                        ),
                    }
                )
        except Exception as e:
            logger.debug(f"Failed to extract tasks info: {e}")
        return tasks_info

    @staticmethod
    def _is_crewai_crew(crew: Any) -> bool:
        """Validate if object is a CrewAI Crew instance.

        Args:
            crew: Object to validate.

        Returns:
            True if crew is a CrewAI Crew instance.
        """
        try:
            return (
                "crewai" in str(type(crew).__module__)
                and "Crew" in type(crew).__name__
            )
        except Exception:
            return False
