import logging
import uuid
from typing import Any, Optional, Union

from memori import Memory, MemoryClient

logger = logging.getLogger(__name__)


class MemoriCrewAIAdapter:
    """
    Adapter to integrate Memori with CrewAI.
    Captures agent conversations, tool calls, tool outputs, reasoning, and task results.
    """

    def __init__(self, memori_client: Optional[Union[Memory, MemoryClient]] = None, **kwargs):
        if memori_client:
            self.client = memori_client
        else:
            api_key = kwargs.get("api_key")
            if api_key:
                self.client = MemoryClient(**kwargs)
            else:
                self.client = Memory(**kwargs)

        # CrewAI callbacks are synchronous, so we must reject async clients to prevent silent coroutine failures.
        client_name = self.client.__class__.__name__
        if "Async" in client_name:
            raise ValueError(f"MemoriCrewAIAdapter only supports synchronous clients, but received {client_name}.")

    def setup_crew(self, crew: Any, user_id: str, run_id: Optional[str] = None):
        """
        Injects callbacks into the CrewAI Crew and its Agents to capture events.
        """
        if run_id is None:
            run_id = str(uuid.uuid4())

        # Inject into Crew tasks
        original_task_callback = crew.task_callback

        def _extract(obj, key, default=""):
            if isinstance(obj, dict):
                return obj.get(key, default)
            return getattr(obj, key, default)

        def wrapped_task_callback(task_output):
            try:
                if task_output is None:
                    return

                agent_role = _extract(task_output, "agent", "system")
                description = _extract(task_output, "description", "Unknown Task")
                raw_output = _extract(task_output, "raw", str(task_output))

                message = f"Task completed: {description}\nOutput: {raw_output}"
                self.client.add(
                    messages=[{"role": "user", "content": message}],
                    user_id=user_id,
                    agent_id=str(agent_role),
                    run_id=run_id,
                    metadata={"event_type": "task_completion", "task_description": str(description)},
                )
            except Exception as e:
                logger.error(f"MemoriCrewAIAdapter task_callback error: {e}")
            if original_task_callback:
                original_task_callback(task_output)

        crew.task_callback = wrapped_task_callback

        # Inject into Agents for step reasoning and tool calls
        if hasattr(crew, "agents"):
            for agent in crew.agents:
                original_step_callback = getattr(agent, "step_callback", None)
                agent_role = getattr(agent, "role", "unknown_agent")

                def create_wrapped_step_callback(role, orig_cb):
                    def wrapped_step_callback(step_output):
                        try:
                            if step_output is None:
                                return

                            # Capture agent step reasoning or tool execution
                            message = f"Agent reasoning/tool execution: {str(step_output)}"
                            self.client.add(
                                messages=[{"role": "assistant", "content": message}],
                                user_id=user_id,
                                agent_id=str(role),
                                run_id=run_id,
                                metadata={"event_type": "agent_step"},
                            )
                        except Exception as e:
                            logger.error(f"MemoriCrewAIAdapter step_callback error: {e}")
                        if orig_cb:
                            orig_cb(step_output)

                    return wrapped_step_callback

                agent.step_callback = create_wrapped_step_callback(agent_role, original_step_callback)

        return crew
