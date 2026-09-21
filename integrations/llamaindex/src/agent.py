"""Memori-aware agent wrapper for LlamaIndex.

Wraps any LlamaIndex agent (e.g. ``ReActAgent``, ``OpenAIAgent``) and adds
persistent long-term memory via Memori.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from memori.integrations.llamaindex.src.config import MemoriLlamaIndexConfig

    from memori import Memori

logger = logging.getLogger(__name__)


class MemoriAgent:
    """A drop-in wrapper that adds Memori long-term memory to any LlamaIndex agent.

    Memory injection and capture are handled transparently by Memori's patched
    LLM client — no manual recall/store calls are needed.

    Args:
        agent: Any LlamaIndex agent instance (e.g. ``ReActAgent``).
        memori: An initialised :class:`~memori.Memori` instance.
        entity_id: Identifies the user/entity whose memories are tracked.
        process_id: Identifies this agent process. Defaults to
            ``"llamaindex-agent"``.
        session_id: Optional explicit session ID.
        config: Optional :class:`MemoriLlamaIndexConfig` that overrides the
            individual keyword arguments above.

    Example::

        from llama_index.core.agent import ReActAgent
        from llama_index.llms.openai import OpenAI
        from memori import Memori
        from memori.integrations.llamaindex import MemoriAgent

        llm   = OpenAI(model="gpt-4o-mini")
        agent = ReActAgent.from_tools([], llm=llm)
        mem   = Memori()

        memory_agent = MemoriAgent(
            agent=agent,
            memori=mem,
            entity_id="user_123",
        )
        response = memory_agent.chat("What did we discuss last time?")
    """

    def __init__(
        self,
        agent: Any,
        memori: Memori,
        entity_id: str,
        process_id: str = "llamaindex-agent",
        session_id: str | None = None,
        config: MemoriLlamaIndexConfig | None = None,
    ) -> None:
        from _helpers import (
            get_llm_from_engine_or_agent,
            register_llamaindex_llm,
        )

        self._agent = agent
        self._memori = memori

        if config is not None:
            entity_id = config.entity_id
            process_id = config.process_id
            session_id = config.session_id

        memori.attribution(entity_id=entity_id, process_id=process_id)

        if session_id is not None:
            memori.set_session(session_id)

        llm = get_llm_from_engine_or_agent(agent)
        if llm is not None:
            register_llamaindex_llm(memori, llm)
        else:
            logger.warning(
                "MemoriAgent: could not find an LLM on the provided "
                "agent (%s). Memory will not be active.",
                type(agent).__name__,
            )

    # ------------------------------------------------------------------
    # Public interface mirrors BaseChatEngine / BaseAgent
    # ------------------------------------------------------------------

    def chat(self, message: str, **kwargs: Any) -> Any:
        """Send *message* to the agent and return its response."""
        return self._agent.chat(message, **kwargs)

    async def achat(self, message: str, **kwargs: Any) -> Any:
        """Async version of :meth:`chat`."""
        return await self._agent.achat(message, **kwargs)

    def query(self, query: str, **kwargs: Any) -> Any:
        """Run a query through the agent (for query-engine compatible agents)."""
        return self._agent.query(query, **kwargs)

    async def aquery(self, query: str, **kwargs: Any) -> Any:
        """Async version of :meth:`query`."""
        return await self._agent.aquery(query, **kwargs)

    def reset(self) -> None:
        """Reset the agent's memory and start a new Memori session."""
        if hasattr(self._agent, "reset"):
            self._agent.reset()
        self._memori.new_session()
        logger.debug("MemoriAgent: agent reset, new Memori session started")

    def new_session(self) -> MemoriAgent:
        """Start a fresh Memori session (keeps agent state intact)."""
        self._memori.new_session()
        return self

    # ------------------------------------------------------------------
    # Attribute pass-through for agent-specific APIs
    # ------------------------------------------------------------------

    def __getattr__(self, name: str) -> Any:
        return getattr(self._agent, name)

    def __repr__(self) -> str:
        return (
            f"MemoriAgent(agent={type(self._agent).__name__!r}, "
            f"entity_id={self._memori.config.entity_id!r})"
        )
