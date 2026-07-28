"""Memori-aware chat engine wrapper for LlamaIndex.

Wraps any :class:`llama_index.core.chat_engine.types.BaseChatEngine`
subclass and adds persistent long-term memory via Memori.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from memori.integrations.llamaindex.src.config import MemoriLlamaIndexConfig

    from memori import Memori

logger = logging.getLogger(__name__)


class MemoriChatEngine:
    """A drop-in wrapper that adds Memori long-term memory to any LlamaIndex chat engine.

    Memory injection (retrieved facts inserted into the system prompt) and
    memory capture (conversation stored after each turn) are handled
    transparently by Memori's patched LLM client — no manual recall/store
    calls are needed.

    Args:
        chat_engine: Any LlamaIndex ``BaseChatEngine`` instance.
        memori: An initialised :class:`~memori.Memori` instance.
        entity_id: Identifies the user/entity whose memories are tracked.
        process_id: Identifies this agent process. Defaults to
            ``"llamaindex-chat"``.
        session_id: Optional explicit session ID. When omitted, Memori
            generates a new session UUID.
        config: Optional :class:`MemoriLlamaIndexConfig` that overrides the
            individual keyword arguments above.

    Example::

        from llama_index.core.chat_engine import SimpleChatEngine
        from llama_index.llms.openai import OpenAI
        from memori import Memori
        from memori.integrations.llamaindex import MemoriChatEngine

        llm  = OpenAI(model="gpt-4o-mini")
        engine = SimpleChatEngine.from_defaults(llm=llm)
        mem  = Memori()

        memory_engine = MemoriChatEngine(
            chat_engine=engine,
            memori=mem,
            entity_id="user_123",
        )
        response = memory_engine.chat("What did we discuss last time?")
    """

    def __init__(
        self,
        chat_engine: Any,
        memori: Memori,
        entity_id: str,
        process_id: str = "llamaindex-chat",
        session_id: str | None = None,
        config: MemoriLlamaIndexConfig | None = None,
    ) -> None:
        from _helpers import (
            get_llm_from_engine_or_agent,
            register_llamaindex_llm,
        )

        self._engine = chat_engine
        self._memori = memori

        # Allow config object to override keyword args.
        if config is not None:
            entity_id = config.entity_id
            process_id = config.process_id
            session_id = config.session_id

        memori.attribution(entity_id=entity_id, process_id=process_id)

        if session_id is not None:
            memori.set_session(session_id)

        # Register the underlying SDK client with Memori.
        llm = get_llm_from_engine_or_agent(chat_engine)
        if llm is not None:
            register_llamaindex_llm(memori, llm)
        else:
            logger.warning(
                "MemoriChatEngine: could not find an LLM on the provided "
                "chat engine (%s). Memory will not be active.",
                type(chat_engine).__name__,
            )

    # ------------------------------------------------------------------
    # Public interface mirrors BaseChatEngine
    # ------------------------------------------------------------------

    def chat(self, message: str, **kwargs: Any) -> Any:
        """Send *message* and return the engine's response.

        Memory injection and capture are handled transparently by the
        Memori-patched LLM client.
        """
        return self._engine.chat(message, **kwargs)

    async def achat(self, message: str, **kwargs: Any) -> Any:
        """Async version of :meth:`chat`."""
        return await self._engine.achat(message, **kwargs)

    def stream_chat(self, message: str, **kwargs: Any) -> Any:
        """Streaming version of :meth:`chat`."""
        return self._engine.stream_chat(message, **kwargs)

    async def astream_chat(self, message: str, **kwargs: Any) -> Any:
        """Async streaming version of :meth:`chat`."""
        return await self._engine.astream_chat(message, **kwargs)

    def reset(self) -> None:
        """Reset the engine's conversation history and start a new Memori session."""
        self._engine.reset()
        self._memori.new_session()
        logger.debug("MemoriChatEngine: conversation reset, new Memori session started")

    # ------------------------------------------------------------------
    # Attribute pass-through for engine-specific APIs
    # ------------------------------------------------------------------

    def __getattr__(self, name: str) -> Any:
        """Proxy attribute access to the wrapped engine."""
        return getattr(self._engine, name)

    def __repr__(self) -> str:
        return (
            f"MemoriChatEngine(engine={type(self._engine).__name__!r}, "
            f"entity_id={self._memori.config.entity_id!r})"
        )
