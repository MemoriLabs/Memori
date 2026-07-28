"""Configuration classes for the Memori × LlamaIndex integration."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class MemoriLlamaIndexConfig:
    """Configuration for the Memori × LlamaIndex integration.

    Args:
        entity_id: Identifies the *user* or entity whose memories are tracked.
            Required — Memori cannot create memories without an entity_id.
        process_id: Identifies the agent or application process.
            Defaults to ``"llamaindex"`` when not provided.
        auto_retrieve_memories: When True (default), Memori injects relevant
            memories into every LLM prompt automatically via the patched SDK
            client.  Set to False to disable automatic injection (memories are
            still captured after each turn).
        retrieval_k: Maximum number of memory facts returned per recall query.
            Passed as ``limit`` to ``memori.recall()``.  Defaults to 5.
        enable_chat_memory: Enable memory for chat-engine wrappers.
        enable_agent_memory: Enable memory for agent wrappers.
        session_id: Optional explicit session ID.  When omitted, Memori
            generates a new session UUID automatically.
        project_id: Optional Memori Cloud project ID used when calling
            ``capture_agent_turn`` for Cloud-mode persistence.
    """

    entity_id: str
    process_id: str = "llamaindex"
    auto_retrieve_memories: bool = True
    retrieval_k: int = 5
    enable_chat_memory: bool = True
    enable_agent_memory: bool = True
    session_id: str | None = None
    project_id: str | None = None
    extra: dict = field(default_factory=dict)
