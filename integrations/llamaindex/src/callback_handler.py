"""Lightweight LlamaIndex callback handler for Memori Cloud agent turn capture.

When using **Memori Cloud** (``api_key`` mode), this optional handler lets you
explicitly capture each agent conversation turn via
:meth:`~memori.Memori.capture_agent_turn`.  This is additive — memory
injection and recall are already handled transparently by the patched LLM
client.  Use this handler when you want explicit, durable turn records in
Memori Cloud (useful for analytics, dashboards, and per-session summaries).

If you are using the self-hosted / BYODB mode, the client patching alone is
sufficient — you do not need this handler.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from memori import Memori

logger = logging.getLogger(__name__)


class MemoriCallbackHandler:
    """Optional LlamaIndex callback handler for Memori Cloud turn capture.

    Attach this to a LlamaIndex ``CallbackManager`` to have each agent
    conversation turn captured via ``memori.capture_agent_turn()``.

    Args:
        memori: An initialised :class:`~memori.Memori` instance in Cloud mode.
        project_id: Memori Cloud project ID used when capturing turns.

    Example::

        from llama_index.core.callbacks import CallbackManager
        from memori import Memori
        from memori.integrations.llamaindex import MemoriCallbackHandler

        mem = Memori()
        mem.attribution(entity_id="user_123", process_id="my-agent")

        handler = MemoriCallbackHandler(mem, project_id="my-project")
        callback_manager = CallbackManager([handler])
    """

    event_starts_to_ignore: list[str] = []
    event_ends_to_ignore: list[str] = []

    def __init__(self, memori: Memori, project_id: str | None = None) -> None:
        self._memori = memori
        self._project_id = project_id
        self._pending_user_message: str | None = None

    # ------------------------------------------------------------------
    # LlamaIndex CBEventType hooks
    # ------------------------------------------------------------------

    def on_event_start(
        self,
        event_type: Any,
        payload: dict[str, Any] | None = None,
        event_id: str = "",
        **kwargs: Any,
    ) -> str:
        """Capture the user message at the start of a chat/LLM event."""
        payload = payload or {}
        try:
            from llama_index.core.callbacks.schema import CBEventType

            if event_type == CBEventType.LLM:
                messages = payload.get("messages", [])
                if messages:
                    # Find the last user message.
                    for msg in reversed(messages):
                        role = getattr(msg, "role", None) or (
                            msg.get("role") if isinstance(msg, dict) else None
                        )
                        content = getattr(msg, "content", None) or (
                            msg.get("content") if isinstance(msg, dict) else None
                        )
                        if role == "user" and content:
                            self._pending_user_message = str(content)
                            break
        except Exception:  # nosec B110
            pass
        return event_id

    def on_event_end(
        self,
        event_type: Any,
        payload: dict[str, Any] | None = None,
        event_id: str = "",
        **kwargs: Any,
    ) -> None:
        """Capture the assistant response and submit the turn to Memori Cloud."""
        payload = payload or {}
        try:
            from llama_index.core.callbacks.schema import CBEventType

            if event_type == CBEventType.LLM:
                response = payload.get("response")
                if response is None:
                    return

                # Extract text from various response types.
                assistant_text: str | None = None
                if isinstance(response, str):
                    assistant_text = response
                elif hasattr(response, "message"):
                    msg = response.message
                    assistant_text = getattr(msg, "content", None) or str(msg)
                elif hasattr(response, "text"):
                    assistant_text = response.text

                if assistant_text and self._pending_user_message and self._project_id:
                    self._memori.capture_agent_turn(
                        user_content=self._pending_user_message,
                        assistant_content=assistant_text,
                        project_id=self._project_id,
                        platform="llamaindex",
                    )
                    logger.debug(
                        "MemoriCallbackHandler: captured agent turn for project %s",
                        self._project_id,
                    )
                    self._pending_user_message = None
        except Exception as exc:  # nosec B110
            logger.debug("MemoriCallbackHandler: failed to capture turn: %s", exc)

    def start_trace(self, trace_id: str | None = None) -> None:
        """No-op; satisfies the BaseCallbackHandler interface."""

    def end_trace(
        self,
        trace_id: str | None = None,
        trace_map: dict[str, list[str]] | None = None,
    ) -> None:
        """No-op; satisfies the BaseCallbackHandler interface."""
