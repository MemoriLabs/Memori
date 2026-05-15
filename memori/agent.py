"""Agent-oriented Memori Cloud helpers."""

from __future__ import annotations

import logging
from typing import Any
from urllib.parse import urlencode

from memori._config import Config
from memori._network import Api, ApiSubdomain

logger = logging.getLogger(__name__)


class Agent:
    """Client for Memori Cloud agent endpoints."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.default_api = Api(config)
        self.collector_api = Api(config, ApiSubdomain.COLLECTOR)

    def recall(
        self,
        *,
        query: str | None = None,
        date_start: str | None = None,
        date_end: str | None = None,
        project_id: str | None = None,
        session_id: str | None = None,
        signal: str | None = None,
        source: str | None = None,
    ) -> dict[str, Any]:
        """Fetch agent memories from ``GET /v1/agent/recall``."""
        if session_id and not project_id:
            raise ValueError("session_id cannot be provided without project_id")

        qs = self._query_string(
            {
                "query": query,
                "date_start": date_start,
                "date_end": date_end,
                "entity_id": self.config.entity_id,
                "project_id": project_id,
                "session_id": session_id,
                "signal": signal,
                "source": source,
            }
        )
        return self.default_api.get(f"agent/recall{qs}")

    def recall_summary(
        self,
        *,
        date_start: str | None = None,
        date_end: str | None = None,
        project_id: str | None = None,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """Fetch agent memory summaries from ``GET /v1/agent/recall/summary``."""
        if session_id and not project_id:
            raise ValueError("session_id cannot be provided without project_id")

        qs = self._query_string(
            {
                "date_start": date_start,
                "date_end": date_end,
                "project_id": project_id,
                "session_id": session_id,
            }
        )
        return self.default_api.get(f"agent/recall/summary{qs}")

    def capture_turn(
        self,
        *,
        user_content: str,
        assistant_content: str,
        project_id: str,
        session_id: str | None = None,
        platform: str = "python",
        trace: dict[str, Any] | None = None,
        summary: str | None = None,
        provider: str | None = None,
        model: str | None = None,
        provider_sdk_version: str | None = None,
    ) -> None:
        """Capture an agent conversation turn and submit it for augmentation.

        The conversation turn write is required to succeed. The collector-side
        augmentation request is best-effort so agent integrations do not fail
        after the durable turn has already been recorded.
        """
        resolved_session_id = str(session_id or self.config.session_id)
        attribution = self._attribution()
        messages = [
            {"role": "user", "content": user_content, "type": "text", "trace": None},
            {
                "role": "assistant",
                "content": assistant_content,
                "type": "text",
                "trace": trace,
            },
        ]

        turn_payload = {
            "attribution": attribution,
            "messages": messages,
            "project": {"id": project_id},
            "session": {"id": resolved_session_id},
        }

        aug_payload = {
            "attribution": attribution,
            "conversation": {"messages": messages},
            "meta": self._meta(
                provider=provider,
                model=model,
                provider_sdk_version=provider_sdk_version,
                platform=platform,
            ),
            "project": {"id": project_id},
            "session": {"id": resolved_session_id, "summary": summary},
            "trace": trace,
        }

        self.default_api.post("agent/conversation/turn", turn_payload)
        try:
            self.collector_api.post("agent/augmentation", aug_payload)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Memori agent augmentation failed: %s", exc)

    def feedback(self, content: str) -> None:
        """Send agent integration feedback to Memori Cloud."""
        self.default_api.post("agent/feedback", {"content": content})

    def _attribution(self) -> dict[str, Any]:
        process = None
        if self.config.process_id is not None:
            process = {"id": self.config.process_id}
        return {
            "entity": {"id": self.config.entity_id},
            "process": process,
        }

    def _meta(
        self,
        *,
        provider: str | None,
        model: str | None,
        provider_sdk_version: str | None,
        platform: str,
    ) -> dict[str, Any]:
        return {
            "sdk": {"lang": "python", "version": self.config.version},
            "framework": {"provider": None},
            "llm": {
                "model": {
                    "provider": provider,
                    "sdk": {"version": provider_sdk_version},
                    "version": model,
                }
            },
            "platform": {"provider": platform},
            "storage": {
                "cockroachdb": self.config.storage_config.cockroachdb,
                "dialect": self._storage_dialect(),
            },
        }

    def _storage_dialect(self) -> str | None:
        storage = self.config.storage
        if storage is None:
            return None
        adapter = getattr(storage, "adapter", None)
        get_dialect = getattr(adapter, "get_dialect", None)
        if callable(get_dialect):
            return get_dialect()
        return None

    @staticmethod
    def _query_string(params: dict[str, Any]) -> str:
        clean = {k: str(v) for k, v in params.items() if v not in (None, "")}
        encoded = urlencode(clean)
        return f"?{encoded}" if encoded else ""
