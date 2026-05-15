"""Small Memori SDK wrapper used by the Hermes provider."""

from __future__ import annotations

from typing import Any

DEFAULT_TIMEOUT_SECS = 30
MEMORI_PLATFORM = "hermes"


class MemoriApiError(RuntimeError):
    """Raised when a Memori SDK request fails."""


class MemoriAgentClient:
    """Thin wrapper around the Memori Python SDK for Hermes."""

    def __init__(
        self,
        *,
        api_key: str,
        entity_id: str,
        project_id: str,
        process_id: str | None = None,
        base_url: str | None = None,
        timeout: int = DEFAULT_TIMEOUT_SECS,
    ) -> None:
        try:
            from memori import Memori
        except ModuleNotFoundError as exc:
            missing = exc.name or "memori"
            raise RuntimeError(
                f"Memori SDK dependency missing: {missing}. Run: pip install memori"
            ) from exc
        except ImportError as exc:
            raise RuntimeError(
                "Memori SDK could not be imported. Run: pip install memori"
            ) from exc

        self.entity_id = entity_id
        self.project_id = project_id
        self.memori = Memori(api_key=api_key, base_url=base_url).attribution(
            entity_id,
            process_id,
        )
        self.memori.config.request_secs_timeout = timeout

    def capture_turn(
        self,
        *,
        user_content: str,
        assistant_content: str,
        session_id: str,
        platform: str,
    ) -> None:
        del platform
        try:
            self.memori.capture_agent_turn(
                user_content=user_content,
                assistant_content=assistant_content,
                project_id=self.project_id,
                session_id=session_id,
                platform=MEMORI_PLATFORM,
            )
        except Exception as exc:  # noqa: BLE001
            raise MemoriApiError(str(exc)) from exc

    def agent_recall(self, params: dict[str, Any]) -> dict[str, Any]:
        try:
            return self.memori.agent_recall(
                query=params.get("query"),
                date_start=params.get("dateStart") or params.get("date_start"),
                date_end=params.get("dateEnd") or params.get("date_end"),
                project_id=params.get("projectId")
                or params.get("project_id")
                or self.project_id,
                session_id=params.get("sessionId") or params.get("session_id"),
                signal=params.get("signal"),
                source=params.get("source"),
            )
        except Exception as exc:  # noqa: BLE001
            raise MemoriApiError(str(exc)) from exc

    def agent_recall_summary(self, params: dict[str, Any]) -> dict[str, Any]:
        try:
            return self.memori.agent_recall_summary(
                date_start=params.get("dateStart") or params.get("date_start"),
                date_end=params.get("dateEnd") or params.get("date_end"),
                project_id=params.get("projectId")
                or params.get("project_id")
                or self.project_id,
                session_id=params.get("sessionId") or params.get("session_id"),
            )
        except Exception as exc:  # noqa: BLE001
            raise MemoriApiError(str(exc)) from exc

    def quota(self) -> dict[str, Any]:
        try:
            return self.memori.agent.default_api.get("sdk/quota")
        except Exception as exc:  # noqa: BLE001
            raise MemoriApiError(str(exc)) from exc

    def signup(self, email: str) -> dict[str, Any]:
        try:
            return self.memori.agent.default_api.post("sdk/account", {"email": email})
        except Exception as exc:  # noqa: BLE001
            raise MemoriApiError(str(exc)) from exc

    def feedback(self, content: str) -> dict[str, Any]:
        try:
            self.memori.agent_feedback(content)
            return {}
        except Exception as exc:  # noqa: BLE001
            raise MemoriApiError(str(exc)) from exc
