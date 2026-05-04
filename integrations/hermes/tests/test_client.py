from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from memori_hermes.client import MemoriAgentClient, MemoriApiError  # noqa: E402


def make_client() -> MemoriAgentClient:
    return MemoriAgentClient(
        api_key="key",
        entity_id="entity",
        project_id="project",
        process_id="process",
        base_url="https://api.example.com",
    )


def test_agent_recall_maps_query_params(monkeypatch: pytest.MonkeyPatch) -> None:
    client = make_client()
    seen = {}

    def fake_agent_recall(**kwargs):
        seen.update(kwargs)
        return {"ok": True}

    monkeypatch.setattr(client.memori, "agent_recall", fake_agent_recall)

    assert client.agent_recall({"query": "hello", "dateStart": "2026-01-01"}) == {
        "ok": True
    }

    assert seen == {
        "query": "hello",
        "date_start": "2026-01-01",
        "date_end": None,
        "project_id": "project",
        "session_id": None,
        "signal": None,
        "source": None,
    }


def test_client_uses_openclaw_consistent_timeout() -> None:
    client = make_client()

    assert client.memori.config.request_secs_timeout == 30


def test_capture_turn_writes_default_then_collector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = make_client()
    seen = {}

    def fake_capture_agent_turn(**kwargs):
        seen.update(kwargs)

    monkeypatch.setattr(client.memori, "capture_agent_turn", fake_capture_agent_turn)

    client.capture_turn(
        user_content="u",
        assistant_content="a",
        session_id="session",
        platform="cli",
    )

    assert seen == {
        "user_content": "u",
        "assistant_content": "a",
        "project_id": "project",
        "session_id": "session",
        "platform": "cli",
    }


def test_summary_rejects_session_without_project() -> None:
    client = make_client()
    client.project_id = ""

    with pytest.raises(MemoriApiError):
        client.agent_recall_summary({"sessionId": "session"})
