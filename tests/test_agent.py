from __future__ import annotations

import pytest

from memori import Memori


def test_memori_accepts_programmatic_api_key(monkeypatch):
    monkeypatch.delenv("MEMORI_API_KEY", raising=False)
    monkeypatch.delenv("MEMORI_COCKROACHDB_CONNECTION_STRING", raising=False)
    monkeypatch.setenv("MEMORI_TEST_MODE", "1")

    mem = Memori(api_key="programmatic-key")

    assert mem.config.api_key == "programmatic-key"
    assert mem.agent.default_api.headers()["Authorization"] == "Bearer programmatic-key"


def test_memori_accepts_programmatic_base_url(monkeypatch):
    monkeypatch.delenv("MEMORI_API_URL_BASE", raising=False)
    monkeypatch.delenv("MEMORI_COCKROACHDB_CONNECTION_STRING", raising=False)
    monkeypatch.setenv("MEMORI_TEST_MODE", "1")

    mem = Memori(api_key="key", base_url="https://api.example.com")

    assert mem.config.api_url_base == "https://api.example.com"
    assert (
        mem.agent.default_api.url("sdk/quota") == "https://api.example.com/v1/sdk/quota"
    )
    assert (
        mem.agent.collector_api.url("agent/augmentation")
        == "https://api.example.com/v1/agent/augmentation"
    )


def test_sync_api_uses_configured_timeout(monkeypatch, mocker):
    monkeypatch.delenv("MEMORI_COCKROACHDB_CONNECTION_STRING", raising=False)
    monkeypatch.setenv("MEMORI_TEST_MODE", "1")
    mem = Memori(api_key="key")
    mem.config.request_secs_timeout = 12

    response = mocker.Mock()
    response.status_code = 200
    response.json.return_value = {"ok": True}
    response.raise_for_status.return_value = None

    session = mocker.Mock()
    session.get.return_value = response
    session_cls = mocker.patch("requests.Session", return_value=session)

    assert mem.agent.default_api.get("sdk/quota") == {"ok": True}
    session_cls.assert_called_once()
    session.get.assert_called_once_with(
        mem.agent.default_api.url("sdk/quota"),
        headers=mem.agent.default_api.headers(),
        timeout=12,
    )


def test_agent_recall_builds_query_string(monkeypatch, mocker):
    monkeypatch.delenv("MEMORI_COCKROACHDB_CONNECTION_STRING", raising=False)
    monkeypatch.setenv("MEMORI_TEST_MODE", "1")
    mem = Memori(api_key="key").attribution("entity", "process")

    get = mocker.patch.object(
        mem.agent.default_api,
        "get",
        return_value={"facts": []},
    )

    result = mem.agent_recall(
        query="deployment",
        date_start="2026-01-01T00:00:00Z",
        project_id="project",
        session_id="session",
        signal="decision",
        source="fact",
    )

    assert result == {"facts": []}
    route = get.call_args.args[0]
    assert route.startswith("agent/recall?")
    assert "query=deployment" in route
    assert "date_start=2026-01-01T00%3A00%3A00Z" in route
    assert "entity_id=entity" in route
    assert "project_id=project" in route
    assert "session_id=session" in route
    assert "signal=decision" in route
    assert "source=fact" in route


def test_agent_recall_rejects_session_without_project(monkeypatch):
    monkeypatch.delenv("MEMORI_COCKROACHDB_CONNECTION_STRING", raising=False)
    monkeypatch.setenv("MEMORI_TEST_MODE", "1")
    mem = Memori(api_key="key")

    with pytest.raises(ValueError, match="session_id cannot be provided"):
        mem.agent_recall(session_id="session")


def test_agent_recall_summary_builds_query_string(monkeypatch, mocker):
    monkeypatch.delenv("MEMORI_COCKROACHDB_CONNECTION_STRING", raising=False)
    monkeypatch.setenv("MEMORI_TEST_MODE", "1")
    mem = Memori(api_key="key")

    get = mocker.patch.object(
        mem.agent.default_api,
        "get",
        return_value={"summaries": []},
    )

    result = mem.agent_recall_summary(project_id="project", session_id="session")

    assert result == {"summaries": []}
    assert get.call_args.args[0] == (
        "agent/recall/summary?project_id=project&session_id=session"
    )


def test_capture_agent_turn_writes_turn_then_collector(monkeypatch, mocker):
    monkeypatch.delenv("MEMORI_COCKROACHDB_CONNECTION_STRING", raising=False)
    monkeypatch.setenv("MEMORI_TEST_MODE", "1")
    mem = Memori(api_key="key").attribution("entity", "process")
    mem.set_session("session")

    default_post = mocker.patch.object(mem.agent.default_api, "post", return_value={})
    collector_post = mocker.patch.object(
        mem.agent.collector_api, "post", return_value={}
    )

    mem.capture_agent_turn(
        user_content="hello",
        assistant_content="hi",
        project_id="project",
        platform="hermes",
        trace={"tools": []},
    )

    assert default_post.call_args.args[0] == "agent/conversation/turn"
    turn_payload = default_post.call_args.args[1]
    assert turn_payload["attribution"] == {
        "entity": {"id": "entity"},
        "process": {"id": "process"},
    }
    assert turn_payload["project"] == {"id": "project"}
    assert turn_payload["session"] == {"id": "session"}
    assert turn_payload["messages"][1]["trace"] == {"tools": []}

    assert collector_post.call_args.args[0] == "agent/augmentation"
    aug_payload = collector_post.call_args.args[1]
    assert aug_payload["meta"]["platform"] == {"provider": "hermes"}
    assert aug_payload["trace"] == {"tools": []}


def test_capture_agent_turn_swallow_collector_failure(monkeypatch, mocker):
    monkeypatch.delenv("MEMORI_COCKROACHDB_CONNECTION_STRING", raising=False)
    monkeypatch.setenv("MEMORI_TEST_MODE", "1")
    mem = Memori(api_key="key")

    default_post = mocker.patch.object(mem.agent.default_api, "post", return_value={})
    mocker.patch.object(
        mem.agent.collector_api,
        "post",
        side_effect=RuntimeError("collector down"),
    )

    mem.capture_agent_turn(
        user_content="hello",
        assistant_content="hi",
        project_id="project",
    )

    default_post.assert_called_once()


def test_agent_feedback_posts_content(monkeypatch, mocker):
    monkeypatch.delenv("MEMORI_COCKROACHDB_CONNECTION_STRING", raising=False)
    monkeypatch.setenv("MEMORI_TEST_MODE", "1")
    mem = Memori(api_key="key")

    post = mocker.patch.object(mem.agent.default_api, "post", return_value={})

    mem.agent_feedback("Great integration")

    post.assert_called_once_with("agent/feedback", {"content": "Great integration"})
