from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from memori_hermes import MemoriMemoryProvider  # noqa: E402


class FakeClient:
    def __init__(self) -> None:
        self.captured = []
        self.recall_params = None

    def capture_turn(
        self,
        *,
        user_content: str,
        assistant_content: str,
        session_id: str,
        platform: str,
        trace=None,
    ) -> None:
        self.captured.append(
            (user_content, assistant_content, session_id, platform, trace)
        )

    def agent_recall(self, params):
        self.recall_params = params
        return {"facts": [{"content": "remembered"}]}

    def agent_recall_summary(self, params):
        return {"summaries": [{"content": "summary"}]}

    def agent_compaction(self, params):
        return {"params": params, "state": {"active_tasks": ["ship compaction"]}}

    def quota(self):
        return {"memories": {"num": 1, "max": 100}}

    def signup(self, email: str):
        return {"content": f"sent to {email}"}

    def feedback(self, content: str):
        return {"ok": bool(content)}


def test_save_config_writes_profile_scoped_memori_json(tmp_path: Path) -> None:
    provider = MemoriMemoryProvider()

    provider.save_config(
        {"entity_id": "user-1", "project_id": "project-1"},
        str(tmp_path),
    )

    data = json.loads((tmp_path / "memori.json").read_text())
    assert data == {"entityId": "user-1", "projectId": "project-1"}


def test_prefetch_does_not_auto_recall() -> None:
    provider = MemoriMemoryProvider(client=FakeClient())

    result = provider.prefetch("database")

    assert result == ""


def test_sync_turn_runs_background_capture() -> None:
    client = FakeClient()
    provider = MemoriMemoryProvider(client=client)
    provider._session_id = "session-1"

    provider.sync_turn("hello", "hi")
    provider.shutdown()

    assert client.captured == [("hello", "hi", "session-1", "hermes", None)]


def test_sync_turn_derives_trace_from_current_hermes_turn_only() -> None:
    client = FakeClient()
    provider = MemoriMemoryProvider(client=client)
    provider._session_id = "session-1"

    provider.sync_turn(
        "run tests",
        "tests passed",
        messages=[
            {"role": "user", "content": "old request"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "old-call",
                        "type": "function",
                        "function": {
                            "name": "terminal",
                            "arguments": '{"command": "old"}',
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "old-call", "content": "old result"},
            {"role": "assistant", "content": "old answer"},
            {"role": "user", "content": "run tests"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {
                            "name": "terminal",
                            "arguments": '{"command": "pytest"}',
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call-1", "content": "passed"},
            {"role": "assistant", "content": "tests passed"},
        ],
    )
    provider.shutdown()

    assert client.captured == [
        (
            "run tests",
            "tests passed",
            "session-1",
            "hermes",
            {
                "tools": [
                    {
                        "name": "terminal",
                        "args": {"command": "pytest"},
                        "result": "passed",
                    }
                ]
            },
        )
    ]


def test_handle_recall_adds_project_default() -> None:
    client = FakeClient()
    provider = MemoriMemoryProvider(client=client)
    provider._project_id = "project-1"

    result = json.loads(provider.handle_tool_call("memori_recall", {"query": "prefs"}))

    assert result == {"facts": [{"content": "remembered"}]}
    assert client.recall_params == {"query": "prefs", "projectId": "project-1"}


def test_handle_compaction_adds_project_default() -> None:
    client = FakeClient()
    provider = MemoriMemoryProvider(client=client)
    provider._project_id = "project-1"

    result = json.loads(
        provider.handle_tool_call("memori_compaction", {"numMessages": 3})
    )

    assert result == {
        "params": {"numMessages": 3, "projectId": "project-1"},
        "state": {"active_tasks": ["ship compaction"]},
    }


def test_tool_schemas_include_compaction() -> None:
    provider = MemoriMemoryProvider()

    names = {schema["name"] for schema in provider.get_tool_schemas()}

    assert "memori_compaction" in names


def test_handle_tool_call_returns_json_error_on_client_failure() -> None:
    class FailingClient(FakeClient):
        def quota(self):
            raise RuntimeError("network unavailable")

    provider = MemoriMemoryProvider(client=FailingClient())

    result = json.loads(provider.handle_tool_call("memori_quota", {}))

    assert result == {"error": "network unavailable"}


def test_config_schema_contains_required_setup_fields() -> None:
    provider = MemoriMemoryProvider()
    schema = provider.get_config_schema()

    keys = {field["key"] for field in schema}
    assert {"api_key", "entity_id", "project_id"} <= keys
    assert schema[0]["env_var"] == "MEMORI_API_KEY"
    assert "default" not in schema[2]
