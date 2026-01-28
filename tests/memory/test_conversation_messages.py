import json

from memori.memory._conversation_messages import parse_payload_conversation_messages


def test_parse_payload_conversation_messages_stringifies(mocker):
    payload = {
        "conversation": {"client": {"provider": "x", "title": "y"}},
    }

    adapter = mocker.Mock()
    adapter.get_formatted_query.return_value = [
        {"role": "system", "content": "ignore"},
        {"role": "user", "content": {"a": 1}},
        {"role": "assistant", "content": ["x", 2]},
    ]
    adapter.get_formatted_response.return_value = [
        {"role": "assistant", "type": "output_text", "text": "ok"},
    ]

    messages = list(parse_payload_conversation_messages(payload, adapter=adapter))

    assert messages == [
        {"role": "system", "type": None, "text": "ignore"},
        {"role": "user", "type": None, "text": json.dumps({"a": 1})},
        {"role": "assistant", "type": None, "text": json.dumps(["x", 2])},
        {"role": "assistant", "type": "output_text", "text": "ok"},
    ]

    adapter.get_formatted_query.assert_called_once_with(payload)
    adapter.get_formatted_response.assert_called_once_with(payload)


def test_parse_payload_conversation_messages_uses_registry_when_no_adapter(mocker):
    payload = {
        "conversation": {"client": {"provider": "agno", "title": "OpenAIChat"}},
    }

    adapter = mocker.Mock()
    adapter.get_formatted_query.return_value = []
    adapter.get_formatted_response.return_value = []

    registry = mocker.Mock()
    registry.adapter.return_value = adapter

    list(parse_payload_conversation_messages(payload, registry=registry))

    registry.adapter.assert_called_once_with("agno", "OpenAIChat")


def test_parse_payload_conversation_messages_passthrough_existing_messages():
    payload = {
        "conversation": {
            "client": {"provider": "x", "title": "y"},
            "messages": [
                {"role": "user", "type": "text", "text": "hi"},
                {"role": "assistant", "type": "text", "text": "ok"},
            ],
        }
    }

    assert list(parse_payload_conversation_messages(payload)) == [
        {"role": "user", "type": "text", "text": "hi"},
        {"role": "assistant", "type": "text", "text": "ok"},
    ]


def test_parse_payload_conversation_messages_preserves_response_type(mocker):
    payload = {
        "conversation": {"client": {"provider": "x", "title": "y"}},
    }

    adapter = mocker.Mock()
    adapter.get_formatted_query.return_value = []
    adapter.get_formatted_response.return_value = [
        {"role": "assistant", "text": "hello"},
        {"role": "assistant", "type": "output_text", "text": "ok"},
    ]

    messages = list(parse_payload_conversation_messages(payload, adapter=adapter))

    assert messages == [
        {"role": "assistant", "type": None, "text": "hello"},
        {"role": "assistant", "type": "output_text", "text": "ok"},
    ]
