from types import SimpleNamespace

from memori.llm._constants import OPENAI_LLM_PROVIDER
from memori.llm.pipelines.conversation_injection import (
    _inject_messages_by_provider,
    _sanitize_history_for_openai_compat,
)


def test_sanitize_drops_role_tool():
    messages = [
        {"role": "user", "content": "what's the weather in Paris?"},
        {"role": "assistant", "content": ""},
        {"role": "tool", "content": '{"temp": 12}'},
        {"role": "assistant", "content": "It's 12C in Paris."},
    ]

    cleaned = _sanitize_history_for_openai_compat(messages)

    assert cleaned == [
        {"role": "user", "content": "what's the weather in Paris?"},
        {"role": "assistant", "content": "It's 12C in Paris."},
    ]


def test_sanitize_drops_empty_assistant():
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": ""},
        {"role": "assistant", "content": "   "},
        {"role": "assistant", "content": "hello!"},
    ]

    cleaned = _sanitize_history_for_openai_compat(messages)

    assert cleaned == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello!"},
    ]


def test_sanitize_rewrites_legacy_model_role():
    messages = [
        {"role": "user", "content": "ping"},
        {"role": "model", "content": "pong"},
    ]

    cleaned = _sanitize_history_for_openai_compat(messages)

    assert cleaned == [
        {"role": "user", "content": "ping"},
        {"role": "assistant", "content": "pong"},
    ]


def test_sanitize_preserves_user_and_system():
    messages = [
        {"role": "system", "content": "you are helpful"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]

    cleaned = _sanitize_history_for_openai_compat(messages)

    assert cleaned == messages


def test_sanitize_preserves_order():
    messages = [
        {"role": "user", "content": "a"},
        {"role": "assistant", "content": "b"},
        {"role": "user", "content": "c"},
        {"role": "tool", "content": "d"},
        {"role": "assistant", "content": "e"},
    ]

    cleaned = _sanitize_history_for_openai_compat(messages)

    assert [m["content"] for m in cleaned] == ["a", "b", "c", "e"]


def test_sanitize_handles_empty_input():
    assert _sanitize_history_for_openai_compat([]) == []


def _openai_config():
    return SimpleNamespace(
        framework=SimpleNamespace(provider=None),
        llm=SimpleNamespace(provider=OPENAI_LLM_PROVIDER),
    )


def test_inject_returns_post_sanitization_count_for_tool_call_history():
    """4 rows in (user, empty assistant, tool, final assistant) but only 2
    survive sanitization. The returned count must reflect what was actually
    prepended, otherwise the downstream slice in _exclude_injected_messages
    will eat the current user message.
    """
    messages = [
        {"role": "user", "content": "what's the weather in Paris?"},
        {"role": "assistant", "content": ""},
        {"role": "tool", "content": '{"temp": 12}'},
        {"role": "assistant", "content": "It's 12C in Paris."},
    ]
    kwargs = {"messages": [{"role": "user", "content": "and in Berlin?"}]}

    kwargs, injected_count = _inject_messages_by_provider(
        _openai_config(), kwargs, messages
    )

    assert injected_count == 2
    assert kwargs["messages"][injected_count:] == [
        {"role": "user", "content": "and in Berlin?"}
    ]
    assert kwargs["messages"][:injected_count] == [
        {"role": "user", "content": "what's the weather in Paris?"},
        {"role": "assistant", "content": "It's 12C in Paris."},
    ]


def test_inject_count_matches_message_count_when_nothing_filtered():
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]
    kwargs = {"messages": [{"role": "user", "content": "next"}]}

    kwargs, injected_count = _inject_messages_by_provider(
        _openai_config(), kwargs, messages
    )

    assert injected_count == 2
    assert len(kwargs["messages"]) == 3


def test_sanitize_handles_missing_role_and_content():
    messages = [
        {"content": "no role defaults to user"},
        {"role": "assistant"},
        {"role": "user"},
    ]

    cleaned = _sanitize_history_for_openai_compat(messages)

    assert cleaned == [
        {"role": "user", "content": "no role defaults to user"},
        {"role": "user", "content": ""},
    ]
