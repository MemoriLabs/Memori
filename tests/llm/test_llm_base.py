import json
from unittest.mock import Mock, patch

import pytest

from memori._config import Config
from memori.llm._base import BaseInvoke, BaseLlmAdaptor
from memori.llm._constants import (
    LANGCHAIN_FRAMEWORK_PROVIDER,
    LANGCHAIN_OPENAI_LLM_PROVIDER,
    OPENAI_LLM_PROVIDER,
)
from memori.llm.helpers.google_system_instruction import (
    append_to_content_dict,
    append_to_google_system_instruction_dict,
    append_to_list,
)
from memori.llm.helpers.query_extraction import (
    extract_from_contents,
    extract_text_from_parts,
    extract_user_query,
)
from memori.llm.helpers.serialization import dict_to_json, get_response_content
from memori.llm.pipelines.conversation_injection import inject_conversation_messages
from memori.llm.pipelines.post_invoke import handle_post_response
from memori.llm.pipelines.recall_injection import inject_recalled_facts


def test_dict_to_json_dict():
    assert dict_to_json({"a": "b", "c": "d"}) == {
        "a": "b",
        "c": "d",
    }


def test_dist_to_json_dict_has_dict():
    assert dict_to_json({"a": {"b": {"c": "d"}, "e": 123}}) == {
        "a": {"b": {"c": "d"}, "e": 123}
    }


def test_configure_for_streaming_usage_openai():
    invoke = BaseInvoke(Config(), "abc")
    invoke.config.llm.provider = OPENAI_LLM_PROVIDER

    assert invoke.configure_for_streaming_usage({"abc": "def", "stream": True}) == {
        "abc": "def",
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    assert invoke.configure_for_streaming_usage(
        {"abc": "def", "stream": True, "stream_options": {}}
    ) == {"abc": "def", "stream": True, "stream_options": {"include_usage": True}}

    assert invoke.configure_for_streaming_usage(
        {"abc": "def", "stream": True, "stream_options": {"include_usage": False}}
    ) == {"abc": "def", "stream": True, "stream_options": {"include_usage": True}}


def test_configure_for_streaming_usage_streaming_options_is_not_dict_openai():
    invoke = BaseInvoke(Config(), "abc")
    invoke.config.llm.provider = OPENAI_LLM_PROVIDER

    assert invoke.configure_for_streaming_usage(
        {"abc": "def", "stream": True, "stream_options": 123}
    ) == {
        "abc": "def",
        "stream": True,
        "stream_options": {"include_usage": True},
    }


def test_configure_for_streaming_usage_only_if_stream_is_true_openai():
    invoke = BaseInvoke(Config(), "abc")
    invoke.config.llm.provider = OPENAI_LLM_PROVIDER

    assert invoke.configure_for_streaming_usage({"abc": "def"}) == {"abc": "def"}


def test_configure_for_streaming_usage_langchain_openai():
    invoke = BaseInvoke(Config(), "abc")
    invoke.config.framework.provider = LANGCHAIN_FRAMEWORK_PROVIDER
    invoke.config.llm.provider = OPENAI_LLM_PROVIDER

    assert invoke.configure_for_streaming_usage({"abc": "def", "stream": True}) == {
        "abc": "def",
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    assert invoke.configure_for_streaming_usage(
        {"abc": "def", "stream": True, "stream_options": {}}
    ) == {"abc": "def", "stream": True, "stream_options": {"include_usage": True}}

    assert invoke.configure_for_streaming_usage(
        {"abc": "def", "stream": True, "stream_options": {"include_usage": False}}
    ) == {"abc": "def", "stream": True, "stream_options": {"include_usage": True}}


def test_configure_for_streaming_usage_streaming_opts_is_not_dict_langchain_openai():
    invoke = BaseInvoke(Config(), "abc")
    invoke.config.framework.provider = LANGCHAIN_FRAMEWORK_PROVIDER
    invoke.config.llm.provider = LANGCHAIN_OPENAI_LLM_PROVIDER

    assert invoke.configure_for_streaming_usage(
        {"abc": "def", "stream": True, "stream_options": 123}
    ) == {
        "abc": "def",
        "stream": True,
        "stream_options": {"include_usage": True},
    }


def test_configure_for_streaming_usage_only_if_stream_is_true_langchain_openai():
    invoke = BaseInvoke(Config(), "abc")
    invoke.config.framework.provider = LANGCHAIN_FRAMEWORK_PROVIDER
    invoke.config.llm.provider = LANGCHAIN_OPENAI_LLM_PROVIDER

    assert invoke.configure_for_streaming_usage({"abc": "def"}) == {"abc": "def"}


def test_get_response_content():
    BaseInvoke(Config(), "abc")

    assert get_response_content({"abc": "def"}) == {"abc": "def"}

    class MockLegacyAPIResponse:
        def __init__(self):
            self.text = json.dumps({"abc": "def"})

    legacy_api_response = MockLegacyAPIResponse()
    legacy_api_response.__class__.__name__ = "LegacyAPIResponse"
    legacy_api_response.__class__.__module__ = "openai._legacy_response"

    assert get_response_content(legacy_api_response) == {"abc": "def"}


def test_exclude_injected_messages():
    adapter = BaseLlmAdaptor()

    # No injected count - returns all messages
    messages = [{"role": "user", "content": "Hello"}]
    payload = {"conversation": {"query": {}}}
    assert adapter._exclude_injected_messages(messages, payload) == messages

    # Injected count of 2 - slices off first 2 messages
    messages = [
        {"role": "user", "content": "injected 1"},
        {"role": "assistant", "content": "injected 2"},
        {"role": "user", "content": "new message"},
    ]
    payload = {"conversation": {"query": {"_memori_injected_count": 2}}}
    assert adapter._exclude_injected_messages(messages, payload) == [
        {"role": "user", "content": "new message"}
    ]

    # Safe navigation - missing keys don't cause errors
    assert adapter._exclude_injected_messages(messages, {}) == messages


def test_handle_post_response_without_augmentation():
    config = Config()
    invoke = BaseInvoke(config, "test_method")
    invoke.set_client("test_provider", "test_title", "1.0.0")

    kwargs = {"messages": [{"role": "user", "content": "Hello"}]}
    start_time = 1234567890.0
    raw_response = {"choices": [{"message": {"content": "Hi"}}]}

    with patch("memori.memory._manager.Manager") as mock_memory_manager:
        mock_manager_instance = Mock()
        mock_memory_manager.return_value = mock_manager_instance

        with patch(
            "memori.memory._conversation_messages.parse_payload_conversation_messages"
        ) as mock_parse:
            mock_parse.return_value = [{"role": "user", "type": None, "text": "Hello"}]

            handle_post_response(invoke, kwargs, start_time, raw_response)

            mock_memory_manager.assert_called_once_with(config)
            mock_manager_instance.execute.assert_called_once()


def test_handle_post_response_with_augmentation_no_conversation():
    config = Config()
    config.augmentation = Mock()
    config.entity_id = "test-entity"
    invoke = BaseInvoke(config, "test_method")
    invoke.set_client("test_provider", "test_title", "1.0.0")

    kwargs = {"messages": [{"role": "user", "content": "Hello"}]}
    start_time = 1234567890.0
    raw_response = {"choices": [{"message": {"content": "Hi"}}]}

    with patch("memori.memory._manager.Manager") as mock_memory_manager:
        mock_manager_instance = Mock()
        mock_memory_manager.return_value = mock_manager_instance

        with patch(
            "memori.memory._conversation_messages.parse_payload_conversation_messages"
        ) as mock_parse:
            mock_parse.return_value = [{"role": "user", "type": None, "text": "Hello"}]

            handle_post_response(invoke, kwargs, start_time, raw_response)

            mock_memory_manager.assert_called_once_with(config)
            mock_manager_instance.execute.assert_called_once()
            config.augmentation.enqueue.assert_called_once()
            call_args = config.augmentation.enqueue.call_args[0][0]
            assert call_args.conversation_id is None
            assert call_args.entity_id == "test-entity"
            assert call_args.conversation_messages[0].role == "user"
            assert call_args.conversation_messages[0].content == "Hello"


def test_handle_post_response_with_augmentation_and_conversation():
    config = Config()
    config.augmentation = Mock()
    config.entity_id = "test-entity"
    config.cache.conversation_id = 123
    invoke = BaseInvoke(config, "test_method")
    invoke.set_client("test_provider", "test_title", "1.0.0")

    kwargs = {"messages": [{"role": "user", "content": "Hello"}]}
    start_time = 1234567890.0
    raw_response = {"choices": [{"message": {"content": "Hi"}}]}

    with patch("memori.memory._manager.Manager") as mock_memory_manager:
        mock_manager_instance = Mock()
        mock_memory_manager.return_value = mock_manager_instance

        with patch(
            "memori.memory._conversation_messages.parse_payload_conversation_messages"
        ) as mock_parse:
            mock_parse.return_value = [{"role": "user", "type": None, "text": "Hello"}]

            handle_post_response(invoke, kwargs, start_time, raw_response)

            mock_memory_manager.assert_called_once_with(config)
            mock_manager_instance.execute.assert_called_once()
            config.augmentation.enqueue.assert_called_once()
            call_args = config.augmentation.enqueue.call_args[0][0]
            assert call_args.conversation_id == 123
            assert call_args.entity_id == "test-entity"
            assert call_args.conversation_messages[0].role == "user"
            assert call_args.conversation_messages[0].content == "Hello"


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        (
            {
                "messages": [
                    {"role": "system", "content": "You are helpful"},
                    {"role": "user", "content": "What is the weather?"},
                ]
            },
            "What is the weather?",
        ),
        (
            {
                "messages": [
                    {"role": "user", "content": "First question"},
                    {"role": "assistant", "content": "First answer"},
                    {"role": "user", "content": "Second question"},
                ]
            },
            "Second question",
        ),
        ({}, ""),
        ({"messages": []}, ""),
        (
            {
                "messages": [
                    {"role": "system", "content": "You are helpful"},
                    {"role": "assistant", "content": "I can help"},
                ]
            },
            "",
        ),
        ({"contents": "What is the weather?"}, "What is the weather?"),
        ({"contents": ["First message", "Second message"]}, "Second message"),
        (
            {
                "contents": [
                    {"role": "user", "parts": [{"text": "First question"}]},
                    {"role": "model", "parts": [{"text": "Answer"}]},
                    {"role": "user", "parts": [{"text": "Second question"}]},
                ]
            },
            "Second question",
        ),
        ({"contents": [{"role": "user", "parts": ["Hello", "World"]}]}, "Hello World"),
        ({"contents": []}, ""),
        ({"contents": ""}, ""),
    ],
)
def test_extract_user_query(kwargs, expected):
    assert extract_user_query(kwargs) == expected


@pytest.mark.parametrize(
    ("parts", "expected"),
    [
        (["Hello", "World"], "Hello World"),
        ([{"text": "Hello"}, {"text": "World"}], "Hello World"),
        (["Hello", {"text": "World"}], "Hello World"),
        ([], ""),
    ],
)
def test_extract_text_from_parts(parts, expected):
    assert extract_text_from_parts(parts) == expected


@pytest.mark.parametrize(
    ("contents", "expected"),
    [
        ("Hello", "Hello"),
        (["First", "Second"], "Second"),
        ([{"role": "user", "parts": [{"text": "Question"}]}], "Question"),
    ],
)
def test_extract_from_contents(contents, expected):
    assert extract_from_contents(contents) == expected


def test_inject_recalled_facts_no_storage():
    config = Config()
    config.storage = None
    invoke = BaseInvoke(config, "test_method")

    kwargs = {"messages": [{"role": "user", "content": "Hello"}]}
    result = inject_recalled_facts(invoke, kwargs)

    assert result == kwargs


def test_inject_recalled_facts_no_entity_id():
    config = Config()
    config.storage = Mock()
    config.entity_id = None
    invoke = BaseInvoke(config, "test_method")

    kwargs = {"messages": [{"role": "user", "content": "Hello"}]}
    result = inject_recalled_facts(invoke, kwargs)

    assert result == kwargs


def test_inject_recalled_facts_no_user_query():
    config = Config()
    config.storage = Mock()
    config.storage.driver = Mock()
    config.storage.driver.entity.create.return_value = 1
    config.entity_id = "test-entity"
    invoke = BaseInvoke(config, "test_method")

    kwargs = {"messages": [{"role": "system", "content": "You are helpful"}]}
    result = inject_recalled_facts(invoke, kwargs)

    assert result == kwargs


def test_inject_recalled_facts_no_facts_found():
    config = Config()
    config.storage = Mock()
    config.storage.driver = Mock()
    config.storage.driver.entity.create.return_value = 1
    config.entity_id = "test-entity"
    invoke = BaseInvoke(config, "test_method")

    kwargs = {"messages": [{"role": "user", "content": "Hello"}]}

    with patch("memori.memory.recall.Recall") as mock_recall:
        mock_recall.return_value.search_facts.return_value = []
        result = inject_recalled_facts(invoke, kwargs)

    assert result == kwargs
    assert len(kwargs["messages"]) == 1


def test_inject_recalled_facts_no_relevant_facts():
    config = Config()
    config.storage = Mock()
    config.storage.driver = Mock()
    config.storage.driver.entity.create.return_value = 1
    config.entity_id = "test-entity"
    invoke = BaseInvoke(config, "test_method")

    kwargs = {"messages": [{"role": "user", "content": "Hello"}]}

    with patch("memori.memory.recall.Recall") as mock_recall:
        mock_recall.return_value.search_facts.return_value = [
            {"content": "Irrelevant fact", "similarity": 0.05}
        ]
        result = inject_recalled_facts(invoke, kwargs)

    assert result == kwargs
    assert len(kwargs["messages"]) == 1


def test_inject_recalled_facts_success():
    config = Config()
    config.storage = Mock()
    config.storage.driver = Mock()
    config.storage.driver.entity.create.return_value = 1
    config.entity_id = "test-entity"
    invoke = BaseInvoke(config, "test_method")

    kwargs = {"messages": [{"role": "user", "content": "What do I like?"}]}

    with patch("memori.memory.recall.Recall") as mock_recall:
        mock_recall.return_value.search_facts.return_value = [
            {
                "content": "User likes pizza",
                "similarity": 0.9,
                "date_created": "2026-01-01 10:30:00",
            },
            {
                "content": "User likes coding",
                "similarity": 0.85,
                "date_created": "2026-01-02 11:15:00",
            },
        ]
        result = inject_recalled_facts(invoke, kwargs)

    assert len(result["messages"]) == 2
    assert result["messages"][0]["role"] == "system"
    assert "User likes pizza" in result["messages"][0]["content"]
    assert (
        "User likes pizza. Stated at 2026-01-01 10:30"
        in result["messages"][0]["content"]
    )
    assert "User likes coding" in result["messages"][0]["content"]
    assert result["messages"][1]["role"] == "user"


def test_inject_recalled_facts_local_includes_summaries():
    config = Config()
    config.storage = Mock()
    config.storage.driver = Mock()
    config.storage.driver.entity.create.return_value = 1
    config.entity_id = "test-entity"
    invoke = BaseInvoke(config, "test_method")

    kwargs = {"messages": [{"role": "user", "content": "What should I remember?"}]}

    with patch("memori.memory.recall.Recall") as mock_recall:
        mock_recall.return_value.search_facts.return_value = [
            {
                "id": 1,
                "content": "User likes structured answers",
                "similarity": 0.92,
                "date_created": "2026-01-01 10:30:00",
                "summaries": [
                    {
                        "content": "Prefers concise bullets",
                        "date_created": "2026-01-02 11:15:00",
                    }
                ],
            }
        ]
        result = inject_recalled_facts(invoke, kwargs)

    assert "User likes structured answers" in result["messages"][0]["content"]
    assert "## Summaries" in result["messages"][0]["content"]
    assert "Prefers concise bullets" in result["messages"][0]["content"]


def test_inject_recalled_facts_local_dedupes_repeated_summaries():
    config = Config()
    config.storage = Mock()
    config.storage.driver = Mock()
    config.storage.driver.entity.create.return_value = 1
    config.entity_id = "test-entity"
    invoke = BaseInvoke(config, "test_method")

    kwargs = {"messages": [{"role": "user", "content": "What should I remember?"}]}

    repeated_summary = {
        "content": "The assistant stated the favorite actor is Viggo Mortensen.",
        "date_created": "2026-03-24 18:01:00",
    }
    with patch("memori.memory.recall.Recall") as mock_recall:
        mock_recall.return_value.search_facts.return_value = [
            {
                "id": 1,
                "content": "Favorite actor is Viggo Mortensen",
                "similarity": 0.95,
                "date_created": "2026-03-24 18:01:00",
                "summaries": [repeated_summary, repeated_summary],
            },
            {
                "id": 2,
                "content": "Favorite movie is Lord of the Rings",
                "similarity": 0.91,
                "date_created": "2026-03-24 18:00:00",
                "summaries": [repeated_summary],
            },
        ]
        result = inject_recalled_facts(invoke, kwargs)

    context = result["messages"][0]["content"]
    assert (
        context.count("The assistant stated the favorite actor is Viggo Mortensen.")
        == 1
    )


def test_inject_recalled_facts_filters_by_relevance():
    config = Config()
    config.storage = Mock()
    config.storage.driver = Mock()
    config.storage.driver.entity.create.return_value = 1
    config.entity_id = "test-entity"
    invoke = BaseInvoke(config, "test_method")

    kwargs = {"messages": [{"role": "user", "content": "Hello"}]}

    with patch("memori.memory.recall.Recall") as mock_recall:
        mock_recall.return_value.search_facts.return_value = [
            {"content": "Relevant fact", "similarity": 0.9},
            {"content": "Irrelevant fact", "similarity": 0.05},
        ]
        result = inject_recalled_facts(invoke, kwargs)

    assert len(result["messages"]) == 2
    assert "Relevant fact" in result["messages"][0]["content"]
    assert "Irrelevant fact" not in result["messages"][0]["content"]


def test_inject_recalled_facts_extends_existing_system_message():
    config = Config()
    config.storage = Mock()
    config.storage.driver = Mock()
    config.storage.driver.entity.create.return_value = 1
    config.entity_id = "test-entity"
    invoke = BaseInvoke(config, "test_method")

    kwargs = {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What do I like?"},
        ]
    }

    with patch("memori.memory.recall.Recall") as mock_recall:
        mock_recall.return_value.search_facts.return_value = [
            {"content": "User likes pizza", "similarity": 0.9},
        ]
        result = inject_recalled_facts(invoke, kwargs)

    # Should still have 2 messages (not 3)
    assert len(result["messages"]) == 2
    # First message should still be system role
    assert result["messages"][0]["role"] == "system"
    # System message should contain both original content and recalled facts
    assert "You are a helpful assistant." in result["messages"][0]["content"]
    assert "User likes pizza" in result["messages"][0]["content"]
    assert "Relevant context about the user" in result["messages"][0]["content"]


def test_inject_recalled_facts_creates_system_message_when_none_exists():
    config = Config()
    config.storage = Mock()
    config.storage.driver = Mock()
    config.storage.driver.entity.create.return_value = 1
    config.entity_id = "test-entity"
    invoke = BaseInvoke(config, "test_method")

    kwargs = {"messages": [{"role": "user", "content": "What do I like?"}]}

    with patch("memori.memory.recall.Recall") as mock_recall:
        mock_recall.return_value.search_facts.return_value = [
            {"content": "User likes pizza", "similarity": 0.9},
        ]
        result = inject_recalled_facts(invoke, kwargs)

    # Should have 2 messages now (system + user)
    assert len(result["messages"]) == 2
    # First message should be system role
    assert result["messages"][0]["role"] == "system"
    # System message should contain recalled facts
    assert "User likes pizza" in result["messages"][0]["content"]
    assert "Relevant context about the user" in result["messages"][0]["content"]


def test_inject_recalled_facts_google_creates_config():
    config = Config()
    config.storage = Mock()
    config.storage.driver = Mock()
    config.storage.driver.entity.create.return_value = 1
    config.entity_id = "test-entity"
    config.framework.provider = "langchain"
    config.llm.provider = "chatgooglegenai"
    invoke = BaseInvoke(config, "test_method")

    kwargs = {"contents": "What do I like?"}

    with patch("memori.memory.recall.Recall") as mock_recall:
        mock_recall.return_value.search_facts.return_value = [
            {"content": "User likes pizza", "similarity": 0.9},
        ]
        result = inject_recalled_facts(invoke, kwargs)

    assert "config" in result
    assert "system_instruction" in result["config"]
    assert "User likes pizza" in result["config"]["system_instruction"]


def test_inject_recalled_facts_google_extends_existing_config():
    config = Config()
    config.storage = Mock()
    config.storage.driver = Mock()
    config.storage.driver.entity.create.return_value = 1
    config.entity_id = "test-entity"
    config.framework.provider = "langchain"
    config.llm.provider = "chatgooglegenai"
    invoke = BaseInvoke(config, "test_method")

    kwargs = {
        "contents": "What do I like?",
        "config": {"system_instruction": "You are helpful."},
    }

    with patch("memori.memory.recall.Recall") as mock_recall:
        mock_recall.return_value.search_facts.return_value = [
            {"content": "User likes pizza", "similarity": 0.9},
        ]
        result = inject_recalled_facts(invoke, kwargs)

    assert "You are helpful." in result["config"]["system_instruction"]
    assert "User likes pizza" in result["config"]["system_instruction"]


def test_inject_recalled_facts_google_with_contents_list():
    config = Config()
    config.storage = Mock()
    config.storage.driver = Mock()
    config.storage.driver.entity.create.return_value = 1
    config.entity_id = "test-entity"
    config.framework.provider = "langchain"
    config.llm.provider = "chatgooglegenai"
    invoke = BaseInvoke(config, "test_method")

    kwargs = {
        "contents": [
            {"role": "user", "parts": [{"text": "What do I like?"}]},
        ]
    }

    with patch("memori.memory.recall.Recall") as mock_recall:
        mock_recall.return_value.search_facts.return_value = [
            {"content": "User likes pizza", "similarity": 0.9},
        ]
        result = inject_recalled_facts(invoke, kwargs)

    assert "config" in result
    assert "system_instruction" in result["config"]
    assert "User likes pizza" in result["config"]["system_instruction"]


def test_append_to_google_system_instruction_dict_empty():
    config = {}
    append_to_google_system_instruction_dict(config, "\n\ntest context")
    assert config["system_instruction"] == "test context"


def test_append_to_google_system_instruction_dict_string():
    config = {"system_instruction": "Existing."}
    append_to_google_system_instruction_dict(config, "\n\ntest context")
    assert config["system_instruction"] == "Existing.\n\ntest context"


def test_append_to_google_system_instruction_dict_list_of_dicts():
    config = {"system_instruction": [{"text": "Existing."}]}
    append_to_google_system_instruction_dict(config, "\n\ntest context")
    assert config["system_instruction"][0]["text"] == "Existing.\n\ntest context"


def test_append_to_google_system_instruction_dict_list_of_strings():
    config = {"system_instruction": ["Existing."]}
    append_to_google_system_instruction_dict(config, "\n\ntest context")
    assert config["system_instruction"][0] == "Existing.\n\ntest context"


def test_append_to_list_empty():
    parent = {"key": []}
    append_to_list(parent["key"], "\n\ntest", parent, "key")
    assert parent["key"] == [{"text": "test"}]


def test_append_to_list_dict_with_text():
    lst = [{"text": "Existing"}]
    parent = {"key": lst}
    append_to_list(lst, "\n\ntest", parent, "key")
    assert lst[0]["text"] == "Existing\n\ntest"


def test_append_to_list_strings():
    lst = ["Existing"]
    parent = {"key": lst}
    append_to_list(lst, "\n\ntest", parent, "key")
    assert lst[0] == "Existing\n\ntest"


def test_append_to_content_dict_with_parts():
    content = {"parts": [{"text": "Existing"}]}
    parent = {"key": content}
    append_to_content_dict(content, "\n\ntest", parent, "key")
    assert content["parts"][0]["text"] == "Existing\n\ntest"


def test_append_to_content_dict_with_text():
    content = {"text": "Existing"}
    parent = {"key": content}
    append_to_content_dict(content, "\n\ntest", parent, "key")
    assert content["text"] == "Existing\n\ntest"


def test_inject_conversation_messages_no_conversation_id():
    config = Config()
    config.cache.conversation_id = None
    invoke = BaseInvoke(config, "test_method")

    kwargs = {"messages": [{"role": "user", "content": "Hello"}]}
    result = inject_conversation_messages(invoke, kwargs)

    assert result == kwargs


def test_inject_conversation_messages_no_storage():
    config = Config()
    config.cache.conversation_id = 123
    config.storage = None
    invoke = BaseInvoke(config, "test_method")

    kwargs = {"messages": [{"role": "user", "content": "Hello"}]}
    result = inject_conversation_messages(invoke, kwargs)

    assert result == kwargs


def test_inject_conversation_messages_no_messages():
    config = Config()
    config.cache.conversation_id = 123
    config.storage = Mock()
    config.storage.driver = Mock()
    config.storage.driver.conversation.messages.read.return_value = []
    invoke = BaseInvoke(config, "test_method")

    kwargs = {"messages": [{"role": "user", "content": "Hello"}]}
    result = inject_conversation_messages(invoke, kwargs)

    assert result == kwargs
    assert invoke._injected_message_count == 0


def test_inject_conversation_messages_openai_success():
    config = Config()
    config.cache.conversation_id = 123
    config.llm.provider = OPENAI_LLM_PROVIDER
    config.storage = Mock()
    config.storage.driver = Mock()
    config.storage.driver.conversation.messages.read.return_value = [
        {"role": "user", "content": "Previous question"},
        {"role": "assistant", "content": "Previous answer"},
    ]
    invoke = BaseInvoke(config, "test_method")

    kwargs = {"messages": [{"role": "user", "content": "New question"}]}
    result = inject_conversation_messages(invoke, kwargs)

    assert len(result["messages"]) == 3
    assert result["messages"][0]["content"] == "Previous question"
    assert result["messages"][1]["content"] == "Previous answer"
    assert result["messages"][2]["content"] == "New question"
    assert invoke._injected_message_count == 2


def test_inject_conversation_messages_cache_miss_loads_from_session(mocker):
    config = Config()
    config.session_id = "session-uuid"
    config.llm.provider = OPENAI_LLM_PROVIDER

    mock_driver = mocker.MagicMock()
    mock_driver.session.read.return_value = 11
    mock_driver.conversation.read_id_by_session_id.return_value = 22
    mock_driver.conversation.messages.read.return_value = [
        {"role": "user", "content": "Previous question"},
        {"role": "assistant", "content": "Previous answer"},
    ]

    mock_storage = mocker.MagicMock()
    mock_storage.driver = mock_driver
    config.storage = mock_storage

    invoke = BaseInvoke(config, "test_method")
    kwargs = {"messages": [{"role": "user", "content": "New question"}]}

    result = inject_conversation_messages(invoke, kwargs)

    assert config.cache.session_id == 11
    assert config.cache.conversation_id == 22
    mock_driver.session.read.assert_called_once_with("session-uuid")
    mock_driver.conversation.read_id_by_session_id.assert_called_once_with(11)
    mock_driver.conversation.messages.read.assert_called_once_with(22)
    assert [m["content"] for m in result["messages"]] == [
        "Previous question",
        "Previous answer",
        "New question",
    ]


def test_inject_conversation_messages_cloud_fetches_from_cloud(mocker):
    config = Config()
    config.cloud = True
    config.session_id = "session-uuid"
    config.entity_id = "entity-id"
    config.llm.provider = OPENAI_LLM_PROVIDER

    invoke = BaseInvoke(config, "test_method")
    kwargs = {"messages": [{"role": "user", "content": "New question"}]}

    mocker.patch(
        "memori.memory.recall.Recall._cloud_recall",
        autospec=True,
        return_value={
            "facts": [],
            "messages": [
                {"role": "user", "content": "cloud previous question"},
                {"role": "assistant", "content": "cloud previous answer"},
            ],
        },
    )

    kwargs = inject_recalled_facts(invoke, kwargs)
    result = inject_conversation_messages(invoke, kwargs)

    assert [m["content"] for m in result["messages"]] == [
        "cloud previous question",
        "cloud previous answer",
        "New question",
    ]
    assert invoke._injected_message_count == 2


def test_inject_recalled_facts_cloud_uses_filtered_summaries():
    config = Config()
    config.cloud = True
    config.session_id = "session-uuid"
    config.entity_id = "entity-id"
    invoke = BaseInvoke(config, "test_method")

    kwargs = {"messages": [{"role": "user", "content": "New question"}]}

    with patch("memori.memory.recall.Recall") as mock_recall:
        mock_recall.return_value.search_facts.return_value = {
            "facts": [
                {
                    "id": 1,
                    "content": "Relevant fact",
                    "similarity": 0.9,
                    "summaries": [
                        {
                            "content": "Relevant summary",
                            "date_created": "2026-03-09 19:50:09",
                        }
                    ],
                },
            ],
            "messages": [],
        }

        result = inject_recalled_facts(invoke, kwargs)

    assert "Relevant fact" in result["messages"][0]["content"]
    assert "## Summaries" in result["messages"][0]["content"]
    assert "[2026-03-09 19:50]" in result["messages"][0]["content"]
    assert "Relevant summary" in result["messages"][0]["content"]
    assert invoke._cloud_summaries == [
        {"content": "Relevant summary", "date_created": "2026-03-09 19:50:09"}
    ]
