import json
from unittest.mock import Mock, patch

from memori._config import Config
from memori.llm._base import BaseInvoke, BaseLlmAdaptor
from memori.llm._constants import (
    LANGCHAIN_FRAMEWORK_PROVIDER,
    LANGCHAIN_OPENAI_LLM_PROVIDER,
    OPENAI_LLM_PROVIDER,
)


def test_dict_to_json_dict():
    assert BaseInvoke(Config(), "abc").dict_to_json({"a": "b", "c": "d"}) == {
        "a": "b",
        "c": "d",
    }


def test_dist_to_json_dict_has_dict():
    assert BaseInvoke(Config(), "abc").dict_to_json(
        {"a": {"b": {"c": "d"}, "e": 123}}
    ) == {"a": {"b": {"c": "d"}, "e": 123}}


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
    invoke = BaseInvoke(Config(), "abc")

    assert invoke.get_response_content({"abc": "def"}) == {"abc": "def"}

    class MockLegacyAPIResponse:
        def __init__(self):
            self.text = json.dumps({"abc": "def"})

    legacy_api_response = MockLegacyAPIResponse()
    legacy_api_response.__class__.__name__ = "LegacyAPIResponse"
    legacy_api_response.__class__.__module__ = "openai._legacy_response"

    assert invoke.get_response_content(legacy_api_response) == {"abc": "def"}


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

            invoke.handle_post_response(kwargs, start_time, raw_response)

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

            invoke.handle_post_response(kwargs, start_time, raw_response)

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

            invoke.handle_post_response(kwargs, start_time, raw_response)

            mock_memory_manager.assert_called_once_with(config)
            mock_manager_instance.execute.assert_called_once()
            config.augmentation.enqueue.assert_called_once()
            call_args = config.augmentation.enqueue.call_args[0][0]
            assert call_args.conversation_id == 123
            assert call_args.entity_id == "test-entity"
            assert call_args.conversation_messages[0].role == "user"
            assert call_args.conversation_messages[0].content == "Hello"


def test_extract_user_query_with_user_message():
    invoke = BaseInvoke(Config(), "test_method")
    kwargs = {
        "messages": [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "What is the weather?"},
        ]
    }
    assert invoke._extract_user_query(kwargs) == "What is the weather?"


def test_extract_user_query_with_multiple_user_messages():
    invoke = BaseInvoke(Config(), "test_method")
    kwargs = {
        "messages": [
            {"role": "user", "content": "First question"},
            {"role": "assistant", "content": "First answer"},
            {"role": "user", "content": "Second question"},
        ]
    }
    assert invoke._extract_user_query(kwargs) == "Second question"


def test_extract_user_query_no_messages():
    invoke = BaseInvoke(Config(), "test_method")
    assert invoke._extract_user_query({}) == ""
    assert invoke._extract_user_query({"messages": []}) == ""


def test_extract_user_query_no_user_messages():
    invoke = BaseInvoke(Config(), "test_method")
    kwargs = {
        "messages": [
            {"role": "system", "content": "You are helpful"},
            {"role": "assistant", "content": "I can help"},
        ]
    }
    assert invoke._extract_user_query(kwargs) == ""


def test_extract_user_query_google_contents_string():
    invoke = BaseInvoke(Config(), "test_method")
    kwargs = {"contents": "What is the weather?"}
    assert invoke._extract_user_query(kwargs) == "What is the weather?"


def test_extract_user_query_google_contents_list_of_strings():
    invoke = BaseInvoke(Config(), "test_method")
    kwargs = {"contents": ["First message", "Second message"]}
    assert invoke._extract_user_query(kwargs) == "Second message"


def test_extract_user_query_google_contents_list_of_dicts():
    invoke = BaseInvoke(Config(), "test_method")
    kwargs = {
        "contents": [
            {"role": "user", "parts": [{"text": "First question"}]},
            {"role": "model", "parts": [{"text": "Answer"}]},
            {"role": "user", "parts": [{"text": "Second question"}]},
        ]
    }
    assert invoke._extract_user_query(kwargs) == "Second question"


def test_extract_user_query_google_contents_with_string_parts():
    invoke = BaseInvoke(Config(), "test_method")
    kwargs = {
        "contents": [
            {"role": "user", "parts": ["Hello", "World"]},
        ]
    }
    assert invoke._extract_user_query(kwargs) == "Hello World"


def test_extract_user_query_google_contents_empty():
    invoke = BaseInvoke(Config(), "test_method")
    assert invoke._extract_user_query({"contents": []}) == ""
    assert invoke._extract_user_query({"contents": ""}) == ""


def test_extract_text_from_parts_with_strings():
    invoke = BaseInvoke(Config(), "test_method")
    parts = ["Hello", "World"]
    assert invoke._extract_text_from_parts(parts) == "Hello World"


def test_extract_text_from_parts_with_dicts():
    invoke = BaseInvoke(Config(), "test_method")
    parts = [{"text": "Hello"}, {"text": "World"}]
    assert invoke._extract_text_from_parts(parts) == "Hello World"


def test_extract_text_from_parts_mixed():
    invoke = BaseInvoke(Config(), "test_method")
    parts = ["Hello", {"text": "World"}]
    assert invoke._extract_text_from_parts(parts) == "Hello World"


def test_extract_text_from_parts_empty():
    invoke = BaseInvoke(Config(), "test_method")
    assert invoke._extract_text_from_parts([]) == ""


def test_extract_from_contents_string():
    invoke = BaseInvoke(Config(), "test_method")
    assert invoke._extract_from_contents("Hello") == "Hello"


def test_extract_from_contents_list_strings():
    invoke = BaseInvoke(Config(), "test_method")
    assert invoke._extract_from_contents(["First", "Second"]) == "Second"


def test_extract_from_contents_list_dicts():
    invoke = BaseInvoke(Config(), "test_method")
    contents = [
        {"role": "user", "parts": [{"text": "Question"}]},
    ]
    assert invoke._extract_from_contents(contents) == "Question"


def test_inject_recalled_facts_no_storage():
    config = Config()
    config.storage = None
    invoke = BaseInvoke(config, "test_method")

    kwargs = {"messages": [{"role": "user", "content": "Hello"}]}
    result = invoke.inject_recalled_facts(kwargs)

    assert result == kwargs


def test_inject_recalled_facts_no_entity_id():
    config = Config()
    config.storage = Mock()
    config.entity_id = None
    invoke = BaseInvoke(config, "test_method")

    kwargs = {"messages": [{"role": "user", "content": "Hello"}]}
    result = invoke.inject_recalled_facts(kwargs)

    assert result == kwargs


def test_inject_recalled_facts_entity_create_returns_none():
    config = Config()
    config.storage = Mock()
    config.storage.driver = Mock()
    config.storage.driver.entity.create.return_value = None
    config.entity_id = "test-entity"
    invoke = BaseInvoke(config, "test_method")

    kwargs = {"messages": [{"role": "user", "content": "Hello"}]}
    result = invoke.inject_recalled_facts(kwargs)

    assert result == kwargs


def test_inject_recalled_facts_no_user_query():
    config = Config()
    config.storage = Mock()
    config.storage.driver = Mock()
    config.storage.driver.entity.create.return_value = 1
    config.entity_id = "test-entity"
    invoke = BaseInvoke(config, "test_method")

    kwargs = {"messages": [{"role": "system", "content": "You are helpful"}]}
    result = invoke.inject_recalled_facts(kwargs)

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
        result = invoke.inject_recalled_facts(kwargs)

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
        result = invoke.inject_recalled_facts(kwargs)

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
        result = invoke.inject_recalled_facts(kwargs)

    assert len(result["messages"]) == 2
    assert result["messages"][0]["role"] == "system"
    assert "User likes pizza" in result["messages"][0]["content"]
    assert (
        "User likes pizza. Stated at 2026-01-01 10:30"
        in result["messages"][0]["content"]
    )
    assert "User likes coding" in result["messages"][0]["content"]
    assert result["messages"][1]["role"] == "user"


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
        result = invoke.inject_recalled_facts(kwargs)

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
        result = invoke.inject_recalled_facts(kwargs)

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
        result = invoke.inject_recalled_facts(kwargs)

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
        result = invoke.inject_recalled_facts(kwargs)

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
        result = invoke.inject_recalled_facts(kwargs)

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
        result = invoke.inject_recalled_facts(kwargs)

    assert "config" in result
    assert "system_instruction" in result["config"]
    assert "User likes pizza" in result["config"]["system_instruction"]


def test_append_to_google_system_instruction_dict_empty():
    invoke = BaseInvoke(Config(), "test_method")
    config = {}
    invoke._append_to_google_system_instruction_dict(config, "\n\ntest context")
    assert config["system_instruction"] == "test context"


def test_append_to_google_system_instruction_dict_string():
    invoke = BaseInvoke(Config(), "test_method")
    config = {"system_instruction": "Existing."}
    invoke._append_to_google_system_instruction_dict(config, "\n\ntest context")
    assert config["system_instruction"] == "Existing.\n\ntest context"


def test_append_to_google_system_instruction_dict_list_of_dicts():
    invoke = BaseInvoke(Config(), "test_method")
    config = {"system_instruction": [{"text": "Existing."}]}
    invoke._append_to_google_system_instruction_dict(config, "\n\ntest context")
    assert config["system_instruction"][0]["text"] == "Existing.\n\ntest context"


def test_append_to_google_system_instruction_dict_list_of_strings():
    invoke = BaseInvoke(Config(), "test_method")
    config = {"system_instruction": ["Existing."]}
    invoke._append_to_google_system_instruction_dict(config, "\n\ntest context")
    assert config["system_instruction"][0] == "Existing.\n\ntest context"


def test_append_to_list_empty():
    invoke = BaseInvoke(Config(), "test_method")
    parent = {"key": []}
    invoke._append_to_list(parent["key"], "\n\ntest", parent, "key")
    assert parent["key"] == [{"text": "test"}]


def test_append_to_list_dict_with_text():
    invoke = BaseInvoke(Config(), "test_method")
    lst = [{"text": "Existing"}]
    parent = {"key": lst}
    invoke._append_to_list(lst, "\n\ntest", parent, "key")
    assert lst[0]["text"] == "Existing\n\ntest"


def test_append_to_list_strings():
    invoke = BaseInvoke(Config(), "test_method")
    lst = ["Existing"]
    parent = {"key": lst}
    invoke._append_to_list(lst, "\n\ntest", parent, "key")
    assert lst[0] == "Existing\n\ntest"


def test_append_to_content_dict_with_parts():
    invoke = BaseInvoke(Config(), "test_method")
    content = {"parts": [{"text": "Existing"}]}
    parent = {"key": content}
    invoke._append_to_content_dict(content, "\n\ntest", parent, "key")
    assert content["parts"][0]["text"] == "Existing\n\ntest"


def test_append_to_content_dict_with_text():
    invoke = BaseInvoke(Config(), "test_method")
    content = {"text": "Existing"}
    parent = {"key": content}
    invoke._append_to_content_dict(content, "\n\ntest", parent, "key")
    assert content["text"] == "Existing\n\ntest"


def test_inject_conversation_messages_no_conversation_id():
    config = Config()
    config.cache.conversation_id = None
    invoke = BaseInvoke(config, "test_method")

    kwargs = {"messages": [{"role": "user", "content": "Hello"}]}
    result = invoke.inject_conversation_messages(kwargs)

    assert result == kwargs


def test_inject_conversation_messages_no_storage():
    config = Config()
    config.cache.conversation_id = 123
    config.storage = None
    invoke = BaseInvoke(config, "test_method")

    kwargs = {"messages": [{"role": "user", "content": "Hello"}]}
    result = invoke.inject_conversation_messages(kwargs)

    assert result == kwargs


def test_inject_conversation_messages_no_messages():
    config = Config()
    config.cache.conversation_id = 123
    config.storage = Mock()
    config.storage.driver = Mock()
    config.storage.driver.conversation.messages.read.return_value = []
    invoke = BaseInvoke(config, "test_method")

    kwargs = {"messages": [{"role": "user", "content": "Hello"}]}
    result = invoke.inject_conversation_messages(kwargs)

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
    result = invoke.inject_conversation_messages(kwargs)

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

    result = invoke.inject_conversation_messages(kwargs)

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


def test_inject_conversation_messages_hosted_fetches_from_hosted(mocker):
    config = Config()
    config.hosted = True
    config.session_id = "session-uuid"
    config.llm.provider = OPENAI_LLM_PROVIDER

    api_instance = mocker.MagicMock()
    api_instance.get.return_value = {
        "messages": [
            {"role": "user", "text": "Hosted previous question", "type": "message"},
            {"role": "assistant", "text": "Hosted previous answer", "type": "message"},
        ]
    }
    mock_api_cls = mocker.patch("memori.llm._base.Api", autospec=True)
    mock_api_cls.return_value = api_instance

    invoke = BaseInvoke(config, "test_method")
    kwargs = {"messages": [{"role": "user", "content": "New question"}]}

    result = invoke.inject_conversation_messages(kwargs)

    mock_api_cls.assert_called_once()
    _, call_kwargs = mock_api_cls.call_args
    assert call_kwargs["subdomain"].value == "hosted-api"
    api_instance.get.assert_called_once_with("conversation/session-uuid/messages")

    assert [m["content"] for m in result["messages"]] == [
        "Hosted previous question",
        "Hosted previous answer",
        "New question",
    ]
    assert invoke._injected_message_count == 2
