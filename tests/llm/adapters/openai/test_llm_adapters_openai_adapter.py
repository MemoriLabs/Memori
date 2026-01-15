"""Unit tests for OpenAI adapter (Chat Completions and Responses API)."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from memori._config import Config
from memori.llm._base import BaseInvoke
from memori.llm._invoke import Invoke, InvokeAsync
from memori.llm._iterator import AsyncIterator, Iterator
from memori.llm.adapters.openai._adapter import Adapter


class MockEvent:
    def __init__(self, event_type: str, response=None):
        self.type = event_type
        if response is not None:
            self.response = response


class MockResponse:
    def __init__(self, output_text: str):
        self.output_text = output_text
        self.output = [
            {
                "type": "message",
                "content": [{"type": "output_text", "text": output_text}],
            }
        ]

    def model_dump(self):
        return {"output_text": self.output_text, "output": self.output}


class MockResponsesResponse:
    def __init__(self):
        self.output = []
        self.output_text = "Test response"

    def model_dump(self):
        return {"output": self.output, "output_text": self.output_text}


# Chat Completions API Tests

def test_get_formatted_query():
    assert Adapter().get_formatted_query({}) == []
    assert Adapter().get_formatted_query({"conversation": {"query": {}}}) == []

    assert Adapter().get_formatted_query(
        {
            "conversation": {
                "query": {
                    "messages": [
                        {"content": "abc", "role": "user"},
                        {"content": "def", "role": "assistant"},
                    ]
                }
            }
        }
    ) == [{"content": "abc", "role": "user"}, {"content": "def", "role": "assistant"}]


def test_get_formatted_response_streamed():
    assert Adapter().get_formatted_response({}) == []
    assert Adapter().get_formatted_query({"conversation": {"response": {}}}) == []

    assert Adapter().get_formatted_response(
        {
            "conversation": {
                "query": {"stream": True},
                "response": {
                    "choices": [
                        {"delta": {"content": "abc", "role": "assistant"}},
                        {"delta": {"content": "def", "role": "assistant"}},
                    ]
                },
            }
        }
    ) == [{"role": "assistant", "text": "abcdef", "type": "text"}]


def test_get_formatted_response_unstreamed():
    assert Adapter().get_formatted_response({}) == []
    assert Adapter().get_formatted_query({"conversation": {"response": {}}}) == []

    assert Adapter().get_formatted_response(
        {
            "conversation": {
                "query": {},
                "response": {
                    "choices": [
                        {"message": {"content": "abc", "role": "assistant"}},
                        {"message": {"content": "def", "role": "assistant"}},
                    ]
                },
            }
        }
    ) == [
        {"role": "assistant", "text": "abc", "type": "text"},
        {"role": "assistant", "text": "def", "type": "text"},
    ]


def test_get_formatted_query_with_injected_messages():
    assert Adapter().get_formatted_query(
        {
            "conversation": {
                "query": {
                    "_memori_injected_count": 2,
                    "messages": [
                        {"content": "injected 1", "role": "user"},
                        {"content": "injected 2", "role": "assistant"},
                        {"content": "new message", "role": "user"},
                        {"content": "new response", "role": "assistant"},
                    ],
                }
            }
        }
    ) == [
        {"content": "new message", "role": "user"},
        {"content": "new response", "role": "assistant"},
    ]


# Responses API Tests

def test_responses_get_formatted_query_string_input():
    payload = {
        "conversation": {
            "query": {
                "input": "Hello, how are you?",
                "instructions": "You are a helpful assistant.",
            }
        }
    }
    result = Adapter().get_formatted_query(payload)
    assert len(result) == 2
    assert result[0] == {"role": "system", "content": "You are a helpful assistant."}
    assert result[1] == {"role": "user", "content": "Hello, how are you?"}


def test_responses_get_formatted_query_list_input():
    payload = {
        "conversation": {
            "query": {
                "input": [
                    {"role": "user", "content": "First message"},
                    {"role": "assistant", "content": "First response"},
                    {"role": "user", "content": "Second message"},
                ]
            }
        }
    }
    result = Adapter().get_formatted_query(payload)
    assert len(result) == 3
    assert result[0] == {"role": "user", "content": "First message"}


def test_responses_get_formatted_query_strips_memori_context():
    payload = {
        "conversation": {
            "query": {
                "input": "Hello",
                "instructions": "Be helpful.\n\n<memori_context>\nUser likes cats.\n</memori_context>",
            }
        }
    }
    result = Adapter().get_formatted_query(payload)
    assert len(result) == 2
    assert result[0]["content"] == "Be helpful."


def test_responses_get_formatted_query_with_injected_messages():
    payload = {
        "conversation": {
            "query": {
                "_memori_injected_count": 2,
                "input": [
                    {"role": "user", "content": "Injected 1"},
                    {"role": "assistant", "content": "Injected 2"},
                    {"role": "user", "content": "Actual query"},
                ],
            }
        }
    }
    result = Adapter().get_formatted_query(payload)
    assert len(result) == 1
    assert result[0] == {"role": "user", "content": "Actual query"}


def test_responses_get_formatted_response_with_output_message():
    payload = {
        "conversation": {
            "query": {},
            "response": {
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": "Hello!"}],
                    }
                ]
            },
        }
    }
    result = Adapter().get_formatted_response(payload)
    assert len(result) == 1
    assert result[0] == {"role": "assistant", "text": "Hello!", "type": "text"}


def test_responses_get_formatted_response_fallback_to_output_text():
    payload = {
        "conversation": {
            "query": {},
            "response": {"output": [], "output_text": "Fallback text"},
        }
    }
    result = Adapter().get_formatted_response(payload)
    assert len(result) == 1
    assert result[0] == {"role": "assistant", "text": "Fallback text", "type": "text"}


# Iterator Tests with Responses API Streaming

class TestIteratorWithResponsesAPI:
    def test_iter_returns_self(self):
        config = Config()
        iterator = Iterator(config, iter([]))
        assert iterator.__iter__() is iterator

    @patch("memori.llm._iterator.MemoryManager")
    def test_yields_all_events(self, mock_memory_manager):
        config = Config()
        events = [
            MockEvent("response.created"),
            MockEvent("response.completed", MockResponse("Hello")),
        ]
        iterator = Iterator(config, iter(events))

        mock_invoke = MagicMock()
        mock_invoke._uses_protobuf = False
        mock_invoke._format_payload.return_value = {}
        mock_invoke._format_kwargs.return_value = {}
        mock_invoke._format_response.return_value = {}
        iterator.configure_invoke(mock_invoke)
        iterator.configure_request({"input": "test"}, 0)

        collected = list(iterator)
        assert len(collected) == 2

    @patch("memori.llm._iterator.MemoryManager")
    def test_captures_response_on_completed_event(self, mock_memory_manager):
        config = Config()
        mock_response = MockResponse("Test output")
        events = [MockEvent("response.completed", mock_response)]
        iterator = Iterator(config, iter(events))

        mock_invoke = MagicMock()
        mock_invoke._uses_protobuf = False
        mock_invoke._format_payload.return_value = {}
        mock_invoke._format_kwargs.return_value = {}
        mock_invoke._format_response.return_value = {}
        iterator.configure_invoke(mock_invoke)
        iterator.configure_request({"input": "test"}, 0)

        list(iterator)
        assert iterator.raw_response == mock_response.model_dump()


class TestAsyncIteratorWithResponsesAPI:
    def test_aiter_returns_self(self):
        config = Config()
        mock_source = MagicMock()
        mock_source.__aiter__.return_value = mock_source
        iterator = AsyncIterator(config, mock_source)
        assert iterator.__aiter__() is iterator

    @pytest.mark.asyncio
    @patch("memori.llm._iterator.MemoryManager")
    async def test_yields_all_events(self, mock_memory_manager):
        config = Config()
        events = [
            MockEvent("response.created"),
            MockEvent("response.completed", MockResponse("Hello")),
        ]

        async def async_gen():
            for event in events:
                yield event

        iterator = AsyncIterator(config, async_gen())

        mock_invoke = MagicMock()
        mock_invoke._uses_protobuf = False
        mock_invoke._format_payload.return_value = {}
        mock_invoke._format_kwargs.return_value = {}
        mock_invoke._format_response.return_value = {}
        iterator.configure_invoke(mock_invoke)
        iterator.configure_request({"input": "test"}, 0)
        iterator.__aiter__()

        collected = []
        async for event in iterator:
            collected.append(event)
        assert len(collected) == 2

    @pytest.mark.asyncio
    async def test_raises_runtime_error_if_not_initialized(self):
        config = Config()
        iterator = AsyncIterator(config, MagicMock())
        with pytest.raises(RuntimeError, match="Iterator not initialized"):
            await iterator.__anext__()


# BaseInvoke Tests with Responses API

class TestExtractUserQueryResponses:
    def test_extract_from_string_input(self):
        config = Config()
        invoke = BaseInvoke(config, lambda **kwargs: None)
        assert invoke._extract_user_query({"input": "What is 2+2?"}) == "What is 2+2?"

    def test_extract_from_list_input(self):
        config = Config()
        invoke = BaseInvoke(config, lambda **kwargs: None)
        kwargs = {
            "input": [
                {"role": "user", "content": "First"},
                {"role": "user", "content": "Second"},
            ]
        }
        assert invoke._extract_user_query(kwargs) == "Second"

    def test_extract_from_missing_input(self):
        config = Config()
        invoke = BaseInvoke(config, lambda **kwargs: None)
        assert invoke._extract_user_query({}) == ""


class TestInjectRecalledFactsResponses:
    def test_returns_kwargs_when_no_storage(self):
        config = Config()
        config.storage = None
        invoke = BaseInvoke(config, lambda **kwargs: None)
        kwargs = {"input": "test", "instructions": "Be helpful"}
        assert invoke.inject_recalled_facts(kwargs) == kwargs

    def test_appends_facts_to_instructions(self):
        config = Config()
        config.storage = MagicMock()
        config.storage.driver = MagicMock()
        config.storage.driver.entity.create.return_value = 1
        config.entity_id = "test-entity"
        config.recall_relevance_threshold = 0.1
        config.llm.provider = "openai_responses"

        invoke = BaseInvoke(config, lambda **kwargs: None)
        invoke.set_client(None, "openai_responses", "1.0.0")

        mock_facts = [{"content": "User likes Python", "similarity": 0.8}]

        with patch("memori.memory.recall.Recall") as MockRecall:
            MockRecall.return_value.search_facts.return_value = mock_facts
            kwargs = {"input": "Test", "instructions": "Be helpful."}
            result = invoke.inject_recalled_facts(kwargs)
            assert "<memori_context>" in result["instructions"]
            assert "User likes Python" in result["instructions"]


class TestInjectConversationMessagesResponses:
    def test_returns_kwargs_when_no_conversation_id(self):
        config = Config()
        config.cache.conversation_id = None
        invoke = BaseInvoke(config, lambda **kwargs: None)
        kwargs = {"input": "test"}
        assert invoke.inject_conversation_messages(kwargs) == kwargs

    def test_converts_string_input_to_list(self):
        config = Config()
        config.cache.conversation_id = 1
        config.storage = MagicMock()
        config.storage.driver = MagicMock()
        config.storage.driver.conversation.messages.read.return_value = [
            {"role": "user", "content": "Previous"},
        ]
        config.llm.provider = "openai_responses"

        invoke = BaseInvoke(config, lambda **kwargs: None)
        invoke.set_client(None, "openai_responses", "1.0.0")

        result = invoke.inject_conversation_messages({"input": "New"})
        assert isinstance(result["input"], list)
        assert len(result["input"]) == 2


class TestInvokeWithResponsesAPI:
    def test_invoke_calls_method(self):
        config = Config()
        config.storage = None

        mock_response = MockResponsesResponse()
        mock_method = MagicMock(return_value=mock_response)

        invoke = Invoke(config, mock_method)
        invoke.set_client(None, "openai_responses", "1.0.0")

        result = invoke.invoke(model="gpt-4o", input="test")
        mock_method.assert_called_once()
        assert result == mock_response


class TestInvokeAsyncWithResponsesAPI:
    @pytest.mark.asyncio
    async def test_async_invoke_calls_method(self):
        config = Config()
        config.storage = None

        mock_response = MockResponsesResponse()

        async def mock_method(**kwargs):
            return mock_response

        invoke = InvokeAsync(config, mock_method)
        invoke.set_client(None, "openai_responses", "1.0.0")

        result = await invoke.invoke(model="gpt-4o", input="test")
        assert result == mock_response
