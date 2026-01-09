"""Integration tests for OpenAI provider with real API calls.

These tests require OPENAI_API_KEY to be set in the environment.

Run with:
    OPENAI_API_KEY=sk-... pytest tests/integration/providers/test_openai.py -v

To run all integration tests:
    OPENAI_API_KEY=sk-... pytest tests/integration/ -v -m integration
"""

import pytest
from openai import AsyncOpenAI, AuthenticationError, NotFoundError, OpenAI

from tests.integration.conftest import requires_openai

# Test configuration constants
MODEL = "gpt-4o-mini"  # Cheapest model for testing
MAX_TOKENS = 50  # Minimize token usage
TEST_PROMPT = "Say 'hello' in one word."  # Minimal prompt


class TestClientRegistration:
    """Tests for verifying client wrapping/registration works correctly."""

    @requires_openai
    @pytest.mark.integration
    def test_sync_client_registration_marks_installed(
        self, memori_instance, openai_api_key
    ):
        """Verify that registering a sync client sets _memori_installed flag."""
        client = OpenAI(api_key=openai_api_key)

        assert not hasattr(client, "_memori_installed")

        memori_instance.llm.register(client)

        assert hasattr(client, "_memori_installed")
        assert getattr(client, "_memori_installed", False) is True

    @requires_openai
    @pytest.mark.integration
    def test_async_client_registration_marks_installed(
        self, memori_instance, openai_api_key
    ):
        """Verify that registering an async client sets _memori_installed flag."""
        client = AsyncOpenAI(api_key=openai_api_key)

        assert not hasattr(client, "_memori_installed")

        memori_instance.llm.register(client)

        assert hasattr(client, "_memori_installed")
        assert getattr(client, "_memori_installed", False) is True

    @requires_openai
    @pytest.mark.integration
    def test_multiple_registrations_are_idempotent(
        self, memori_instance, openai_api_key
    ):
        """Verify that registering the same client multiple times is safe."""
        client = OpenAI(api_key=openai_api_key)

        memori_instance.llm.register(client)
        original_create = client.chat.completions.create

        # Register again
        memori_instance.llm.register(client)

        # Should still be the same wrapped method
        assert client.chat.completions.create is original_create
        assert getattr(client, "_memori_installed", False) is True

    @requires_openai
    @pytest.mark.integration
    def test_registration_preserves_original_methods(
        self, memori_instance, openai_api_key
    ):
        """Verify that original methods are backed up after registration."""
        client = OpenAI(api_key=openai_api_key)

        memori_instance.llm.register(client)

        assert hasattr(client.chat, "_completions_create")
        assert hasattr(client.beta, "_chat_completions_parse")


class TestSyncChatCompletions:
    """Tests for synchronous chat.completions.create() calls."""

    @requires_openai
    @pytest.mark.integration
    def test_sync_chat_completion_returns_response(self, registered_openai_client):
        """Verify basic sync chat completion works and returns valid response."""
        response = registered_openai_client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": TEST_PROMPT}],
            max_tokens=MAX_TOKENS,
        )

        assert response is not None
        assert hasattr(response, "choices")
        assert len(response.choices) > 0
        assert response.choices[0].message.content is not None

    @requires_openai
    @pytest.mark.integration
    def test_sync_chat_completion_response_structure(self, registered_openai_client):
        """Verify response has expected OpenAI structure."""
        response = registered_openai_client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": TEST_PROMPT}],
            max_tokens=MAX_TOKENS,
        )

        # Validate response structure
        assert hasattr(response, "id")
        assert hasattr(response, "model")
        assert hasattr(response, "choices")
        assert hasattr(response, "usage")

        # Validate choice structure
        choice = response.choices[0]
        assert hasattr(choice, "message")
        assert hasattr(choice, "finish_reason")
        assert hasattr(choice.message, "role")
        assert hasattr(choice.message, "content")
        assert choice.message.role == "assistant"

    @requires_openai
    @pytest.mark.integration
    def test_sync_chat_completion_with_system_message(self, registered_openai_client):
        """Verify sync completion works with system message."""
        response = registered_openai_client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": TEST_PROMPT},
            ],
            max_tokens=MAX_TOKENS,
        )

        assert response is not None
        assert response.choices[0].message.content is not None

    @requires_openai
    @pytest.mark.integration
    def test_sync_chat_completion_multi_turn(self, registered_openai_client):
        """Verify multi-turn conversation works."""
        response = registered_openai_client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "user", "content": "My name is Alice."},
                {"role": "assistant", "content": "Nice to meet you, Alice!"},
                {"role": "user", "content": "What is my name?"},
            ],
            max_tokens=MAX_TOKENS,
        )

        assert response is not None
        content = response.choices[0].message.content.lower()
        assert "alice" in content


class TestAsyncChatCompletions:
    """Tests for asynchronous chat.completions.create() calls."""

    @requires_openai
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_async_chat_completion_returns_response(
        self, registered_async_openai_client
    ):
        """Verify basic async chat completion works and returns valid response."""
        response = await registered_async_openai_client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": TEST_PROMPT}],
            max_tokens=MAX_TOKENS,
        )

        assert response is not None
        assert hasattr(response, "choices")
        assert len(response.choices) > 0
        assert response.choices[0].message.content is not None

    @requires_openai
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_async_chat_completion_response_structure(
        self, registered_async_openai_client
    ):
        """Verify async response has expected OpenAI structure."""
        response = await registered_async_openai_client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": TEST_PROMPT}],
            max_tokens=MAX_TOKENS,
        )

        # Validate response structure
        assert hasattr(response, "id")
        assert hasattr(response, "model")
        assert hasattr(response, "choices")
        assert hasattr(response, "usage")

        # Validate choice structure
        choice = response.choices[0]
        assert hasattr(choice, "message")
        assert choice.message.role == "assistant"

    @requires_openai
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_async_chat_completion_with_system_message(
        self, registered_async_openai_client
    ):
        """Verify async completion works with system message."""
        response = await registered_async_openai_client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": TEST_PROMPT},
            ],
            max_tokens=MAX_TOKENS,
        )

        assert response is not None
        assert response.choices[0].message.content is not None


class TestSyncStreaming:
    """Tests for synchronous streaming responses."""

    @requires_openai
    @pytest.mark.integration
    def test_sync_streaming_returns_chunks(self, registered_streaming_openai_client):
        """Verify sync streaming returns iterable chunks."""
        stream = registered_streaming_openai_client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": TEST_PROMPT}],
            max_tokens=MAX_TOKENS,
            stream=True,
        )

        chunks = list(stream)

        assert len(chunks) > 0

    @requires_openai
    @pytest.mark.integration
    def test_sync_streaming_assembles_content(self, registered_streaming_openai_client):
        """Verify sync streaming content can be assembled."""
        stream = registered_streaming_openai_client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": TEST_PROMPT}],
            max_tokens=MAX_TOKENS,
            stream=True,
        )

        content_parts = []
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                content_parts.append(chunk.choices[0].delta.content)

        full_content = "".join(content_parts)
        assert len(full_content) > 0

    @requires_openai
    @pytest.mark.integration
    def test_sync_streaming_chunk_structure(self, registered_streaming_openai_client):
        """Verify streaming chunks have expected structure."""
        stream = registered_streaming_openai_client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": TEST_PROMPT}],
            max_tokens=MAX_TOKENS,
            stream=True,
        )

        for chunk in stream:
            assert hasattr(chunk, "choices")
            if chunk.choices:
                assert hasattr(chunk.choices[0], "delta")


class TestAsyncStreaming:
    """Tests for asynchronous streaming responses."""

    @requires_openai
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_async_streaming_returns_chunks(
        self, registered_async_streaming_client
    ):
        """Verify async streaming returns async iterable chunks."""
        stream = await registered_async_streaming_client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": TEST_PROMPT}],
            max_tokens=MAX_TOKENS,
            stream=True,
        )

        chunks = []
        async for chunk in stream:
            chunks.append(chunk)

        assert len(chunks) > 0

    @requires_openai
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_async_streaming_assembles_content(
        self, registered_async_streaming_client
    ):
        """Verify async streaming content can be assembled."""
        stream = await registered_async_streaming_client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": TEST_PROMPT}],
            max_tokens=MAX_TOKENS,
            stream=True,
        )

        content_parts = []
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                content_parts.append(chunk.choices[0].delta.content)

        full_content = "".join(content_parts)
        assert len(full_content) > 0

    @requires_openai
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_async_streaming_chunk_structure(
        self, registered_async_streaming_client
    ):
        """Verify async streaming chunks have expected structure."""
        stream = await registered_async_streaming_client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": TEST_PROMPT}],
            max_tokens=MAX_TOKENS,
            stream=True,
        )

        async for chunk in stream:
            # Each chunk should have standard OpenAI structure
            assert hasattr(chunk, "choices")
            assert hasattr(chunk, "id")
            assert hasattr(chunk, "model")

    @requires_openai
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_async_streaming_with_usage_info(
        self, registered_async_streaming_client
    ):
        """Verify async streaming includes usage when requested."""
        stream = await registered_async_streaming_client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": TEST_PROMPT}],
            max_tokens=MAX_TOKENS,
            stream=True,
            stream_options={"include_usage": True},
        )

        last_chunk = None
        async for chunk in stream:
            last_chunk = chunk

        # The last chunk should exist
        assert last_chunk is not None


class TestErrorHandling:
    """Tests for error handling scenarios."""

    @pytest.mark.integration
    def test_invalid_api_key_raises_authentication_error(self, memori_instance):
        """Verify invalid API key raises AuthenticationError."""
        client = OpenAI(api_key="invalid-key-12345")
        memori_instance.llm.register(client)

        with pytest.raises(AuthenticationError):
            client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": TEST_PROMPT}],
                max_tokens=MAX_TOKENS,
            )

    @requires_openai
    @pytest.mark.integration
    def test_invalid_model_raises_error(self, registered_openai_client):
        """Verify invalid model name raises appropriate error."""
        with pytest.raises(NotFoundError):
            registered_openai_client.chat.completions.create(
                model="nonexistent-model-xyz",
                messages=[{"role": "user", "content": TEST_PROMPT}],
                max_tokens=MAX_TOKENS,
            )

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_async_invalid_api_key_raises_error(self, memori_instance):
        """Verify async client with invalid API key raises error."""
        client = AsyncOpenAI(api_key="invalid-key-12345")
        memori_instance.llm.register(client)

        with pytest.raises(AuthenticationError):
            await client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": TEST_PROMPT}],
                max_tokens=MAX_TOKENS,
            )


class TestResponseFormatValidation:
    """Tests for validating response formats and content."""

    @requires_openai
    @pytest.mark.integration
    def test_response_contains_usage_metadata(self, registered_openai_client):
        """Verify response contains token usage metadata."""
        response = registered_openai_client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": TEST_PROMPT}],
            max_tokens=MAX_TOKENS,
        )

        assert response.usage is not None
        assert response.usage.prompt_tokens > 0
        assert response.usage.completion_tokens > 0
        assert response.usage.total_tokens > 0

    @requires_openai
    @pytest.mark.integration
    def test_response_model_matches_request(self, registered_openai_client):
        """Verify response model matches or is related to requested model."""
        response = registered_openai_client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": TEST_PROMPT}],
            max_tokens=MAX_TOKENS,
        )

        # Model should contain the base model name
        assert "gpt-4o-mini" in response.model

    @requires_openai
    @pytest.mark.integration
    def test_response_finish_reason_is_valid(self, registered_openai_client):
        """Verify finish_reason is one of expected values."""
        response = registered_openai_client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": TEST_PROMPT}],
            max_tokens=MAX_TOKENS,
        )

        valid_reasons = {
            "stop",
            "length",
            "content_filter",
            "tool_calls",
            "function_call",
        }
        assert response.choices[0].finish_reason in valid_reasons

    @requires_openai
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_async_response_contains_usage_metadata(
        self, registered_async_openai_client
    ):
        """Verify async response contains token usage metadata."""
        response = await registered_async_openai_client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": TEST_PROMPT}],
            max_tokens=MAX_TOKENS,
        )

        assert response.usage is not None
        assert response.usage.prompt_tokens > 0
        assert response.usage.completion_tokens > 0


class TestMemoriIntegration:
    """Tests to verify Memori-specific integration functionality."""

    @requires_openai
    @pytest.mark.integration
    def test_memori_wrapper_does_not_modify_response_type(
        self, openai_api_key, memori_instance
    ):
        """Verify Memori wrapper doesn't alter the response type."""
        # Get response without Memori wrapper
        unwrapped_client = OpenAI(api_key=openai_api_key)

        # Get response with Memori wrapper
        wrapped_client = OpenAI(api_key=openai_api_key)
        memori_instance.llm.register(wrapped_client)
        memori_instance.attribution(entity_id="test", process_id="test")

        # Both should return valid responses
        unwrapped_response = unwrapped_client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": TEST_PROMPT}],
            max_tokens=MAX_TOKENS,
        )

        wrapped_response = wrapped_client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": TEST_PROMPT}],
            max_tokens=MAX_TOKENS,
        )

        # Response types should be identical
        assert type(unwrapped_response) is type(wrapped_response)

    @requires_openai
    @pytest.mark.integration
    def test_config_captures_provider_info(self, memori_instance, openai_api_key):
        """Verify Memori config captures OpenAI provider information."""
        client = OpenAI(api_key=openai_api_key)
        memori_instance.llm.register(client)

        # Provider SDK version should be set after registration
        assert memori_instance.config.llm.provider_sdk_version is not None

    @requires_openai
    @pytest.mark.integration
    def test_attribution_is_preserved_across_calls(
        self, registered_openai_client, memori_instance
    ):
        """Verify attribution remains set across multiple API calls."""
        memori_instance.attribution(entity_id="user-123", process_id="process-456")

        # Make first call
        registered_openai_client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": TEST_PROMPT}],
            max_tokens=MAX_TOKENS,
        )

        assert memori_instance.config.entity_id == "user-123"
        assert memori_instance.config.process_id == "process-456"

        # Make second call
        registered_openai_client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": TEST_PROMPT}],
            max_tokens=MAX_TOKENS,
        )

        # Attribution should still be set
        assert memori_instance.config.entity_id == "user-123"
        assert memori_instance.config.process_id == "process-456"


class TestBetaApi:
    """Tests for OpenAI beta API features like structured output."""

    @requires_openai
    @pytest.mark.integration
    def test_beta_parse_registration(self, memori_instance, openai_api_key):
        """Verify beta.chat.completions.parse is wrapped."""
        client = OpenAI(api_key=openai_api_key)
        memori_instance.llm.register(client)

        assert hasattr(client.beta, "_chat_completions_parse")


class TestStorageVerification:
    """Tests to verify conversations are stored in the database."""

    @requires_openai
    @pytest.mark.integration
    def test_conversation_stored_after_sync_call(
        self, registered_openai_client, memori_instance
    ):
        """After chat.completions.create(), verify conversation record exists."""
        # Make an API call
        registered_openai_client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": TEST_PROMPT}],
            max_tokens=MAX_TOKENS,
        )

        # Verify conversation was created
        conversation_id = memori_instance.config.cache.conversation_id
        assert conversation_id is not None

        # Read conversation from storage
        conversation = memori_instance.config.storage.driver.conversation.read(
            conversation_id
        )
        assert conversation is not None
        assert conversation["id"] == conversation_id

    @requires_openai
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_conversation_stored_after_async_call(
        self, registered_async_openai_client, memori_instance
    ):
        """After async chat.completions.create(), verify conversation exists."""
        # Make an async API call
        await registered_async_openai_client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": TEST_PROMPT}],
            max_tokens=MAX_TOKENS,
        )

        # Verify conversation was created
        conversation_id = memori_instance.config.cache.conversation_id
        assert conversation_id is not None

        # Read conversation from storage
        conversation = memori_instance.config.storage.driver.conversation.read(
            conversation_id
        )
        assert conversation is not None

    @requires_openai
    @pytest.mark.integration
    def test_messages_stored_with_content(
        self, registered_openai_client, memori_instance
    ):
        """Verify user query and assistant response are stored."""
        test_query = "What is 2 + 2?"

        # Make an API call
        registered_openai_client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": test_query}],
            max_tokens=MAX_TOKENS,
        )

        # Get conversation ID
        conversation_id = memori_instance.config.cache.conversation_id
        assert conversation_id is not None

        # Read messages from storage
        messages = memori_instance.config.storage.driver.conversation.messages.read(
            conversation_id
        )

        # Should have at least user and assistant messages
        assert len(messages) >= 2

        # Find user message
        user_messages = [m for m in messages if m["role"] == "user"]
        assert len(user_messages) >= 1
        assert test_query in user_messages[0]["content"]

        # Find assistant message
        assistant_messages = [m for m in messages if m["role"] == "assistant"]
        assert len(assistant_messages) >= 1
        assert len(assistant_messages[0]["content"]) > 0

    @requires_openai
    @pytest.mark.integration
    def test_multiple_calls_accumulate_messages(
        self, registered_openai_client, memori_instance
    ):
        """Verify multiple API calls store multiple message pairs."""
        # First call
        registered_openai_client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": "First question"}],
            max_tokens=MAX_TOKENS,
        )

        conversation_id = memori_instance.config.cache.conversation_id
        messages_after_first = (
            memori_instance.config.storage.driver.conversation.messages.read(
                conversation_id
            )
        )
        count_after_first = len(messages_after_first)

        # Second call
        registered_openai_client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": "Second question"}],
            max_tokens=MAX_TOKENS,
        )

        messages_after_second = (
            memori_instance.config.storage.driver.conversation.messages.read(
                conversation_id
            )
        )
        count_after_second = len(messages_after_second)

        # Should have more messages after second call
        assert count_after_second > count_after_first
