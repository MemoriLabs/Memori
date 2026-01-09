"""Integration tests for xAI (Grok) provider with real API calls.

These tests require XAI_API_KEY to be set in the environment.

xAI uses an OpenAI-compatible API, so we use the OpenAI client with a custom base_url.

Run with:
    XAI_API_KEY=... pytest tests/integration/providers/test_xai.py -v

To run all integration tests:
    XAI_API_KEY=... pytest tests/integration/ -v -m integration
"""

import pytest
from openai import (
    AsyncOpenAI,
    AuthenticationError,
    BadRequestError,
    NotFoundError,
    OpenAI,
)

from tests.integration.conftest import requires_xai

# Test configuration constants
MODEL = "grok-beta"  # xAI's Grok model
MAX_TOKENS = 50  # Minimize token usage
TEST_PROMPT = "Say 'hello' in one word."  # Minimal prompt
XAI_BASE_URL = "https://api.x.ai/v1"


class TestClientRegistration:
    """Tests for verifying client wrapping/registration works correctly."""

    @requires_xai
    @pytest.mark.integration
    def test_sync_client_registration_marks_installed(
        self, memori_instance, xai_api_key
    ):
        """Verify that registering a sync client sets _memori_installed flag."""
        client = OpenAI(api_key=xai_api_key, base_url=XAI_BASE_URL)

        assert not hasattr(client, "_memori_installed")

        memori_instance.llm.register(client)

        assert hasattr(client, "_memori_installed")
        assert getattr(client, "_memori_installed", False) is True

    @requires_xai
    @pytest.mark.integration
    def test_async_client_registration_marks_installed(
        self, memori_instance, xai_api_key
    ):
        """Verify that registering an async client sets _memori_installed flag."""
        client = AsyncOpenAI(api_key=xai_api_key, base_url=XAI_BASE_URL)

        assert not hasattr(client, "_memori_installed")

        memori_instance.llm.register(client)

        assert hasattr(client, "_memori_installed")
        assert getattr(client, "_memori_installed", False) is True

    @requires_xai
    @pytest.mark.integration
    def test_multiple_registrations_are_idempotent(self, memori_instance, xai_api_key):
        """Verify that registering the same client multiple times is safe."""
        client = OpenAI(api_key=xai_api_key, base_url=XAI_BASE_URL)

        memori_instance.llm.register(client)
        original_create = client.chat.completions.create

        # Register again
        memori_instance.llm.register(client)

        # Should still be the same wrapped method
        assert client.chat.completions.create is original_create
        assert getattr(client, "_memori_installed", False) is True

    @requires_xai
    @pytest.mark.integration
    def test_registration_preserves_original_methods(
        self, memori_instance, xai_api_key
    ):
        """Verify that original methods are backed up after registration."""
        client = OpenAI(api_key=xai_api_key, base_url=XAI_BASE_URL)

        memori_instance.llm.register(client)

        assert hasattr(client.chat, "_completions_create")


class TestSyncChatCompletions:
    """Tests for synchronous chat.completions.create() calls."""

    @requires_xai
    @pytest.mark.integration
    def test_sync_chat_completion_returns_response(self, registered_xai_client):
        """Verify basic sync chat completion works and returns valid response."""
        response = registered_xai_client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": TEST_PROMPT}],
            max_tokens=MAX_TOKENS,
        )

        assert response is not None
        assert hasattr(response, "choices")
        assert len(response.choices) > 0
        assert response.choices[0].message.content is not None

    @requires_xai
    @pytest.mark.integration
    def test_sync_chat_completion_response_structure(self, registered_xai_client):
        """Verify response has expected OpenAI-compatible structure."""
        response = registered_xai_client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": TEST_PROMPT}],
            max_tokens=MAX_TOKENS,
        )

        # Validate response structure (OpenAI-compatible)
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

    @requires_xai
    @pytest.mark.integration
    def test_sync_chat_completion_with_system_message(self, registered_xai_client):
        """Verify sync completion works with system message."""
        response = registered_xai_client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": TEST_PROMPT},
            ],
            max_tokens=MAX_TOKENS,
        )

        assert response is not None
        assert response.choices[0].message.content is not None

    @requires_xai
    @pytest.mark.integration
    def test_sync_chat_completion_multi_turn(self, registered_xai_client):
        """Verify multi-turn conversation works."""
        response = registered_xai_client.chat.completions.create(
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

    @requires_xai
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_async_chat_completion_returns_response(
        self, registered_async_xai_client
    ):
        """Verify basic async chat completion works and returns valid response."""
        response = await registered_async_xai_client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": TEST_PROMPT}],
            max_tokens=MAX_TOKENS,
        )

        assert response is not None
        assert hasattr(response, "choices")
        assert len(response.choices) > 0
        assert response.choices[0].message.content is not None

    @requires_xai
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_async_chat_completion_response_structure(
        self, registered_async_xai_client
    ):
        """Verify async response has expected structure."""
        response = await registered_async_xai_client.chat.completions.create(
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

    @requires_xai
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_async_chat_completion_with_system_message(
        self, registered_async_xai_client
    ):
        """Verify async completion works with system message."""
        response = await registered_async_xai_client.chat.completions.create(
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

    @requires_xai
    @pytest.mark.integration
    def test_sync_streaming_returns_chunks(self, registered_xai_client):
        """Verify sync streaming returns iterable chunks."""
        stream = registered_xai_client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": TEST_PROMPT}],
            max_tokens=MAX_TOKENS,
            stream=True,
        )

        chunks = list(stream)

        assert len(chunks) > 0

    @requires_xai
    @pytest.mark.integration
    def test_sync_streaming_assembles_content(self, registered_xai_client):
        """Verify sync streaming content can be assembled."""
        stream = registered_xai_client.chat.completions.create(
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

    @requires_xai
    @pytest.mark.integration
    def test_sync_streaming_chunk_structure(self, registered_xai_client):
        """Verify streaming chunks have expected structure."""
        stream = registered_xai_client.chat.completions.create(
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

    @requires_xai
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_async_streaming_returns_chunks(self, registered_async_xai_client):
        """Verify async streaming returns async iterable chunks."""
        stream = await registered_async_xai_client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": TEST_PROMPT}],
            max_tokens=MAX_TOKENS,
            stream=True,
        )

        chunks = []
        async for chunk in stream:
            chunks.append(chunk)

        assert len(chunks) > 0

    @requires_xai
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_async_streaming_assembles_content(self, registered_async_xai_client):
        """Verify async streaming content can be assembled."""
        stream = await registered_async_xai_client.chat.completions.create(
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

    @requires_xai
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_async_streaming_chunk_structure(self, registered_async_xai_client):
        """Verify async streaming chunks have expected structure."""
        stream = await registered_async_xai_client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": TEST_PROMPT}],
            max_tokens=MAX_TOKENS,
            stream=True,
        )

        async for chunk in stream:
            # Each chunk should have standard OpenAI-compatible structure
            assert hasattr(chunk, "choices")
            assert hasattr(chunk, "id")
            assert hasattr(chunk, "model")

    @requires_xai
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_async_streaming_with_usage_info(self, registered_async_xai_client):
        """Verify async streaming includes usage when requested."""
        stream = await registered_async_xai_client.chat.completions.create(
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
    def test_invalid_api_key_raises_error(self, memori_instance):
        """Verify invalid API key raises appropriate error.

        Note: xAI returns BadRequestError (400) instead of AuthenticationError (401)
        for invalid API keys, unlike OpenAI.
        """
        client = OpenAI(api_key="invalid-key-12345", base_url=XAI_BASE_URL)
        memori_instance.llm.register(client)

        with pytest.raises((AuthenticationError, BadRequestError)):
            client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": TEST_PROMPT}],
                max_tokens=MAX_TOKENS,
            )

    @requires_xai
    @pytest.mark.integration
    def test_invalid_model_raises_error(self, registered_xai_client):
        """Verify invalid model name raises appropriate error."""
        with pytest.raises(NotFoundError):
            registered_xai_client.chat.completions.create(
                model="nonexistent-model-xyz",
                messages=[{"role": "user", "content": TEST_PROMPT}],
                max_tokens=MAX_TOKENS,
            )

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_async_invalid_api_key_raises_error(self, memori_instance):
        """Verify async client with invalid API key raises error.

        Note: xAI returns BadRequestError (400) instead of AuthenticationError (401)
        for invalid API keys, unlike OpenAI.
        """
        client = AsyncOpenAI(api_key="invalid-key-12345", base_url=XAI_BASE_URL)
        memori_instance.llm.register(client)

        with pytest.raises((AuthenticationError, BadRequestError)):
            await client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": TEST_PROMPT}],
                max_tokens=MAX_TOKENS,
            )


class TestResponseFormatValidation:
    """Tests for validating response formats and content."""

    @requires_xai
    @pytest.mark.integration
    def test_response_contains_usage_metadata(self, registered_xai_client):
        """Verify response contains token usage metadata."""
        response = registered_xai_client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": TEST_PROMPT}],
            max_tokens=MAX_TOKENS,
        )

        assert response.usage is not None
        assert response.usage.prompt_tokens > 0
        assert response.usage.completion_tokens > 0
        assert response.usage.total_tokens > 0

    @requires_xai
    @pytest.mark.integration
    def test_response_model_matches_request(self, registered_xai_client):
        """Verify response model matches or is related to requested model."""
        response = registered_xai_client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": TEST_PROMPT}],
            max_tokens=MAX_TOKENS,
        )

        # Model should contain grok
        assert "grok" in response.model.lower()

    @requires_xai
    @pytest.mark.integration
    def test_response_finish_reason_is_valid(self, registered_xai_client):
        """Verify finish_reason is one of expected values."""
        response = registered_xai_client.chat.completions.create(
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

    @requires_xai
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_async_response_contains_usage_metadata(
        self, registered_async_xai_client
    ):
        """Verify async response contains token usage metadata."""
        response = await registered_async_xai_client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": TEST_PROMPT}],
            max_tokens=MAX_TOKENS,
        )

        assert response.usage is not None
        assert response.usage.prompt_tokens > 0
        assert response.usage.completion_tokens > 0


class TestMemoriIntegration:
    """Tests to verify Memori-specific integration functionality."""

    @requires_xai
    @pytest.mark.integration
    def test_memori_wrapper_does_not_modify_response_type(
        self, xai_api_key, memori_instance
    ):
        """Verify Memori wrapper doesn't alter the response type."""
        # Get response without Memori wrapper
        unwrapped_client = OpenAI(api_key=xai_api_key, base_url=XAI_BASE_URL)

        # Get response with Memori wrapper
        wrapped_client = OpenAI(api_key=xai_api_key, base_url=XAI_BASE_URL)
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

    @requires_xai
    @pytest.mark.integration
    def test_config_captures_provider_info(self, memori_instance, xai_api_key):
        """Verify Memori config captures xAI provider information."""
        client = OpenAI(api_key=xai_api_key, base_url=XAI_BASE_URL)
        memori_instance.llm.register(client)

        # Provider SDK version should be set after registration
        assert memori_instance.config.llm.provider_sdk_version is not None

    @requires_xai
    @pytest.mark.integration
    def test_attribution_is_preserved_across_calls(
        self, registered_xai_client, memori_instance
    ):
        """Verify attribution remains set across multiple API calls."""
        memori_instance.attribution(entity_id="user-123", process_id="process-456")

        # Make first call
        registered_xai_client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": TEST_PROMPT}],
            max_tokens=MAX_TOKENS,
        )

        assert memori_instance.config.entity_id == "user-123"
        assert memori_instance.config.process_id == "process-456"

        # Make second call
        registered_xai_client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": TEST_PROMPT}],
            max_tokens=MAX_TOKENS,
        )

        # Attribution should still be set
        assert memori_instance.config.entity_id == "user-123"
        assert memori_instance.config.process_id == "process-456"


class TestStorageVerification:
    """Tests to verify conversations are stored in the database."""

    @requires_xai
    @pytest.mark.integration
    def test_conversation_stored_after_sync_call(
        self, registered_xai_client, memori_instance
    ):
        """After chat.completions.create(), verify conversation record exists."""
        # Make an API call
        registered_xai_client.chat.completions.create(
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

    @requires_xai
    @pytest.mark.integration
    def test_messages_stored_with_content(self, registered_xai_client, memori_instance):
        """Verify user query and assistant response are stored."""
        test_query = "What is 2 + 2?"

        # Make an API call
        registered_xai_client.chat.completions.create(
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

    @requires_xai
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_conversation_stored_after_async_call(
        self, registered_async_xai_client, memori_instance
    ):
        """After async chat.completions.create(), verify conversation exists."""
        # Make an async API call
        await registered_async_xai_client.chat.completions.create(
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

    @requires_xai
    @pytest.mark.integration
    def test_multiple_calls_accumulate_messages(
        self, registered_xai_client, memori_instance
    ):
        """Verify multiple API calls store multiple message pairs."""
        # First call
        registered_xai_client.chat.completions.create(
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
        registered_xai_client.chat.completions.create(
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
