"""Integration tests for Anthropic provider with real API calls.

These tests require ANTHROPIC_API_KEY to be set in the environment.

Run with:
    ANTHROPIC_API_KEY=sk-... pytest tests/integration/providers/test_anthropic.py -v

To run all integration tests:
    ANTHROPIC_API_KEY=sk-... pytest tests/integration/ -v -m integration
"""

import pytest
from anthropic import Anthropic, APIStatusError, AsyncAnthropic, AuthenticationError

from tests.integration.conftest import requires_anthropic

# Test configuration constants
MODEL = "claude-3-haiku-20240307"  # Cheapest/fastest Claude model
MAX_TOKENS = 50  # Minimize token usage
TEST_PROMPT = "Say 'hello' in one word."  # Minimal prompt


class TestClientRegistration:
    """Tests for verifying client wrapping/registration works correctly."""

    @requires_anthropic
    @pytest.mark.integration
    def test_sync_client_registration_marks_installed(
        self, memori_instance, anthropic_api_key
    ):
        """Verify that registering a sync client sets _memori_installed flag."""
        client = Anthropic(api_key=anthropic_api_key)

        assert not hasattr(client, "_memori_installed")

        memori_instance.llm.register(client)

        assert hasattr(client, "_memori_installed")
        assert getattr(client, "_memori_installed", False) is True

    @requires_anthropic
    @pytest.mark.integration
    def test_async_client_registration_marks_installed(
        self, memori_instance, anthropic_api_key
    ):
        """Verify that registering an async client sets _memori_installed flag."""
        client = AsyncAnthropic(api_key=anthropic_api_key)

        assert not hasattr(client, "_memori_installed")

        memori_instance.llm.register(client)

        assert hasattr(client, "_memori_installed")
        assert getattr(client, "_memori_installed", False) is True

    @requires_anthropic
    @pytest.mark.integration
    def test_multiple_registrations_are_idempotent(
        self, memori_instance, anthropic_api_key
    ):
        """Verify that registering the same client multiple times is safe."""
        client = Anthropic(api_key=anthropic_api_key)

        memori_instance.llm.register(client)
        original_create = client.messages.create

        # Register again
        memori_instance.llm.register(client)

        # Should still be the same wrapped method
        assert client.messages.create is original_create
        assert getattr(client, "_memori_installed", False) is True

    @requires_anthropic
    @pytest.mark.integration
    def test_registration_preserves_original_methods(
        self, memori_instance, anthropic_api_key
    ):
        """Verify that original methods are backed up after registration."""
        client = Anthropic(api_key=anthropic_api_key)

        memori_instance.llm.register(client)

        # Anthropic uses messages.create, so original should be backed up
        assert hasattr(client, "_memori_installed")


class TestSyncMessages:
    """Tests for synchronous messages.create() calls."""

    @requires_anthropic
    @pytest.mark.integration
    def test_sync_message_returns_response(self, registered_anthropic_client):
        """Verify basic sync message works and returns valid response."""
        response = registered_anthropic_client.messages.create(
            model=MODEL,
            messages=[{"role": "user", "content": TEST_PROMPT}],
            max_tokens=MAX_TOKENS,
        )

        assert response is not None
        assert hasattr(response, "content")
        assert len(response.content) > 0
        assert response.content[0].text is not None

    @requires_anthropic
    @pytest.mark.integration
    def test_sync_message_response_structure(self, registered_anthropic_client):
        """Verify response has expected Anthropic structure."""
        response = registered_anthropic_client.messages.create(
            model=MODEL,
            messages=[{"role": "user", "content": TEST_PROMPT}],
            max_tokens=MAX_TOKENS,
        )

        # Validate response structure
        assert hasattr(response, "id")
        assert hasattr(response, "model")
        assert hasattr(response, "content")
        assert hasattr(response, "usage")
        assert hasattr(response, "stop_reason")

        # Validate content structure
        assert len(response.content) > 0
        assert hasattr(response.content[0], "type")
        assert hasattr(response.content[0], "text")
        assert response.content[0].type == "text"
        assert response.role == "assistant"

    @requires_anthropic
    @pytest.mark.integration
    def test_sync_message_with_system_message(self, registered_anthropic_client):
        """Verify sync message works with system parameter."""
        response = registered_anthropic_client.messages.create(
            model=MODEL,
            system="You are a helpful assistant.",
            messages=[{"role": "user", "content": TEST_PROMPT}],
            max_tokens=MAX_TOKENS,
        )

        assert response is not None
        assert response.content[0].text is not None

    @requires_anthropic
    @pytest.mark.integration
    def test_sync_message_multi_turn(self, registered_anthropic_client):
        """Verify multi-turn conversation works."""
        response = registered_anthropic_client.messages.create(
            model=MODEL,
            messages=[
                {"role": "user", "content": "My name is Alice."},
                {"role": "assistant", "content": "Nice to meet you, Alice!"},
                {"role": "user", "content": "What is my name?"},
            ],
            max_tokens=MAX_TOKENS,
        )

        assert response is not None
        content = response.content[0].text.lower()
        assert "alice" in content


class TestAsyncMessages:
    """Tests for asynchronous messages.create() calls."""

    @requires_anthropic
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_async_message_returns_response(
        self, registered_async_anthropic_client
    ):
        """Verify basic async message works and returns valid response."""
        response = await registered_async_anthropic_client.messages.create(
            model=MODEL,
            messages=[{"role": "user", "content": TEST_PROMPT}],
            max_tokens=MAX_TOKENS,
        )

        assert response is not None
        assert hasattr(response, "content")
        assert len(response.content) > 0
        assert response.content[0].text is not None

    @requires_anthropic
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_async_message_response_structure(
        self, registered_async_anthropic_client
    ):
        """Verify async response has expected Anthropic structure."""
        response = await registered_async_anthropic_client.messages.create(
            model=MODEL,
            messages=[{"role": "user", "content": TEST_PROMPT}],
            max_tokens=MAX_TOKENS,
        )

        # Validate response structure
        assert hasattr(response, "id")
        assert hasattr(response, "model")
        assert hasattr(response, "content")
        assert hasattr(response, "usage")

        # Validate content structure
        assert len(response.content) > 0
        assert response.content[0].type == "text"

    @requires_anthropic
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_async_message_with_system(self, registered_async_anthropic_client):
        """Verify async message works with system parameter."""
        response = await registered_async_anthropic_client.messages.create(
            model=MODEL,
            system="You are a helpful assistant.",
            messages=[{"role": "user", "content": TEST_PROMPT}],
            max_tokens=MAX_TOKENS,
        )

        assert response is not None
        assert response.content[0].text is not None


class TestSyncStreaming:
    """Tests for synchronous streaming responses."""

    @requires_anthropic
    @pytest.mark.integration
    def test_sync_streaming_returns_events(self, registered_anthropic_client):
        """Verify sync streaming returns iterable events."""
        with registered_anthropic_client.messages.stream(
            model=MODEL,
            messages=[{"role": "user", "content": TEST_PROMPT}],
            max_tokens=MAX_TOKENS,
        ) as stream:
            events = list(stream.text_stream)

        assert len(events) > 0

    @requires_anthropic
    @pytest.mark.integration
    def test_sync_streaming_assembles_content(self, registered_anthropic_client):
        """Verify sync streaming content can be assembled."""
        with registered_anthropic_client.messages.stream(
            model=MODEL,
            messages=[{"role": "user", "content": TEST_PROMPT}],
            max_tokens=MAX_TOKENS,
        ) as stream:
            full_content = "".join(stream.text_stream)

        assert len(full_content) > 0

    @requires_anthropic
    @pytest.mark.integration
    def test_sync_streaming_event_structure(self, registered_anthropic_client):
        """Verify streaming events have expected structure."""
        with registered_anthropic_client.messages.stream(
            model=MODEL,
            messages=[{"role": "user", "content": TEST_PROMPT}],
            max_tokens=MAX_TOKENS,
        ) as stream:
            events = list(stream.text_stream)

        # Should have at least one text chunk
        assert len(events) > 0
        assert all(isinstance(e, str) for e in events)


class TestAsyncStreaming:
    """Tests for asynchronous streaming responses."""

    @requires_anthropic
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_async_streaming_returns_events(
        self, registered_async_anthropic_client
    ):
        """Verify async streaming returns async iterable events."""
        async with registered_async_anthropic_client.messages.stream(
            model=MODEL,
            messages=[{"role": "user", "content": TEST_PROMPT}],
            max_tokens=MAX_TOKENS,
        ) as stream:
            events = []
            async for text in stream.text_stream:
                events.append(text)

        assert len(events) > 0

    @requires_anthropic
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_async_streaming_assembles_content(
        self, registered_async_anthropic_client
    ):
        """Verify async streaming content can be assembled."""
        async with registered_async_anthropic_client.messages.stream(
            model=MODEL,
            messages=[{"role": "user", "content": TEST_PROMPT}],
            max_tokens=MAX_TOKENS,
        ) as stream:
            content_parts = []
            async for text in stream.text_stream:
                content_parts.append(text)

        full_content = "".join(content_parts)
        assert len(full_content) > 0

    @requires_anthropic
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_async_streaming_chunk_structure(
        self, registered_async_anthropic_client
    ):
        """Verify async streaming events have expected structure."""
        async with registered_async_anthropic_client.messages.stream(
            model=MODEL,
            messages=[{"role": "user", "content": TEST_PROMPT}],
            max_tokens=MAX_TOKENS,
        ) as stream:
            async for _ in stream.text_stream:
                pass
            final_message = await stream.get_final_message()

        # Verify message structure
        assert hasattr(final_message, "id")
        assert hasattr(final_message, "model")
        assert hasattr(final_message, "content")

    @requires_anthropic
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_async_streaming_final_message(
        self, registered_async_anthropic_client
    ):
        """Verify async streaming provides final message."""
        async with registered_async_anthropic_client.messages.stream(
            model=MODEL,
            messages=[{"role": "user", "content": TEST_PROMPT}],
            max_tokens=MAX_TOKENS,
        ) as stream:
            async for _ in stream.text_stream:
                pass
            final_message = await stream.get_final_message()

        assert final_message is not None
        assert hasattr(final_message, "content")
        assert len(final_message.content) > 0

    @requires_anthropic
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_async_streaming_with_usage_info(
        self, registered_async_anthropic_client
    ):
        """Verify async streaming includes usage information."""
        async with registered_async_anthropic_client.messages.stream(
            model=MODEL,
            messages=[{"role": "user", "content": TEST_PROMPT}],
            max_tokens=MAX_TOKENS,
        ) as stream:
            async for _ in stream.text_stream:
                pass
            final_message = await stream.get_final_message()

        # Anthropic includes usage in the final message
        assert final_message is not None
        assert hasattr(final_message, "usage")
        assert final_message.usage is not None
        assert hasattr(final_message.usage, "input_tokens")
        assert hasattr(final_message.usage, "output_tokens")


class TestErrorHandling:
    """Tests for error handling scenarios."""

    @pytest.mark.integration
    def test_invalid_api_key_raises_authentication_error(self, memori_instance):
        """Verify invalid API key raises AuthenticationError."""
        client = Anthropic(api_key="invalid-key-12345")
        memori_instance.llm.register(client)

        with pytest.raises(AuthenticationError):
            client.messages.create(
                model=MODEL,
                messages=[{"role": "user", "content": TEST_PROMPT}],
                max_tokens=MAX_TOKENS,
            )

    @requires_anthropic
    @pytest.mark.integration
    def test_invalid_model_raises_error(self, registered_anthropic_client):
        """Verify invalid model name raises appropriate error."""
        with pytest.raises(APIStatusError):
            registered_anthropic_client.messages.create(
                model="nonexistent-model-xyz",
                messages=[{"role": "user", "content": TEST_PROMPT}],
                max_tokens=MAX_TOKENS,
            )

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_async_invalid_api_key_raises_error(self, memori_instance):
        """Verify async client with invalid API key raises error."""
        client = AsyncAnthropic(api_key="invalid-key-12345")
        memori_instance.llm.register(client)

        with pytest.raises(AuthenticationError):
            await client.messages.create(
                model=MODEL,
                messages=[{"role": "user", "content": TEST_PROMPT}],
                max_tokens=MAX_TOKENS,
            )


class TestResponseFormatValidation:
    """Tests for validating response formats and content."""

    @requires_anthropic
    @pytest.mark.integration
    def test_response_contains_usage_metadata(self, registered_anthropic_client):
        """Verify response contains token usage metadata."""
        response = registered_anthropic_client.messages.create(
            model=MODEL,
            messages=[{"role": "user", "content": TEST_PROMPT}],
            max_tokens=MAX_TOKENS,
        )

        assert response.usage is not None
        assert response.usage.input_tokens > 0
        assert response.usage.output_tokens > 0

    @requires_anthropic
    @pytest.mark.integration
    def test_response_model_matches_request(self, registered_anthropic_client):
        """Verify response model matches requested model."""
        response = registered_anthropic_client.messages.create(
            model=MODEL,
            messages=[{"role": "user", "content": TEST_PROMPT}],
            max_tokens=MAX_TOKENS,
        )

        # Model should contain the base model name
        assert "claude" in response.model.lower()

    @requires_anthropic
    @pytest.mark.integration
    def test_response_stop_reason_is_valid(self, registered_anthropic_client):
        """Verify stop_reason is one of expected values."""
        response = registered_anthropic_client.messages.create(
            model=MODEL,
            messages=[{"role": "user", "content": TEST_PROMPT}],
            max_tokens=MAX_TOKENS,
        )

        valid_reasons = {"end_turn", "max_tokens", "stop_sequence", "tool_use"}
        assert response.stop_reason in valid_reasons

    @requires_anthropic
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_async_response_contains_usage_metadata(
        self, registered_async_anthropic_client
    ):
        """Verify async response contains token usage metadata."""
        response = await registered_async_anthropic_client.messages.create(
            model=MODEL,
            messages=[{"role": "user", "content": TEST_PROMPT}],
            max_tokens=MAX_TOKENS,
        )

        assert response.usage is not None
        assert response.usage.input_tokens > 0
        assert response.usage.output_tokens > 0


class TestMemoriIntegration:
    """Tests to verify Memori-specific integration functionality."""

    @requires_anthropic
    @pytest.mark.integration
    def test_memori_wrapper_does_not_modify_response_type(
        self, anthropic_api_key, memori_instance
    ):
        """Verify Memori wrapper doesn't alter the response type."""
        # Get response without Memori wrapper
        unwrapped_client = Anthropic(api_key=anthropic_api_key)

        # Get response with Memori wrapper
        wrapped_client = Anthropic(api_key=anthropic_api_key)
        memori_instance.llm.register(wrapped_client)
        memori_instance.attribution(entity_id="test", process_id="test")

        # Both should return valid responses
        unwrapped_response = unwrapped_client.messages.create(
            model=MODEL,
            messages=[{"role": "user", "content": TEST_PROMPT}],
            max_tokens=MAX_TOKENS,
        )

        wrapped_response = wrapped_client.messages.create(
            model=MODEL,
            messages=[{"role": "user", "content": TEST_PROMPT}],
            max_tokens=MAX_TOKENS,
        )

        # Response types should be identical
        assert type(unwrapped_response) is type(wrapped_response)

    @requires_anthropic
    @pytest.mark.integration
    def test_config_captures_provider_info(self, memori_instance, anthropic_api_key):
        """Verify Memori config captures Anthropic provider information."""
        client = Anthropic(api_key=anthropic_api_key)
        memori_instance.llm.register(client)

        # Provider SDK version should be set after registration
        assert memori_instance.config.llm.provider_sdk_version is not None

    @requires_anthropic
    @pytest.mark.integration
    def test_attribution_is_preserved_across_calls(
        self, registered_anthropic_client, memori_instance
    ):
        """Verify attribution remains set across multiple API calls."""
        memori_instance.attribution(entity_id="user-123", process_id="process-456")

        # Make first call
        registered_anthropic_client.messages.create(
            model=MODEL,
            messages=[{"role": "user", "content": TEST_PROMPT}],
            max_tokens=MAX_TOKENS,
        )

        assert memori_instance.config.entity_id == "user-123"
        assert memori_instance.config.process_id == "process-456"

        # Make second call
        registered_anthropic_client.messages.create(
            model=MODEL,
            messages=[{"role": "user", "content": TEST_PROMPT}],
            max_tokens=MAX_TOKENS,
        )

        # Attribution should still be set
        assert memori_instance.config.entity_id == "user-123"
        assert memori_instance.config.process_id == "process-456"


class TestStorageVerification:
    """Tests to verify conversations are stored in the database."""

    @requires_anthropic
    @pytest.mark.integration
    def test_conversation_stored_after_sync_call(
        self, registered_anthropic_client, memori_instance
    ):
        """After messages.create(), verify conversation record exists."""
        # Make an API call
        registered_anthropic_client.messages.create(
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

    @requires_anthropic
    @pytest.mark.integration
    def test_messages_stored_with_content(
        self, registered_anthropic_client, memori_instance
    ):
        """Verify user query and assistant response are stored."""
        test_query = "What is 2 + 2?"

        # Make an API call
        registered_anthropic_client.messages.create(
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

    @requires_anthropic
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_conversation_stored_after_async_call(
        self, registered_async_anthropic_client, memori_instance
    ):
        """After async messages.create(), verify conversation exists."""
        # Make an async API call
        await registered_async_anthropic_client.messages.create(
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

    @requires_anthropic
    @pytest.mark.integration
    def test_multiple_calls_accumulate_messages(
        self, registered_anthropic_client, memori_instance
    ):
        """Verify multiple API calls store multiple message pairs."""
        # First call
        registered_anthropic_client.messages.create(
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
        registered_anthropic_client.messages.create(
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
