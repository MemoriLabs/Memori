import pytest
from anthropic import Anthropic, APIStatusError, AsyncAnthropic, AuthenticationError

from tests.integration.conftest import requires_anthropic

MODEL = "claude-3-haiku-20240307"
MAX_TOKENS = 50
TEST_PROMPT = "Say 'hello' in one word."


class TestClientRegistration:
    @requires_anthropic
    @pytest.mark.integration
    def test_sync_client_registration_marks_installed(
        self, memori_instance, anthropic_api_key
    ):
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
        client = Anthropic(api_key=anthropic_api_key)

        memori_instance.llm.register(client)
        original_create = client.messages.create

        memori_instance.llm.register(client)

        assert client.messages.create is original_create
        assert getattr(client, "_memori_installed", False) is True

    @requires_anthropic
    @pytest.mark.integration
    def test_registration_preserves_original_methods(
        self, memori_instance, anthropic_api_key
    ):
        client = Anthropic(api_key=anthropic_api_key)

        memori_instance.llm.register(client)

        assert hasattr(client, "_memori_installed")


class TestSyncMessages:
    @requires_anthropic
    @pytest.mark.integration
    def test_sync_message_returns_response(self, registered_anthropic_client):
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
        response = registered_anthropic_client.messages.create(
            model=MODEL,
            messages=[{"role": "user", "content": TEST_PROMPT}],
            max_tokens=MAX_TOKENS,
        )

        assert hasattr(response, "id")
        assert hasattr(response, "model")
        assert hasattr(response, "content")
        assert hasattr(response, "usage")
        assert hasattr(response, "stop_reason")

        assert len(response.content) > 0
        assert hasattr(response.content[0], "type")
        assert hasattr(response.content[0], "text")
        assert response.content[0].type == "text"
        assert response.role == "assistant"

    @requires_anthropic
    @pytest.mark.integration
    def test_sync_message_with_system_message(self, registered_anthropic_client):
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
    @requires_anthropic
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_async_message_returns_response(
        self, registered_async_anthropic_client
    ):
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
        response = await registered_async_anthropic_client.messages.create(
            model=MODEL,
            messages=[{"role": "user", "content": TEST_PROMPT}],
            max_tokens=MAX_TOKENS,
        )

        assert hasattr(response, "id")
        assert hasattr(response, "model")
        assert hasattr(response, "content")
        assert hasattr(response, "usage")

        assert len(response.content) > 0
        assert response.content[0].type == "text"

    @requires_anthropic
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_async_message_with_system(self, registered_async_anthropic_client):
        response = await registered_async_anthropic_client.messages.create(
            model=MODEL,
            system="You are a helpful assistant.",
            messages=[{"role": "user", "content": TEST_PROMPT}],
            max_tokens=MAX_TOKENS,
        )

        assert response is not None
        assert response.content[0].text is not None


class TestSyncStreaming:
    @requires_anthropic
    @pytest.mark.integration
    def test_sync_streaming_returns_events(self, registered_anthropic_client):
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
        with registered_anthropic_client.messages.stream(
            model=MODEL,
            messages=[{"role": "user", "content": TEST_PROMPT}],
            max_tokens=MAX_TOKENS,
        ) as stream:
            events = list(stream.text_stream)

        assert len(events) > 0
        assert all(isinstance(e, str) for e in events)


class TestAsyncStreaming:
    @requires_anthropic
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_async_streaming_returns_events(
        self, registered_async_anthropic_client
    ):
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
        async with registered_async_anthropic_client.messages.stream(
            model=MODEL,
            messages=[{"role": "user", "content": TEST_PROMPT}],
            max_tokens=MAX_TOKENS,
        ) as stream:
            async for _ in stream.text_stream:
                pass
            final_message = await stream.get_final_message()

        assert hasattr(final_message, "id")
        assert hasattr(final_message, "model")
        assert hasattr(final_message, "content")

    @requires_anthropic
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_async_streaming_final_message(
        self, registered_async_anthropic_client
    ):
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
        async with registered_async_anthropic_client.messages.stream(
            model=MODEL,
            messages=[{"role": "user", "content": TEST_PROMPT}],
            max_tokens=MAX_TOKENS,
        ) as stream:
            async for _ in stream.text_stream:
                pass
            final_message = await stream.get_final_message()

        assert final_message is not None
        assert hasattr(final_message, "usage")
        assert final_message.usage is not None
        assert hasattr(final_message.usage, "input_tokens")
        assert hasattr(final_message.usage, "output_tokens")


class TestErrorHandling:
    @pytest.mark.integration
    def test_invalid_api_key_raises_authentication_error(self, memori_instance):
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
        with pytest.raises(APIStatusError):
            registered_anthropic_client.messages.create(
                model="nonexistent-model-xyz",
                messages=[{"role": "user", "content": TEST_PROMPT}],
                max_tokens=MAX_TOKENS,
            )

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_async_invalid_api_key_raises_error(self, memori_instance):
        client = AsyncAnthropic(api_key="invalid-key-12345")
        memori_instance.llm.register(client)

        with pytest.raises(AuthenticationError):
            await client.messages.create(
                model=MODEL,
                messages=[{"role": "user", "content": TEST_PROMPT}],
                max_tokens=MAX_TOKENS,
            )


class TestResponseFormatValidation:
    @requires_anthropic
    @pytest.mark.integration
    def test_response_contains_usage_metadata(self, registered_anthropic_client):
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
        response = registered_anthropic_client.messages.create(
            model=MODEL,
            messages=[{"role": "user", "content": TEST_PROMPT}],
            max_tokens=MAX_TOKENS,
        )

        assert "claude" in response.model.lower()

    @requires_anthropic
    @pytest.mark.integration
    def test_response_stop_reason_is_valid(self, registered_anthropic_client):
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
        response = await registered_async_anthropic_client.messages.create(
            model=MODEL,
            messages=[{"role": "user", "content": TEST_PROMPT}],
            max_tokens=MAX_TOKENS,
        )

        assert response.usage is not None
        assert response.usage.input_tokens > 0
        assert response.usage.output_tokens > 0


class TestMemoriIntegration:
    @requires_anthropic
    @pytest.mark.integration
    def test_memori_wrapper_does_not_modify_response_type(
        self, anthropic_api_key, memori_instance
    ):
        unwrapped_client = Anthropic(api_key=anthropic_api_key)

        wrapped_client = Anthropic(api_key=anthropic_api_key)
        memori_instance.llm.register(wrapped_client)
        memori_instance.attribution(entity_id="test", process_id="test")

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

        assert type(unwrapped_response) is type(wrapped_response)

    @requires_anthropic
    @pytest.mark.integration
    def test_config_captures_provider_info(self, memori_instance, anthropic_api_key):
        client = Anthropic(api_key=anthropic_api_key)
        memori_instance.llm.register(client)

        assert memori_instance.config.llm.provider_sdk_version is not None

    @requires_anthropic
    @pytest.mark.integration
    def test_attribution_is_preserved_across_calls(
        self, registered_anthropic_client, memori_instance
    ):
        memori_instance.attribution(entity_id="user-123", process_id="process-456")

        registered_anthropic_client.messages.create(
            model=MODEL,
            messages=[{"role": "user", "content": TEST_PROMPT}],
            max_tokens=MAX_TOKENS,
        )

        assert memori_instance.config.entity_id == "user-123"
        assert memori_instance.config.process_id == "process-456"

        registered_anthropic_client.messages.create(
            model=MODEL,
            messages=[{"role": "user", "content": TEST_PROMPT}],
            max_tokens=MAX_TOKENS,
        )

        assert memori_instance.config.entity_id == "user-123"
        assert memori_instance.config.process_id == "process-456"


class TestStorageVerification:
    @requires_anthropic
    @pytest.mark.integration
    def test_conversation_stored_after_sync_call(
        self, registered_anthropic_client, memori_instance
    ):
        registered_anthropic_client.messages.create(
            model=MODEL,
            messages=[{"role": "user", "content": TEST_PROMPT}],
            max_tokens=MAX_TOKENS,
        )

        conversation_id = memori_instance.config.cache.conversation_id
        assert conversation_id is not None

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
        test_query = "What is 2 + 2?"

        registered_anthropic_client.messages.create(
            model=MODEL,
            messages=[{"role": "user", "content": test_query}],
            max_tokens=MAX_TOKENS,
        )

        conversation_id = memori_instance.config.cache.conversation_id
        assert conversation_id is not None

        messages = memori_instance.config.storage.driver.conversation.messages.read(
            conversation_id
        )

        assert len(messages) >= 2

        user_messages = [m for m in messages if m["role"] == "user"]
        assert len(user_messages) >= 1
        assert test_query in user_messages[0]["content"]

        assistant_messages = [m for m in messages if m["role"] == "assistant"]
        assert len(assistant_messages) >= 1
        assert len(assistant_messages[0]["content"]) > 0

    @requires_anthropic
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_conversation_stored_after_async_call(
        self, registered_async_anthropic_client, memori_instance
    ):
        await registered_async_anthropic_client.messages.create(
            model=MODEL,
            messages=[{"role": "user", "content": TEST_PROMPT}],
            max_tokens=MAX_TOKENS,
        )

        conversation_id = memori_instance.config.cache.conversation_id
        assert conversation_id is not None

        conversation = memori_instance.config.storage.driver.conversation.read(
            conversation_id
        )
        assert conversation is not None

    @requires_anthropic
    @pytest.mark.integration
    def test_multiple_calls_accumulate_messages(
        self, registered_anthropic_client, memori_instance
    ):
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

        assert count_after_second > count_after_first
