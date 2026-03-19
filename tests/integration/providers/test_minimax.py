import pytest
from openai import (
    AsyncOpenAI,
    OpenAI,
)

from tests.integration.conftest import requires_minimax

MODEL = "MiniMax-M2.5"
MODEL_HIGHSPEED = "MiniMax-M2.5-highspeed"
MAX_TOKENS = 50
TEST_PROMPT = "Say 'hello' in one word."


class TestClientRegistration:
    @requires_minimax
    @pytest.mark.integration
    def test_sync_client_registration_marks_installed(
        self, memori_instance, minimax_api_key
    ):
        client = OpenAI(
            api_key=minimax_api_key,
            base_url="https://api.minimax.io/v1",
        )

        assert not hasattr(client, "_memori_installed")

        memori_instance.llm.register(client)

        assert hasattr(client, "_memori_installed")
        assert getattr(client, "_memori_installed", False) is True

    @requires_minimax
    @pytest.mark.integration
    def test_async_client_registration_marks_installed(
        self, memori_instance, minimax_api_key
    ):
        client = AsyncOpenAI(
            api_key=minimax_api_key,
            base_url="https://api.minimax.io/v1",
        )

        assert not hasattr(client, "_memori_installed")

        memori_instance.llm.register(client)

        assert hasattr(client, "_memori_installed")
        assert getattr(client, "_memori_installed", False) is True

    @requires_minimax
    @pytest.mark.integration
    def test_platform_detected_as_minimax(self, memori_instance, minimax_api_key):
        client = OpenAI(
            api_key=minimax_api_key,
            base_url="https://api.minimax.io/v1",
        )
        memori_instance.llm.register(client)

        assert memori_instance.config.platform.provider == "minimax"

    @requires_minimax
    @pytest.mark.integration
    def test_multiple_registrations_are_idempotent(
        self, memori_instance, minimax_api_key
    ):
        client = OpenAI(
            api_key=minimax_api_key,
            base_url="https://api.minimax.io/v1",
        )

        memori_instance.llm.register(client)
        original_create = client.chat.completions.create

        memori_instance.llm.register(client)

        assert client.chat.completions.create is original_create
        assert getattr(client, "_memori_installed", False) is True


class TestSyncChatCompletions:
    @requires_minimax
    @pytest.mark.integration
    def test_sync_chat_completion_returns_response(self, registered_minimax_client):
        response = registered_minimax_client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": TEST_PROMPT}],
            max_tokens=MAX_TOKENS,
        )

        assert response is not None
        assert hasattr(response, "choices")
        assert len(response.choices) > 0
        assert response.choices[0].message.content is not None

    @requires_minimax
    @pytest.mark.integration
    def test_sync_chat_completion_response_structure(self, registered_minimax_client):
        response = registered_minimax_client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": TEST_PROMPT}],
            max_tokens=MAX_TOKENS,
        )

        assert hasattr(response, "id")
        assert hasattr(response, "model")
        assert hasattr(response, "choices")
        assert hasattr(response, "usage")

        choice = response.choices[0]
        assert hasattr(choice, "message")
        assert hasattr(choice, "finish_reason")
        assert hasattr(choice.message, "role")
        assert hasattr(choice.message, "content")
        assert choice.message.role == "assistant"

    @requires_minimax
    @pytest.mark.integration
    def test_sync_chat_completion_with_system_message(self, registered_minimax_client):
        response = registered_minimax_client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": TEST_PROMPT},
            ],
            max_tokens=MAX_TOKENS,
        )

        assert response is not None
        assert response.choices[0].message.content is not None

    @requires_minimax
    @pytest.mark.integration
    def test_sync_chat_completion_with_highspeed_model(self, registered_minimax_client):
        response = registered_minimax_client.chat.completions.create(
            model=MODEL_HIGHSPEED,
            messages=[{"role": "user", "content": TEST_PROMPT}],
            max_tokens=MAX_TOKENS,
        )

        assert response is not None
        assert hasattr(response, "choices")
        assert len(response.choices) > 0
        assert response.choices[0].message.content is not None

    @requires_minimax
    @pytest.mark.integration
    def test_sync_chat_completion_multi_turn(self, registered_minimax_client):
        response = registered_minimax_client.chat.completions.create(
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
    @requires_minimax
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_async_chat_completion_returns_response(
        self, registered_async_minimax_client
    ):
        response = await registered_async_minimax_client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": TEST_PROMPT}],
            max_tokens=MAX_TOKENS,
        )

        assert response is not None
        assert hasattr(response, "choices")
        assert len(response.choices) > 0
        assert response.choices[0].message.content is not None

    @requires_minimax
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_async_chat_completion_response_structure(
        self, registered_async_minimax_client
    ):
        response = await registered_async_minimax_client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": TEST_PROMPT}],
            max_tokens=MAX_TOKENS,
        )

        assert hasattr(response, "id")
        assert hasattr(response, "model")
        assert hasattr(response, "choices")
        assert hasattr(response, "usage")

        choice = response.choices[0]
        assert hasattr(choice, "message")
        assert choice.message.role == "assistant"


class TestSyncStreaming:
    @requires_minimax
    @pytest.mark.integration
    def test_sync_streaming_returns_chunks(self, registered_minimax_client):
        stream = registered_minimax_client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": TEST_PROMPT}],
            max_tokens=MAX_TOKENS,
            stream=True,
        )

        chunks = list(stream)

        assert len(chunks) > 0

    @requires_minimax
    @pytest.mark.integration
    def test_sync_streaming_assembles_content(self, registered_minimax_client):
        stream = registered_minimax_client.chat.completions.create(
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


class TestAsyncStreaming:
    @requires_minimax
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_async_streaming_returns_chunks(self, registered_async_minimax_client):
        stream = await registered_async_minimax_client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": TEST_PROMPT}],
            max_tokens=MAX_TOKENS,
            stream=True,
        )

        chunks = []
        async for chunk in stream:
            chunks.append(chunk)

        assert len(chunks) > 0

    @requires_minimax
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_async_streaming_assembles_content(
        self, registered_async_minimax_client
    ):
        stream = await registered_async_minimax_client.chat.completions.create(
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


class TestResponseFormatValidation:
    @requires_minimax
    @pytest.mark.integration
    def test_response_contains_usage_metadata(self, registered_minimax_client):
        response = registered_minimax_client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": TEST_PROMPT}],
            max_tokens=MAX_TOKENS,
        )

        assert response.usage is not None
        assert response.usage.prompt_tokens > 0
        assert response.usage.completion_tokens > 0
        assert response.usage.total_tokens > 0

    @requires_minimax
    @pytest.mark.integration
    def test_response_finish_reason_is_valid(self, registered_minimax_client):
        response = registered_minimax_client.chat.completions.create(
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


class TestMemoriIntegration:
    @requires_minimax
    @pytest.mark.integration
    def test_config_captures_provider_info(self, memori_instance, minimax_api_key):
        client = OpenAI(
            api_key=minimax_api_key,
            base_url="https://api.minimax.io/v1",
        )
        memori_instance.llm.register(client)

        assert memori_instance.config.llm.provider_sdk_version is not None

    @requires_minimax
    @pytest.mark.integration
    def test_attribution_is_preserved_across_calls(
        self, registered_minimax_client, memori_instance
    ):
        memori_instance.attribution(entity_id="user-123", process_id="process-456")

        registered_minimax_client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": TEST_PROMPT}],
            max_tokens=MAX_TOKENS,
        )

        assert memori_instance.config.entity_id == "user-123"
        assert memori_instance.config.process_id == "process-456"


class TestStorageVerification:
    @requires_minimax
    @pytest.mark.integration
    def test_conversation_stored_after_sync_call(
        self, registered_minimax_client, memori_instance
    ):
        registered_minimax_client.chat.completions.create(
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

    @requires_minimax
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_conversation_stored_after_async_call(
        self, registered_async_minimax_client, memori_instance
    ):
        await registered_async_minimax_client.chat.completions.create(
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

    @requires_minimax
    @pytest.mark.integration
    def test_messages_stored_with_content(
        self, registered_minimax_client, memori_instance
    ):
        test_query = "What is 2 + 2?"

        registered_minimax_client.chat.completions.create(
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
