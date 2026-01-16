import pytest
from openai import AsyncOpenAI, AuthenticationError,BadRequestError, NotFoundError, OpenAI

from tests.integration.conftest import requires_openai

MODEL = "gpt-4o-mini"
MAX_TOKENS = 50
MAX_OUTPUT_TOKENS = 50
TEST_PROMPT = "Say 'hello' in one word."


class TestClientRegistration:
    @requires_openai
    @pytest.mark.integration
    def test_sync_client_registration_marks_installed(
        self, memori_instance, openai_api_key
    ):
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
        client = OpenAI(api_key=openai_api_key)

        memori_instance.llm.register(client)
        original_create = client.chat.completions.create

        memori_instance.llm.register(client)

        assert client.chat.completions.create is original_create
        assert getattr(client, "_memori_installed", False) is True

    @requires_openai
    @pytest.mark.integration
    def test_registration_preserves_original_methods(
        self, memori_instance, openai_api_key
    ):
        client = OpenAI(api_key=openai_api_key)

        memori_instance.llm.register(client)

        assert hasattr(client.chat, "_completions_create")
        assert hasattr(client.beta, "_chat_completions_parse")

    @requires_openai
    @pytest.mark.integration
    def test_sync_client_registration_wraps_responses(
        self, memori_instance, openai_api_key
    ):
        client = OpenAI(api_key=openai_api_key)

        assert not hasattr(client, "_memori_installed")
        assert not hasattr(client, "_responses_create")

        memori_instance.llm.register(client)

        assert hasattr(client, "_memori_installed")
        assert hasattr(client, "_responses_create")
        assert getattr(client, "_memori_installed", False) is True

    @requires_openai
    @pytest.mark.integration
    def test_async_client_registration_wraps_responses(
        self, memori_instance, openai_api_key
    ):
        client = AsyncOpenAI(api_key=openai_api_key)

        assert not hasattr(client, "_memori_installed")

        memori_instance.llm.register(client)

        assert hasattr(client, "_memori_installed")
        assert hasattr(client, "_responses_create")

    @requires_openai
    @pytest.mark.integration
    def test_multiple_registrations_preserve_responses_wrapper(
        self, memori_instance, openai_api_key
    ):
        client = OpenAI(api_key=openai_api_key)

        memori_instance.llm.register(client)
        original_responses_create = client.responses.create

        memori_instance.llm.register(client)

        assert client.responses.create is original_responses_create


class TestSyncChatCompletions:
    @requires_openai
    @pytest.mark.integration
    def test_sync_chat_completion_returns_response(self, registered_openai_client):
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
        response = registered_openai_client.chat.completions.create(
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

    @requires_openai
    @pytest.mark.integration
    def test_sync_chat_completion_with_system_message(self, registered_openai_client):
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
    @requires_openai
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_async_chat_completion_returns_response(
        self, registered_async_openai_client
    ):
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
        response = await registered_async_openai_client.chat.completions.create(
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

    @requires_openai
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_async_chat_completion_with_system_message(
        self, registered_async_openai_client
    ):
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
    @requires_openai
    @pytest.mark.integration
    def test_sync_streaming_returns_chunks(self, registered_openai_client):
        stream = registered_openai_client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": TEST_PROMPT}],
            max_tokens=MAX_TOKENS,
            stream=True,
        )

        chunks = list(stream)

        assert len(chunks) > 0

    @requires_openai
    @pytest.mark.integration
    def test_sync_streaming_assembles_content(self, registered_openai_client):
        stream = registered_openai_client.chat.completions.create(
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
    def test_sync_streaming_chunk_structure(self, registered_openai_client):
        stream = registered_openai_client.chat.completions.create(
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
    @requires_openai
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_async_streaming_returns_chunks(self, registered_async_openai_client):
        stream = await registered_async_openai_client.chat.completions.create(
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
        self, registered_async_openai_client
    ):
        stream = await registered_async_openai_client.chat.completions.create(
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
        self, registered_async_openai_client
    ):
        stream = await registered_async_openai_client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": TEST_PROMPT}],
            max_tokens=MAX_TOKENS,
            stream=True,
        )

        async for chunk in stream:
            assert hasattr(chunk, "choices")
            assert hasattr(chunk, "id")
            assert hasattr(chunk, "model")

    @requires_openai
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_async_streaming_with_usage_info(
        self, registered_async_openai_client
    ):
        stream = await registered_async_openai_client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": TEST_PROMPT}],
            max_tokens=MAX_TOKENS,
            stream=True,
            stream_options={"include_usage": True},
        )

        last_chunk = None
        async for chunk in stream:
            last_chunk = chunk

        assert last_chunk is not None


class TestErrorHandling:
    @pytest.mark.integration
    def test_invalid_api_key_raises_authentication_error(self, memori_instance):
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
        with pytest.raises(NotFoundError):
            registered_openai_client.chat.completions.create(
                model="nonexistent-model-xyz",
                messages=[{"role": "user", "content": TEST_PROMPT}],
                max_tokens=MAX_TOKENS,
            )

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_async_invalid_api_key_raises_error(self, memori_instance):
        client = AsyncOpenAI(api_key="invalid-key-12345")
        memori_instance.llm.register(client)

        with pytest.raises(AuthenticationError):
            await client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": TEST_PROMPT}],
                max_tokens=MAX_TOKENS,
            )


class TestResponseFormatValidation:
    @requires_openai
    @pytest.mark.integration
    def test_response_contains_usage_metadata(self, registered_openai_client):
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
        response = registered_openai_client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": TEST_PROMPT}],
            max_tokens=MAX_TOKENS,
        )

        assert "gpt-4o-mini" in response.model

    @requires_openai
    @pytest.mark.integration
    def test_response_finish_reason_is_valid(self, registered_openai_client):
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
        response = await registered_async_openai_client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": TEST_PROMPT}],
            max_tokens=MAX_TOKENS,
        )

        assert response.usage is not None
        assert response.usage.prompt_tokens > 0
        assert response.usage.completion_tokens > 0


class TestMemoriIntegration:
    @requires_openai
    @pytest.mark.integration
    def test_memori_wrapper_does_not_modify_response_type(
        self, openai_api_key, memori_instance
    ):
        unwrapped_client = OpenAI(api_key=openai_api_key)

        wrapped_client = OpenAI(api_key=openai_api_key)
        memori_instance.llm.register(wrapped_client)
        memori_instance.attribution(entity_id="test", process_id="test")

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

        assert type(unwrapped_response) is type(wrapped_response)

    @requires_openai
    @pytest.mark.integration
    def test_config_captures_provider_info(self, memori_instance, openai_api_key):
        client = OpenAI(api_key=openai_api_key)
        memori_instance.llm.register(client)

        assert memori_instance.config.llm.provider_sdk_version is not None

    @requires_openai
    @pytest.mark.integration
    def test_attribution_is_preserved_across_calls(
        self, registered_openai_client, memori_instance
    ):
        memori_instance.attribution(entity_id="user-123", process_id="process-456")

        registered_openai_client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": TEST_PROMPT}],
            max_tokens=MAX_TOKENS,
        )

        assert memori_instance.config.entity_id == "user-123"
        assert memori_instance.config.process_id == "process-456"

        registered_openai_client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": TEST_PROMPT}],
            max_tokens=MAX_TOKENS,
        )

        assert memori_instance.config.entity_id == "user-123"
        assert memori_instance.config.process_id == "process-456"


class TestBetaApi:
    @requires_openai
    @pytest.mark.integration
    def test_beta_parse_registration(self, memori_instance, openai_api_key):
        client = OpenAI(api_key=openai_api_key)
        memori_instance.llm.register(client)

        assert hasattr(client.beta, "_chat_completions_parse")


class TestStorageVerification:
    @requires_openai
    @pytest.mark.integration
    def test_conversation_stored_after_sync_call(
        self, registered_openai_client, memori_instance
    ):
        registered_openai_client.chat.completions.create(
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

    @requires_openai
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_conversation_stored_after_async_call(
        self, registered_async_openai_client, memori_instance
    ):
        await registered_async_openai_client.chat.completions.create(
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

    @requires_openai
    @pytest.mark.integration
    def test_messages_stored_with_content(
        self, registered_openai_client, memori_instance
    ):
        test_query = "What is 2 + 2?"

        registered_openai_client.chat.completions.create(
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

    @requires_openai
    @pytest.mark.integration
    def test_multiple_calls_accumulate_messages(
        self, registered_openai_client, memori_instance
    ):
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

        assert count_after_second > count_after_first


# Responses API Tests


class TestSyncResponses:
    @requires_openai
    @pytest.mark.integration
    def test_sync_responses_returns_response(self, registered_openai_client):
        response = registered_openai_client.responses.create(
            model=MODEL,
            input=TEST_PROMPT,
            max_output_tokens=MAX_OUTPUT_TOKENS,
        )

        assert response is not None
        assert hasattr(response, "output")
        assert hasattr(response, "output_text")
        assert len(response.output_text) > 0

    @requires_openai
    @pytest.mark.integration
    def test_sync_responses_response_structure(self, registered_openai_client):
        response = registered_openai_client.responses.create(
            model=MODEL,
            input=TEST_PROMPT,
            max_output_tokens=MAX_OUTPUT_TOKENS,
        )

        assert hasattr(response, "id")
        assert hasattr(response, "model")
        assert hasattr(response, "output")
        assert hasattr(response, "status")
        assert response.status == "completed"

    @requires_openai
    @pytest.mark.integration
    def test_sync_responses_with_instructions(self, registered_openai_client):
        response = registered_openai_client.responses.create(
            model=MODEL,
            input="What is 2+2?",
            instructions="You are a math tutor. Be very brief.",
            max_output_tokens=MAX_OUTPUT_TOKENS,
        )

        assert response is not None
        assert len(response.output_text) > 0

    @requires_openai
    @pytest.mark.integration
    def test_sync_responses_simple_math(self, registered_openai_client):
        response = registered_openai_client.responses.create(
            model=MODEL,
            input="What is 5 + 3? Answer with just the number.",
            instructions="Be very brief. Answer with just the number.",
            max_output_tokens=MAX_OUTPUT_TOKENS,
        )

        assert response is not None
        assert "8" in response.output_text


class TestAsyncResponses:
    @requires_openai
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_async_responses_returns_response(
        self, registered_async_openai_client
    ):
        response = await registered_async_openai_client.responses.create(
            model=MODEL,
            input=TEST_PROMPT,
            max_output_tokens=MAX_OUTPUT_TOKENS,
        )

        assert response is not None
        assert hasattr(response, "output")
        assert hasattr(response, "output_text")
        assert len(response.output_text) > 0

    @requires_openai
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_async_responses_response_structure(
        self, registered_async_openai_client
    ):
        response = await registered_async_openai_client.responses.create(
            model=MODEL,
            input=TEST_PROMPT,
            max_output_tokens=MAX_OUTPUT_TOKENS,
        )

        assert hasattr(response, "id")
        assert hasattr(response, "model")
        assert hasattr(response, "output")
        assert hasattr(response, "status")

    @requires_openai
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_async_responses_with_instructions(
        self, registered_async_openai_client
    ):
        response = await registered_async_openai_client.responses.create(
            model=MODEL,
            input="What is the capital of France?",
            instructions="Be brief.",
            max_output_tokens=MAX_OUTPUT_TOKENS,
        )

        assert response is not None
        assert "paris" in response.output_text.lower()


class TestResponsesConversationStorage:
    @requires_openai
    @pytest.mark.integration
    def test_conversation_id_created_after_call(self, memori_instance, openai_api_key):
        client = OpenAI(api_key=openai_api_key)
        memori_instance.llm.register(client)
        memori_instance.attribution(entity_id="test-entity", process_id="test-process")

        assert memori_instance.config.cache.conversation_id is None

        client.responses.create(
            model=MODEL,
            input="Hello",
            max_output_tokens=MAX_OUTPUT_TOKENS,
        )

        assert memori_instance.config.cache.conversation_id is not None

    @requires_openai
    @pytest.mark.integration
    def test_conversation_continuity(self, memori_instance, openai_api_key):
        client = OpenAI(api_key=openai_api_key)
        memori_instance.llm.register(client)
        memori_instance.attribution(entity_id="test-entity", process_id="test-process")

        client.responses.create(
            model=MODEL,
            input="My favorite color is blue.",
            instructions="Remember what the user tells you.",
            max_output_tokens=MAX_OUTPUT_TOKENS,
        )

        response = client.responses.create(
            model=MODEL,
            input="What is my favorite color?",
            instructions="Answer based on what the user previously told you.",
            max_output_tokens=MAX_OUTPUT_TOKENS,
        )

        assert "blue" in response.output_text.lower()


class TestResponsesErrorHandling:
    @pytest.mark.integration
    def test_invalid_api_key_raises_error(self, memori_instance):
        client = OpenAI(api_key="invalid-key-12345")
        memori_instance.llm.register(client)

        with pytest.raises(AuthenticationError):
            client.responses.create(
                model=MODEL,
                input=TEST_PROMPT,
                max_output_tokens=MAX_OUTPUT_TOKENS,
            )

    @requires_openai
    @pytest.mark.integration
    def test_invalid_model_raises_error(self, registered_openai_client):
        with pytest.raises(BadRequestError):
            registered_openai_client.responses.create(
                model="nonexistent-model-xyz",
                input=TEST_PROMPT,
                max_output_tokens=MAX_OUTPUT_TOKENS,
            )

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_async_invalid_api_key_raises_error(self, memori_instance):
        client = AsyncOpenAI(api_key="invalid-key-12345")
        memori_instance.llm.register(client)

        with pytest.raises(AuthenticationError):
            await client.responses.create(
                model=MODEL,
                input=TEST_PROMPT,
                max_output_tokens=MAX_OUTPUT_TOKENS,
            )


class TestResponsesWithChatCompletionsCoexistence:
    @requires_openai
    @pytest.mark.integration
    def test_both_apis_work_with_same_client(self, registered_openai_client):
        responses_result = registered_openai_client.responses.create(
            model=MODEL,
            input="Say 'responses' in one word.",
            max_output_tokens=MAX_OUTPUT_TOKENS,
        )
        assert responses_result is not None
        assert len(responses_result.output_text) > 0

        chat_result = registered_openai_client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": "Say 'chat' in one word."}],
            max_tokens=MAX_OUTPUT_TOKENS,
        )
        assert chat_result is not None
        assert len(chat_result.choices[0].message.content) > 0

    @requires_openai
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_async_both_apis_work_with_same_client(
        self, registered_async_openai_client
    ):
        responses_result = await registered_async_openai_client.responses.create(
            model=MODEL,
            input="Say 'async-responses' in one word.",
            max_output_tokens=MAX_OUTPUT_TOKENS,
        )
        assert responses_result is not None
        assert len(responses_result.output_text) > 0

        chat_result = await registered_async_openai_client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": "Say 'async-chat' in one word."}],
            max_tokens=MAX_OUTPUT_TOKENS,
        )
        assert chat_result is not None
        assert len(chat_result.choices[0].message.content) > 0


class TestResponsesStreaming:
    @requires_openai
    @pytest.mark.integration
    def test_sync_streaming_returns_events(self, registered_openai_client):
        stream = registered_openai_client.responses.create(
            model=MODEL,
            input=TEST_PROMPT,
            max_output_tokens=MAX_OUTPUT_TOKENS,
            stream=True,
        )

        events = list(stream)

        assert len(events) > 0
        event_types = [getattr(e, "type", None) for e in events]
        assert "response.completed" in event_types

    @requires_openai
    @pytest.mark.integration
    def test_sync_streaming_final_response_structure(self, registered_openai_client):
        stream = registered_openai_client.responses.create(
            model=MODEL,
            input=TEST_PROMPT,
            max_output_tokens=MAX_OUTPUT_TOKENS,
            stream=True,
        )

        events = list(stream)
        completed_events = [
            e for e in events if hasattr(e, "type") and e.type == "response.completed"
        ]

        assert len(completed_events) == 1
        completed_event = completed_events[0]

        assert hasattr(completed_event, "response")
        response = completed_event.response
        assert hasattr(response, "id")
        assert hasattr(response, "model")
        assert hasattr(response, "output")
        assert hasattr(response, "output_text")
        assert hasattr(response, "status")
        assert response.status == "completed"
        assert len(response.output_text) > 0

    @requires_openai
    @pytest.mark.integration
    def test_sync_streaming_yields_text_deltas(self, registered_openai_client):
        stream = registered_openai_client.responses.create(
            model=MODEL,
            input="Count from 1 to 3.",
            max_output_tokens=MAX_OUTPUT_TOKENS,
            stream=True,
        )

        delta_events = []
        for event in stream:
            if hasattr(event, "type") and "delta" in event.type:
                delta_events.append(event)

        assert len(delta_events) > 0

    @requires_openai
    @pytest.mark.integration
    def test_sync_streaming_context_manager(self, registered_openai_client):
        with registered_openai_client.responses.create(
            model=MODEL,
            input=TEST_PROMPT,
            max_output_tokens=MAX_OUTPUT_TOKENS,
            stream=True,
        ) as stream:
            events = list(stream)

        assert len(events) > 0

    @requires_openai
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_async_streaming_returns_events(self, registered_async_openai_client):
        stream = await registered_async_openai_client.responses.create(
            model=MODEL,
            input=TEST_PROMPT,
            max_output_tokens=MAX_OUTPUT_TOKENS,
            stream=True,
        )

        events = []
        async for event in stream:
            events.append(event)

        assert len(events) > 0
        event_types = [getattr(e, "type", None) for e in events]
        assert "response.completed" in event_types

    @requires_openai
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_async_streaming_final_response_structure(
        self, registered_async_openai_client
    ):
        stream = await registered_async_openai_client.responses.create(
            model=MODEL,
            input=TEST_PROMPT,
            max_output_tokens=MAX_OUTPUT_TOKENS,
            stream=True,
        )

        events = []
        async for event in stream:
            events.append(event)

        completed_events = [
            e for e in events if hasattr(e, "type") and e.type == "response.completed"
        ]

        assert len(completed_events) == 1
        completed_event = completed_events[0]

        assert hasattr(completed_event, "response")
        response = completed_event.response
        assert hasattr(response, "id")
        assert hasattr(response, "model")
        assert hasattr(response, "output")
        assert hasattr(response, "output_text")
        assert hasattr(response, "status")
        assert response.status == "completed"
        assert len(response.output_text) > 0

    @requires_openai
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_async_streaming_context_manager(
        self, registered_async_openai_client
    ):
        async with await registered_async_openai_client.responses.create(
            model=MODEL,
            input=TEST_PROMPT,
            max_output_tokens=MAX_OUTPUT_TOKENS,
            stream=True,
        ) as stream:
            events = []
            async for event in stream:
                events.append(event)

        assert len(events) > 0

    @requires_openai
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_async_streaming_event_structure(
        self, registered_async_openai_client
    ):
        stream = await registered_async_openai_client.responses.create(
            model=MODEL,
            input=TEST_PROMPT,
            max_output_tokens=MAX_OUTPUT_TOKENS,
            stream=True,
        )

        async for event in stream:
            assert hasattr(event, "type")

    @requires_openai
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_async_streaming_yields_text_deltas(
        self, registered_async_openai_client
    ):
        stream = await registered_async_openai_client.responses.create(
            model=MODEL,
            input="Count from 1 to 3.",
            max_output_tokens=MAX_OUTPUT_TOKENS,
            stream=True,
        )

        delta_events = []
        async for event in stream:
            if hasattr(event, "type") and "delta" in event.type:
                delta_events.append(event)

        assert len(delta_events) > 0


class TestResponsesInputFormats:
    @requires_openai
    @pytest.mark.integration
    def test_string_input(self, registered_openai_client):
        response = registered_openai_client.responses.create(
            model=MODEL,
            input="Hello, world!",
            max_output_tokens=MAX_OUTPUT_TOKENS,
        )

        assert response is not None
        assert len(response.output_text) > 0

    @requires_openai
    @pytest.mark.integration
    def test_list_input_with_messages(self, registered_openai_client):
        response = registered_openai_client.responses.create(
            model=MODEL,
            input=[
                {"role": "user", "content": "My name is Alice."},
                {"role": "assistant", "content": "Hello Alice!"},
                {"role": "user", "content": "What is my name?"},
            ],
            max_output_tokens=MAX_OUTPUT_TOKENS,
        )

        assert response is not None
        assert "alice" in response.output_text.lower()


class TestResponsesFormatValidation:
    @requires_openai
    @pytest.mark.integration
    def test_responses_contains_usage_metadata(self, registered_openai_client):
        response = registered_openai_client.responses.create(
            model=MODEL,
            input=TEST_PROMPT,
            max_output_tokens=MAX_OUTPUT_TOKENS,
        )

        assert response.usage is not None
        assert response.usage.input_tokens > 0
        assert response.usage.output_tokens > 0
        assert response.usage.total_tokens > 0

    @requires_openai
    @pytest.mark.integration
    def test_responses_model_matches_request(self, registered_openai_client):
        response = registered_openai_client.responses.create(
            model=MODEL,
            input=TEST_PROMPT,
            max_output_tokens=MAX_OUTPUT_TOKENS,
        )

        assert "gpt-4o-mini" in response.model

    @requires_openai
    @pytest.mark.integration
    def test_responses_status_is_valid(self, registered_openai_client):
        response = registered_openai_client.responses.create(
            model=MODEL,
            input=TEST_PROMPT,
            max_output_tokens=MAX_OUTPUT_TOKENS,
        )

        valid_statuses = {"completed", "failed", "incomplete", "in_progress"}
        assert response.status in valid_statuses

    @requires_openai
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_async_responses_contains_usage_metadata(
        self, registered_async_openai_client
    ):
        response = await registered_async_openai_client.responses.create(
            model=MODEL,
            input=TEST_PROMPT,
            max_output_tokens=MAX_OUTPUT_TOKENS,
        )

        assert response.usage is not None
        assert response.usage.input_tokens > 0
        assert response.usage.output_tokens > 0


class TestResponsesMemoriIntegration:
    @requires_openai
    @pytest.mark.integration
    def test_memori_wrapper_does_not_modify_response_type(
        self, openai_api_key, memori_instance
    ):
        unwrapped_client = OpenAI(api_key=openai_api_key)

        wrapped_client = OpenAI(api_key=openai_api_key)
        memori_instance.llm.register(wrapped_client)
        memori_instance.attribution(entity_id="test", process_id="test")

        unwrapped_response = unwrapped_client.responses.create(
            model=MODEL,
            input=TEST_PROMPT,
            max_output_tokens=MAX_OUTPUT_TOKENS,
        )

        wrapped_response = wrapped_client.responses.create(
            model=MODEL,
            input=TEST_PROMPT,
            max_output_tokens=MAX_OUTPUT_TOKENS,
        )

        assert type(unwrapped_response) is type(wrapped_response)

    @requires_openai
    @pytest.mark.integration
    def test_config_captures_provider_info(self, memori_instance, openai_api_key):
        client = OpenAI(api_key=openai_api_key)
        memori_instance.llm.register(client)

        assert memori_instance.config.llm.provider_sdk_version is not None

    @requires_openai
    @pytest.mark.integration
    def test_attribution_is_preserved_across_calls(
        self, registered_openai_client, memori_instance
    ):
        memori_instance.attribution(entity_id="user-123", process_id="process-456")

        registered_openai_client.responses.create(
            model=MODEL,
            input=TEST_PROMPT,
            max_output_tokens=MAX_OUTPUT_TOKENS,
        )

        assert memori_instance.config.entity_id == "user-123"
        assert memori_instance.config.process_id == "process-456"

        registered_openai_client.responses.create(
            model=MODEL,
            input="Another question",
            max_output_tokens=MAX_OUTPUT_TOKENS,
        )

        assert memori_instance.config.entity_id == "user-123"
        assert memori_instance.config.process_id == "process-456"


class TestResponsesStorageVerification:
    @requires_openai
    @pytest.mark.integration
    def test_conversation_stored_after_sync_call(
        self, registered_openai_client, memori_instance
    ):
        registered_openai_client.responses.create(
            model=MODEL,
            input=TEST_PROMPT,
            max_output_tokens=MAX_OUTPUT_TOKENS,
        )

        conversation_id = memori_instance.config.cache.conversation_id
        assert conversation_id is not None

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
        await registered_async_openai_client.responses.create(
            model=MODEL,
            input=TEST_PROMPT,
            max_output_tokens=MAX_OUTPUT_TOKENS,
        )

        conversation_id = memori_instance.config.cache.conversation_id
        assert conversation_id is not None

        conversation = memori_instance.config.storage.driver.conversation.read(
            conversation_id
        )
        assert conversation is not None

    @requires_openai
    @pytest.mark.integration
    def test_messages_stored_with_content(
        self, registered_openai_client, memori_instance
    ):
        test_query = "What is 2 + 2?"

        registered_openai_client.responses.create(
            model=MODEL,
            input=test_query,
            max_output_tokens=MAX_OUTPUT_TOKENS,
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

    @requires_openai
    @pytest.mark.integration
    def test_multiple_calls_accumulate_messages(
        self, registered_openai_client, memori_instance
    ):
        registered_openai_client.responses.create(
            model=MODEL,
            input="First question",
            max_output_tokens=MAX_OUTPUT_TOKENS,
        )

        conversation_id = memori_instance.config.cache.conversation_id
        messages_after_first = (
            memori_instance.config.storage.driver.conversation.messages.read(
                conversation_id
            )
        )
        count_after_first = len(messages_after_first)

        registered_openai_client.responses.create(
            model=MODEL,
            input="Second question",
            max_output_tokens=MAX_OUTPUT_TOKENS,
        )

        messages_after_second = (
            memori_instance.config.storage.driver.conversation.messages.read(
                conversation_id
            )
        )
        count_after_second = len(messages_after_second)

        assert count_after_second > count_after_first
