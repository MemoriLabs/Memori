import asyncio

import pytest
from openai import AsyncOpenAI, OpenAI

from tests.integration.conftest import requires_openai

MODEL = "gpt-4o-mini"
MAX_TOKENS = 50
MAX_OUTPUT_TOKENS = 50
TEST_PROMPT = "Say 'hello' in one word."
AA_WAIT_TIMEOUT = 15.0


class TestHostedOpenAISync:
    @requires_openai
    @pytest.mark.integration
    def test_sync_completion_through_hosted_pipeline(
        self, hosted_memori_instance, openai_api_key
    ):
        client = OpenAI(api_key=openai_api_key)
        hosted_memori_instance.llm.register(client)
        hosted_memori_instance.attribution(
            entity_id="hosted-test-user", process_id="hosted-test"
        )

        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": TEST_PROMPT}],
            max_tokens=MAX_TOKENS,
        )

        assert response is not None
        assert response.choices[0].message.content is not None

        hosted_memori_instance.config.augmentation.wait(timeout=AA_WAIT_TIMEOUT)

    @requires_openai
    @pytest.mark.integration
    def test_sync_completion_stores_conversation(
        self, hosted_registered_openai_client, hosted_memori_instance
    ):
        hosted_registered_openai_client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": TEST_PROMPT}],
            max_tokens=MAX_TOKENS,
        )

        hosted_memori_instance.config.augmentation.wait(timeout=AA_WAIT_TIMEOUT)

        conversation_id = hosted_memori_instance.config.cache.conversation_id
        assert conversation_id is not None

        conversation = hosted_memori_instance.config.storage.driver.conversation.read(
            conversation_id
        )
        assert conversation is not None
        assert conversation["id"] == conversation_id

    @requires_openai
    @pytest.mark.integration
    def test_sync_completion_stores_messages(
        self, hosted_registered_openai_client, hosted_memori_instance
    ):
        test_query = "What is 2 + 2?"

        hosted_registered_openai_client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": test_query}],
            max_tokens=MAX_TOKENS,
        )

        hosted_memori_instance.config.augmentation.wait(timeout=AA_WAIT_TIMEOUT)

        conversation_id = hosted_memori_instance.config.cache.conversation_id
        messages = (
            hosted_memori_instance.config.storage.driver.conversation.messages.read(
                conversation_id
            )
        )

        assert len(messages) >= 2

        user_messages = [m for m in messages if m["role"] == "user"]
        assert len(user_messages) >= 1
        assert test_query in user_messages[0]["content"]

        assistant_messages = [m for m in messages if m["role"] == "assistant"]
        assert len(assistant_messages) >= 1
        assert len(assistant_messages[0]["content"]) > 0


class TestHostedOpenAIAsync:
    @requires_openai
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_async_completion_through_hosted_pipeline(
        self, hosted_memori_instance, openai_api_key
    ):
        client = AsyncOpenAI(api_key=openai_api_key)
        hosted_memori_instance.llm.register(client)
        hosted_memori_instance.attribution(
            entity_id="hosted-async-user", process_id="hosted-async-test"
        )

        response = await client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": TEST_PROMPT}],
            max_tokens=MAX_TOKENS,
        )

        assert response is not None
        assert response.choices[0].message.content is not None

        await asyncio.sleep(0.5)
        hosted_memori_instance.config.augmentation.wait(timeout=AA_WAIT_TIMEOUT)

    @requires_openai
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_async_completion_stores_conversation(
        self, hosted_registered_async_openai_client, hosted_memori_instance
    ):
        await hosted_registered_async_openai_client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": TEST_PROMPT}],
            max_tokens=MAX_TOKENS,
        )

        await asyncio.sleep(0.5)
        hosted_memori_instance.config.augmentation.wait(timeout=AA_WAIT_TIMEOUT)

        conversation_id = hosted_memori_instance.config.cache.conversation_id
        assert conversation_id is not None

        conversation = hosted_memori_instance.config.storage.driver.conversation.read(
            conversation_id
        )
        assert conversation is not None


class TestHostedOpenAIStreaming:
    @requires_openai
    @pytest.mark.integration
    def test_sync_streaming_through_hosted_pipeline(
        self, hosted_registered_openai_client, hosted_memori_instance
    ):
        stream = hosted_registered_openai_client.chat.completions.create(
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

        hosted_memori_instance.config.augmentation.wait(timeout=AA_WAIT_TIMEOUT)

    @requires_openai
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_async_streaming_through_hosted_pipeline(
        self, hosted_registered_async_openai_client, hosted_memori_instance
    ):
        stream = await hosted_registered_async_openai_client.chat.completions.create(
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

        await asyncio.sleep(0.5)
        hosted_memori_instance.config.augmentation.wait(timeout=AA_WAIT_TIMEOUT)


class TestHostedOpenAIAugmentation:
    @requires_openai
    @pytest.mark.integration
    def test_augmentation_completes_without_error(
        self, hosted_registered_openai_client, hosted_memori_instance
    ):
        hosted_registered_openai_client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": TEST_PROMPT}],
            max_tokens=MAX_TOKENS,
        )

        hosted_memori_instance.config.augmentation.wait(timeout=AA_WAIT_TIMEOUT)

    @requires_openai
    @pytest.mark.integration
    def test_multi_turn_triggers_augmentation(
        self, hosted_registered_openai_client, hosted_memori_instance
    ):
        hosted_registered_openai_client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "My name is Alice."},
                {"role": "assistant", "content": "Nice to meet you, Alice!"},
                {"role": "user", "content": "What is my name?"},
            ],
            max_tokens=MAX_TOKENS,
        )

        hosted_memori_instance.config.augmentation.wait(timeout=AA_WAIT_TIMEOUT)

    @requires_openai
    @pytest.mark.integration
    def test_hosted_memori_instance_is_configured(self, hosted_memori_instance):
        assert hosted_memori_instance.config is not None
        assert hosted_memori_instance.config.augmentation is not None
        assert hosted_memori_instance.config.storage is not None


class TestHostedOpenAIResponses:
    @requires_openai
    @pytest.mark.integration
    def test_responses_api_through_hosted_pipeline(
        self, hosted_registered_openai_client, hosted_memori_instance
    ):
        response = hosted_registered_openai_client.responses.create(
            model=MODEL,
            input=TEST_PROMPT,
            max_output_tokens=MAX_OUTPUT_TOKENS,
        )

        assert response is not None
        assert hasattr(response, "output_text")
        assert len(response.output_text) > 0

        hosted_memori_instance.config.augmentation.wait(timeout=AA_WAIT_TIMEOUT)

    @requires_openai
    @pytest.mark.integration
    def test_responses_streaming_through_hosted_pipeline(
        self, hosted_registered_openai_client, hosted_memori_instance
    ):
        stream = hosted_registered_openai_client.responses.create(
            model=MODEL,
            input=TEST_PROMPT,
            max_output_tokens=MAX_OUTPUT_TOKENS,
            stream=True,
        )

        events = list(stream)
        assert len(events) > 0

        event_types = [getattr(e, "type", None) for e in events]
        assert "response.completed" in event_types

        hosted_memori_instance.config.augmentation.wait(timeout=AA_WAIT_TIMEOUT)


class TestHostedOpenAISessionManagement:
    @requires_openai
    @pytest.mark.integration
    def test_multiple_calls_same_session(
        self, hosted_registered_openai_client, hosted_memori_instance
    ):
        for i in range(3):
            response = hosted_registered_openai_client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": f"Say the number {i}"}],
                max_tokens=MAX_TOKENS,
            )
            assert response is not None

        hosted_memori_instance.config.augmentation.wait(timeout=AA_WAIT_TIMEOUT)

    @requires_openai
    @pytest.mark.integration
    def test_new_session_resets_context(
        self, hosted_registered_openai_client, hosted_memori_instance
    ):
        hosted_registered_openai_client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": TEST_PROMPT}],
            max_tokens=MAX_TOKENS,
        )

        first_conversation_id = hosted_memori_instance.config.cache.conversation_id
        assert first_conversation_id is not None

        hosted_memori_instance.config.augmentation.wait(timeout=AA_WAIT_TIMEOUT)

        hosted_memori_instance.new_session()

        hosted_registered_openai_client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": TEST_PROMPT}],
            max_tokens=MAX_TOKENS,
        )

        second_conversation_id = hosted_memori_instance.config.cache.conversation_id
        assert second_conversation_id is not None
        assert first_conversation_id != second_conversation_id

        hosted_memori_instance.config.augmentation.wait(timeout=AA_WAIT_TIMEOUT)
