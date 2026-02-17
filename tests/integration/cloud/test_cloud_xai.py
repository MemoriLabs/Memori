import asyncio

import pytest
from openai import AsyncOpenAI, OpenAI

from tests.integration.conftest import requires_xai

MODEL = "grok-beta"
MAX_TOKENS = 50
TEST_PROMPT = "Say 'hello' in one word."
XAI_BASE_URL = "https://api.x.ai/v1"
AA_WAIT_TIMEOUT = 15.0


class TestcloudXAISync:
    @requires_xai
    @pytest.mark.integration
    def test_sync_completion_through_cloud_pipeline(
        self, cloud_memori_instance, xai_api_key
    ):
        client = OpenAI(api_key=xai_api_key, base_url=XAI_BASE_URL)
        cloud_memori_instance.llm.register(client)
        cloud_memori_instance.attribution(
            entity_id="cloud-test-user", process_id="cloud-test"
        )

        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": TEST_PROMPT}],
            max_tokens=MAX_TOKENS,
        )

        assert response is not None
        assert response.choices[0].message.content is not None

        cloud_memori_instance.config.augmentation.wait(timeout=AA_WAIT_TIMEOUT)

    @requires_xai
    @pytest.mark.integration
    def test_sync_completion_stores_conversation(
        self, cloud_registered_xai_client, cloud_memori_instance
    ):
        cloud_registered_xai_client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": TEST_PROMPT}],
            max_tokens=MAX_TOKENS,
        )

        cloud_memori_instance.config.augmentation.wait(timeout=AA_WAIT_TIMEOUT)

        conversation_id = cloud_memori_instance.config.cache.conversation_id
        assert conversation_id is not None

        conversation = cloud_memori_instance.config.storage.driver.conversation.read(
            conversation_id
        )
        assert conversation is not None
        assert conversation["id"] == conversation_id

    @requires_xai
    @pytest.mark.integration
    def test_sync_completion_stores_messages(
        self, cloud_registered_xai_client, cloud_memori_instance
    ):
        test_query = "What is 2 + 2?"

        cloud_registered_xai_client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": test_query}],
            max_tokens=MAX_TOKENS,
        )

        cloud_memori_instance.config.augmentation.wait(timeout=AA_WAIT_TIMEOUT)

        conversation_id = cloud_memori_instance.config.cache.conversation_id
        messages = (
            cloud_memori_instance.config.storage.driver.conversation.messages.read(
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


class TestcloudXAIAsync:
    @requires_xai
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_async_completion_through_cloud_pipeline(
        self, cloud_memori_instance, xai_api_key
    ):
        client = AsyncOpenAI(api_key=xai_api_key, base_url=XAI_BASE_URL)
        cloud_memori_instance.llm.register(client)
        cloud_memori_instance.attribution(
            entity_id="cloud-async-user", process_id="cloud-async-test"
        )

        response = await client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": TEST_PROMPT}],
            max_tokens=MAX_TOKENS,
        )

        assert response is not None
        assert response.choices[0].message.content is not None

        await asyncio.sleep(0.5)
        cloud_memori_instance.config.augmentation.wait(timeout=AA_WAIT_TIMEOUT)

    @requires_xai
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_async_completion_stores_conversation(
        self, cloud_registered_async_xai_client, cloud_memori_instance
    ):
        await cloud_registered_async_xai_client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": TEST_PROMPT}],
            max_tokens=MAX_TOKENS,
        )

        await asyncio.sleep(0.5)
        cloud_memori_instance.config.augmentation.wait(timeout=AA_WAIT_TIMEOUT)

        conversation_id = cloud_memori_instance.config.cache.conversation_id
        assert conversation_id is not None

        conversation = cloud_memori_instance.config.storage.driver.conversation.read(
            conversation_id
        )
        assert conversation is not None


class TestcloudXAIStreaming:
    @requires_xai
    @pytest.mark.integration
    def test_sync_streaming_through_cloud_pipeline(
        self, cloud_registered_xai_client, cloud_memori_instance
    ):
        stream = cloud_registered_xai_client.chat.completions.create(
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

        cloud_memori_instance.config.augmentation.wait(timeout=AA_WAIT_TIMEOUT)

    @requires_xai
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_async_streaming_through_cloud_pipeline(
        self, cloud_registered_async_xai_client, cloud_memori_instance
    ):
        stream = await cloud_registered_async_xai_client.chat.completions.create(
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
        cloud_memori_instance.config.augmentation.wait(timeout=AA_WAIT_TIMEOUT)


class TestcloudXAIAugmentation:
    @requires_xai
    @pytest.mark.integration
    def test_augmentation_completes_without_error(
        self, cloud_registered_xai_client, cloud_memori_instance
    ):
        cloud_registered_xai_client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": TEST_PROMPT}],
            max_tokens=MAX_TOKENS,
        )

        cloud_memori_instance.config.augmentation.wait(timeout=AA_WAIT_TIMEOUT)

    @requires_xai
    @pytest.mark.integration
    def test_multi_turn_triggers_augmentation(
        self, cloud_registered_xai_client, cloud_memori_instance
    ):
        cloud_registered_xai_client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "My name is Alice."},
                {"role": "assistant", "content": "Nice to meet you, Alice!"},
                {"role": "user", "content": "What is my name?"},
            ],
            max_tokens=MAX_TOKENS,
        )

        cloud_memori_instance.config.augmentation.wait(timeout=AA_WAIT_TIMEOUT)


class TestcloudXAISessionManagement:
    @requires_xai
    @pytest.mark.integration
    def test_multiple_calls_same_session(
        self, cloud_registered_xai_client, cloud_memori_instance
    ):
        for i in range(3):
            response = cloud_registered_xai_client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": f"Say the number {i}"}],
                max_tokens=MAX_TOKENS,
            )
            assert response is not None

        cloud_memori_instance.config.augmentation.wait(timeout=AA_WAIT_TIMEOUT)

    @requires_xai
    @pytest.mark.integration
    def test_new_session_resets_context(
        self, cloud_registered_xai_client, cloud_memori_instance
    ):
        cloud_registered_xai_client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": TEST_PROMPT}],
            max_tokens=MAX_TOKENS,
        )

        first_conversation_id = cloud_memori_instance.config.cache.conversation_id
        assert first_conversation_id is not None

        cloud_memori_instance.config.augmentation.wait(timeout=AA_WAIT_TIMEOUT)

        cloud_memori_instance.new_session()

        cloud_registered_xai_client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": TEST_PROMPT}],
            max_tokens=MAX_TOKENS,
        )

        second_conversation_id = cloud_memori_instance.config.cache.conversation_id
        assert second_conversation_id is not None
        assert first_conversation_id != second_conversation_id

        cloud_memori_instance.config.augmentation.wait(timeout=AA_WAIT_TIMEOUT)
