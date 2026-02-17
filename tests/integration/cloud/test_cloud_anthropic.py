import asyncio

import pytest
from anthropic import Anthropic, AsyncAnthropic

from tests.integration.conftest import requires_anthropic

MODEL = "claude-3-haiku-20240307"
MAX_TOKENS = 50
TEST_PROMPT = "Say 'hello' in one word."
AA_WAIT_TIMEOUT = 15.0


class TestCloudAnthropicSync:
    @requires_anthropic
    @pytest.mark.integration
    def test_sync_message_through_cloud_pipeline(
        self, cloud_memori_instance, anthropic_api_key
    ):
        client = Anthropic(api_key=anthropic_api_key)
        cloud_memori_instance.llm.register(client)
        cloud_memori_instance.attribution(
            entity_id="cloud-test-user", process_id="cloud-test"
        )

        response = client.messages.create(
            model=MODEL,
            messages=[{"role": "user", "content": TEST_PROMPT}],
            max_tokens=MAX_TOKENS,
        )

        assert response is not None
        assert len(response.content) > 0
        assert response.content[0].text is not None

        cloud_memori_instance.config.augmentation.wait(timeout=AA_WAIT_TIMEOUT)

    @requires_anthropic
    @pytest.mark.integration
    def test_sync_message_stores_conversation(
        self, cloud_registered_anthropic_client, cloud_memori_instance
    ):
        cloud_registered_anthropic_client.messages.create(
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

    @requires_anthropic
    @pytest.mark.integration
    def test_sync_message_stores_messages(
        self, cloud_registered_anthropic_client, cloud_memori_instance
    ):
        test_query = "What is 2 + 2?"

        cloud_registered_anthropic_client.messages.create(
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


class TestCloudAnthropicAsync:
    @requires_anthropic
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_async_message_through_cloud_pipeline(
        self, cloud_memori_instance, anthropic_api_key
    ):
        client = AsyncAnthropic(api_key=anthropic_api_key)
        cloud_memori_instance.llm.register(client)
        cloud_memori_instance.attribution(
            entity_id="cloud-async-user", process_id="cloud-async-test"
        )

        response = await client.messages.create(
            model=MODEL,
            messages=[{"role": "user", "content": TEST_PROMPT}],
            max_tokens=MAX_TOKENS,
        )

        assert response is not None
        assert len(response.content) > 0
        assert response.content[0].text is not None

        await asyncio.sleep(0.5)
        cloud_memori_instance.config.augmentation.wait(timeout=AA_WAIT_TIMEOUT)

    @requires_anthropic
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_async_message_stores_conversation(
        self, cloud_registered_async_anthropic_client, cloud_memori_instance
    ):
        await cloud_registered_async_anthropic_client.messages.create(
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


class TestCloudAnthropicStreaming:
    @requires_anthropic
    @pytest.mark.integration
    def test_sync_streaming_through_cloud_pipeline(
        self, cloud_registered_anthropic_client, cloud_memori_instance
    ):
        with cloud_registered_anthropic_client.messages.stream(
            model=MODEL,
            messages=[{"role": "user", "content": TEST_PROMPT}],
            max_tokens=MAX_TOKENS,
        ) as stream:
            full_content = "".join(stream.text_stream)

        assert len(full_content) > 0

        cloud_memori_instance.config.augmentation.wait(timeout=AA_WAIT_TIMEOUT)

    @requires_anthropic
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_async_streaming_through_cloud_pipeline(
        self, cloud_registered_async_anthropic_client, cloud_memori_instance
    ):
        async with cloud_registered_async_anthropic_client.messages.stream(
            model=MODEL,
            messages=[{"role": "user", "content": TEST_PROMPT}],
            max_tokens=MAX_TOKENS,
        ) as stream:
            content_parts = []
            async for text in stream.text_stream:
                content_parts.append(text)

        full_content = "".join(content_parts)
        assert len(full_content) > 0

        await asyncio.sleep(0.5)
        cloud_memori_instance.config.augmentation.wait(timeout=AA_WAIT_TIMEOUT)


class TestCloudAnthropicAugmentation:
    @requires_anthropic
    @pytest.mark.integration
    def test_augmentation_completes_without_error(
        self, cloud_registered_anthropic_client, cloud_memori_instance
    ):
        cloud_registered_anthropic_client.messages.create(
            model=MODEL,
            messages=[{"role": "user", "content": TEST_PROMPT}],
            max_tokens=MAX_TOKENS,
        )

        cloud_memori_instance.config.augmentation.wait(timeout=AA_WAIT_TIMEOUT)

    @requires_anthropic
    @pytest.mark.integration
    def test_multi_turn_triggers_augmentation(
        self, cloud_registered_anthropic_client, cloud_memori_instance
    ):
        cloud_registered_anthropic_client.messages.create(
            model=MODEL,
            messages=[
                {"role": "user", "content": "My name is Alice."},
                {"role": "assistant", "content": "Nice to meet you, Alice!"},
                {"role": "user", "content": "What is my name?"},
            ],
            max_tokens=MAX_TOKENS,
        )

        cloud_memori_instance.config.augmentation.wait(timeout=AA_WAIT_TIMEOUT)


class TestCloudAnthropicSessionManagement:
    @requires_anthropic
    @pytest.mark.integration
    def test_multiple_calls_same_session(
        self, cloud_registered_anthropic_client, cloud_memori_instance
    ):
        for i in range(3):
            response = cloud_registered_anthropic_client.messages.create(
                model=MODEL,
                messages=[{"role": "user", "content": f"Say the number {i}"}],
                max_tokens=MAX_TOKENS,
            )
            assert response is not None

        cloud_memori_instance.config.augmentation.wait(timeout=AA_WAIT_TIMEOUT)

    @requires_anthropic
    @pytest.mark.integration
    def test_new_session_resets_context(
        self, cloud_registered_anthropic_client, cloud_memori_instance
    ):
        cloud_registered_anthropic_client.messages.create(
            model=MODEL,
            messages=[{"role": "user", "content": TEST_PROMPT}],
            max_tokens=MAX_TOKENS,
        )

        first_conversation_id = cloud_memori_instance.config.cache.conversation_id
        assert first_conversation_id is not None

        cloud_memori_instance.config.augmentation.wait(timeout=AA_WAIT_TIMEOUT)

        cloud_memori_instance.new_session()

        cloud_registered_anthropic_client.messages.create(
            model=MODEL,
            messages=[{"role": "user", "content": TEST_PROMPT}],
            max_tokens=MAX_TOKENS,
        )

        second_conversation_id = cloud_memori_instance.config.cache.conversation_id
        assert second_conversation_id is not None
        assert first_conversation_id != second_conversation_id

        cloud_memori_instance.config.augmentation.wait(timeout=AA_WAIT_TIMEOUT)
