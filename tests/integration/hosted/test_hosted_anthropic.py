import asyncio

import pytest
from anthropic import Anthropic, AsyncAnthropic

from tests.integration.conftest import requires_anthropic

MODEL = "claude-3-haiku-20240307"
MAX_TOKENS = 50
TEST_PROMPT = "Say 'hello' in one word."
AA_WAIT_TIMEOUT = 15.0


class TestHostedAnthropicSync:
    @requires_anthropic
    @pytest.mark.integration
    def test_sync_message_through_hosted_pipeline(
        self, hosted_memori_instance, anthropic_api_key
    ):
        client = Anthropic(api_key=anthropic_api_key)
        hosted_memori_instance.llm.register(client)
        hosted_memori_instance.attribution(
            entity_id="hosted-test-user", process_id="hosted-test"
        )

        response = client.messages.create(
            model=MODEL,
            messages=[{"role": "user", "content": TEST_PROMPT}],
            max_tokens=MAX_TOKENS,
        )

        assert response is not None
        assert len(response.content) > 0
        assert response.content[0].text is not None

        hosted_memori_instance.config.augmentation.wait(timeout=AA_WAIT_TIMEOUT)

    @requires_anthropic
    @pytest.mark.integration
    def test_sync_message_stores_conversation(
        self, hosted_registered_anthropic_client, hosted_memori_instance
    ):
        hosted_registered_anthropic_client.messages.create(
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

    @requires_anthropic
    @pytest.mark.integration
    def test_sync_message_stores_messages(
        self, hosted_registered_anthropic_client, hosted_memori_instance
    ):
        test_query = "What is 2 + 2?"

        hosted_registered_anthropic_client.messages.create(
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


class TestHostedAnthropicAsync:
    @requires_anthropic
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_async_message_through_hosted_pipeline(
        self, hosted_memori_instance, anthropic_api_key
    ):
        client = AsyncAnthropic(api_key=anthropic_api_key)
        hosted_memori_instance.llm.register(client)
        hosted_memori_instance.attribution(
            entity_id="hosted-async-user", process_id="hosted-async-test"
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
        hosted_memori_instance.config.augmentation.wait(timeout=AA_WAIT_TIMEOUT)

    @requires_anthropic
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_async_message_stores_conversation(
        self, hosted_registered_async_anthropic_client, hosted_memori_instance
    ):
        await hosted_registered_async_anthropic_client.messages.create(
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


class TestHostedAnthropicStreaming:
    @requires_anthropic
    @pytest.mark.integration
    def test_sync_streaming_through_hosted_pipeline(
        self, hosted_registered_anthropic_client, hosted_memori_instance
    ):
        with hosted_registered_anthropic_client.messages.stream(
            model=MODEL,
            messages=[{"role": "user", "content": TEST_PROMPT}],
            max_tokens=MAX_TOKENS,
        ) as stream:
            full_content = "".join(stream.text_stream)

        assert len(full_content) > 0

        hosted_memori_instance.config.augmentation.wait(timeout=AA_WAIT_TIMEOUT)

    @requires_anthropic
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_async_streaming_through_hosted_pipeline(
        self, hosted_registered_async_anthropic_client, hosted_memori_instance
    ):
        async with hosted_registered_async_anthropic_client.messages.stream(
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
        hosted_memori_instance.config.augmentation.wait(timeout=AA_WAIT_TIMEOUT)


class TestHostedAnthropicAugmentation:
    @requires_anthropic
    @pytest.mark.integration
    def test_augmentation_completes_without_error(
        self, hosted_registered_anthropic_client, hosted_memori_instance
    ):
        hosted_registered_anthropic_client.messages.create(
            model=MODEL,
            messages=[{"role": "user", "content": TEST_PROMPT}],
            max_tokens=MAX_TOKENS,
        )

        hosted_memori_instance.config.augmentation.wait(timeout=AA_WAIT_TIMEOUT)

    @requires_anthropic
    @pytest.mark.integration
    def test_multi_turn_triggers_augmentation(
        self, hosted_registered_anthropic_client, hosted_memori_instance
    ):
        hosted_registered_anthropic_client.messages.create(
            model=MODEL,
            messages=[
                {"role": "user", "content": "My name is Alice."},
                {"role": "assistant", "content": "Nice to meet you, Alice!"},
                {"role": "user", "content": "What is my name?"},
            ],
            max_tokens=MAX_TOKENS,
        )

        hosted_memori_instance.config.augmentation.wait(timeout=AA_WAIT_TIMEOUT)


class TestHostedAnthropicSessionManagement:
    @requires_anthropic
    @pytest.mark.integration
    def test_multiple_calls_same_session(
        self, hosted_registered_anthropic_client, hosted_memori_instance
    ):
        for i in range(3):
            response = hosted_registered_anthropic_client.messages.create(
                model=MODEL,
                messages=[{"role": "user", "content": f"Say the number {i}"}],
                max_tokens=MAX_TOKENS,
            )
            assert response is not None

        hosted_memori_instance.config.augmentation.wait(timeout=AA_WAIT_TIMEOUT)

    @requires_anthropic
    @pytest.mark.integration
    def test_new_session_resets_context(
        self, hosted_registered_anthropic_client, hosted_memori_instance
    ):
        hosted_registered_anthropic_client.messages.create(
            model=MODEL,
            messages=[{"role": "user", "content": TEST_PROMPT}],
            max_tokens=MAX_TOKENS,
        )

        first_conversation_id = hosted_memori_instance.config.cache.conversation_id
        assert first_conversation_id is not None

        hosted_memori_instance.config.augmentation.wait(timeout=AA_WAIT_TIMEOUT)

        hosted_memori_instance.new_session()

        hosted_registered_anthropic_client.messages.create(
            model=MODEL,
            messages=[{"role": "user", "content": TEST_PROMPT}],
            max_tokens=MAX_TOKENS,
        )

        second_conversation_id = hosted_memori_instance.config.cache.conversation_id
        assert second_conversation_id is not None
        assert first_conversation_id != second_conversation_id

        hosted_memori_instance.config.augmentation.wait(timeout=AA_WAIT_TIMEOUT)
