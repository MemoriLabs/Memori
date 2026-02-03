import asyncio

import pytest

from tests.integration.conftest import GOOGLE_SDK_AVAILABLE, requires_google

pytestmark = pytest.mark.skipif(
    not GOOGLE_SDK_AVAILABLE,
    reason="google-genai package not installed (pip install google-genai)",
)

MODEL = "gemini-2.0-flash"
TEST_PROMPT = "Say 'hello' in one word."
AA_WAIT_TIMEOUT = 15.0


class TestHostedGeminiSync:
    @requires_google
    @pytest.mark.integration
    def test_sync_generation_through_hosted_pipeline(
        self, hosted_memori_instance, google_api_key
    ):
        from google import genai

        client = genai.Client(api_key=google_api_key)
        hosted_memori_instance.llm.register(client)
        hosted_memori_instance.attribution(
            entity_id="hosted-test-user", process_id="hosted-test"
        )

        response = client.models.generate_content(
            model=MODEL,
            contents=TEST_PROMPT,
        )

        assert response is not None
        assert hasattr(response, "text")
        assert len(response.text) > 0

        hosted_memori_instance.config.augmentation.wait(timeout=AA_WAIT_TIMEOUT)

        client.close()

    @requires_google
    @pytest.mark.integration
    def test_sync_generation_stores_conversation(
        self, hosted_registered_google_client, hosted_memori_instance
    ):
        hosted_registered_google_client.models.generate_content(
            model=MODEL,
            contents=TEST_PROMPT,
        )

        hosted_memori_instance.config.augmentation.wait(timeout=AA_WAIT_TIMEOUT)

        conversation_id = hosted_memori_instance.config.cache.conversation_id
        assert conversation_id is not None

        conversation = (
            hosted_memori_instance.config.storage.driver.conversation.read(
                conversation_id
            )
        )
        assert conversation is not None
        assert conversation["id"] == conversation_id

    @requires_google
    @pytest.mark.integration
    def test_sync_generation_stores_messages(
        self, hosted_registered_google_client, hosted_memori_instance
    ):
        test_query = "What is 2 + 2?"

        hosted_registered_google_client.models.generate_content(
            model=MODEL,
            contents=test_query,
        )

        hosted_memori_instance.config.augmentation.wait(timeout=AA_WAIT_TIMEOUT)

        conversation_id = hosted_memori_instance.config.cache.conversation_id
        messages = (
            hosted_memori_instance.config.storage.driver.conversation.messages.read(
                conversation_id
            )
        )

        assert len(messages) >= 1

        user_messages = [m for m in messages if m["role"] == "user"]
        assert len(user_messages) >= 1
        assert test_query in user_messages[0]["content"]


class TestHostedGeminiAsync:
    @requires_google
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_async_generation_through_hosted_pipeline(
        self, hosted_memori_instance, google_api_key
    ):
        from google import genai

        client = genai.Client(api_key=google_api_key)
        hosted_memori_instance.llm.register(client)
        hosted_memori_instance.attribution(
            entity_id="hosted-async-user", process_id="hosted-async-test"
        )

        response = await client.aio.models.generate_content(
            model=MODEL,
            contents=TEST_PROMPT,
        )

        assert response is not None
        assert hasattr(response, "text")
        assert len(response.text) > 0

        await asyncio.sleep(0.5)
        hosted_memori_instance.config.augmentation.wait(timeout=AA_WAIT_TIMEOUT)

        client.close()

    @requires_google
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_async_generation_stores_conversation(
        self, hosted_registered_google_client, hosted_memori_instance
    ):
        await hosted_registered_google_client.aio.models.generate_content(
            model=MODEL,
            contents=TEST_PROMPT,
        )

        await asyncio.sleep(0.5)
        hosted_memori_instance.config.augmentation.wait(timeout=AA_WAIT_TIMEOUT)

        conversation_id = hosted_memori_instance.config.cache.conversation_id
        assert conversation_id is not None

        conversation = (
            hosted_memori_instance.config.storage.driver.conversation.read(
                conversation_id
            )
        )
        assert conversation is not None


class TestHostedGeminiStreaming:
    @requires_google
    @pytest.mark.integration
    def test_sync_streaming_through_hosted_pipeline(
        self, hosted_registered_google_client, hosted_memori_instance
    ):
        stream = hosted_registered_google_client.models.generate_content_stream(
            model=MODEL,
            contents=TEST_PROMPT,
        )

        content_parts = []
        for chunk in stream:
            if hasattr(chunk, "text") and chunk.text:
                content_parts.append(chunk.text)

        full_content = "".join(content_parts)
        assert len(full_content) > 0

        hosted_memori_instance.config.augmentation.wait(timeout=AA_WAIT_TIMEOUT)

    @requires_google
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_async_streaming_through_hosted_pipeline(
        self, hosted_registered_google_client, hosted_memori_instance
    ):
        stream = (
            await hosted_registered_google_client.aio.models.generate_content_stream(
                model=MODEL,
                contents=TEST_PROMPT,
            )
        )

        content_parts = []
        async for chunk in stream:
            if hasattr(chunk, "text") and chunk.text:
                content_parts.append(chunk.text)

        full_content = "".join(content_parts)
        assert len(full_content) > 0

        await asyncio.sleep(0.5)
        hosted_memori_instance.config.augmentation.wait(timeout=AA_WAIT_TIMEOUT)


class TestHostedGeminiAugmentation:
    @requires_google
    @pytest.mark.integration
    def test_augmentation_completes_without_error(
        self, hosted_registered_google_client, hosted_memori_instance
    ):
        hosted_registered_google_client.models.generate_content(
            model=MODEL,
            contents=TEST_PROMPT,
        )

        hosted_memori_instance.config.augmentation.wait(timeout=AA_WAIT_TIMEOUT)

    @requires_google
    @pytest.mark.integration
    def test_multi_turn_triggers_augmentation(
        self, hosted_registered_google_client, hosted_memori_instance
    ):
        from google.genai.types import Content, Part

        hosted_registered_google_client.models.generate_content(
            model=MODEL,
            contents=[
                Content(role="user", parts=[Part(text="My name is Alice.")]),
                Content(role="model", parts=[Part(text="Nice to meet you, Alice!")]),
                Content(role="user", parts=[Part(text="What is my name?")]),
            ],
        )

        hosted_memori_instance.config.augmentation.wait(timeout=AA_WAIT_TIMEOUT)


class TestHostedGeminiSessionManagement:
    @requires_google
    @pytest.mark.integration
    def test_multiple_calls_same_session(
        self, hosted_registered_google_client, hosted_memori_instance
    ):
        for i in range(3):
            response = hosted_registered_google_client.models.generate_content(
                model=MODEL,
                contents=f"Say the number {i}",
            )
            assert response is not None

        hosted_memori_instance.config.augmentation.wait(timeout=AA_WAIT_TIMEOUT)

    @requires_google
    @pytest.mark.integration
    def test_new_session_resets_context(
        self, hosted_registered_google_client, hosted_memori_instance
    ):
        hosted_registered_google_client.models.generate_content(
            model=MODEL,
            contents=TEST_PROMPT,
        )

        first_conversation_id = hosted_memori_instance.config.cache.conversation_id
        assert first_conversation_id is not None

        hosted_memori_instance.config.augmentation.wait(timeout=AA_WAIT_TIMEOUT)

        hosted_memori_instance.new_session()

        hosted_registered_google_client.models.generate_content(
            model=MODEL,
            contents=TEST_PROMPT,
        )

        second_conversation_id = hosted_memori_instance.config.cache.conversation_id
        assert second_conversation_id is not None
        assert first_conversation_id != second_conversation_id

        hosted_memori_instance.config.augmentation.wait(timeout=AA_WAIT_TIMEOUT)
