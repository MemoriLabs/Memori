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


class TestCloudGeminiSync:
    @requires_google
    @pytest.mark.integration
    def test_sync_generation_through_cloud_pipeline(
        self, cloud_memori_instance, google_api_key
    ):
        from google import genai

        client = genai.Client(api_key=google_api_key)
        cloud_memori_instance.llm.register(client)
        cloud_memori_instance.attribution(
            entity_id="cloud-test-user", process_id="cloud-test"
        )

        response = client.models.generate_content(
            model=MODEL,
            contents=TEST_PROMPT,
        )

        assert response is not None
        assert hasattr(response, "text")
        assert response.text is not None
        assert len(response.text) > 0

        cloud_memori_instance.config.augmentation.wait(timeout=AA_WAIT_TIMEOUT)

        client.close()

    @requires_google
    @pytest.mark.integration
    def test_sync_generation_stores_conversation(
        self, cloud_registered_google_client, cloud_memori_instance
    ):
        cloud_registered_google_client.models.generate_content(
            model=MODEL,
            contents=TEST_PROMPT,
        )

        cloud_memori_instance.config.augmentation.wait(timeout=AA_WAIT_TIMEOUT)

        conversation_id = cloud_memori_instance.config.cache.conversation_id
        assert conversation_id is not None

        conversation = cloud_memori_instance.config.storage.driver.conversation.read(
            conversation_id
        )
        assert conversation is not None
        assert conversation["id"] == conversation_id

    @requires_google
    @pytest.mark.integration
    def test_sync_generation_stores_messages(
        self, cloud_registered_google_client, cloud_memori_instance
    ):
        test_query = "What is 2 + 2?"

        cloud_registered_google_client.models.generate_content(
            model=MODEL,
            contents=test_query,
        )

        cloud_memori_instance.config.augmentation.wait(timeout=AA_WAIT_TIMEOUT)

        conversation_id = cloud_memori_instance.config.cache.conversation_id
        messages = (
            cloud_memori_instance.config.storage.driver.conversation.messages.read(
                conversation_id
            )
        )

        assert len(messages) >= 1

        user_messages = [m for m in messages if m["role"] == "user"]
        assert len(user_messages) >= 1
        assert test_query in user_messages[0]["content"]


class TestCloudGeminiAsync:
    @requires_google
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_async_generation_through_cloud_pipeline(
        self, cloud_memori_instance, google_api_key
    ):
        from google import genai

        client = genai.Client(api_key=google_api_key)
        cloud_memori_instance.llm.register(client)
        cloud_memori_instance.attribution(
            entity_id="cloud-async-user", process_id="cloud-async-test"
        )

        response = await client.aio.models.generate_content(
            model=MODEL,
            contents=TEST_PROMPT,
        )

        assert response is not None
        assert hasattr(response, "text")
        assert response.text is not None
        assert len(response.text) > 0

        await asyncio.sleep(0.5)
        cloud_memori_instance.config.augmentation.wait(timeout=AA_WAIT_TIMEOUT)

        client.close()

    @requires_google
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_async_generation_stores_conversation(
        self, cloud_registered_google_client, cloud_memori_instance
    ):
        await cloud_registered_google_client.aio.models.generate_content(
            model=MODEL,
            contents=TEST_PROMPT,
        )

        await asyncio.sleep(0.5)
        cloud_memori_instance.config.augmentation.wait(timeout=AA_WAIT_TIMEOUT)

        conversation_id = cloud_memori_instance.config.cache.conversation_id
        assert conversation_id is not None

        conversation = cloud_memori_instance.config.storage.driver.conversation.read(
            conversation_id
        )
        assert conversation is not None


class TestCloudGeminiStreaming:
    @requires_google
    @pytest.mark.integration
    def test_sync_streaming_through_cloud_pipeline(
        self, cloud_registered_google_client, cloud_memori_instance
    ):
        stream = cloud_registered_google_client.models.generate_content_stream(
            model=MODEL,
            contents=TEST_PROMPT,
        )

        content_parts = []
        for chunk in stream:
            if hasattr(chunk, "text") and chunk.text:
                content_parts.append(chunk.text)

        full_content = "".join(content_parts)
        assert len(full_content) > 0

        cloud_memori_instance.config.augmentation.wait(timeout=AA_WAIT_TIMEOUT)

    @requires_google
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_async_streaming_through_cloud_pipeline(
        self, cloud_registered_google_client, cloud_memori_instance
    ):
        stream = (
            await cloud_registered_google_client.aio.models.generate_content_stream(
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
        cloud_memori_instance.config.augmentation.wait(timeout=AA_WAIT_TIMEOUT)


class TestCloudGeminiAugmentation:
    @requires_google
    @pytest.mark.integration
    def test_augmentation_completes_without_error(
        self, cloud_registered_google_client, cloud_memori_instance
    ):
        cloud_registered_google_client.models.generate_content(
            model=MODEL,
            contents=TEST_PROMPT,
        )

        cloud_memori_instance.config.augmentation.wait(timeout=AA_WAIT_TIMEOUT)

    @requires_google
    @pytest.mark.integration
    def test_multi_turn_triggers_augmentation(
        self, cloud_registered_google_client, cloud_memori_instance
    ):
        from google.genai.types import Content, Part

        cloud_registered_google_client.models.generate_content(
            model=MODEL,
            contents=[
                Content(role="user", parts=[Part(text="My name is Alice.")]),
                Content(role="model", parts=[Part(text="Nice to meet you, Alice!")]),
                Content(role="user", parts=[Part(text="What is my name?")]),
            ],
        )

        cloud_memori_instance.config.augmentation.wait(timeout=AA_WAIT_TIMEOUT)


class TestCloudGeminiSessionManagement:
    @requires_google
    @pytest.mark.integration
    def test_multiple_calls_same_session(
        self, cloud_registered_google_client, cloud_memori_instance
    ):
        for i in range(3):
            response = cloud_registered_google_client.models.generate_content(
                model=MODEL,
                contents=f"Say the number {i}",
            )
            assert response is not None

        cloud_memori_instance.config.augmentation.wait(timeout=AA_WAIT_TIMEOUT)

    @requires_google
    @pytest.mark.integration
    def test_new_session_resets_context(
        self, cloud_registered_google_client, cloud_memori_instance
    ):
        cloud_registered_google_client.models.generate_content(
            model=MODEL,
            contents=TEST_PROMPT,
        )

        first_conversation_id = cloud_memori_instance.config.cache.conversation_id
        assert first_conversation_id is not None

        cloud_memori_instance.config.augmentation.wait(timeout=AA_WAIT_TIMEOUT)

        cloud_memori_instance.new_session()

        cloud_registered_google_client.models.generate_content(
            model=MODEL,
            contents=TEST_PROMPT,
        )

        second_conversation_id = cloud_memori_instance.config.cache.conversation_id
        assert second_conversation_id is not None
        assert first_conversation_id != second_conversation_id

        cloud_memori_instance.config.augmentation.wait(timeout=AA_WAIT_TIMEOUT)
