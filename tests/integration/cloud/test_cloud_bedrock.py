import asyncio

import pytest

from tests.integration.conftest import BEDROCK_SDK_AVAILABLE, requires_bedrock

pytestmark = pytest.mark.skipif(
    not BEDROCK_SDK_AVAILABLE,
    reason="langchain-aws package not installed (pip install langchain-aws)",
)

MODEL_ID = "anthropic.claude-3-haiku-20240307-v1:0"
TEST_PROMPT = "Say 'hello' in one word."
AA_WAIT_TIMEOUT = 15.0


class TestCloudBedrockSync:
    @requires_bedrock
    @pytest.mark.integration
    def test_sync_invocation_through_cloud_pipeline(
        self, cloud_memori_instance, aws_credentials
    ):
        from langchain_aws import ChatBedrock

        client = ChatBedrock(
            model=MODEL_ID,
            region_name=aws_credentials["region_name"],
        )
        cloud_memori_instance.llm.register(chatbedrock=client)
        cloud_memori_instance.attribution(
            entity_id="cloud-test-user", process_id="cloud-test"
        )

        response = client.invoke(TEST_PROMPT)

        assert response is not None
        assert hasattr(response, "content")
        assert len(response.content) > 0

        cloud_memori_instance.config.augmentation.wait(timeout=AA_WAIT_TIMEOUT)

    @requires_bedrock
    @pytest.mark.integration
    def test_sync_invocation_stores_conversation(
        self, cloud_registered_bedrock_client, cloud_memori_instance
    ):
        cloud_registered_bedrock_client.invoke(TEST_PROMPT)

        cloud_memori_instance.config.augmentation.wait(timeout=AA_WAIT_TIMEOUT)

        conversation_id = cloud_memori_instance.config.cache.conversation_id
        assert conversation_id is not None

        conversation = cloud_memori_instance.config.storage.driver.conversation.read(
            conversation_id
        )
        assert conversation is not None
        assert conversation["id"] == conversation_id

    @requires_bedrock
    @pytest.mark.integration
    def test_sync_invocation_stores_messages(
        self, cloud_registered_bedrock_client, cloud_memori_instance
    ):
        test_query = "What is 2 + 2?"

        cloud_registered_bedrock_client.invoke(test_query)

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


class TestCloudBedrockAsync:
    @requires_bedrock
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_async_invocation_through_cloud_pipeline(
        self, cloud_memori_instance, aws_credentials
    ):
        from langchain_aws import ChatBedrock

        client = ChatBedrock(
            model=MODEL_ID,
            region_name=aws_credentials["region_name"],
        )
        cloud_memori_instance.llm.register(chatbedrock=client)
        cloud_memori_instance.attribution(
            entity_id="cloud-async-user", process_id="cloud-async-test"
        )

        response = await client.ainvoke(TEST_PROMPT)

        assert response is not None
        assert hasattr(response, "content")
        assert len(response.content) > 0

        await asyncio.sleep(0.5)
        cloud_memori_instance.config.augmentation.wait(timeout=AA_WAIT_TIMEOUT)

    @requires_bedrock
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_async_invocation_stores_conversation(
        self, cloud_registered_bedrock_client, cloud_memori_instance
    ):
        await cloud_registered_bedrock_client.ainvoke(TEST_PROMPT)

        await asyncio.sleep(0.5)
        cloud_memori_instance.config.augmentation.wait(timeout=AA_WAIT_TIMEOUT)

        conversation_id = cloud_memori_instance.config.cache.conversation_id
        assert conversation_id is not None

        conversation = cloud_memori_instance.config.storage.driver.conversation.read(
            conversation_id
        )
        assert conversation is not None


class TestCloudBedrockStreaming:
    @requires_bedrock
    @pytest.mark.integration
    def test_sync_streaming_through_cloud_pipeline(
        self, cloud_registered_bedrock_client, cloud_memori_instance
    ):
        content_parts = []
        for chunk in cloud_registered_bedrock_client.stream(TEST_PROMPT):
            if hasattr(chunk, "content") and chunk.content:
                content_parts.append(chunk.content)

        full_content = "".join(content_parts)
        assert len(full_content) > 0

        cloud_memori_instance.config.augmentation.wait(timeout=AA_WAIT_TIMEOUT)

    @requires_bedrock
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_async_streaming_through_cloud_pipeline(
        self, cloud_registered_bedrock_client, cloud_memori_instance
    ):
        content_parts = []
        async for chunk in cloud_registered_bedrock_client.astream(TEST_PROMPT):
            if hasattr(chunk, "content") and chunk.content:
                content_parts.append(chunk.content)

        full_content = "".join(content_parts)
        assert len(full_content) > 0

        await asyncio.sleep(0.5)
        cloud_memori_instance.config.augmentation.wait(timeout=AA_WAIT_TIMEOUT)


class TestCloudBedrockAugmentation:
    @requires_bedrock
    @pytest.mark.integration
    def test_augmentation_completes_without_error(
        self, cloud_registered_bedrock_client, cloud_memori_instance
    ):
        cloud_registered_bedrock_client.invoke(TEST_PROMPT)

        cloud_memori_instance.config.augmentation.wait(timeout=AA_WAIT_TIMEOUT)

    @requires_bedrock
    @pytest.mark.integration
    def test_multi_turn_triggers_augmentation(
        self, cloud_registered_bedrock_client, cloud_memori_instance
    ):
        from langchain_core.messages import AIMessage, HumanMessage

        cloud_registered_bedrock_client.invoke(
            [
                HumanMessage(content="My name is Alice."),
                AIMessage(content="Nice to meet you, Alice!"),
                HumanMessage(content="What is my name?"),
            ]
        )

        cloud_memori_instance.config.augmentation.wait(timeout=AA_WAIT_TIMEOUT)


class TestCloudBedrockSessionManagement:
    @requires_bedrock
    @pytest.mark.integration
    def test_multiple_calls_same_session(
        self, cloud_registered_bedrock_client, cloud_memori_instance
    ):
        for i in range(3):
            response = cloud_registered_bedrock_client.invoke(f"Say the number {i}")
            assert response is not None

        cloud_memori_instance.config.augmentation.wait(timeout=AA_WAIT_TIMEOUT)

    @requires_bedrock
    @pytest.mark.integration
    def test_new_session_resets_context(
        self, cloud_registered_bedrock_client, cloud_memori_instance
    ):
        cloud_registered_bedrock_client.invoke(TEST_PROMPT)

        first_conversation_id = cloud_memori_instance.config.cache.conversation_id
        assert first_conversation_id is not None

        cloud_memori_instance.config.augmentation.wait(timeout=AA_WAIT_TIMEOUT)

        cloud_memori_instance.new_session()

        cloud_registered_bedrock_client.invoke(TEST_PROMPT)

        second_conversation_id = cloud_memori_instance.config.cache.conversation_id
        assert second_conversation_id is not None
        assert first_conversation_id != second_conversation_id

        cloud_memori_instance.config.augmentation.wait(timeout=AA_WAIT_TIMEOUT)
