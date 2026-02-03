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


class TestHostedBedrockSync:
    @requires_bedrock
    @pytest.mark.integration
    def test_sync_invocation_through_hosted_pipeline(
        self, hosted_memori_instance, aws_credentials
    ):
        from langchain_aws import ChatBedrock

        client = ChatBedrock(
            model=MODEL_ID,
            region_name=aws_credentials["region_name"],
        )
        hosted_memori_instance.llm.register(chatbedrock=client)
        hosted_memori_instance.attribution(
            entity_id="hosted-test-user", process_id="hosted-test"
        )

        response = client.invoke(TEST_PROMPT)

        assert response is not None
        assert hasattr(response, "content")
        assert len(response.content) > 0

        hosted_memori_instance.config.augmentation.wait(timeout=AA_WAIT_TIMEOUT)

    @requires_bedrock
    @pytest.mark.integration
    def test_sync_invocation_stores_conversation(
        self, hosted_registered_bedrock_client, hosted_memori_instance
    ):
        hosted_registered_bedrock_client.invoke(TEST_PROMPT)

        hosted_memori_instance.config.augmentation.wait(timeout=AA_WAIT_TIMEOUT)

        conversation_id = hosted_memori_instance.config.cache.conversation_id
        assert conversation_id is not None

        conversation = hosted_memori_instance.config.storage.driver.conversation.read(
            conversation_id
        )
        assert conversation is not None
        assert conversation["id"] == conversation_id

    @requires_bedrock
    @pytest.mark.integration
    def test_sync_invocation_stores_messages(
        self, hosted_registered_bedrock_client, hosted_memori_instance
    ):
        test_query = "What is 2 + 2?"

        hosted_registered_bedrock_client.invoke(test_query)

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


class TestHostedBedrockAsync:
    @requires_bedrock
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_async_invocation_through_hosted_pipeline(
        self, hosted_memori_instance, aws_credentials
    ):
        from langchain_aws import ChatBedrock

        client = ChatBedrock(
            model=MODEL_ID,
            region_name=aws_credentials["region_name"],
        )
        hosted_memori_instance.llm.register(chatbedrock=client)
        hosted_memori_instance.attribution(
            entity_id="hosted-async-user", process_id="hosted-async-test"
        )

        response = await client.ainvoke(TEST_PROMPT)

        assert response is not None
        assert hasattr(response, "content")
        assert len(response.content) > 0

        await asyncio.sleep(0.5)
        hosted_memori_instance.config.augmentation.wait(timeout=AA_WAIT_TIMEOUT)

    @requires_bedrock
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_async_invocation_stores_conversation(
        self, hosted_registered_bedrock_client, hosted_memori_instance
    ):
        await hosted_registered_bedrock_client.ainvoke(TEST_PROMPT)

        await asyncio.sleep(0.5)
        hosted_memori_instance.config.augmentation.wait(timeout=AA_WAIT_TIMEOUT)

        conversation_id = hosted_memori_instance.config.cache.conversation_id
        assert conversation_id is not None

        conversation = hosted_memori_instance.config.storage.driver.conversation.read(
            conversation_id
        )
        assert conversation is not None


class TestHostedBedrockStreaming:
    @requires_bedrock
    @pytest.mark.integration
    def test_sync_streaming_through_hosted_pipeline(
        self, hosted_registered_bedrock_client, hosted_memori_instance
    ):
        content_parts = []
        for chunk in hosted_registered_bedrock_client.stream(TEST_PROMPT):
            if hasattr(chunk, "content") and chunk.content:
                content_parts.append(chunk.content)

        full_content = "".join(content_parts)
        assert len(full_content) > 0

        hosted_memori_instance.config.augmentation.wait(timeout=AA_WAIT_TIMEOUT)

    @requires_bedrock
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_async_streaming_through_hosted_pipeline(
        self, hosted_registered_bedrock_client, hosted_memori_instance
    ):
        content_parts = []
        async for chunk in hosted_registered_bedrock_client.astream(TEST_PROMPT):
            if hasattr(chunk, "content") and chunk.content:
                content_parts.append(chunk.content)

        full_content = "".join(content_parts)
        assert len(full_content) > 0

        await asyncio.sleep(0.5)
        hosted_memori_instance.config.augmentation.wait(timeout=AA_WAIT_TIMEOUT)


class TestHostedBedrockAugmentation:
    @requires_bedrock
    @pytest.mark.integration
    def test_augmentation_completes_without_error(
        self, hosted_registered_bedrock_client, hosted_memori_instance
    ):
        hosted_registered_bedrock_client.invoke(TEST_PROMPT)

        hosted_memori_instance.config.augmentation.wait(timeout=AA_WAIT_TIMEOUT)

    @requires_bedrock
    @pytest.mark.integration
    def test_multi_turn_triggers_augmentation(
        self, hosted_registered_bedrock_client, hosted_memori_instance
    ):
        from langchain_core.messages import AIMessage, HumanMessage

        hosted_registered_bedrock_client.invoke(
            [
                HumanMessage(content="My name is Alice."),
                AIMessage(content="Nice to meet you, Alice!"),
                HumanMessage(content="What is my name?"),
            ]
        )

        hosted_memori_instance.config.augmentation.wait(timeout=AA_WAIT_TIMEOUT)


class TestHostedBedrockSessionManagement:
    @requires_bedrock
    @pytest.mark.integration
    def test_multiple_calls_same_session(
        self, hosted_registered_bedrock_client, hosted_memori_instance
    ):
        for i in range(3):
            response = hosted_registered_bedrock_client.invoke(f"Say the number {i}")
            assert response is not None

        hosted_memori_instance.config.augmentation.wait(timeout=AA_WAIT_TIMEOUT)

    @requires_bedrock
    @pytest.mark.integration
    def test_new_session_resets_context(
        self, hosted_registered_bedrock_client, hosted_memori_instance
    ):
        hosted_registered_bedrock_client.invoke(TEST_PROMPT)

        first_conversation_id = hosted_memori_instance.config.cache.conversation_id
        assert first_conversation_id is not None

        hosted_memori_instance.config.augmentation.wait(timeout=AA_WAIT_TIMEOUT)

        hosted_memori_instance.new_session()

        hosted_registered_bedrock_client.invoke(TEST_PROMPT)

        second_conversation_id = hosted_memori_instance.config.cache.conversation_id
        assert second_conversation_id is not None
        assert first_conversation_id != second_conversation_id

        hosted_memori_instance.config.augmentation.wait(timeout=AA_WAIT_TIMEOUT)
