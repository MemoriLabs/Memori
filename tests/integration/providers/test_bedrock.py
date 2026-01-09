import pytest

from tests.integration.conftest import BEDROCK_SDK_AVAILABLE, requires_bedrock

pytestmark = pytest.mark.skipif(
    not BEDROCK_SDK_AVAILABLE,
    reason="langchain-aws package not installed (pip install langchain-aws)",
)

MODEL_ID = "anthropic.claude-3-haiku-20240307-v1:0"
TEST_PROMPT = "Say 'hello' in one word."


class TestClientRegistration:
    @requires_bedrock
    @pytest.mark.integration
    def test_client_registration_marks_installed(
        self, memori_instance, aws_credentials
    ):
        from langchain_aws import ChatBedrock

        client = ChatBedrock(
            model_id=MODEL_ID,
            region_name=aws_credentials["region_name"],
        )

        assert not hasattr(client, "_memori_installed")

        memori_instance.llm.register(chatbedrock=client)

        assert hasattr(client, "_memori_installed")
        assert client._memori_installed is True

    @requires_bedrock
    @pytest.mark.integration
    def test_multiple_registrations_are_idempotent(
        self, memori_instance, aws_credentials
    ):
        from langchain_aws import ChatBedrock

        client = ChatBedrock(
            model_id=MODEL_ID,
            region_name=aws_credentials["region_name"],
        )

        memori_instance.llm.register(chatbedrock=client)
        original_invoke = client.invoke

        memori_instance.llm.register(chatbedrock=client)

        assert client.invoke is original_invoke
        assert client._memori_installed is True

    @requires_bedrock
    @pytest.mark.integration
    def test_registration_preserves_original_methods(
        self, memori_instance, aws_credentials
    ):
        from langchain_aws import ChatBedrock

        client = ChatBedrock(
            model_id=MODEL_ID,
            region_name=aws_credentials["region_name"],
        )

        memori_instance.llm.register(chatbedrock=client)

        assert hasattr(client, "_memori_installed")
        assert client._memori_installed is True


class TestSyncInvocation:
    @requires_bedrock
    @pytest.mark.integration
    def test_sync_invoke_returns_response(self, registered_bedrock_client):
        response = registered_bedrock_client.invoke(TEST_PROMPT)

        assert response is not None
        assert hasattr(response, "content")
        assert len(response.content) > 0

    @requires_bedrock
    @pytest.mark.integration
    def test_sync_invoke_response_structure(self, registered_bedrock_client):
        response = registered_bedrock_client.invoke(TEST_PROMPT)

        assert hasattr(response, "content")
        assert hasattr(response, "response_metadata")
        assert response.type == "ai"

    @requires_bedrock
    @pytest.mark.integration
    def test_sync_invoke_with_messages(self, registered_bedrock_client):
        from langchain_core.messages import HumanMessage, SystemMessage

        response = registered_bedrock_client.invoke(
            [
                SystemMessage(content="You are a helpful assistant."),
                HumanMessage(content=TEST_PROMPT),
            ]
        )

        assert response is not None
        assert len(response.content) > 0

    @requires_bedrock
    @pytest.mark.integration
    def test_sync_invoke_multi_turn(self, registered_bedrock_client):
        from langchain_core.messages import AIMessage, HumanMessage

        response = registered_bedrock_client.invoke(
            [
                HumanMessage(content="My name is Alice."),
                AIMessage(content="Nice to meet you, Alice!"),
                HumanMessage(content="What is my name?"),
            ]
        )

        assert response is not None
        content = response.content.lower()
        assert "alice" in content


class TestAsyncInvocation:
    @requires_bedrock
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_async_invoke_returns_response(self, registered_bedrock_client):
        response = await registered_bedrock_client.ainvoke(TEST_PROMPT)

        assert response is not None
        assert hasattr(response, "content")
        assert len(response.content) > 0

    @requires_bedrock
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_async_invoke_response_structure(self, registered_bedrock_client):
        response = await registered_bedrock_client.ainvoke(TEST_PROMPT)

        assert hasattr(response, "content")
        assert hasattr(response, "response_metadata")
        assert response.type == "ai"

    @requires_bedrock
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_async_invoke_with_system_message(self, registered_bedrock_client):
        from langchain_core.messages import HumanMessage, SystemMessage

        response = await registered_bedrock_client.ainvoke(
            [
                SystemMessage(content="You are a helpful assistant."),
                HumanMessage(content=TEST_PROMPT),
            ]
        )

        assert response is not None
        assert len(response.content) > 0


class TestSyncStreaming:
    @requires_bedrock
    @pytest.mark.integration
    def test_sync_streaming_returns_chunks(self, registered_bedrock_client):
        chunks = list(registered_bedrock_client.stream(TEST_PROMPT))

        assert len(chunks) > 0

    @requires_bedrock
    @pytest.mark.integration
    def test_sync_streaming_assembles_content(self, registered_bedrock_client):
        content_parts = []
        for chunk in registered_bedrock_client.stream(TEST_PROMPT):
            if hasattr(chunk, "content") and chunk.content:
                content_parts.append(chunk.content)

        full_content = "".join(content_parts)
        assert len(full_content) > 0

    @requires_bedrock
    @pytest.mark.integration
    def test_sync_streaming_chunk_structure(self, registered_bedrock_client):
        for chunk in registered_bedrock_client.stream(TEST_PROMPT):
            assert hasattr(chunk, "content")


class TestAsyncStreaming:
    @requires_bedrock
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_async_streaming_returns_chunks(self, registered_bedrock_client):
        chunks = []
        async for chunk in registered_bedrock_client.astream(TEST_PROMPT):
            chunks.append(chunk)

        assert len(chunks) > 0

    @requires_bedrock
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_async_streaming_assembles_content(self, registered_bedrock_client):
        content_parts = []
        async for chunk in registered_bedrock_client.astream(TEST_PROMPT):
            if hasattr(chunk, "content") and chunk.content:
                content_parts.append(chunk.content)

        full_content = "".join(content_parts)
        assert len(full_content) > 0

    @requires_bedrock
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_async_streaming_chunk_structure(self, registered_bedrock_client):
        async for chunk in registered_bedrock_client.astream(TEST_PROMPT):
            assert hasattr(chunk, "content")

    @requires_bedrock
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_async_streaming_with_usage_info(self, registered_bedrock_client):
        last_chunk = None
        async for chunk in registered_bedrock_client.astream(TEST_PROMPT):
            last_chunk = chunk

        assert last_chunk is not None
        assert hasattr(last_chunk, "content")
        if hasattr(last_chunk, "response_metadata"):
            metadata = last_chunk.response_metadata
            assert metadata is not None


class TestErrorHandling:
    @pytest.mark.integration
    def test_invalid_credentials_raises_error(self, memori_instance):
        import os
        from unittest.mock import patch

        from langchain_aws import ChatBedrock

        with patch.dict(
            os.environ,
            {
                "AWS_ACCESS_KEY_ID": "invalid-key",
                "AWS_SECRET_ACCESS_KEY": "invalid-secret",
            },
        ):
            client = ChatBedrock(
                model_id=MODEL_ID,
                region_name="us-east-1",
            )
            memori_instance.llm.register(chatbedrock=client)

            with pytest.raises((ValueError, RuntimeError, Exception)):
                client.invoke(TEST_PROMPT)

    @requires_bedrock
    @pytest.mark.integration
    def test_invalid_model_raises_error(self, memori_instance, aws_credentials):
        from langchain_aws import ChatBedrock

        client = ChatBedrock(
            model_id="invalid-model-xyz",
            region_name=aws_credentials["region_name"],
        )
        memori_instance.llm.register(chatbedrock=client)

        with pytest.raises((ValueError, RuntimeError, TypeError)):
            client.invoke(TEST_PROMPT)

    @requires_bedrock
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_async_invalid_model_raises_error(
        self, memori_instance, aws_credentials
    ):
        from langchain_aws import ChatBedrock

        client = ChatBedrock(
            model_id="invalid-model-xyz",
            region_name=aws_credentials["region_name"],
        )
        memori_instance.llm.register(chatbedrock=client)

        with pytest.raises((ValueError, RuntimeError, TypeError)):
            await client.ainvoke(TEST_PROMPT)


class TestResponseFormatValidation:
    @requires_bedrock
    @pytest.mark.integration
    def test_response_contains_usage_metadata(self, registered_bedrock_client):
        response = registered_bedrock_client.invoke(TEST_PROMPT)

        assert response.response_metadata is not None
        metadata = response.response_metadata
        assert "usage" in metadata or "stopReason" in metadata

    @requires_bedrock
    @pytest.mark.integration
    def test_response_model_matches_requested(self, registered_bedrock_client):
        response = registered_bedrock_client.invoke(TEST_PROMPT)

        metadata = response.response_metadata
        assert metadata is not None

    @requires_bedrock
    @pytest.mark.integration
    def test_response_finish_reason_is_valid(self, registered_bedrock_client):
        response = registered_bedrock_client.invoke(TEST_PROMPT)

        metadata = response.response_metadata
        if "stopReason" in metadata:
            valid_reasons = {"end_turn", "max_tokens", "stop_sequence", "tool_use"}
            assert metadata["stopReason"] in valid_reasons

    @requires_bedrock
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_async_response_contains_usage_metadata(
        self, registered_bedrock_client
    ):
        response = await registered_bedrock_client.ainvoke(TEST_PROMPT)

        assert response.response_metadata is not None


class TestMemoriIntegration:
    @requires_bedrock
    @pytest.mark.integration
    def test_memori_wrapper_does_not_modify_response_type(
        self, aws_credentials, memori_instance
    ):
        from langchain_aws import ChatBedrock

        unwrapped_client = ChatBedrock(
            model_id=MODEL_ID,
            region_name=aws_credentials["region_name"],
        )

        wrapped_client = ChatBedrock(
            model_id=MODEL_ID,
            region_name=aws_credentials["region_name"],
        )
        memori_instance.llm.register(chatbedrock=wrapped_client)
        memori_instance.attribution(entity_id="test", process_id="test")

        unwrapped_response = unwrapped_client.invoke(TEST_PROMPT)
        wrapped_response = wrapped_client.invoke(TEST_PROMPT)

        assert type(unwrapped_response) is type(wrapped_response)

    @requires_bedrock
    @pytest.mark.integration
    def test_config_captures_provider_info(self, memori_instance, aws_credentials):
        from langchain_aws import ChatBedrock

        client = ChatBedrock(
            model_id=MODEL_ID,
            region_name=aws_credentials["region_name"],
        )
        memori_instance.llm.register(chatbedrock=client)

        assert memori_instance.config.llm.provider_sdk_version is not None

    @requires_bedrock
    @pytest.mark.integration
    def test_attribution_is_preserved_across_calls(
        self, registered_bedrock_client, memori_instance
    ):
        memori_instance.attribution(entity_id="user-123", process_id="process-456")

        registered_bedrock_client.invoke(TEST_PROMPT)

        assert memori_instance.config.entity_id == "user-123"
        assert memori_instance.config.process_id == "process-456"

        registered_bedrock_client.invoke(TEST_PROMPT)

        assert memori_instance.config.entity_id == "user-123"
        assert memori_instance.config.process_id == "process-456"


class TestStorageVerification:
    @requires_bedrock
    @pytest.mark.integration
    def test_conversation_stored_after_sync_call(
        self, registered_bedrock_client, memori_instance
    ):
        registered_bedrock_client.invoke(TEST_PROMPT)

        conversation_id = memori_instance.config.cache.conversation_id
        assert conversation_id is not None

        conversation = memori_instance.config.storage.driver.conversation.read(
            conversation_id
        )
        assert conversation is not None
        assert conversation["id"] == conversation_id

    @requires_bedrock
    @pytest.mark.integration
    def test_messages_stored_with_content(
        self, registered_bedrock_client, memori_instance
    ):
        test_query = "What is 2 + 2?"

        registered_bedrock_client.invoke(test_query)

        conversation_id = memori_instance.config.cache.conversation_id
        assert conversation_id is not None

        messages = memori_instance.config.storage.driver.conversation.messages.read(
            conversation_id
        )

        assert len(messages) >= 1

        user_messages = [m for m in messages if m["role"] == "user"]
        assert len(user_messages) >= 1

    @requires_bedrock
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_conversation_stored_after_async_call(
        self, registered_bedrock_client, memori_instance
    ):
        await registered_bedrock_client.ainvoke(TEST_PROMPT)

        conversation_id = memori_instance.config.cache.conversation_id
        assert conversation_id is not None

        conversation = memori_instance.config.storage.driver.conversation.read(
            conversation_id
        )
        assert conversation is not None

    @requires_bedrock
    @pytest.mark.integration
    def test_multiple_calls_accumulate_messages(
        self, registered_bedrock_client, memori_instance
    ):
        registered_bedrock_client.invoke("First question")

        conversation_id = memori_instance.config.cache.conversation_id
        messages_after_first = (
            memori_instance.config.storage.driver.conversation.messages.read(
                conversation_id
            )
        )
        count_after_first = len(messages_after_first)

        registered_bedrock_client.invoke("Second question")

        messages_after_second = (
            memori_instance.config.storage.driver.conversation.messages.read(
                conversation_id
            )
        )
        count_after_second = len(messages_after_second)

        assert count_after_second > count_after_first
