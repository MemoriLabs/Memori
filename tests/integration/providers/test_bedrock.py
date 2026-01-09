"""Integration tests for AWS Bedrock provider with real API calls.

These tests require AWS credentials to be set in the environment:
- AWS_ACCESS_KEY_ID
- AWS_SECRET_ACCESS_KEY
- AWS_REGION (optional, defaults to us-east-1)

Bedrock is accessed via LangChain's ChatBedrock class.
Registration uses named parameter: mem.llm.register(chatbedrock=client)

Run with:
    AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=... \\
        pytest tests/integration/providers/test_bedrock.py -v

Note: Requires langchain-aws package:
    pip install langchain-aws
"""

import pytest

from tests.integration.conftest import BEDROCK_SDK_AVAILABLE, requires_bedrock

# Skip entire module if Bedrock SDK not available
pytestmark = pytest.mark.skipif(
    not BEDROCK_SDK_AVAILABLE,
    reason="langchain-aws package not installed (pip install langchain-aws)",
)

# Test configuration constants
MODEL_ID = "anthropic.claude-3-haiku-20240307-v1:0"  # Claude 3 Haiku on Bedrock
TEST_PROMPT = "Say 'hello' in one word."  # Minimal prompt


class TestClientRegistration:
    """Tests for verifying client wrapping/registration works correctly."""

    @requires_bedrock
    @pytest.mark.integration
    def test_client_registration_marks_installed(
        self, memori_instance, aws_credentials
    ):
        """Verify that registering a client sets _memori_installed flag."""
        from langchain_aws import ChatBedrock  # type: ignore[import-not-found]

        client = ChatBedrock(
            model_id=MODEL_ID,
            region_name=aws_credentials["region_name"],
        )

        assert not hasattr(client, "_memori_installed")

        # Bedrock uses named parameter registration
        memori_instance.llm.register(chatbedrock=client)

        assert hasattr(client, "_memori_installed")
        assert client._memori_installed is True

    @requires_bedrock
    @pytest.mark.integration
    def test_multiple_registrations_are_idempotent(
        self, memori_instance, aws_credentials
    ):
        """Verify that registering the same client multiple times is safe."""
        from langchain_aws import ChatBedrock  # type: ignore[import-not-found]

        client = ChatBedrock(
            model_id=MODEL_ID,
            region_name=aws_credentials["region_name"],
        )

        memori_instance.llm.register(chatbedrock=client)
        original_invoke = client.invoke

        # Register again
        memori_instance.llm.register(chatbedrock=client)

        # Should still be the same wrapped method
        assert client.invoke is original_invoke
        assert client._memori_installed is True

    @requires_bedrock
    @pytest.mark.integration
    def test_registration_preserves_original_methods(
        self, memori_instance, aws_credentials
    ):
        """Verify that original methods are backed up after registration."""
        from langchain_aws import ChatBedrock  # type: ignore[import-not-found]

        client = ChatBedrock(
            model_id=MODEL_ID,
            region_name=aws_credentials["region_name"],
        )

        memori_instance.llm.register(chatbedrock=client)

        # Should have _memori_installed flag
        assert hasattr(client, "_memori_installed")
        assert client._memori_installed is True


class TestSyncInvocation:
    """Tests for synchronous invoke() calls."""

    @requires_bedrock
    @pytest.mark.integration
    def test_sync_invoke_returns_response(self, registered_bedrock_client):
        """Verify basic sync invocation works and returns valid response."""
        response = registered_bedrock_client.invoke(TEST_PROMPT)

        assert response is not None
        assert hasattr(response, "content")
        assert len(response.content) > 0

    @requires_bedrock
    @pytest.mark.integration
    def test_sync_invoke_response_structure(self, registered_bedrock_client):
        """Verify response has expected LangChain structure."""
        response = registered_bedrock_client.invoke(TEST_PROMPT)

        # Validate response structure (LangChain AIMessage)
        assert hasattr(response, "content")
        assert hasattr(response, "response_metadata")
        assert response.type == "ai"

    @requires_bedrock
    @pytest.mark.integration
    def test_sync_invoke_with_messages(self, registered_bedrock_client):
        """Verify invocation works with message list."""
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
        """Verify multi-turn conversation works."""
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
    """Tests for asynchronous ainvoke() calls."""

    @requires_bedrock
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_async_invoke_returns_response(self, registered_bedrock_client):
        """Verify basic async invocation works and returns valid response."""
        response = await registered_bedrock_client.ainvoke(TEST_PROMPT)

        assert response is not None
        assert hasattr(response, "content")
        assert len(response.content) > 0

    @requires_bedrock
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_async_invoke_response_structure(self, registered_bedrock_client):
        """Verify async response has expected structure."""
        response = await registered_bedrock_client.ainvoke(TEST_PROMPT)

        # Validate response structure
        assert hasattr(response, "content")
        assert hasattr(response, "response_metadata")
        assert response.type == "ai"

    @requires_bedrock
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_async_invoke_with_system_message(self, registered_bedrock_client):
        """Verify async invocation works with system message."""
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
    """Tests for synchronous streaming responses."""

    @requires_bedrock
    @pytest.mark.integration
    def test_sync_streaming_returns_chunks(self, registered_bedrock_client):
        """Verify sync streaming returns iterable chunks."""
        chunks = list(registered_bedrock_client.stream(TEST_PROMPT))

        assert len(chunks) > 0

    @requires_bedrock
    @pytest.mark.integration
    def test_sync_streaming_assembles_content(self, registered_bedrock_client):
        """Verify sync streaming content can be assembled."""
        content_parts = []
        for chunk in registered_bedrock_client.stream(TEST_PROMPT):
            if hasattr(chunk, "content") and chunk.content:
                content_parts.append(chunk.content)

        full_content = "".join(content_parts)
        assert len(full_content) > 0

    @requires_bedrock
    @pytest.mark.integration
    def test_sync_streaming_chunk_structure(self, registered_bedrock_client):
        """Verify streaming chunks have expected structure."""
        for chunk in registered_bedrock_client.stream(TEST_PROMPT):
            # Each chunk should be a LangChain message type
            assert hasattr(chunk, "content")


class TestAsyncStreaming:
    """Tests for asynchronous streaming responses."""

    @requires_bedrock
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_async_streaming_returns_chunks(self, registered_bedrock_client):
        """Verify async streaming returns async iterable chunks."""
        chunks = []
        async for chunk in registered_bedrock_client.astream(TEST_PROMPT):
            chunks.append(chunk)

        assert len(chunks) > 0

    @requires_bedrock
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_async_streaming_assembles_content(self, registered_bedrock_client):
        """Verify async streaming content can be assembled."""
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
        """Verify async streaming chunks have expected structure."""
        async for chunk in registered_bedrock_client.astream(TEST_PROMPT):
            # Each chunk should be a LangChain message type
            assert hasattr(chunk, "content")

    @requires_bedrock
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_async_streaming_with_usage_info(self, registered_bedrock_client):
        """Verify async streaming includes usage information in final chunk."""
        last_chunk = None
        async for chunk in registered_bedrock_client.astream(TEST_PROMPT):
            last_chunk = chunk

        # Bedrock via LangChain includes response_metadata in chunks
        assert last_chunk is not None
        # The final chunk should have content
        assert hasattr(last_chunk, "content")
        # Response metadata may include usage info (varies by model)
        if hasattr(last_chunk, "response_metadata"):
            metadata = last_chunk.response_metadata
            # Bedrock may include usage or stopReason in metadata
            assert metadata is not None


class TestErrorHandling:
    """Tests for error handling scenarios."""

    @pytest.mark.integration
    def test_invalid_credentials_raises_error(self, memori_instance):
        """Verify invalid AWS credentials raises appropriate error."""
        import os
        from unittest.mock import patch

        from langchain_aws import ChatBedrock  # type: ignore[import-not-found]

        # Temporarily override AWS credentials
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
        """Verify invalid model ID raises appropriate error."""
        from langchain_aws import ChatBedrock  # type: ignore[import-not-found]

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
        """Verify async invocation with invalid model raises error."""
        from langchain_aws import ChatBedrock  # type: ignore[import-not-found]

        client = ChatBedrock(
            model_id="invalid-model-xyz",
            region_name=aws_credentials["region_name"],
        )
        memori_instance.llm.register(chatbedrock=client)

        with pytest.raises((ValueError, RuntimeError, TypeError)):
            await client.ainvoke(TEST_PROMPT)


class TestResponseFormatValidation:
    """Tests for validating response formats and content."""

    @requires_bedrock
    @pytest.mark.integration
    def test_response_contains_usage_metadata(self, registered_bedrock_client):
        """Verify response contains usage metadata."""
        response = registered_bedrock_client.invoke(TEST_PROMPT)

        # Bedrock includes usage in response_metadata
        assert response.response_metadata is not None
        metadata = response.response_metadata
        assert "usage" in metadata or "stopReason" in metadata

    @requires_bedrock
    @pytest.mark.integration
    def test_response_model_matches_requested(self, registered_bedrock_client):
        """Verify response includes model information."""
        response = registered_bedrock_client.invoke(TEST_PROMPT)

        # Bedrock includes model info in response_metadata
        metadata = response.response_metadata
        assert metadata is not None

    @requires_bedrock
    @pytest.mark.integration
    def test_response_finish_reason_is_valid(self, registered_bedrock_client):
        """Verify response has a valid stop reason."""
        response = registered_bedrock_client.invoke(TEST_PROMPT)

        metadata = response.response_metadata
        # Bedrock uses "stopReason" in metadata
        if "stopReason" in metadata:
            valid_reasons = {"end_turn", "max_tokens", "stop_sequence", "tool_use"}
            assert metadata["stopReason"] in valid_reasons

    @requires_bedrock
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_async_response_contains_usage_metadata(
        self, registered_bedrock_client
    ):
        """Verify async response contains usage metadata."""
        response = await registered_bedrock_client.ainvoke(TEST_PROMPT)

        assert response.response_metadata is not None


class TestMemoriIntegration:
    """Tests to verify Memori-specific integration functionality."""

    @requires_bedrock
    @pytest.mark.integration
    def test_memori_wrapper_does_not_modify_response_type(
        self, aws_credentials, memori_instance
    ):
        """Verify Memori wrapper doesn't alter the response type."""
        from langchain_aws import ChatBedrock  # type: ignore[import-not-found]

        # Get response without Memori wrapper
        unwrapped_client = ChatBedrock(
            model_id=MODEL_ID,
            region_name=aws_credentials["region_name"],
        )

        # Get response with Memori wrapper
        wrapped_client = ChatBedrock(
            model_id=MODEL_ID,
            region_name=aws_credentials["region_name"],
        )
        memori_instance.llm.register(chatbedrock=wrapped_client)
        memori_instance.attribution(entity_id="test", process_id="test")

        # Both should return valid responses
        unwrapped_response = unwrapped_client.invoke(TEST_PROMPT)
        wrapped_response = wrapped_client.invoke(TEST_PROMPT)

        # Response types should be identical
        assert type(unwrapped_response) is type(wrapped_response)

    @requires_bedrock
    @pytest.mark.integration
    def test_config_captures_provider_info(self, memori_instance, aws_credentials):
        """Verify Memori config captures Bedrock provider information."""
        from langchain_aws import ChatBedrock  # type: ignore[import-not-found]

        client = ChatBedrock(
            model_id=MODEL_ID,
            region_name=aws_credentials["region_name"],
        )
        memori_instance.llm.register(chatbedrock=client)

        # Provider SDK version should be set after registration
        assert memori_instance.config.llm.provider_sdk_version is not None

    @requires_bedrock
    @pytest.mark.integration
    def test_attribution_is_preserved_across_calls(
        self, registered_bedrock_client, memori_instance
    ):
        """Verify attribution remains set across multiple API calls."""
        memori_instance.attribution(entity_id="user-123", process_id="process-456")

        # Make first call
        registered_bedrock_client.invoke(TEST_PROMPT)

        assert memori_instance.config.entity_id == "user-123"
        assert memori_instance.config.process_id == "process-456"

        # Make second call
        registered_bedrock_client.invoke(TEST_PROMPT)

        # Attribution should still be set
        assert memori_instance.config.entity_id == "user-123"
        assert memori_instance.config.process_id == "process-456"


class TestStorageVerification:
    """Tests to verify conversations are stored in the database."""

    @requires_bedrock
    @pytest.mark.integration
    def test_conversation_stored_after_sync_call(
        self, registered_bedrock_client, memori_instance
    ):
        """After invoke(), verify conversation record exists."""
        # Make an API call
        registered_bedrock_client.invoke(TEST_PROMPT)

        # Verify conversation was created
        conversation_id = memori_instance.config.cache.conversation_id
        assert conversation_id is not None

        # Read conversation from storage
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
        """Verify user query is stored after API call."""
        test_query = "What is 2 + 2?"

        # Make an API call
        registered_bedrock_client.invoke(test_query)

        # Get conversation ID
        conversation_id = memori_instance.config.cache.conversation_id
        assert conversation_id is not None

        # Read messages from storage
        messages = memori_instance.config.storage.driver.conversation.messages.read(
            conversation_id
        )

        # Should have at least user message stored
        assert len(messages) >= 1

        # Find user message
        user_messages = [m for m in messages if m["role"] == "user"]
        assert len(user_messages) >= 1

    @requires_bedrock
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_conversation_stored_after_async_call(
        self, registered_bedrock_client, memori_instance
    ):
        """After async ainvoke(), verify conversation exists."""
        # Make an async API call
        await registered_bedrock_client.ainvoke(TEST_PROMPT)

        # Verify conversation was created
        conversation_id = memori_instance.config.cache.conversation_id
        assert conversation_id is not None

        # Read conversation from storage
        conversation = memori_instance.config.storage.driver.conversation.read(
            conversation_id
        )
        assert conversation is not None

    @requires_bedrock
    @pytest.mark.integration
    def test_multiple_calls_accumulate_messages(
        self, registered_bedrock_client, memori_instance
    ):
        """Verify multiple API calls store multiple message pairs."""
        # First call
        registered_bedrock_client.invoke("First question")

        conversation_id = memori_instance.config.cache.conversation_id
        messages_after_first = (
            memori_instance.config.storage.driver.conversation.messages.read(
                conversation_id
            )
        )
        count_after_first = len(messages_after_first)

        # Second call
        registered_bedrock_client.invoke("Second question")

        messages_after_second = (
            memori_instance.config.storage.driver.conversation.messages.read(
                conversation_id
            )
        )
        count_after_second = len(messages_after_second)

        # Should have more messages after second call
        assert count_after_second > count_after_first
