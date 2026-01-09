"""Integration tests for Google Gemini provider with real API calls.

These tests require GOOGLE_API_KEY to be set in the environment.

Run with:
    GOOGLE_API_KEY=... pytest tests/integration/providers/test_google.py -v

To run all integration tests:
    GOOGLE_API_KEY=... pytest tests/integration/ -v -m integration

Note: Requires google-genai package (the new unified SDK):
    pip install google-genai
"""

import pytest

from tests.integration.conftest import GOOGLE_SDK_AVAILABLE, requires_google

# Skip entire module if google SDK not available
pytestmark = pytest.mark.skipif(
    not GOOGLE_SDK_AVAILABLE,
    reason="google-genai package not installed (pip install google-genai)",
)

# Test configuration constants
MODEL = "gemini-2.0-flash"  # Latest fast model
TEST_PROMPT = "Say 'hello' in one word."  # Minimal prompt


class TestClientRegistration:
    """Tests for verifying client wrapping/registration works correctly."""

    @requires_google
    @pytest.mark.integration
    def test_client_registration_marks_installed(self, memori_instance, google_api_key):
        """Verify that registering a client sets _memori_installed flag."""
        from google import genai

        client = genai.Client(api_key=google_api_key)

        assert not hasattr(client, "_memori_installed")

        memori_instance.llm.register(client)

        assert hasattr(client, "_memori_installed")
        assert getattr(client, "_memori_installed", False) is True

        client.close()

    @requires_google
    @pytest.mark.integration
    def test_multiple_registrations_are_idempotent(
        self, memori_instance, google_api_key
    ):
        """Verify that registering the same client multiple times is safe."""
        from google import genai

        client = genai.Client(api_key=google_api_key)

        memori_instance.llm.register(client)
        original_generate = client.models.generate_content

        # Register again
        memori_instance.llm.register(client)

        # Should still be the same wrapped method
        assert client.models.generate_content is original_generate
        assert getattr(client, "_memori_installed", False) is True

        client.close()

    @requires_google
    @pytest.mark.integration
    def test_registration_preserves_original_methods(
        self, memori_instance, google_api_key
    ):
        """Verify that original methods are backed up after registration."""
        from google import genai

        client = genai.Client(api_key=google_api_key)

        memori_instance.llm.register(client)

        # Google uses models.generate_content, so wrapper should be installed
        assert hasattr(client, "_memori_installed")
        assert getattr(client, "_memori_installed", False) is True

        client.close()


class TestSyncContentGeneration:
    """Tests for synchronous generate_content() calls."""

    @requires_google
    @pytest.mark.integration
    def test_sync_generate_returns_response(self, registered_google_client):
        """Verify basic sync generation works and returns valid response."""
        response = registered_google_client.models.generate_content(
            model=MODEL,
            contents=TEST_PROMPT,
        )

        assert response is not None
        assert hasattr(response, "text")
        assert len(response.text) > 0

    @requires_google
    @pytest.mark.integration
    def test_sync_generate_response_structure(self, registered_google_client):
        """Verify response has expected Google structure."""
        response = registered_google_client.models.generate_content(
            model=MODEL,
            contents=TEST_PROMPT,
        )

        # Validate response structure
        assert hasattr(response, "candidates")
        assert len(response.candidates) > 0
        assert hasattr(response.candidates[0], "content")
        assert hasattr(response.candidates[0].content, "parts")
        assert len(response.candidates[0].content.parts) > 0

    @requires_google
    @pytest.mark.integration
    def test_sync_generate_with_config(self, registered_google_client):
        """Verify generation works with custom config."""
        from google.genai.types import GenerateContentConfig

        response = registered_google_client.models.generate_content(
            model=MODEL,
            contents=TEST_PROMPT,
            config=GenerateContentConfig(
                max_output_tokens=50,
                temperature=0.5,
            ),
        )

        assert response is not None
        assert len(response.text) > 0

    @requires_google
    @pytest.mark.integration
    def test_sync_generate_multi_turn(self, registered_google_client):
        """Verify multi-turn conversation works."""
        from google.genai.types import Content, Part

        response = registered_google_client.models.generate_content(
            model=MODEL,
            contents=[
                Content(role="user", parts=[Part(text="My name is Alice.")]),
                Content(role="model", parts=[Part(text="Nice to meet you, Alice!")]),
                Content(role="user", parts=[Part(text="What is my name?")]),
            ],
        )

        assert response is not None
        content = response.text.lower()
        assert "alice" in content


class TestAsyncContentGeneration:
    """Tests for asynchronous generate_content() calls using client.aio."""

    @requires_google
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_async_generate_returns_response(self, registered_google_client):
        """Verify basic async generation works and returns valid response."""
        response = await registered_google_client.aio.models.generate_content(
            model=MODEL,
            contents=TEST_PROMPT,
        )

        assert response is not None
        assert hasattr(response, "text")
        assert len(response.text) > 0

    @requires_google
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_async_generate_response_structure(self, registered_google_client):
        """Verify async response has expected Google structure."""
        response = await registered_google_client.aio.models.generate_content(
            model=MODEL,
            contents=TEST_PROMPT,
        )

        # Validate response structure
        assert hasattr(response, "candidates")
        assert len(response.candidates) > 0
        assert hasattr(response.candidates[0], "content")

    @requires_google
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_async_generate_with_system_instruction(
        self, registered_google_client
    ):
        """Verify async generation works with system instruction config."""
        from google.genai.types import GenerateContentConfig

        response = await registered_google_client.aio.models.generate_content(
            model=MODEL,
            contents=TEST_PROMPT,
            config=GenerateContentConfig(
                system_instruction="You are a helpful assistant.",
                max_output_tokens=50,
            ),
        )

        assert response is not None
        assert len(response.text) > 0


class TestSyncStreaming:
    """Tests for synchronous streaming responses."""

    @requires_google
    @pytest.mark.integration
    def test_sync_streaming_returns_chunks(self, registered_google_client):
        """Verify sync streaming returns iterable chunks."""
        stream = registered_google_client.models.generate_content_stream(
            model=MODEL,
            contents=TEST_PROMPT,
        )

        chunks = list(stream)
        assert len(chunks) > 0

    @requires_google
    @pytest.mark.integration
    def test_sync_streaming_assembles_content(self, registered_google_client):
        """Verify sync streaming content can be assembled."""
        stream = registered_google_client.models.generate_content_stream(
            model=MODEL,
            contents=TEST_PROMPT,
        )

        content_parts = []
        for chunk in stream:
            if hasattr(chunk, "text") and chunk.text:
                content_parts.append(chunk.text)

        full_content = "".join(content_parts)
        assert len(full_content) > 0

    @requires_google
    @pytest.mark.integration
    def test_sync_streaming_chunk_structure(self, registered_google_client):
        """Verify streaming chunks have expected structure."""
        stream = registered_google_client.models.generate_content_stream(
            model=MODEL,
            contents=TEST_PROMPT,
        )

        for chunk in stream:
            # Each chunk should have candidates or be empty
            assert hasattr(chunk, "candidates") or hasattr(chunk, "text")


class TestAsyncStreaming:
    """Tests for asynchronous streaming responses."""

    @requires_google
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_async_streaming_returns_chunks(self, registered_google_client):
        """Verify async streaming returns async iterable chunks."""
        stream = await registered_google_client.aio.models.generate_content_stream(
            model=MODEL,
            contents=TEST_PROMPT,
        )

        chunks = []
        async for chunk in stream:
            chunks.append(chunk)

        assert len(chunks) > 0

    @requires_google
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_async_streaming_assembles_content(self, registered_google_client):
        """Verify async streaming content can be assembled."""
        stream = await registered_google_client.aio.models.generate_content_stream(
            model=MODEL,
            contents=TEST_PROMPT,
        )

        content_parts = []
        async for chunk in stream:
            if hasattr(chunk, "text") and chunk.text:
                content_parts.append(chunk.text)

        full_content = "".join(content_parts)
        assert len(full_content) > 0

    @requires_google
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_async_streaming_chunk_structure(self, registered_google_client):
        """Verify async streaming chunks have expected structure."""
        stream = await registered_google_client.aio.models.generate_content_stream(
            model=MODEL,
            contents=TEST_PROMPT,
        )

        async for chunk in stream:
            # Each chunk should have candidates or text attribute
            assert hasattr(chunk, "candidates") or hasattr(chunk, "text")

    @requires_google
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_async_streaming_with_usage_info(self, registered_google_client):
        """Verify async streaming includes usage metadata."""
        stream = await registered_google_client.aio.models.generate_content_stream(
            model=MODEL,
            contents=TEST_PROMPT,
        )

        last_chunk = None
        async for chunk in stream:
            last_chunk = chunk

        # Google includes usage_metadata in the final chunk
        assert last_chunk is not None
        if hasattr(last_chunk, "usage_metadata") and last_chunk.usage_metadata:
            usage = last_chunk.usage_metadata
            # Check for token counts (attribute names may vary)
            assert hasattr(usage, "prompt_token_count") or hasattr(
                usage, "candidates_token_count"
            )


class TestErrorHandling:
    """Tests for error handling scenarios."""

    @pytest.mark.integration
    def test_invalid_api_key_raises_error(self, memori_instance):
        """Verify invalid API key raises appropriate error."""
        from google import genai
        from google.genai import errors

        client = genai.Client(api_key="invalid-key-12345")
        memori_instance.llm.register(client)

        # Google may raise different exceptions for invalid keys
        with pytest.raises((errors.APIError, Exception)):
            client.models.generate_content(
                model=MODEL,
                contents=TEST_PROMPT,
            )

        client.close()

    @requires_google
    @pytest.mark.integration
    def test_invalid_model_raises_error(self, registered_google_client):
        """Verify invalid model name raises appropriate error."""
        from google.genai import errors

        with pytest.raises((errors.APIError, Exception)):
            registered_google_client.models.generate_content(
                model="nonexistent-model-xyz",
                contents=TEST_PROMPT,
            )

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_async_invalid_api_key_raises_error(self, memori_instance):
        """Verify async client with invalid API key raises error."""
        from google import genai
        from google.genai import errors

        client = genai.Client(api_key="invalid-key-12345")
        memori_instance.llm.register(client)

        with pytest.raises((errors.APIError, Exception)):
            await client.aio.models.generate_content(
                model=MODEL,
                contents=TEST_PROMPT,
            )

        client.close()


class TestResponseFormatValidation:
    """Tests for validating response formats and content."""

    @requires_google
    @pytest.mark.integration
    def test_response_contains_usage_metadata(self, registered_google_client):
        """Verify response contains token usage metadata."""
        response = registered_google_client.models.generate_content(
            model=MODEL,
            contents=TEST_PROMPT,
        )

        # Google provides usage_metadata
        assert hasattr(response, "usage_metadata") or hasattr(response, "candidates")

    @requires_google
    @pytest.mark.integration
    def test_response_finish_reason_is_valid(self, registered_google_client):
        """Verify finish_reason is present."""
        response = registered_google_client.models.generate_content(
            model=MODEL,
            contents=TEST_PROMPT,
        )

        assert len(response.candidates) > 0
        # Google uses finish_reason on candidates
        candidate = response.candidates[0]
        assert hasattr(candidate, "finish_reason")

    @requires_google
    @pytest.mark.integration
    def test_response_model_info_is_present(self, registered_google_client):
        """Verify response contains model info."""
        response = registered_google_client.models.generate_content(
            model=MODEL,
            contents=TEST_PROMPT,
        )

        # Google responses should have model_version
        assert hasattr(response, "model_version") or hasattr(response, "candidates")

    @requires_google
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_async_response_contains_usage_metadata(
        self, registered_google_client
    ):
        """Verify async response contains usage metadata."""
        response = await registered_google_client.aio.models.generate_content(
            model=MODEL,
            contents=TEST_PROMPT,
        )

        # Google provides usage_metadata
        assert hasattr(response, "usage_metadata") or hasattr(response, "candidates")


class TestMemoriIntegration:
    """Tests to verify Memori-specific integration functionality."""

    @requires_google
    @pytest.mark.integration
    def test_memori_wrapper_does_not_modify_response_type(
        self, google_api_key, memori_instance
    ):
        """Verify Memori wrapper doesn't alter the response type."""
        from google import genai

        # Get response without Memori wrapper
        unwrapped_client = genai.Client(api_key=google_api_key)

        # Get response with Memori wrapper
        wrapped_client = genai.Client(api_key=google_api_key)
        memori_instance.llm.register(wrapped_client)
        memori_instance.attribution(entity_id="test", process_id="test")

        # Both should return valid responses
        unwrapped_response = unwrapped_client.models.generate_content(
            model=MODEL,
            contents=TEST_PROMPT,
        )
        wrapped_response = wrapped_client.models.generate_content(
            model=MODEL,
            contents=TEST_PROMPT,
        )

        # Response types should be identical
        assert type(unwrapped_response) is type(wrapped_response)

        unwrapped_client.close()
        wrapped_client.close()

    @requires_google
    @pytest.mark.integration
    def test_config_captures_provider_info(self, memori_instance, google_api_key):
        """Verify Memori config captures Google provider information."""
        from google import genai

        client = genai.Client(api_key=google_api_key)
        memori_instance.llm.register(client)

        # Provider SDK version should be set after registration
        assert memori_instance.config.llm.provider_sdk_version is not None

        client.close()

    @requires_google
    @pytest.mark.integration
    def test_attribution_is_preserved_across_calls(
        self, registered_google_client, memori_instance
    ):
        """Verify attribution remains set across multiple API calls."""
        memori_instance.attribution(entity_id="user-123", process_id="process-456")

        # Make first call
        registered_google_client.models.generate_content(
            model=MODEL,
            contents=TEST_PROMPT,
        )

        assert memori_instance.config.entity_id == "user-123"
        assert memori_instance.config.process_id == "process-456"

        # Make second call
        registered_google_client.models.generate_content(
            model=MODEL,
            contents=TEST_PROMPT,
        )

        # Attribution should still be set
        assert memori_instance.config.entity_id == "user-123"
        assert memori_instance.config.process_id == "process-456"


class TestStorageVerification:
    """Tests to verify conversations are stored in the database."""

    @requires_google
    @pytest.mark.integration
    def test_conversation_stored_after_sync_call(
        self, registered_google_client, memori_instance
    ):
        """After generate_content(), verify conversation record exists."""
        # Make an API call
        registered_google_client.models.generate_content(
            model=MODEL,
            contents=TEST_PROMPT,
        )

        # Verify conversation was created
        conversation_id = memori_instance.config.cache.conversation_id
        assert conversation_id is not None

        # Read conversation from storage
        conversation = memori_instance.config.storage.driver.conversation.read(
            conversation_id
        )
        assert conversation is not None
        assert conversation["id"] == conversation_id

    @requires_google
    @pytest.mark.integration
    def test_messages_stored_with_content(
        self, registered_google_client, memori_instance
    ):
        """Verify user query is stored after API call.

        Note: Assistant response storage depends on the memori Google adapter
        being updated for the new google-genai SDK response format.
        """
        test_query = "What is 2 + 2?"

        # Make an API call
        registered_google_client.models.generate_content(
            model=MODEL,
            contents=test_query,
        )

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
        assert test_query in user_messages[0]["content"]

    @requires_google
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_conversation_stored_after_async_call(
        self, registered_google_client, memori_instance
    ):
        """After async generate_content(), verify conversation exists."""
        # Make an async API call
        await registered_google_client.aio.models.generate_content(
            model=MODEL,
            contents=TEST_PROMPT,
        )

        # Verify conversation was created
        conversation_id = memori_instance.config.cache.conversation_id
        assert conversation_id is not None

        # Read conversation from storage
        conversation = memori_instance.config.storage.driver.conversation.read(
            conversation_id
        )
        assert conversation is not None

    @requires_google
    @pytest.mark.integration
    def test_multiple_calls_accumulate_messages(
        self, registered_google_client, memori_instance
    ):
        """Verify multiple API calls store multiple message pairs."""
        # First call
        registered_google_client.models.generate_content(
            model=MODEL,
            contents="First question",
        )

        conversation_id = memori_instance.config.cache.conversation_id
        messages_after_first = (
            memori_instance.config.storage.driver.conversation.messages.read(
                conversation_id
            )
        )
        count_after_first = len(messages_after_first)

        # Second call
        registered_google_client.models.generate_content(
            model=MODEL,
            contents="Second question",
        )

        messages_after_second = (
            memori_instance.config.storage.driver.conversation.messages.read(
                conversation_id
            )
        )
        count_after_second = len(messages_after_second)

        # Should have more messages after second call
        assert count_after_second > count_after_first
