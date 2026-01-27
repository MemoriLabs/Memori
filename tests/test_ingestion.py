from unittest.mock import AsyncMock, MagicMock

import pytest

from memori.seed._ingest import (
    DEFAULT_MAX_CHARS_PER_REQUEST,
    DEFAULT_MAX_MESSAGES_PER_REQUEST,
    DEFAULT_MAX_RETRIES,
    DEFAULT_RETRY_DELAY,
    ConversationResult,
    SeedClient,
    SeedConfig,
    SeedData,
    SeedResult,
    SeedType,
    estimate_conversation_size,
    validate_conversation,
    validate_message,
)


class TestSeedType:
    def test_conversation_type_exists(self):
        assert SeedType.CONVERSATION.value == "conversation"

    def test_seed_type_is_enum(self):
        assert isinstance(SeedType.CONVERSATION, SeedType)


class TestSeedData:
    def test_for_conversations_creation(self):
        conversations = [
            {"id": "conv-1", "messages": [{"role": "user", "content": "Hello"}]}
        ]
        seed_data = SeedData.for_conversations(
            entity_id="user-123",
            conversations=conversations,
        )

        assert seed_data.seed_type == SeedType.CONVERSATION
        assert seed_data.entity_id == "user-123"
        assert seed_data.process_id is None
        assert seed_data.data == conversations

    def test_for_conversations_with_process_id(self):
        conversations = [{"id": "conv-1", "messages": []}]
        seed_data = SeedData.for_conversations(
            entity_id="user-123",
            conversations=conversations,
            process_id="my-process",
        )

        assert seed_data.process_id == "my-process"

    def test_entity_id_required(self):
        conversations = [{"id": "conv-1", "messages": []}]
        seed_data = SeedData.for_conversations(
            entity_id="",
            conversations=conversations,
        )
        assert seed_data.entity_id == ""

    def test_conversations_stored_as_data(self):
        conversations = [
            {"id": "1", "messages": [{"role": "user", "content": "a"}]},
            {"id": "2", "messages": [{"role": "user", "content": "b"}]},
        ]
        seed_data = SeedData.for_conversations(
            entity_id="user-123",
            conversations=conversations,
        )
        assert len(seed_data.data) == 2


class TestSeedConfig:
    def test_default_values(self):
        config = SeedConfig()

        assert config.max_messages_per_request == DEFAULT_MAX_MESSAGES_PER_REQUEST
        assert config.max_chars_per_request == DEFAULT_MAX_CHARS_PER_REQUEST
        assert config.max_retries == DEFAULT_MAX_RETRIES
        assert config.retry_delay == DEFAULT_RETRY_DELAY
        assert config.chunk_large_conversations is True

    def test_custom_values(self):
        config = SeedConfig(
            max_messages_per_request=1000,
            max_chars_per_request=100_000,
            max_retries=5,
            retry_delay=1.0,
            chunk_large_conversations=False,
        )

        assert config.max_messages_per_request == 1000
        assert config.max_chars_per_request == 100_000
        assert config.max_retries == 5
        assert config.retry_delay == 1.0
        assert config.chunk_large_conversations is False

    def test_defaults_are_safe_margins(self):
        assert DEFAULT_MAX_MESSAGES_PER_REQUEST == 4000
        assert DEFAULT_MAX_CHARS_PER_REQUEST == 800_000


class TestSeedResult:
    def test_default_values(self):
        result = SeedResult()

        assert result.total == 0
        assert result.successful == 0
        assert result.failed == 0
        assert result.total_triples == 0
        assert result.duration_ms == 0
        assert result.conversations == []
        assert result.chunked_conversations == 0

    def test_success_rate_zero_total(self):
        result = SeedResult(total=0)
        assert result.success_rate == 0.0

    def test_success_rate_all_successful(self):
        result = SeedResult(total=10, successful=10)
        assert result.success_rate == 100.0

    def test_success_rate_partial(self):
        result = SeedResult(total=10, successful=7, failed=3)
        assert result.success_rate == 70.0

    def test_success_rate_none_successful(self):
        result = SeedResult(total=5, successful=0, failed=5)
        assert result.success_rate == 0.0


class TestConversationResult:
    def test_successful_result(self):
        result = ConversationResult(
            conversation_id="conv-1",
            success=True,
            triples_count=5,
            summary="Test summary",
            duration_ms=100.0,
        )

        assert result.conversation_id == "conv-1"
        assert result.success is True
        assert result.triples_count == 5
        assert result.summary == "Test summary"
        assert result.error is None
        assert result.duration_ms == 100.0
        assert result.chunks_processed == 1
        assert result.warnings == []

    def test_failed_result(self):
        result = ConversationResult(
            conversation_id="conv-1",
            success=False,
            error="Something went wrong",
        )

        assert result.success is False
        assert result.error == "Something went wrong"
        assert result.triples_count == 0

    def test_chunked_result(self):
        result = ConversationResult(
            conversation_id="conv-1",
            success=True,
            chunks_processed=3,
            warnings=["Large conversation processed in 3 chunks"],
        )

        assert result.chunks_processed == 3
        assert len(result.warnings) == 1


class TestValidateMessage:
    def test_valid_user_message(self):
        message = {"role": "user", "content": "Hello!"}
        errors = validate_message(message, 0)
        assert errors == []

    def test_valid_assistant_message(self):
        message = {"role": "assistant", "content": "Hi there!"}
        errors = validate_message(message, 0)
        assert errors == []

    def test_valid_system_message(self):
        message = {"role": "system", "content": "You are a helpful assistant."}
        errors = validate_message(message, 0)
        assert errors == []

    def test_missing_role(self):
        message = {"content": "Hello!"}
        errors = validate_message(message, 0)
        assert len(errors) == 1
        assert "missing required field 'role'" in errors[0]

    def test_invalid_role(self):
        message = {"role": "invalid", "content": "Hello!"}
        errors = validate_message(message, 0)
        assert len(errors) == 1
        assert "invalid role 'invalid'" in errors[0]

    def test_missing_content(self):
        message = {"role": "user"}
        errors = validate_message(message, 0)
        assert len(errors) == 1
        assert "missing required field 'content'" in errors[0]

    def test_non_string_content(self):
        message = {"role": "user", "content": 123}
        errors = validate_message(message, 0)
        assert len(errors) == 1
        assert "'content' must be a string" in errors[0]

    def test_empty_content(self):
        message = {"role": "user", "content": "   "}
        errors = validate_message(message, 0)
        assert len(errors) == 1
        assert "'content' is empty" in errors[0]

    def test_non_dict_message(self):
        message = "not a dict"
        errors = validate_message(message, 0)
        assert len(errors) == 1
        assert "must be a dictionary" in errors[0]

    def test_multiple_errors(self):
        message = {"role": "invalid"}
        errors = validate_message(message, 0)
        assert len(errors) == 2


class TestValidateConversation:
    def test_valid_conversation(self):
        conversation = {
            "id": "conv-1",
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi!"},
            ],
        }
        is_valid, errors = validate_conversation(conversation)
        assert is_valid is True
        assert errors == []

    def test_non_dict_conversation(self):
        is_valid, errors = validate_conversation("not a dict")
        assert is_valid is False
        assert "must be a dictionary" in errors[0]

    def test_empty_messages(self):
        conversation = {"id": "conv-1", "messages": []}
        is_valid, errors = validate_conversation(conversation)
        assert is_valid is False
        assert "no messages" in errors[0]

    def test_missing_messages(self):
        conversation = {"id": "conv-1"}
        is_valid, errors = validate_conversation(conversation)
        assert is_valid is False
        assert "no messages" in errors[0]

    def test_messages_not_list(self):
        conversation = {"id": "conv-1", "messages": "not a list"}
        is_valid, errors = validate_conversation(conversation)
        assert is_valid is False
        assert "must be a list" in errors[0]

    def test_invalid_messages_in_conversation(self):
        conversation = {
            "id": "conv-1",
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "invalid", "content": ""},
            ],
        }
        is_valid, errors = validate_conversation(conversation)
        assert is_valid is False
        assert len(errors) == 2


class TestEstimateConversationSize:
    def test_empty_messages(self):
        msg_count, char_count = estimate_conversation_size([])
        assert msg_count == 0
        assert char_count == 0

    def test_single_message(self):
        messages = [{"role": "user", "content": "Hello"}]
        msg_count, char_count = estimate_conversation_size(messages)
        assert msg_count == 1
        assert char_count == 5

    def test_multiple_messages(self):
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        msg_count, char_count = estimate_conversation_size(messages)
        assert msg_count == 2
        assert char_count == 14

    def test_message_without_content(self):
        messages = [{"role": "user"}]
        msg_count, char_count = estimate_conversation_size(messages)
        assert msg_count == 1
        assert char_count == 0


class TestSeedClientInit:
    def test_batch_size_validation(self):
        config = MagicMock()
        driver = MagicMock()

        with pytest.raises(ValueError, match="batch_size must be >= 1"):
            SeedClient(
                config=config,
                driver=driver,
                entity_id="user-123",
                batch_size=0,
            )

    def test_batch_size_negative(self):
        config = MagicMock()
        driver = MagicMock()

        with pytest.raises(ValueError, match="batch_size must be >= 1"):
            SeedClient(
                config=config,
                driver=driver,
                entity_id="user-123",
                batch_size=-1,
            )

    def test_default_seed_config(self):
        config = MagicMock()
        driver = MagicMock()

        client = SeedClient(
            config=config,
            driver=driver,
            entity_id="user-123",
        )

        assert client.seed_config is not None
        assert isinstance(client.seed_config, SeedConfig)


class TestSeedClientChunking:
    @pytest.fixture
    def client(self):
        config = MagicMock()
        driver = MagicMock()
        driver.conversation.conn.get_dialect.return_value = "postgresql"
        return SeedClient(
            config=config,
            driver=driver,
            entity_id="user-123",
            seed_config=SeedConfig(
                max_messages_per_request=5,
                max_chars_per_request=100,
            ),
        )

    def test_should_not_chunk_small_conversation(self, client):
        messages = [{"role": "user", "content": "Hi"}]
        assert client._should_chunk_conversation(messages) is False

    def test_should_chunk_by_message_count(self, client):
        messages = [{"role": "user", "content": "Hi"} for _ in range(10)]
        assert client._should_chunk_conversation(messages) is True

    def test_should_chunk_by_character_count(self, client):
        messages = [{"role": "user", "content": "x" * 200}]
        assert client._should_chunk_conversation(messages) is True

    def test_chunk_messages_by_count(self, client):
        messages = [{"role": "user", "content": "Hi"} for _ in range(12)]
        chunks = client._chunk_messages(messages)

        assert len(chunks) == 3
        assert len(chunks[0]) == 5
        assert len(chunks[1]) == 5
        assert len(chunks[2]) == 2

    def test_chunk_messages_by_characters(self, client):
        messages = [{"role": "user", "content": "x" * 50} for _ in range(5)]
        chunks = client._chunk_messages(messages)

        assert len(chunks) == 3

    def test_chunk_preserves_order(self, client):
        messages = [{"role": "user", "content": f"msg-{i}"} for i in range(10)]
        chunks = client._chunk_messages(messages)

        reconstructed = []
        for chunk in chunks:
            reconstructed.extend(chunk)

        for i, msg in enumerate(reconstructed):
            assert msg["content"] == f"msg-{i}"


class TestSeedClientDuplicateDetection:
    @pytest.fixture
    def client(self):
        config = MagicMock()
        driver = MagicMock()
        driver.conversation.conn.get_dialect.return_value = "postgresql"
        return SeedClient(
            config=config,
            driver=driver,
            entity_id="user-123",
        )

    @pytest.mark.asyncio
    async def test_duplicate_ids_raises_error(self, client):
        conversations = [
            {"id": "conv-1", "messages": [{"role": "user", "content": "Hi"}]},
            {"id": "conv-2", "messages": [{"role": "user", "content": "Hello"}]},
            {"id": "conv-1", "messages": [{"role": "user", "content": "Hey"}]},
        ]

        with pytest.raises(ValueError, match="Duplicate conversation IDs"):
            await client.seed(conversations)

    @pytest.mark.asyncio
    async def test_multiple_duplicates_reported(self, client):
        conversations = [
            {"id": "dup-1", "messages": [{"role": "user", "content": "1"}]},
            {"id": "dup-1", "messages": [{"role": "user", "content": "2"}]},
            {"id": "dup-2", "messages": [{"role": "user", "content": "3"}]},
            {"id": "dup-2", "messages": [{"role": "user", "content": "4"}]},
        ]

        with pytest.raises(ValueError) as exc_info:
            await client.seed(conversations)

        error_msg = str(exc_info.value)
        assert "dup-1" in error_msg or "dup-2" in error_msg


class TestSeedClientMissingId:
    @pytest.fixture
    def client(self):
        config = MagicMock()
        driver = MagicMock()
        driver.conversation.conn.get_dialect.return_value = "postgresql"
        return SeedClient(
            config=config,
            driver=driver,
            entity_id="user-123",
        )

    @pytest.mark.asyncio
    async def test_missing_id_returns_error(self, client):
        client.api.augmentation_async = AsyncMock()

        conversations = [
            {"messages": [{"role": "user", "content": "Hi"}]},
        ]

        result = await client.seed(conversations)

        assert result.failed == 1
        assert result.successful == 0
        assert "missing required 'id' field" in result.conversations[0].error


class TestProgressCallback:
    @pytest.fixture
    def client(self):
        config = MagicMock()
        driver = MagicMock()
        driver.conversation.conn.get_dialect.return_value = "postgresql"
        driver.entity.create.return_value = 1
        return SeedClient(
            config=config,
            driver=driver,
            entity_id="user-123",
        )

    @pytest.mark.asyncio
    async def test_progress_callback_called(self, client):
        progress_calls = []

        def on_progress(processed, total, result):
            progress_calls.append((processed, total, result.conversation_id))

        client.on_progress = on_progress
        client.api.augmentation_async = AsyncMock(
            return_value={
                "entity": {"triples": []},
                "conversation": {"summary": "test"},
                "process": {"attributes": []},
            }
        )

        conversations = [
            {"id": "conv-1", "messages": [{"role": "user", "content": "Hi"}]},
            {"id": "conv-2", "messages": [{"role": "user", "content": "Hello"}]},
        ]

        await client.seed(conversations)

        assert len(progress_calls) == 2
        assert progress_calls[0] == (1, 2, "conv-1")
        assert progress_calls[1] == (2, 2, "conv-2")


class TestModuleExports:
    def test_exports_from_seed_init(self):
        from memori.seed import (
            ConversationResult,
            SeedClient,
            SeedConfig,
            SeedData,
            SeedResult,
            SeedType,
            seed_conversations,
        )

        assert SeedType is not None
        assert SeedData is not None
        assert SeedResult is not None
        assert SeedConfig is not None
        assert ConversationResult is not None
        assert SeedClient is not None
        assert seed_conversations is not None

    def test_exports_from_memori_init(self):
        from memori import SeedConfig, SeedData, SeedResult, SeedType

        assert SeedType is not None
        assert SeedData is not None
        assert SeedResult is not None
        assert SeedConfig is not None


class TestSeedConversations:
    @pytest.fixture
    def mock_config(self):
        config = MagicMock()
        config.embeddings.model = "test-model"
        config.framework.provider = "test"
        config.llm.provider = "test"
        config.llm.provider_sdk_version = "1.0.0"
        config.llm.version = "1.0"
        config.platform.provider = "test"
        config.storage_config.cockroachdb = False
        config.version = "1.0.0"
        return config

    @pytest.fixture
    def mock_driver(self):
        driver = MagicMock()
        driver.conversation.conn.get_dialect.return_value = "postgresql"
        driver.entity.create.return_value = 1
        driver.process.create.return_value = 1
        return driver

    @pytest.mark.asyncio
    async def test_seed_conversations_success(self, mock_config, mock_driver, mocker):
        from memori.seed._ingest import seed_conversations

        mocker.patch(
            "memori.seed._client.Api.augmentation_async",
            new_callable=AsyncMock,
            return_value={
                "entity": {"triples": [], "facts": []},
                "conversation": {"summary": "test"},
                "process": {"attributes": []},
            },
        )

        conversations = [
            {"id": "conv-1", "messages": [{"role": "user", "content": "Hello"}]},
        ]

        result = await seed_conversations(
            config=mock_config,
            driver=mock_driver,
            entity_id="user-123",
            conversations=conversations,
        )

        assert result.total == 1
        assert result.successful == 1
        assert result.failed == 0


class TestSeedFromFile:
    @pytest.fixture
    def mock_config(self):
        config = MagicMock()
        config.embeddings.model = "test-model"
        config.framework.provider = "test"
        config.llm.provider = "test"
        config.llm.provider_sdk_version = "1.0.0"
        config.llm.version = "1.0"
        config.platform.provider = "test"
        config.storage_config.cockroachdb = False
        config.version = "1.0.0"
        return config

    @pytest.fixture
    def mock_driver(self):
        driver = MagicMock()
        driver.conversation.conn.get_dialect.return_value = "postgresql"
        driver.entity.create.return_value = 1
        return driver

    def test_seed_from_file_not_found(self, mock_config, mock_driver):
        from memori.seed._ingest import seed_from_file

        with pytest.raises(FileNotFoundError):
            seed_from_file(
                config=mock_config,
                driver=mock_driver,
                file_path="/nonexistent/file.json",
                entity_id="user-123",
            )

    def test_seed_from_file_missing_entity_id(self, mock_config, mock_driver, tmp_path):
        import json

        from memori.seed._ingest import seed_from_file

        file_path = tmp_path / "test.json"
        file_path.write_text(json.dumps({"conversations": []}))

        with pytest.raises(ValueError, match="entity_id must be provided"):
            seed_from_file(
                config=mock_config,
                driver=mock_driver,
                file_path=str(file_path),
            )

    def test_seed_from_file_no_conversations(self, mock_config, mock_driver, tmp_path):
        import json

        from memori.seed._ingest import seed_from_file

        file_path = tmp_path / "test.json"
        file_path.write_text(json.dumps({"entity_id": "user-123", "conversations": []}))

        with pytest.raises(ValueError, match="No conversations found"):
            seed_from_file(
                config=mock_config,
                driver=mock_driver,
                file_path=str(file_path),
            )

    def test_seed_from_file_uses_file_entity_id(
        self, mock_config, mock_driver, tmp_path, mocker
    ):
        import json

        from memori.seed._ingest import seed_from_file

        mocker.patch(
            "memori.seed._client.Api.augmentation_async",
            new_callable=AsyncMock,
            return_value={
                "entity": {"triples": [], "facts": []},
                "conversation": {"summary": "test"},
                "process": {"attributes": []},
            },
        )

        file_path = tmp_path / "test.json"
        file_path.write_text(
            json.dumps(
                {
                    "entity_id": "file-entity",
                    "conversations": [
                        {
                            "id": "conv-1",
                            "messages": [{"role": "user", "content": "Hi"}],
                        }
                    ],
                }
            )
        )

        result = seed_from_file(
            config=mock_config,
            driver=mock_driver,
            file_path=str(file_path),
        )

        assert result.total == 1


class TestSeedClientRetry:
    @pytest.fixture
    def client(self):
        config = MagicMock()
        config.embeddings.model = "test-model"
        driver = MagicMock()
        driver.conversation.conn.get_dialect.return_value = "postgresql"
        return SeedClient(
            config=config,
            driver=driver,
            entity_id="user-123",
            seed_config=SeedConfig(max_retries=3, retry_delay=0.01),
        )

    @pytest.mark.asyncio
    async def test_retry_on_transient_error(self, client, mocker):
        call_count = 0

        async def mock_augmentation(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise Exception("Transient error")
            return {
                "entity": {"triples": []},
                "conversation": {"summary": "test"},
                "process": {"attributes": []},
            }

        client.api.augmentation_async = mock_augmentation

        result = await client._call_aa_with_retry({"test": "payload"})

        assert call_count == 2
        assert result is not None

    @pytest.mark.asyncio
    async def test_no_retry_on_validation_error(self, client):
        client.api.augmentation_async = AsyncMock(
            side_effect=Exception("422 validation error")
        )

        with pytest.raises(Exception, match="422"):
            await client._call_aa_with_retry({"test": "payload"})

    @pytest.mark.asyncio
    async def test_no_retry_on_quota_error(self, client):
        client.api.augmentation_async = AsyncMock(
            side_effect=Exception("429 quota exceeded")
        )

        with pytest.raises(Exception, match="429"):
            await client._call_aa_with_retry({"test": "payload"})

    @pytest.mark.asyncio
    async def test_raises_after_max_retries(self, client):
        client.api.augmentation_async = AsyncMock(
            side_effect=Exception("Persistent error")
        )

        with pytest.raises(Exception, match="Persistent error"):
            await client._call_aa_with_retry({"test": "payload"})


class TestSeedClientProcessing:
    @pytest.fixture
    def client(self):
        config = MagicMock()
        config.embeddings.model = "test-model"
        driver = MagicMock()
        driver.conversation.conn.get_dialect.return_value = "postgresql"
        driver.entity.create.return_value = 1
        driver.process.create.return_value = 1
        return SeedClient(
            config=config,
            driver=driver,
            entity_id="user-123",
            process_id="process-1",
        )

    @pytest.mark.asyncio
    async def test_process_single_conversation_success(self, client, mocker):
        client.api.augmentation_async = AsyncMock(
            return_value={
                "entity": {"triples": [], "facts": ["fact1"]},
                "conversation": {"summary": "test summary"},
                "process": {"attributes": ["attr1"]},
            }
        )
        mocker.patch(
            "memori.seed._client.embed_texts",
            new_callable=AsyncMock,
            return_value=[[0.1, 0.2]],
        )

        conversation = {
            "id": "conv-1",
            "messages": [{"role": "user", "content": "Hello"}],
        }

        result = await client._process_single_conversation(conversation)

        assert result.success is True
        assert result.conversation_id == "conv-1"

    @pytest.mark.asyncio
    async def test_process_single_conversation_validation_error(self, client):
        conversation = {
            "id": "conv-1",
            "messages": [],
        }

        result = await client._process_single_conversation(conversation)

        assert result.success is False
        assert "Validation failed" in result.error

    @pytest.mark.asyncio
    async def test_process_single_conversation_empty_response(self, client):
        client.api.augmentation_async = AsyncMock(return_value=None)

        conversation = {
            "id": "conv-1",
            "messages": [{"role": "user", "content": "Hello"}],
        }

        result = await client._process_single_conversation(conversation)

        assert result.success is False
        assert "Empty response" in result.error

    @pytest.mark.asyncio
    async def test_process_single_conversation_exception(self, client):
        client.api.augmentation_async = AsyncMock(side_effect=Exception("API error"))

        conversation = {
            "id": "conv-1",
            "messages": [{"role": "user", "content": "Hello"}],
        }

        result = await client._process_single_conversation(conversation)

        assert result.success is False
        assert "API error" in result.error


class TestSeedClientStorage:
    @pytest.fixture
    def client(self):
        config = MagicMock()
        config.embeddings.model = "test-model"
        driver = MagicMock()
        driver.conversation.conn.get_dialect.return_value = "postgresql"
        driver.entity.create.return_value = 1
        driver.process.create.return_value = 1
        return SeedClient(
            config=config,
            driver=driver,
            entity_id="user-123",
            process_id="process-1",
        )

    @pytest.mark.asyncio
    async def test_store_memories_entity_creation_fails(self, client):
        from memori.memory._struct import Memories

        client.driver.entity.create.return_value = None

        memories = Memories()

        error = await client._store_memories("conv-1", memories)

        assert error is not None
        assert "Failed to create entity" in error

    @pytest.mark.asyncio
    async def test_store_memories_with_facts(self, client, mocker):
        from memori.memory._struct import Memories

        mocker.patch(
            "memori.seed._client.embed_texts",
            new_callable=AsyncMock,
            return_value=[[0.1, 0.2]],
        )

        memories = Memories()
        memories.entity.facts = ["fact1"]
        memories.entity.fact_embeddings = [[0.1, 0.2]]

        error = await client._store_memories("conv-1", memories)

        assert error is None
        client.driver.entity_fact.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_store_memories_with_process_attributes(self, client):
        from memori.memory._struct import Memories

        memories = Memories()
        memories.process.attributes = ["attr1", "attr2"]

        error = await client._store_memories("conv-1", memories)

        assert error is None
        client.driver.process_attribute.create.assert_called_once()


class TestSeedClientChunkingDisabled:
    @pytest.fixture
    def client(self):
        config = MagicMock()
        driver = MagicMock()
        driver.conversation.conn.get_dialect.return_value = "postgresql"
        return SeedClient(
            config=config,
            driver=driver,
            entity_id="user-123",
            seed_config=SeedConfig(
                max_messages_per_request=5,
                chunk_large_conversations=False,
            ),
        )

    @pytest.mark.asyncio
    async def test_large_conversation_fails_when_chunking_disabled(self, client):
        conversation = {
            "id": "conv-1",
            "messages": [{"role": "user", "content": "Hi"} for _ in range(10)],
        }

        result = await client._process_single_conversation(conversation)

        assert result.success is False
        assert "too large" in result.error
        assert "chunking disabled" in result.error


class TestBuildPayload:
    @pytest.fixture
    def client(self):
        config = MagicMock()
        config.framework.provider = "test-framework"
        config.llm.provider = "openai"
        config.llm.provider_sdk_version = "1.0.0"
        config.llm.version = "gpt-4"
        config.platform.provider = "test-platform"
        config.storage_config.cockroachdb = False
        config.version = "1.0.0"
        driver = MagicMock()
        driver.conversation.conn.get_dialect.return_value = "postgresql"
        return SeedClient(
            config=config,
            driver=driver,
            entity_id="user-123",
        )

    def test_build_payload_structure(self, client):
        messages = [{"role": "user", "content": "Hello"}]

        payload = client._build_payload(messages)

        assert "conversation" in payload
        assert "meta" in payload
        assert payload["conversation"]["messages"] == messages

    def test_build_payload_with_summary(self, client):
        messages = [{"role": "user", "content": "Hello"}]

        payload = client._build_payload(messages, summary="Previous context")

        assert payload["conversation"]["summary"] == "Previous context"


class TestSeedManager:
    @pytest.fixture
    def seed_manager(self):
        from memori._config import Config
        from memori.seed._cli import SeedManager

        config = Config()
        return SeedManager(config)

    def test_create_parser(self, seed_manager):
        parser = seed_manager._create_parser()

        assert parser is not None
        assert parser.prog == "python -m memori seed"

    def test_parser_has_required_arguments(self, seed_manager):
        parser = seed_manager._create_parser()

        args = parser.parse_args(["test.json"])
        assert args.file == "test.json"

    def test_parser_optional_arguments(self, seed_manager):
        parser = seed_manager._create_parser()

        args = parser.parse_args(
            [
                "test.json",
                "--batch-size",
                "20",
                "--dry-run",
            ]
        )

        assert args.batch_size == 20
        assert args.dry_run is True

    def test_print_preview(self, seed_manager, capsys):
        conversations = [
            {
                "id": "conv-1",
                "messages": [{"role": "user", "content": "Hello world"}],
            },
            {
                "id": "conv-2",
                "messages": [{"role": "assistant", "content": "Hi there"}],
            },
        ]

        seed_manager._print_preview(conversations)

        captured = capsys.readouterr()
        assert "conv-1" in captured.out
        assert "conv-2" in captured.out
