from unittest.mock import MagicMock

import pytest

from memori.memory.augmentation._models import (
    AttributionData,
    AugmentationPayload,
    ConversationData,
    EntityData,
    FrameworkData,
    LlmData,
    MetaData,
    ModelData,
    PlatformData,
    ProcessData,
    SdkData,
    SdkVersionData,
    StorageData,
    hash_id,
)


class TestHashId:
    def test_hash_id_returns_64_chars(self):
        result = hash_id("test-user-123")
        assert result is not None
        assert len(result) == 64
        assert all(c in "0123456789abcdef" for c in result)

    def test_hash_id_returns_none_for_none(self):
        assert hash_id(None) is None

    def test_hash_id_returns_none_for_empty_string(self):
        assert hash_id("") is None

    def test_hash_id_is_deterministic(self):
        hash1 = hash_id("consistent-user")
        hash2 = hash_id("consistent-user")
        assert hash1 == hash2

    def test_hash_id_different_inputs_different_hashes(self):
        hash1 = hash_id("user-1")
        hash2 = hash_id("user-2")
        assert hash1 != hash2


class TestDataclassModels:
    def test_conversation_data_structure(self):
        messages = [{"role": "user", "content": "Hello"}]
        conv = ConversationData(messages=messages, summary="A greeting")

        assert conv.messages == messages
        assert conv.summary == "A greeting"

    def test_conversation_data_summary_optional(self):
        conv = ConversationData(messages=[])
        assert conv.summary is None

    def test_entity_data_structure(self):
        entity = EntityData(id=hash_id("user-123"))
        assert entity.id is not None
        assert len(entity.id) == 64

    def test_attribution_data_structure(self):
        attr = AttributionData(
            entity=EntityData(id=hash_id("user")),
            process=ProcessData(id=hash_id("process")),
        )
        assert attr.entity.id is not None
        assert attr.process.id is not None

    def test_meta_data_has_all_required_fields(self):
        meta = MetaData()

        assert hasattr(meta, "attribution")
        assert hasattr(meta, "framework")
        assert hasattr(meta, "llm")
        assert hasattr(meta, "platform")
        assert hasattr(meta, "sdk")
        assert hasattr(meta, "storage")

    def test_sdk_data_defaults_to_python(self):
        sdk = SdkData()
        assert sdk.lang == "python"

    def test_storage_data_defaults(self):
        storage = StorageData()
        assert storage.cockroachdb is False
        assert storage.dialect is None


class TestAugmentationPayloadToDict:
    def test_payload_has_required_top_level_keys(self):
        payload = AugmentationPayload(
            conversation=ConversationData(messages=[]),
            meta=MetaData(),
        )
        result = payload.to_dict()

        assert "conversation" in result
        assert "meta" in result

    def test_payload_conversation_structure(self):
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        payload = AugmentationPayload(
            conversation=ConversationData(messages=messages, summary="A conversation"),
            meta=MetaData(),
        )
        result = payload.to_dict()

        assert "messages" in result["conversation"]
        assert isinstance(result["conversation"]["messages"], list)
        assert len(result["conversation"]["messages"]) == 3
        assert result["conversation"]["summary"] == "A conversation"

    def test_payload_meta_has_all_required_keys(self):
        payload = AugmentationPayload(
            conversation=ConversationData(messages=[]),
            meta=MetaData(
                attribution=AttributionData(
                    entity=EntityData(id=hash_id("user")),
                    process=ProcessData(id=hash_id("proc")),
                ),
                framework=FrameworkData(provider="openai"),
                llm=LlmData(
                    model=ModelData(
                        provider="openai",
                        sdk=SdkVersionData(version="1.0.0"),
                        version="gpt-4",
                    )
                ),
                platform=PlatformData(provider="python"),
                sdk=SdkData(lang="python", version="0.1.0"),
                storage=StorageData(cockroachdb=False, dialect="sqlite"),
            ),
        )
        result = payload.to_dict()

        meta = result["meta"]
        assert "attribution" in meta
        assert "framework" in meta
        assert "llm" in meta
        assert "platform" in meta
        assert "sdk" in meta
        assert "storage" in meta

    def test_payload_attribution_structure(self):
        payload = AugmentationPayload(
            conversation=ConversationData(messages=[]),
            meta=MetaData(
                attribution=AttributionData(
                    entity=EntityData(id=hash_id("entity-123")),
                    process=ProcessData(id=hash_id("process-456")),
                )
            ),
        )
        result = payload.to_dict()

        attr = result["meta"]["attribution"]
        assert "entity" in attr
        assert "id" in attr["entity"]
        assert len(attr["entity"]["id"]) == 64

        assert "process" in attr
        assert "id" in attr["process"]
        assert len(attr["process"]["id"]) == 64

    def test_payload_llm_structure(self):
        payload = AugmentationPayload(
            conversation=ConversationData(messages=[]),
            meta=MetaData(
                llm=LlmData(
                    model=ModelData(
                        provider="anthropic",
                        sdk=SdkVersionData(version="0.30.0"),
                        version="claude-3-opus",
                    )
                )
            ),
        )
        result = payload.to_dict()

        llm = result["meta"]["llm"]
        assert "model" in llm
        assert llm["model"]["provider"] == "anthropic"
        assert llm["model"]["version"] == "claude-3-opus"
        assert llm["model"]["sdk"]["version"] == "0.30.0"

    def test_payload_sdk_structure(self):
        payload = AugmentationPayload(
            conversation=ConversationData(messages=[]),
            meta=MetaData(sdk=SdkData(lang="python", version="1.2.3")),
        )
        result = payload.to_dict()

        sdk = result["meta"]["sdk"]
        assert sdk["lang"] == "python"
        assert sdk["version"] == "1.2.3"

    def test_payload_storage_structure(self):
        payload = AugmentationPayload(
            conversation=ConversationData(messages=[]),
            meta=MetaData(storage=StorageData(cockroachdb=True, dialect="postgresql")),
        )
        result = payload.to_dict()

        storage = result["meta"]["storage"]
        assert storage["cockroachdb"] is True
        assert storage["dialect"] == "postgresql"


class TestBuildApiPayload:
    @pytest.fixture
    def mock_config(self):
        config = MagicMock()
        config.framework.provider = "openai"
        config.llm.provider = "openai"
        config.llm.provider_sdk_version = "1.50.0"
        config.llm.version = "gpt-4o-mini"
        config.platform.provider = "python"
        config.version = "0.1.0"
        config.storage_config.cockroachdb = False
        return config

    @pytest.fixture
    def augmentation(self, mock_config):
        from memori.memory.augmentation.augmentations.memori._augmentation import (
            AdvancedAugmentation,
        )

        aug = AdvancedAugmentation(config=mock_config)
        return aug

    def test_build_payload_returns_dict(self, augmentation):
        payload = augmentation._build_api_payload(
            messages=[{"role": "user", "content": "test"}],
            summary=None,
            system_prompt=None,
            dialect="sqlite",
            entity_id="user-123",
            process_id="proc-456",
        )

        assert isinstance(payload, dict)

    def test_build_payload_has_required_keys(self, augmentation):
        payload = augmentation._build_api_payload(
            messages=[{"role": "user", "content": "test"}],
            summary=None,
            system_prompt=None,
            dialect="sqlite",
            entity_id="user-123",
            process_id="proc-456",
        )

        assert "conversation" in payload
        assert "meta" in payload

    def test_build_payload_hashes_entity_id(self, augmentation):
        payload = augmentation._build_api_payload(
            messages=[],
            summary=None,
            system_prompt=None,
            dialect="sqlite",
            entity_id="my-user-id",
            process_id="my-process-id",
        )

        entity_id = payload["meta"]["attribution"]["entity"]["id"]
        assert entity_id is not None
        assert len(entity_id) == 64
        assert entity_id != "my-user-id"

    def test_build_payload_hashes_process_id(self, augmentation):
        payload = augmentation._build_api_payload(
            messages=[],
            summary=None,
            system_prompt=None,
            dialect="sqlite",
            entity_id="user",
            process_id="my-process-id",
        )

        process_id = payload["meta"]["attribution"]["process"]["id"]
        assert process_id is not None
        assert len(process_id) == 64
        assert process_id != "my-process-id"

    def test_build_payload_includes_messages(self, augmentation):
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
        ]

        payload = augmentation._build_api_payload(
            messages=messages,
            summary=None,
            system_prompt=None,
            dialect="sqlite",
            entity_id="user",
            process_id="proc",
        )

        assert payload["conversation"]["messages"] == messages

    def test_build_payload_includes_summary(self, augmentation):
        payload = augmentation._build_api_payload(
            messages=[{"role": "user", "content": "test"}],
            summary="This is a test conversation",
            system_prompt=None,
            dialect="sqlite",
            entity_id="user",
            process_id="proc",
        )

        assert payload["conversation"]["summary"] == "This is a test conversation"

    def test_build_payload_includes_dialect(self, augmentation):
        payload = augmentation._build_api_payload(
            messages=[],
            summary=None,
            system_prompt=None,
            dialect="postgresql",
            entity_id="user",
            process_id="proc",
        )

        assert payload["meta"]["storage"]["dialect"] == "postgresql"

    def test_build_payload_includes_llm_provider(self, augmentation):
        payload = augmentation._build_api_payload(
            messages=[],
            summary=None,
            system_prompt=None,
            dialect="sqlite",
            entity_id="user",
            process_id="proc",
        )

        assert payload["meta"]["llm"]["model"]["provider"] == "openai"
        assert payload["meta"]["llm"]["model"]["version"] == "gpt-4o-mini"

    def test_build_payload_includes_sdk_info(self, augmentation):
        payload = augmentation._build_api_payload(
            messages=[],
            summary=None,
            system_prompt=None,
            dialect="sqlite",
            entity_id="user",
            process_id="proc",
        )

        assert payload["meta"]["sdk"]["lang"] == "python"
        assert payload["meta"]["sdk"]["version"] == "0.1.0"

    def test_build_payload_includes_framework_provider(self, augmentation):
        payload = augmentation._build_api_payload(
            messages=[],
            summary=None,
            system_prompt=None,
            dialect="sqlite",
            entity_id="user",
            process_id="proc",
        )

        assert payload["meta"]["framework"]["provider"] == "openai"

    def test_build_payload_none_entity_id(self, augmentation):
        payload = augmentation._build_api_payload(
            messages=[],
            summary=None,
            system_prompt=None,
            dialect="sqlite",
            entity_id=None,
            process_id="proc",
        )

        assert payload["meta"]["attribution"]["entity"]["id"] is None

    def test_build_payload_none_process_id(self, augmentation):
        payload = augmentation._build_api_payload(
            messages=[],
            summary=None,
            system_prompt=None,
            dialect="sqlite",
            entity_id="user",
            process_id=None,
        )

        assert payload["meta"]["attribution"]["process"]["id"] is None


class TestPayloadValidator:
    def validate_payload_structure(self, payload: dict) -> list[str]:
        errors = []

        if not payload:
            return ["No payload to validate"]

        if "conversation" not in payload:
            errors.append("Missing 'conversation' key")
        if "meta" not in payload:
            errors.append("Missing 'meta' key")

        if "conversation" in payload:
            conv = payload["conversation"]
            if "messages" not in conv:
                errors.append("Missing 'conversation.messages'")
            elif not isinstance(conv["messages"], list):
                errors.append("'conversation.messages' must be a list")

        if "meta" in payload:
            meta = payload["meta"]

            required_meta = [
                "attribution",
                "framework",
                "llm",
                "platform",
                "sdk",
                "storage",
            ]
            for key in required_meta:
                if key not in meta:
                    errors.append(f"Missing 'meta.{key}'")

            if "attribution" in meta:
                attr = meta["attribution"]
                if "entity" not in attr or "id" not in attr.get("entity", {}):
                    errors.append("Missing 'meta.attribution.entity.id'")
                if "process" not in attr or "id" not in attr.get("process", {}):
                    errors.append("Missing 'meta.attribution.process.id'")

                entity_id = attr.get("entity", {}).get("id")
                if entity_id is not None and len(entity_id) != 64:
                    errors.append(
                        f"Entity ID not hashed: {len(entity_id)} chars, expected 64"
                    )

                process_id = attr.get("process", {}).get("id")
                if process_id is not None and len(process_id) != 64:
                    errors.append(
                        f"Process ID not hashed: {len(process_id)} chars, expected 64"
                    )

            if "llm" in meta:
                llm = meta["llm"]
                if "model" not in llm:
                    errors.append("Missing 'meta.llm.model'")
                elif "provider" not in llm.get("model", {}):
                    errors.append("Missing 'meta.llm.model.provider'")

            if "sdk" in meta:
                sdk = meta["sdk"]
                if sdk.get("lang") != "python":
                    lang = sdk.get("lang")
                    errors.append(f"Expected sdk.lang='python', got '{lang}'")

            if "storage" in meta:
                storage = meta["storage"]
                if "dialect" not in storage:
                    errors.append("Missing 'meta.storage.dialect'")
                if "cockroachdb" not in storage:
                    errors.append("Missing 'meta.storage.cockroachdb'")

        return errors

    def test_valid_payload_passes_validation(self):
        payload = AugmentationPayload(
            conversation=ConversationData(
                messages=[{"role": "user", "content": "Hello"}],
                summary=None,
            ),
            meta=MetaData(
                attribution=AttributionData(
                    entity=EntityData(id=hash_id("user-123")),
                    process=ProcessData(id=hash_id("proc-456")),
                ),
                framework=FrameworkData(provider="openai"),
                llm=LlmData(
                    model=ModelData(
                        provider="openai",
                        sdk=SdkVersionData(version="1.0.0"),
                        version="gpt-4",
                    )
                ),
                platform=PlatformData(provider="python"),
                sdk=SdkData(lang="python", version="0.1.0"),
                storage=StorageData(cockroachdb=False, dialect="sqlite"),
            ),
        )

        errors = self.validate_payload_structure(payload.to_dict())
        assert len(errors) == 0, f"Validation errors: {errors}"

    def test_missing_conversation_fails_validation(self):
        payload = {"meta": {}}
        errors = self.validate_payload_structure(payload)
        assert "Missing 'conversation' key" in errors

    def test_missing_meta_fails_validation(self):
        payload = {"conversation": {"messages": []}}
        errors = self.validate_payload_structure(payload)
        assert "Missing 'meta' key" in errors

    def test_unhashed_entity_id_fails_validation(self):
        payload = {
            "conversation": {"messages": []},
            "meta": {
                "attribution": {
                    "entity": {"id": "raw-user-id"},
                    "process": {"id": hash_id("proc")},
                },
                "framework": {"provider": "openai"},
                "llm": {"model": {"provider": "openai"}},
                "platform": {"provider": "python"},
                "sdk": {"lang": "python", "version": "0.1.0"},
                "storage": {"cockroachdb": False, "dialect": "sqlite"},
            },
        }

        errors = self.validate_payload_structure(payload)
        assert any("Entity ID not hashed" in e for e in errors)


class TestProviderSpecificPayloads:
    @pytest.fixture
    def make_augmentation(self):
        def _make(provider: str, sdk_version: str = "1.0.0", model: str = "test-model"):
            from memori.memory.augmentation.augmentations.memori._augmentation import (
                AdvancedAugmentation,
            )

            config = MagicMock()
            config.framework.provider = provider
            config.llm.provider = provider
            config.llm.provider_sdk_version = sdk_version
            config.llm.version = model
            config.platform.provider = "python"
            config.version = "0.1.0"
            config.storage_config.cockroachdb = False

            return AdvancedAugmentation(config=config)

        return _make

    def test_openai_payload(self, make_augmentation):
        aug = make_augmentation("openai", "1.50.0", "gpt-4o-mini")
        payload = aug._build_api_payload(
            messages=[{"role": "user", "content": "test"}],
            summary=None,
            system_prompt=None,
            dialect="sqlite",
            entity_id="user",
            process_id="proc",
        )

        assert payload["meta"]["framework"]["provider"] == "openai"
        assert payload["meta"]["llm"]["model"]["provider"] == "openai"
        assert payload["meta"]["llm"]["model"]["version"] == "gpt-4o-mini"

    def test_anthropic_payload(self, make_augmentation):
        aug = make_augmentation("anthropic", "0.30.0", "claude-3-opus-20240229")
        payload = aug._build_api_payload(
            messages=[{"role": "user", "content": "test"}],
            summary=None,
            system_prompt=None,
            dialect="postgresql",
            entity_id="user",
            process_id="proc",
        )

        assert payload["meta"]["framework"]["provider"] == "anthropic"
        assert payload["meta"]["llm"]["model"]["provider"] == "anthropic"
        assert payload["meta"]["llm"]["model"]["version"] == "claude-3-opus-20240229"

    def test_google_payload(self, make_augmentation):
        aug = make_augmentation("google", "1.0.0", "gemini-1.5-flash")
        payload = aug._build_api_payload(
            messages=[{"role": "user", "content": "test"}],
            summary=None,
            system_prompt=None,
            dialect="mysql",
            entity_id="user",
            process_id="proc",
        )

        assert payload["meta"]["framework"]["provider"] == "google"
        assert payload["meta"]["llm"]["model"]["provider"] == "google"

    def test_bedrock_payload(self, make_augmentation):
        aug = make_augmentation(
            "bedrock", "0.2.0", "anthropic.claude-3-sonnet-20240229-v1:0"
        )
        payload = aug._build_api_payload(
            messages=[{"role": "user", "content": "test"}],
            summary=None,
            system_prompt=None,
            dialect="sqlite",
            entity_id="user",
            process_id="proc",
        )

        assert payload["meta"]["framework"]["provider"] == "bedrock"
        assert payload["meta"]["llm"]["model"]["provider"] == "bedrock"

    def test_xai_payload(self, make_augmentation):
        aug = make_augmentation("xai", "1.0.0", "grok-beta")
        payload = aug._build_api_payload(
            messages=[{"role": "user", "content": "test"}],
            summary=None,
            system_prompt=None,
            dialect="sqlite",
            entity_id="user",
            process_id="proc",
        )

        assert payload["meta"]["framework"]["provider"] == "xai"
        assert payload["meta"]["llm"]["model"]["provider"] == "xai"
