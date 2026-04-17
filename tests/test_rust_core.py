import base64
import json
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from memori import _rust_core
from memori._config import Config


@contextmanager
def _fake_connection_context(_conn_factory, driver):
    yield None, None, driver


def test_fetch_embeddings_callback_serializes_binary_embeddings(mocker):
    config = Config()
    config.storage = SimpleNamespace(conn_factory=object)
    driver = SimpleNamespace(
        entity=SimpleNamespace(create=mocker.Mock(return_value=42)),
        entity_fact=SimpleNamespace(
            get_embeddings=mocker.Mock(
                return_value=[
                    {
                        "id": 1,
                        "content_embedding": b"\x00\x00\x80?\x00\x00\x00@",
                    }
                ]
            )
        ),
    )

    mocker.patch(
        "memori._rust_core.connection_context",
        side_effect=lambda conn_factory: _fake_connection_context(conn_factory, driver),
    )

    callback = _rust_core.RustCoreAdapter._fetch_embeddings_cb(config)
    output = json.loads(callback(json.dumps({"entity_id": "entity-abc", "limit": 10})))

    assert len(output) == 1
    assert output[0]["id"] == 1
    assert (
        base64.b64decode(output[0]["content_embedding_b64"])
        == b"\x00\x00\x80?\x00\x00\x00@"
    )
    driver.entity.create.assert_called_once_with("entity-abc")
    driver.entity_fact.get_embeddings.assert_called_once_with(42, 10)


def test_write_batch_callback_maps_process_attribute_dict(mocker):
    config = Config()
    config.storage = SimpleNamespace(conn_factory=object)
    driver = SimpleNamespace(
        process=SimpleNamespace(create=mocker.Mock(return_value=7)),
        process_attribute=SimpleNamespace(create=mocker.Mock()),
    )

    mocker.patch(
        "memori._rust_core.connection_context",
        side_effect=lambda conn_factory: _fake_connection_context(conn_factory, driver),
    )

    callback = _rust_core.RustCoreAdapter._write_batch_cb(config)
    response = json.loads(
        callback(
            json.dumps(
                {
                    "ops": [
                        {
                            "op_type": "process_attribute.create",
                            "payload": {
                                "process_id": "proc-1",
                                "attributes": {"tone": "friendly", "lang": "en"},
                            },
                        }
                    ]
                }
            )
        )
    )

    assert response["written_ops"] == 1
    driver.process.create.assert_called_once_with("proc-1")
    driver.process_attribute.create.assert_called_once_with(
        7,
        ["tone:friendly", "lang:en"],
    )


def test_write_batch_callback_embeds_entity_facts(mocker):
    config = Config()
    config.storage = SimpleNamespace(conn_factory=object)
    config.embeddings = SimpleNamespace(model="all-MiniLM-L6-v2")
    driver = SimpleNamespace(
        entity=SimpleNamespace(create=mocker.Mock(return_value=42)),
        entity_fact=SimpleNamespace(create=mocker.Mock()),
    )

    mocker.patch(
        "memori._rust_core.connection_context",
        side_effect=lambda conn_factory: _fake_connection_context(conn_factory, driver),
    )
    embed = mocker.patch("memori._rust_core.embed_texts", return_value=[[0.1, 0.2]])

    callback = _rust_core.RustCoreAdapter._write_batch_cb(config)
    response = json.loads(
        callback(
            json.dumps(
                {
                    "ops": [
                        {
                            "op_type": "entity_fact.create",
                            "payload": {
                                "entity_id": "entity-1",
                                "facts": ["The user's favorite color is blue."],
                                "conversation_id": "5",
                            },
                        }
                    ]
                }
            )
        )
    )

    assert response["written_ops"] == 1
    embed.assert_called_once_with(
        ["The user's favorite color is blue."], model="all-MiniLM-L6-v2"
    )
    driver.entity_fact.create.assert_called_once_with(
        42,
        ["The user's favorite color is blue."],
        fact_embeddings=[[0.1, 0.2]],
        conversation_id=5,
    )


def test_write_batch_callback_rejects_malformed_json():
    callback = _rust_core.RustCoreAdapter._write_batch_cb(
        SimpleNamespace(storage=SimpleNamespace(conn_factory=object))
    )
    with pytest.raises(_rust_core.RustCoreAdapterError, match="Invalid JSON"):
        callback("{not-json")


def test_normalize_model_name_default_alias():
    assert _rust_core._normalize_model_name("all-MiniLM-L6-v2") is None
    assert _rust_core._normalize_model_name("AllMiniLML6V2") is None
    assert (
        _rust_core._normalize_model_name("BAAI/bge-small-en-v1.5")
        == "BAAI/bge-small-en-v1.5"
    )


def test_submit_augmentation_sends_live_request_payload(mocker):
    config = Config()
    config.framework.provider = "langchain"
    config.llm.provider = "openai"
    config.llm.provider_sdk_version = "1.2.3"
    config.llm.version = "gpt-4o-mini"
    config.platform.provider = "local"
    config.storage_config.dialect = "sqlite"
    config.storage_config.cockroachdb = False
    config.version = "3.2.8"
    engine = mocker.Mock()
    engine.submit_augmentation.return_value = "12"
    adapter = _rust_core.RustCoreAdapter(config=config, _engine=engine)

    job_id = adapter.submit_augmentation(
        entity_id="entity-1",
        process_id="process-1",
        conversation_id="1",
        conversation_messages=[{"role": "user", "content": "hello"}],
        llm_provider="openai",
        llm_model="gpt-4o-mini",
        llm_provider_sdk_version="1.2.3",
        framework="langchain",
        platform_provider="local",
        storage_dialect="sqlite",
        storage_cockroachdb=False,
        sdk_version="3.2.8",
    )

    assert job_id == 12
    submitted = json.loads(engine.submit_augmentation.call_args.args[0])
    assert "use_mock_response" not in submitted
    assert "mock_response" not in submitted
    assert submitted["llm_provider_sdk_version"] == "1.2.3"
    assert submitted["platform_provider"] == "local"
    assert submitted["storage_dialect"] == "sqlite"
    assert submitted["storage_cockroachdb"] is False


def test_submit_augmentation_resolves_storage_dialect_from_adapter(mocker):
    config = Config()
    config.framework.provider = "langchain"
    config.llm.provider = "openai"
    config.llm.provider_sdk_version = "1.2.3"
    config.llm.version = "gpt-4o-mini"
    config.platform.provider = "local"
    config.storage_config.dialect = None
    config.storage = SimpleNamespace(
        adapter=SimpleNamespace(get_dialect=lambda: "sqlite")
    )
    engine = mocker.Mock()
    engine.submit_augmentation.return_value = "1"
    adapter = _rust_core.RustCoreAdapter(config=config, _engine=engine)

    adapter.submit_augmentation(
        entity_id="entity-1",
        process_id="process-1",
        conversation_id="1",
        conversation_messages=[{"role": "user", "content": "hello"}],
        llm_provider="openai",
        llm_model="gpt-4o-mini",
        llm_provider_sdk_version="1.2.3",
        framework="langchain",
        platform_provider="local",
        storage_dialect=None,
        storage_cockroachdb=False,
        sdk_version="3.2.8",
    )

    submitted = json.loads(engine.submit_augmentation.call_args.args[0])
    assert submitted["storage_dialect"] == "sqlite"


def test_wait_for_augmentation_forwards_timeout_ms(mocker):
    config = Config()
    engine = mocker.Mock()
    engine.wait_for_augmentation.return_value = True
    adapter = _rust_core.RustCoreAdapter(config=config, _engine=engine)

    result = adapter.wait_for_augmentation(timeout=1.25)

    assert result is True
    engine.wait_for_augmentation.assert_called_once_with(1250)
