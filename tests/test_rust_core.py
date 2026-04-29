import base64
import json
import os
import shutil
import zipfile
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


def test_maybe_create_defers_engine_import(mocker):
    config = Config()
    config.byodb = True
    config.use_rust_core = True
    config.storage = SimpleNamespace(conn_factory=object)
    import_memori_python = mocker.patch("memori._rust_core._try_import_memori_python")

    adapter = _rust_core.RustCoreAdapter.maybe_create(config)

    assert adapter is not None
    assert adapter._engine is None
    import_memori_python.assert_not_called()


def test_retrieve_facts_initializes_engine_on_first_use(mocker):
    config = Config()
    config.storage = SimpleNamespace(conn_factory=object)
    engine = mocker.Mock()
    engine.retrieve.return_value = "[]"
    adapter = _rust_core.RustCoreAdapter(config=config)
    create_engine = mocker.patch.object(adapter, "_create_engine", return_value=engine)

    assert (
        adapter.retrieve_facts(
            query="hello",
            entity_id="entity-1",
            limit=5,
            dense_limit=10,
        )
        == []
    )

    create_engine.assert_called_once_with()
    engine.retrieve.assert_called_once()


def test_wait_for_augmentation_does_not_initialize_idle_engine(mocker):
    config = Config()
    adapter = _rust_core.RustCoreAdapter(config=config)
    create_engine = mocker.patch.object(adapter, "_create_engine")

    assert adapter.wait_for_augmentation(timeout=1.25) is True
    create_engine.assert_not_called()


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


def test_onnxruntime_asset_mapping_for_supported_platforms(mocker):
    mocker.patch("memori._rust_core.platform.system", return_value="Linux")
    mocker.patch("memori._rust_core.platform.machine", return_value="x86_64")
    assert _rust_core._onnxruntime_asset_for_current_platform() == (
        "onnxruntime-linux-x64-1.23.2.tgz",
        "1fa4dcaef22f6f7d5cd81b28c2800414350c10116f5fdd46a2160082551c5f9b",
    )

    mocker.patch("memori._rust_core.platform.system", return_value="Darwin")
    mocker.patch("memori._rust_core.platform.machine", return_value="arm64")
    assert _rust_core._onnxruntime_asset_for_current_platform() == (
        "onnxruntime-osx-arm64-1.23.2.tgz",
        "b4d513ab2b26f088c66891dbbc1408166708773d7cc4163de7bdca0e9bbb7856",
    )

    mocker.patch("memori._rust_core.sys.platform", "android")
    mocker.patch("memori._rust_core.platform.machine", return_value="aarch64")
    assert _rust_core._onnxruntime_asset_for_current_platform() == (
        "onnxruntime-android-1.23.2.aar",
        "82048d1f462218adae4ba76477089ab0ba76093d84f733540066db1a8ba6b827",
    )


def test_resolve_onnxruntime_lib_path_selects_android_abi(mocker, tmp_path):
    mocker.patch("memori._rust_core.sys.platform", "android")
    mocker.patch("memori._rust_core.platform.machine", return_value="aarch64")
    selected = tmp_path / "jni" / "arm64-v8a" / "libonnxruntime.so"
    other = tmp_path / "jni" / "x86_64" / "libonnxruntime.so"
    selected.parent.mkdir(parents=True)
    other.parent.mkdir(parents=True)
    selected.write_text("arm64")
    other.write_text("x64")

    assert _rust_core._resolve_onnxruntime_lib_path(tmp_path) == selected


def test_download_urls_for_android_asset_use_maven_central():
    assert _rust_core._download_urls_for_asset("onnxruntime-android-1.23.2.aar") == (
        "https://repo1.maven.org/maven2/com/microsoft/onnxruntime/"
        "onnxruntime-android/1.23.2/onnxruntime-android-1.23.2.aar",
        "https://repo1.maven.org/maven2/com/microsoft/onnxruntime/"
        "onnxruntime-android/1.23.2/onnxruntime-android-1.23.2.aar",
    )


def test_ensure_onnxruntime_dylib_uses_cached_binary(mocker, tmp_path):
    cache_root = tmp_path / ".cache" / "memori" / "onnxruntime" / "1.23.2"
    lib_path = cache_root / "onnxruntime-linux-x64-1.23.2" / "lib" / "libonnxruntime.so"
    lib_path.parent.mkdir(parents=True)
    lib_path.write_text("placeholder")

    mocker.patch("memori._rust_core.platform.system", return_value="Linux")
    mocker.patch("memori._rust_core.platform.machine", return_value="x86_64")
    mocker.patch("memori._rust_core.Path.home", return_value=tmp_path)
    mock_get = mocker.patch("memori._rust_core.requests.get")

    os.environ.pop("ORT_DYLIB_PATH", None)
    _rust_core._ensure_onnxruntime_dylib()

    assert os.environ["ORT_DYLIB_PATH"] == str(lib_path)
    mock_get.assert_not_called()


def test_ensure_onnxruntime_dylib_uses_versioned_cached_binary(mocker, tmp_path):
    cache_root = tmp_path / ".cache" / "memori" / "onnxruntime" / "1.23.2"
    lib_path = (
        cache_root
        / "onnxruntime-osx-arm64-1.23.2"
        / "lib"
        / "libonnxruntime.1.23.2.dylib"
    )
    lib_path.parent.mkdir(parents=True)
    lib_path.write_text("placeholder")

    mocker.patch("memori._rust_core.platform.system", return_value="Darwin")
    mocker.patch("memori._rust_core.platform.machine", return_value="arm64")
    mocker.patch("memori._rust_core.Path.home", return_value=tmp_path)
    mock_get = mocker.patch("memori._rust_core.requests.get")

    os.environ.pop("ORT_DYLIB_PATH", None)
    _rust_core._ensure_onnxruntime_dylib()

    assert os.environ["ORT_DYLIB_PATH"] == str(lib_path)
    mock_get.assert_not_called()


def test_ensure_onnxruntime_dylib_uses_cached_android_aar_binary(mocker, tmp_path):
    cache_root = tmp_path / ".cache" / "memori" / "onnxruntime" / "1.23.2"
    lib_path = (
        cache_root
        / "onnxruntime-android-1.23.2"
        / "jni"
        / "arm64-v8a"
        / "libonnxruntime.so"
    )
    lib_path.parent.mkdir(parents=True)
    lib_path.write_text("placeholder")

    mocker.patch("memori._rust_core.sys.platform", "android")
    mocker.patch("memori._rust_core.platform.machine", return_value="aarch64")
    mocker.patch("memori._rust_core.Path.home", return_value=tmp_path)
    mock_get = mocker.patch("memori._rust_core.requests.get")

    os.environ.pop("ORT_DYLIB_PATH", None)
    _rust_core._ensure_onnxruntime_dylib()

    assert os.environ["ORT_DYLIB_PATH"] == str(lib_path)
    mock_get.assert_not_called()


def test_ensure_onnxruntime_dylib_extracts_android_aar_without_root(mocker, tmp_path):
    archive_path = tmp_path / "onnxruntime-android-test.aar"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("jni/arm64-v8a/libonnxruntime.so", "placeholder")
    expected_sha = _rust_core._compute_sha256(archive_path)

    def _copy_archive(_asset_name, destination):
        shutil.copyfile(archive_path, destination)
        return True

    mocker.patch.dict(
        "memori._rust_core._ORT_ASSET_BY_PLATFORM",
        {("android", "aarch64"): ("onnxruntime-android-test.aar", expected_sha)},
    )
    mocker.patch("memori._rust_core.sys.platform", "android")
    mocker.patch("memori._rust_core.platform.machine", return_value="aarch64")
    mocker.patch("memori._rust_core.Path.home", return_value=tmp_path)
    mocker.patch(
        "memori._rust_core._download_asset_with_retries", side_effect=_copy_archive
    )

    os.environ.pop("ORT_DYLIB_PATH", None)
    _rust_core._ensure_onnxruntime_dylib()

    expected_lib_path = (
        tmp_path
        / ".cache"
        / "memori"
        / "onnxruntime"
        / "1.23.2"
        / "onnxruntime-android-test"
        / "jni"
        / "arm64-v8a"
        / "libonnxruntime.so"
    )
    assert os.environ["ORT_DYLIB_PATH"] == str(expected_lib_path)


def test_compute_sha256_produces_expected_digest(tmp_path):
    target = tmp_path / "payload.bin"
    target.write_bytes(b"memori")
    assert (
        _rust_core._compute_sha256(target)
        == "e2092aab4fc7f734b716bd2eaccd02e6c8a83a7aeb4955acab115716847bb7f1"
    )
