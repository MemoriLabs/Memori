import pytest

from memori import Memori
from memori._exceptions import MissingMemoriApiKeyError


def test_cloud_false_when_conn_provided(mocker):
    mock_conn = mocker.Mock(spec=["cursor", "commit", "rollback"])
    mock_conn.__module__ = "psycopg"
    type(mock_conn).__module__ = "psycopg"
    mock_cursor = mocker.MagicMock()
    mock_conn.cursor = mocker.MagicMock(return_value=mock_cursor)

    mem = Memori(conn=lambda: mock_conn)
    assert mem.config.cloud is False


def test_memori_initializes_rust_core_adapter_in_byodb(mocker, monkeypatch):
    monkeypatch.setenv("MEMORI_USE_RUST_CORE", "true")
    mock_conn = mocker.Mock(spec=["cursor", "commit", "rollback"])
    mock_conn.__module__ = "psycopg"
    type(mock_conn).__module__ = "psycopg"
    mock_cursor = mocker.MagicMock()
    mock_conn.cursor = mocker.MagicMock(return_value=mock_cursor)

    maybe_create = mocker.patch(
        "memori._rust_core.RustCoreAdapter.maybe_create", return_value=object()
    )

    mem = Memori(conn=lambda: mock_conn)

    maybe_create.assert_called_once_with(mem.config)
    assert mem.config.byodb is True
    assert mem.config.cloud is False
    assert mem.config.rust_core is not None


def test_cloud_true_when_no_conn(monkeypatch):
    monkeypatch.delenv("MEMORI_COCKROACHDB_CONNECTION_STRING", raising=False)
    monkeypatch.setenv("MEMORI_API_KEY", "test-api-key")
    monkeypatch.setenv("MEMORI_TEST_MODE", "1")
    mem = Memori()
    assert mem.config.cloud is True


def test_cloud_raises_error_when_no_api_key(monkeypatch):
    monkeypatch.delenv("MEMORI_COCKROACHDB_CONNECTION_STRING", raising=False)
    monkeypatch.delenv("MEMORI_API_KEY", raising=False)
    with pytest.raises(MissingMemoriApiKeyError) as e:
        Memori()
    assert e.value.env_var == "MEMORI_API_KEY"
    assert "MEMORI_API_KEY" in str(e.value)


def test_cloud_false_when_connection_string_set(monkeypatch, mocker):
    monkeypatch.setenv(
        "MEMORI_COCKROACHDB_CONNECTION_STRING",
        "postgresql://user:pass@localhost:26257/defaultdb?sslmode=disable",
    )

    mock_conn = mocker.Mock(spec=["cursor", "commit", "rollback"])
    mock_conn.__module__ = "psycopg"
    type(mock_conn).__module__ = "psycopg"
    mock_cursor = mocker.MagicMock()
    mock_conn.cursor = mocker.MagicMock(return_value=mock_cursor)

    mocker.patch("psycopg.connect", return_value=mock_conn)

    mem = Memori(conn=None)
    assert mem.config.cloud is False


def test_attribution_exceptions(mocker):
    mock_conn = mocker.Mock(spec=["cursor", "commit", "rollback"])
    mock_conn.__module__ = "psycopg"
    type(mock_conn).__module__ = "psycopg"
    mock_cursor = mocker.MagicMock()
    mock_conn.cursor = mocker.MagicMock(return_value=mock_cursor)

    with pytest.raises(RuntimeError) as e:
        Memori(conn=lambda: mock_conn).attribution(entity_id="a" * 101)

    assert str(e.value) == "entity_id cannot be greater than 100 characters"

    with pytest.raises(RuntimeError) as e:
        Memori(conn=lambda: mock_conn).attribution(
            entity_id="entity", process_id="a" * 101
        )

    assert str(e.value) == "process_id cannot be greater than 100 characters"


def test_attribution_requires_string_entity_id(mocker):
    mock_conn = mocker.Mock(spec=["cursor", "commit", "rollback"])
    mock_conn.__module__ = "psycopg"
    type(mock_conn).__module__ = "psycopg"
    mock_cursor = mocker.MagicMock()
    mock_conn.cursor = mocker.MagicMock(return_value=mock_cursor)

    with pytest.raises(TypeError) as e:
        Memori(conn=lambda: mock_conn).attribution(entity_id=123)

    assert str(e.value) == "entity_id must be a string"


def test_attribution_requires_string_or_none_process_id(mocker):
    mock_conn = mocker.Mock(spec=["cursor", "commit", "rollback"])
    mock_conn.__module__ = "psycopg"
    type(mock_conn).__module__ = "psycopg"
    mock_cursor = mocker.MagicMock()
    mock_conn.cursor = mocker.MagicMock(return_value=mock_cursor)

    with pytest.raises(TypeError) as e:
        Memori(conn=lambda: mock_conn).attribution(entity_id="user-1", process_id=123)

    assert str(e.value) == "process_id must be a string or None"

    mem = Memori(conn=lambda: mock_conn).attribution(
        entity_id="user-1", process_id=None
    )
    assert mem.config.process_id is None


def test_new_session(mocker):
    mock_conn = mocker.Mock(spec=["cursor", "commit", "rollback"])
    mock_conn.__module__ = "psycopg"
    type(mock_conn).__module__ = "psycopg"
    mock_cursor = mocker.MagicMock()
    mock_conn.cursor = mocker.MagicMock(return_value=mock_cursor)

    mem = Memori(conn=lambda: mock_conn)

    session_id = mem.config.session_id
    assert session_id is not None

    mem.new_session()

    assert mem.config.session_id is not None
    assert mem.config.session_id != session_id


def test_set_session(mocker):
    mock_conn = mocker.Mock(spec=["cursor", "commit", "rollback"])
    mock_conn.__module__ = "psycopg"
    type(mock_conn).__module__ = "psycopg"
    mock_cursor = mocker.MagicMock()
    mock_conn.cursor = mocker.MagicMock(return_value=mock_cursor)

    mem = Memori(conn=lambda: mock_conn).set_session(
        "66cf2a0b-7503-4dcd-b717-b29c826fa1db"
    )
    assert mem.config.session_id == "66cf2a0b-7503-4dcd-b717-b29c826fa1db"


def test_set_session_resets_cache(mocker):
    mock_conn = mocker.Mock(spec=["cursor", "commit", "rollback"])
    mock_conn.__module__ = "psycopg"
    type(mock_conn).__module__ = "psycopg"
    mock_cursor = mocker.MagicMock()
    mock_conn.cursor = mocker.MagicMock(return_value=mock_cursor)

    mem = Memori(conn=lambda: mock_conn)
    mem.config.cache.conversation_id = 123
    mem.config.cache.session_id = 456

    mem.new_session()

    assert mem.config.cache.conversation_id is None
    assert mem.config.cache.session_id is None


def test_embed_texts_uses_config_defaults(mocker):
    mock_conn = mocker.Mock(spec=["cursor", "commit", "rollback"])
    mock_conn.__module__ = "psycopg"
    type(mock_conn).__module__ = "psycopg"
    mock_cursor = mocker.MagicMock()
    mock_conn.cursor = mocker.MagicMock(return_value=mock_cursor)

    mem = Memori(conn=lambda: mock_conn)
    mem.config.embeddings.model = "test-model"

    mock_embed = mocker.patch("memori.embed_texts", return_value=[[1.0, 2.0, 3.0]])

    out = mem.embed_texts("hello")

    assert out == [[1.0, 2.0, 3.0]]
    mock_embed.assert_called_once_with(
        "hello",
        model="test-model",
        async_=False,
    )


def test_recall_defaults_to_config_limit_in_cloud(monkeypatch, mocker):
    monkeypatch.delenv("MEMORI_COCKROACHDB_CONNECTION_STRING", raising=False)
    monkeypatch.setenv("MEMORI_API_KEY", "test-api-key")
    monkeypatch.setenv("MEMORI_TEST_MODE", "1")

    mem = Memori().attribution(entity_id="entity-id", process_id="process-id")
    mem.config.recall_facts_limit = 10

    post = mocker.patch(
        "memori.memory.recall.Api.post",
        autospec=True,
        return_value={"facts": [], "messages": []},
    )

    mem.recall("test query")

    assert post.call_args[0][1] == "cloud/recall"
    payload = post.call_args[0][2]
    assert payload["limit"] == 10

    mem.recall("test query", limit=3)
    payload = post.call_args[0][2]
    assert payload["limit"] == 3


def test_delete_entity_memories_supported_in_explicit_conn_mode(mocker):
    mock_conn = mocker.Mock(spec=["cursor", "commit", "rollback"])
    mock_conn.__module__ = "psycopg"
    type(mock_conn).__module__ = "psycopg"
    mock_cursor = mocker.MagicMock()
    mock_conn.cursor = mocker.MagicMock(return_value=mock_cursor)

    delete_memories = mocker.patch(
        "memori.memory.recall.Recall.delete_entity_memories",
        autospec=True,
    )
    mem = Memori(conn=lambda: mock_conn)
    mem.delete_entity_memories("entity-id")

    assert mem.config.byodb is True
    assert mem.config.cloud is False
    delete_memories.assert_called_once()
    assert delete_memories.call_args[0][1] == "entity-id"


def test_delete_entity_memories_rejected_in_cloud_mode(monkeypatch):
    monkeypatch.delenv("MEMORI_COCKROACHDB_CONNECTION_STRING", raising=False)
    monkeypatch.setenv("MEMORI_API_KEY", "test-api-key")
    monkeypatch.setenv("MEMORI_TEST_MODE", "1")
    mem = Memori()

    with pytest.raises(RuntimeError) as e:
        mem.delete_entity_memories("entity-id")

    assert str(e.value) == "delete_entity_memories is only available in BYODB mode"


def test_delete_entity_memories_rejected_for_cockroach_connection_string(
    monkeypatch, mocker
):
    monkeypatch.setenv(
        "MEMORI_COCKROACHDB_CONNECTION_STRING",
        "postgresql://user:pass@localhost:26257/defaultdb?sslmode=disable",
    )

    mock_conn = mocker.Mock(spec=["cursor", "commit", "rollback"])
    mock_conn.__module__ = "psycopg"
    type(mock_conn).__module__ = "psycopg"
    mock_cursor = mocker.MagicMock()
    mock_conn.cursor = mocker.MagicMock(return_value=mock_cursor)

    mocker.patch("psycopg.connect", return_value=mock_conn)

    mem = Memori(conn=None)
    with pytest.raises(RuntimeError) as e:
        mem.delete_entity_memories("entity-id")

    assert mem.config.byodb is False
    assert str(e.value) == "delete_entity_memories is only available in BYODB mode"
