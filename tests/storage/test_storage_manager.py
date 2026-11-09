import sqlite3
from unittest.mock import Mock, patch

from memori._config import Config
from memori.storage import Manager
from memori.storage._connection import ConnectionFactory


def test_persistent_connection_survives_context_exits():
    conn = sqlite3.connect(":memory:")
    manager = Manager(Config())
    manager.conn_factory = ConnectionFactory(lambda: conn, should_close=False)

    with manager.conn as (first_conn, _, _):
        assert first_conn.execute("select 1").fetchone() == (1,)

    with manager.conn as (second_conn, _, _):
        assert second_conn.execute("select 1").fetchone() == (1,)

    assert conn.execute("select 1").fetchone() == (1,)


def test_manager_start_does_not_close_a_user_provided_connection():
    config = Config()
    manager = Manager(config)

    class Connection:
        pass

    raw_conn = Connection()
    mock_adapter = Mock()
    mock_adapter.get_dialect.return_value = "sqlite"
    mock_driver = Mock()

    with patch("memori.storage._manager.Registry") as mock_registry_class:
        mock_registry = Mock()
        mock_registry.adapter.return_value = mock_adapter
        mock_registry.driver.return_value = mock_driver
        mock_registry_class.return_value = mock_registry

        manager.start(raw_conn)

    assert manager.conn_factory() is raw_conn
    assert manager.conn_factory.should_close is False


def test_manager_start_sets_cockroachdb_flag_for_cockroachdb():
    config = Config()
    manager = Manager(config)

    mock_conn = Mock()
    mock_adapter = Mock()
    mock_adapter.get_dialect.return_value = "cockroachdb"
    mock_driver = Mock()

    with patch("memori.storage._manager.Registry") as mock_registry_class:
        mock_registry = Mock()
        mock_registry.adapter.return_value = mock_adapter
        mock_registry.driver.return_value = mock_driver
        mock_registry_class.return_value = mock_registry

        manager.start(mock_conn)

    assert config.storage_config.cockroachdb is True


def test_manager_start_sets_cockroachdb_flag_for_postgresql():
    config = Config()
    manager = Manager(config)

    mock_conn = Mock()
    mock_adapter = Mock()
    mock_adapter.get_dialect.return_value = "postgresql"
    mock_driver = Mock()

    with patch("memori.storage._manager.Registry") as mock_registry_class:
        mock_registry = Mock()
        mock_registry.adapter.return_value = mock_adapter
        mock_registry.driver.return_value = mock_driver
        mock_registry_class.return_value = mock_registry

        manager.start(mock_conn)

    assert config.storage_config.cockroachdb is False


def test_manager_start_sets_cockroachdb_flag_for_mysql():
    config = Config()
    manager = Manager(config)

    mock_conn = Mock()
    mock_adapter = Mock()
    mock_adapter.get_dialect.return_value = "mysql"
    mock_driver = Mock()

    with patch("memori.storage._manager.Registry") as mock_registry_class:
        mock_registry = Mock()
        mock_registry.adapter.return_value = mock_adapter
        mock_registry.driver.return_value = mock_driver
        mock_registry_class.return_value = mock_registry

        manager.start(mock_conn)

    assert config.storage_config.cockroachdb is False


def test_manager_start_sets_cockroachdb_flag_for_oceanbase():
    config = Config()
    manager = Manager(config)

    mock_conn = Mock()
    mock_adapter = Mock()
    mock_adapter.get_dialect.return_value = "oceanbase"
    mock_driver = Mock()

    with patch("memori.storage._manager.Registry") as mock_registry_class:
        mock_registry = Mock()
        mock_registry.adapter.return_value = mock_adapter
        mock_registry.driver.return_value = mock_driver
        mock_registry_class.return_value = mock_registry

        manager.start(mock_conn)

    assert config.storage_config.cockroachdb is False


def test_manager_start_sets_cockroachdb_flag_for_mongodb():
    config = Config()
    manager = Manager(config)

    mock_conn = Mock()
    mock_adapter = Mock()
    mock_adapter.get_dialect.return_value = "mongodb"
    mock_driver = Mock()

    with patch("memori.storage._manager.Registry") as mock_registry_class:
        mock_registry = Mock()
        mock_registry.adapter.return_value = mock_adapter
        mock_registry.driver.return_value = mock_driver
        mock_registry_class.return_value = mock_registry

        manager.start(mock_conn)

    assert config.storage_config.cockroachdb is False


def test_manager_start_with_none_conn_does_not_set_flag():
    config = Config()
    manager = Manager(config)

    original_value = config.storage_config.cockroachdb
    manager.start(None)

    assert config.storage_config.cockroachdb == original_value
