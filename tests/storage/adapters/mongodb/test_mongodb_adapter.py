import pytest

from memori.storage.adapters.mongodb._adapter import Adapter


@pytest.fixture
def mock_mongo_client(mocker):
    """MongoClient instance passed directly as conn."""
    client = mocker.MagicMock()
    client.list_collection_names = mocker.MagicMock(return_value=[])
    client.database = mocker.MagicMock()
    client.get_default_database = mocker.MagicMock(return_value=mocker.MagicMock())
    # Make isinstance(client, MongoClient) work by patching the class
    from pymongo.synchronous.mongo_client import MongoClient

    mocker.patch(
        "memori.storage.adapters.mongodb._adapter.MongoClient",
        MongoClient,
    )
    client.__class__ = MongoClient
    return client


@pytest.fixture
def mock_mongo_db(mocker):
    """IMongoDatabase-style object with a .client attribute."""
    db = mocker.MagicMock()
    db.list_collection_names = mocker.MagicMock(return_value=[])
    db.database = mocker.MagicMock()
    client = mocker.MagicMock()
    db.client = client
    return db


def test_append_metadata_called_when_conn_is_mongo_client(mocker, mock_mongo_client):
    """Adapter calls append_metadata on a MongoClient passed directly."""
    Adapter(lambda: mock_mongo_client)
    mock_mongo_client.append_metadata.assert_called_once()
    call_arg = mock_mongo_client.append_metadata.call_args[0][0]
    assert call_arg.name == "Memori"


def test_append_metadata_called_when_conn_is_database(mocker, mock_mongo_db):
    """Adapter calls append_metadata on client retrieved from a database object."""
    Adapter(lambda: mock_mongo_db)
    mock_mongo_db.client.append_metadata.assert_called_once()
    call_arg = mock_mongo_db.client.append_metadata.call_args[0][0]
    assert call_arg.name == "Memori"


def test_append_metadata_skipped_when_not_available(mocker, mock_mongo_db):
    """Adapter does not raise if append_metadata is absent (older PyMongo)."""
    del mock_mongo_db.client.append_metadata
    # Should not raise
    Adapter(lambda: mock_mongo_db)


def test_append_metadata_includes_version(mocker, mock_mongo_db):
    """DriverInfo passed to append_metadata carries a non-empty version string."""
    Adapter(lambda: mock_mongo_db)
    call_arg = mock_mongo_db.client.append_metadata.call_args[0][0]
    assert call_arg.version is not None
    assert call_arg.version != ""


def test_execute(mongodb_conn):
    """Test MongoDB adapter execute method."""
    adapter = Adapter(lambda: mongodb_conn)

    adapter.execute("test_collection", "find_one", {"test": "value"})
    adapter.execute("test_collection", "insert_one", {"test": "value"})


def test_get_dialect(mongodb_conn):
    """Test MongoDB adapter get_dialect method."""
    adapter = Adapter(lambda: mongodb_conn)
    assert adapter.get_dialect() == "mongodb"


def test_execute_with_args(mongodb_conn):
    """Test MongoDB adapter execute method with various arguments."""
    adapter = Adapter(lambda: mongodb_conn)

    adapter.execute(
        "test_collection", "find", {"test": "value"}, {"field": 1, "_id": 0}
    )
    adapter.execute("test_collection", "delete_many", {"test": "value"})


def test_execute_with_kwargs(mongodb_conn):
    """Test MongoDB adapter execute method with keyword arguments."""
    adapter = Adapter(lambda: mongodb_conn)

    adapter.execute(
        "test_collection",
        "update_one",
        {"test": "value"},
        {"$set": {"updated": True}},
        upsert=True,
    )
