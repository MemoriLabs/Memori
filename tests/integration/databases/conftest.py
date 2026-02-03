import os
import time

import pytest
from pymongo import MongoClient
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool

# Database URL environment variables
SQLITE_DATABASE_URL = os.environ.get("SQLITE_DATABASE_URL")
POSTGRES_DATABASE_URL = os.environ.get("POSTGRES_DATABASE_URL")
MYSQL_DATABASE_URL = os.environ.get("MYSQL_DATABASE_URL")
MONGODB_URL = os.environ.get("MONGODB_URL")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# Skip markers for database availability
requires_sqlite = pytest.mark.skipif(
    not SQLITE_DATABASE_URL,
    reason="SQLITE_DATABASE_URL environment variable not set",
)

requires_postgres = pytest.mark.skipif(
    not POSTGRES_DATABASE_URL,
    reason="POSTGRES_DATABASE_URL environment variable not set",
)

requires_mysql = pytest.mark.skipif(
    not MYSQL_DATABASE_URL,
    reason="MYSQL_DATABASE_URL environment variable not set",
)

requires_mongodb = pytest.mark.skipif(
    not MONGODB_URL,
    reason="MONGODB_URL environment variable not set",
)

requires_openai = pytest.mark.skipif(
    not OPENAI_API_KEY,
    reason="OPENAI_API_KEY environment variable not set",
)


@pytest.fixture
def sqlite_session_factory(tmp_path):
    """Create a SQLite session factory for testing."""
    db_path = tmp_path / "test_memori.db"
    engine = create_engine(
        f"sqlite:///{db_path}",
        connect_args={"check_same_thread": False},
        poolclass=NullPool,
    )

    @event.listens_for(engine, "connect")
    def set_sqlite_pragma(dbapi_conn, connection_record):
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.close()

    Session = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    yield Session

    time.sleep(0.2)
    engine.dispose()


@pytest.fixture
def postgres_session_factory():
    """Create a PostgreSQL session factory for testing."""
    if not POSTGRES_DATABASE_URL:
        pytest.skip("POSTGRES_DATABASE_URL not set")

    engine = create_engine(
        POSTGRES_DATABASE_URL,
        pool_pre_ping=True,
        pool_recycle=300,
    )

    Session = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    yield Session

    engine.dispose()


@pytest.fixture
def mysql_session_factory():
    """Create a MySQL session factory for testing."""
    if not MYSQL_DATABASE_URL:
        pytest.skip("MYSQL_DATABASE_URL not set")

    engine = create_engine(
        MYSQL_DATABASE_URL,
        pool_pre_ping=True,
        pool_recycle=300,
    )

    Session = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    yield Session

    engine.dispose()


@pytest.fixture
def mongodb_client():
    """Create a MongoDB client for testing."""
    if not MONGODB_URL:
        pytest.skip("MONGODB_URL not set")

    client = MongoClient(MONGODB_URL)

    yield client

    client.close()


@pytest.fixture
def memori_test_mode():
    """Enable Memori test mode."""
    original = os.environ.get("MEMORI_TEST_MODE")
    os.environ["MEMORI_TEST_MODE"] = "1"
    yield
    if original is None:
        os.environ.pop("MEMORI_TEST_MODE", None)
    else:
        os.environ["MEMORI_TEST_MODE"] = original


@pytest.fixture
def sqlite_memori(sqlite_session_factory, memori_test_mode):
    """Create a Memori instance with SQLite backend."""
    from memori import Memori

    mem = Memori(conn=sqlite_session_factory)
    mem.config.storage.build()

    yield mem

    time.sleep(0.1)
    mem.close()


@pytest.fixture
def postgres_memori(postgres_session_factory, memori_test_mode):
    """Create a Memori instance with PostgreSQL backend."""
    from memori import Memori

    mem = Memori(conn=postgres_session_factory)
    mem.config.storage.build()

    yield mem

    time.sleep(0.1)
    mem.close()


@pytest.fixture
def mysql_memori(mysql_session_factory, memori_test_mode):
    """Create a Memori instance with MySQL backend."""
    from memori import Memori

    mem = Memori(conn=mysql_session_factory)
    mem.config.storage.build()

    yield mem

    time.sleep(0.1)
    mem.close()


@pytest.fixture
def mongodb_memori(mongodb_client, memori_test_mode):
    """Create a Memori instance with MongoDB backend."""
    from memori import Memori

    mem = Memori(conn=mongodb_client)
    mem.config.storage.build()

    yield mem

    time.sleep(0.1)
    mem.close()


@pytest.fixture(scope="session")
def openai_api_key():
    """Get OpenAI API key."""
    if not OPENAI_API_KEY:
        pytest.skip("OPENAI_API_KEY not set")
    return OPENAI_API_KEY


@pytest.fixture
def openai_client(openai_api_key):
    """Create an OpenAI client."""
    from openai import OpenAI

    return OpenAI(api_key=openai_api_key)
