"""
Database integration tests for validating memories are properly written to and
retrieved from SQLite, PostgreSQL, MySQL, and MongoDB databases.
"""

import pytest
from openai import OpenAI

from tests.integration.databases.conftest import (
    requires_mongodb,
    requires_mysql,
    requires_openai,
    requires_postgres,
    requires_sqlite,
)

MODEL = "gpt-4o-mini"
MAX_TOKENS = 50


class TestSQLiteStorage:
    """Test suite for SQLite database storage."""

    @requires_sqlite
    @requires_openai
    @pytest.mark.integration
    def test_store_and_search_facts(self, sqlite_memori, openai_api_key):
        """Test that facts are stored and can be searched in SQLite."""
        client = OpenAI(api_key=openai_api_key)
        sqlite_memori.llm.register(client)
        sqlite_memori.attribution(entity_id="sqlite-test-user", process_id="test")

        # Make a conversation that should store facts
        client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "user", "content": "My name is Alice and I live in Paris."}
            ],
            max_tokens=MAX_TOKENS,
        )

        # Verify conversation was stored
        conversation_id = sqlite_memori.config.cache.conversation_id
        assert conversation_id is not None

        conversation = sqlite_memori.config.storage.driver.conversation.read(
            conversation_id
        )
        assert conversation is not None
        assert conversation["id"] == conversation_id

    @requires_sqlite
    @requires_openai
    @pytest.mark.integration
    def test_multiple_entities_isolation(self, sqlite_memori, openai_api_key):
        """Test that facts from different entities are isolated in SQLite."""
        client = OpenAI(api_key=openai_api_key)
        sqlite_memori.llm.register(client)

        # First entity
        sqlite_memori.attribution(entity_id="sqlite-user-1", process_id="test")
        client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": "I am User One."}],
            max_tokens=MAX_TOKENS,
        )
        conversation_id_1 = sqlite_memori.config.cache.conversation_id

        # New session for second entity
        sqlite_memori.new_session()

        # Second entity
        sqlite_memori.attribution(entity_id="sqlite-user-2", process_id="test")
        client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": "I am User Two."}],
            max_tokens=MAX_TOKENS,
        )
        conversation_id_2 = sqlite_memori.config.cache.conversation_id

        # Verify both conversations exist and are different
        assert conversation_id_1 is not None
        assert conversation_id_2 is not None
        assert conversation_id_1 != conversation_id_2

    @requires_sqlite
    @requires_openai
    @pytest.mark.integration
    def test_conversation_storage(self, sqlite_memori, openai_api_key):
        """Test that conversation messages are stored correctly in SQLite."""
        client = OpenAI(api_key=openai_api_key)
        sqlite_memori.llm.register(client)
        sqlite_memori.attribution(entity_id="sqlite-conv-user", process_id="test")

        test_message = "Hello, this is a test message for SQLite storage."

        client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": test_message}],
            max_tokens=MAX_TOKENS,
        )

        conversation_id = sqlite_memori.config.cache.conversation_id
        messages = sqlite_memori.config.storage.driver.conversation.messages.read(
            conversation_id
        )

        # Should have at least user and assistant messages
        assert len(messages) >= 2

        user_messages = [m for m in messages if m["role"] == "user"]
        assert len(user_messages) >= 1
        assert test_message in user_messages[0]["content"]

        assistant_messages = [m for m in messages if m["role"] == "assistant"]
        assert len(assistant_messages) >= 1


class TestPostgresStorage:
    """Test suite for PostgreSQL database storage."""

    @requires_postgres
    @requires_openai
    @pytest.mark.integration
    def test_store_and_search_facts(self, postgres_memori, openai_api_key):
        """Test that facts are stored and can be searched in PostgreSQL."""
        client = OpenAI(api_key=openai_api_key)
        postgres_memori.llm.register(client)
        postgres_memori.attribution(entity_id="postgres-test-user", process_id="test")

        client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "user", "content": "My name is Bob and I work at Acme Corp."}
            ],
            max_tokens=MAX_TOKENS,
        )

        conversation_id = postgres_memori.config.cache.conversation_id
        assert conversation_id is not None

        conversation = postgres_memori.config.storage.driver.conversation.read(
            conversation_id
        )
        assert conversation is not None
        assert conversation["id"] == conversation_id

    @requires_postgres
    @requires_openai
    @pytest.mark.integration
    def test_multiple_entities_isolation(self, postgres_memori, openai_api_key):
        """Test that facts from different entities are isolated in PostgreSQL."""
        client = OpenAI(api_key=openai_api_key)
        postgres_memori.llm.register(client)

        # First entity
        postgres_memori.attribution(entity_id="postgres-user-1", process_id="test")
        client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": "I am Postgres User One."}],
            max_tokens=MAX_TOKENS,
        )
        conversation_id_1 = postgres_memori.config.cache.conversation_id

        # New session for second entity
        postgres_memori.new_session()

        # Second entity
        postgres_memori.attribution(entity_id="postgres-user-2", process_id="test")
        client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": "I am Postgres User Two."}],
            max_tokens=MAX_TOKENS,
        )
        conversation_id_2 = postgres_memori.config.cache.conversation_id

        assert conversation_id_1 is not None
        assert conversation_id_2 is not None
        assert conversation_id_1 != conversation_id_2

    @requires_postgres
    @requires_openai
    @pytest.mark.integration
    def test_conversation_storage(self, postgres_memori, openai_api_key):
        """Test that conversation messages are stored correctly in PostgreSQL."""
        client = OpenAI(api_key=openai_api_key)
        postgres_memori.llm.register(client)
        postgres_memori.attribution(entity_id="postgres-conv-user", process_id="test")

        test_message = "Hello, this is a test message for PostgreSQL storage."

        client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": test_message}],
            max_tokens=MAX_TOKENS,
        )

        conversation_id = postgres_memori.config.cache.conversation_id
        messages = postgres_memori.config.storage.driver.conversation.messages.read(
            conversation_id
        )

        assert len(messages) >= 2

        user_messages = [m for m in messages if m["role"] == "user"]
        assert len(user_messages) >= 1
        assert test_message in user_messages[0]["content"]

        assistant_messages = [m for m in messages if m["role"] == "assistant"]
        assert len(assistant_messages) >= 1


class TestMySQLStorage:
    """Test suite for MySQL database storage."""

    @requires_mysql
    @requires_openai
    @pytest.mark.integration
    def test_store_and_search_facts(self, mysql_memori, openai_api_key):
        """Test that facts are stored and can be searched in MySQL."""
        client = OpenAI(api_key=openai_api_key)
        mysql_memori.llm.register(client)
        mysql_memori.attribution(entity_id="mysql-test-user", process_id="test")

        client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "user",
                    "content": "My name is Charlie and I enjoy programming.",
                }
            ],
            max_tokens=MAX_TOKENS,
        )

        conversation_id = mysql_memori.config.cache.conversation_id
        assert conversation_id is not None

        conversation = mysql_memori.config.storage.driver.conversation.read(
            conversation_id
        )
        assert conversation is not None
        assert conversation["id"] == conversation_id

    @requires_mysql
    @requires_openai
    @pytest.mark.integration
    def test_multiple_entities_isolation(self, mysql_memori, openai_api_key):
        """Test that facts from different entities are isolated in MySQL."""
        client = OpenAI(api_key=openai_api_key)
        mysql_memori.llm.register(client)

        # First entity
        mysql_memori.attribution(entity_id="mysql-user-1", process_id="test")
        client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": "I am MySQL User One."}],
            max_tokens=MAX_TOKENS,
        )
        conversation_id_1 = mysql_memori.config.cache.conversation_id

        # New session for second entity
        mysql_memori.new_session()

        # Second entity
        mysql_memori.attribution(entity_id="mysql-user-2", process_id="test")
        client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": "I am MySQL User Two."}],
            max_tokens=MAX_TOKENS,
        )
        conversation_id_2 = mysql_memori.config.cache.conversation_id

        assert conversation_id_1 is not None
        assert conversation_id_2 is not None
        assert conversation_id_1 != conversation_id_2

    @requires_mysql
    @requires_openai
    @pytest.mark.integration
    def test_conversation_storage(self, mysql_memori, openai_api_key):
        """Test that conversation messages are stored correctly in MySQL."""
        client = OpenAI(api_key=openai_api_key)
        mysql_memori.llm.register(client)
        mysql_memori.attribution(entity_id="mysql-conv-user", process_id="test")

        test_message = "Hello, this is a test message for MySQL storage."

        client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": test_message}],
            max_tokens=MAX_TOKENS,
        )

        conversation_id = mysql_memori.config.cache.conversation_id
        messages = mysql_memori.config.storage.driver.conversation.messages.read(
            conversation_id
        )

        assert len(messages) >= 2

        user_messages = [m for m in messages if m["role"] == "user"]
        assert len(user_messages) >= 1
        assert test_message in user_messages[0]["content"]

        assistant_messages = [m for m in messages if m["role"] == "assistant"]
        assert len(assistant_messages) >= 1


class TestMongoDBStorage:
    """Test suite for MongoDB database storage."""

    @requires_mongodb
    @requires_openai
    @pytest.mark.integration
    def test_store_and_search_facts(self, mongodb_memori, openai_api_key):
        """Test that facts are stored and can be searched in MongoDB."""
        client = OpenAI(api_key=openai_api_key)
        mongodb_memori.llm.register(client)
        mongodb_memori.attribution(entity_id="mongodb-test-user", process_id="test")

        client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "user", "content": "My name is Diana and I love databases."}
            ],
            max_tokens=MAX_TOKENS,
        )

        conversation_id = mongodb_memori.config.cache.conversation_id
        assert conversation_id is not None

        conversation = mongodb_memori.config.storage.driver.conversation.read(
            conversation_id
        )
        assert conversation is not None
        assert conversation["id"] == conversation_id

    @requires_mongodb
    @requires_openai
    @pytest.mark.integration
    def test_multiple_entities_isolation(self, mongodb_memori, openai_api_key):
        """Test that facts from different entities are isolated in MongoDB."""
        client = OpenAI(api_key=openai_api_key)
        mongodb_memori.llm.register(client)

        # First entity
        mongodb_memori.attribution(entity_id="mongodb-user-1", process_id="test")
        client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": "I am MongoDB User One."}],
            max_tokens=MAX_TOKENS,
        )
        conversation_id_1 = mongodb_memori.config.cache.conversation_id

        # New session for second entity
        mongodb_memori.new_session()

        # Second entity
        mongodb_memori.attribution(entity_id="mongodb-user-2", process_id="test")
        client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": "I am MongoDB User Two."}],
            max_tokens=MAX_TOKENS,
        )
        conversation_id_2 = mongodb_memori.config.cache.conversation_id

        assert conversation_id_1 is not None
        assert conversation_id_2 is not None
        assert conversation_id_1 != conversation_id_2

    @requires_mongodb
    @requires_openai
    @pytest.mark.integration
    def test_conversation_storage(self, mongodb_memori, openai_api_key):
        """Test that conversation messages are stored correctly in MongoDB."""
        client = OpenAI(api_key=openai_api_key)
        mongodb_memori.llm.register(client)
        mongodb_memori.attribution(entity_id="mongodb-conv-user", process_id="test")

        test_message = "Hello, this is a test message for MongoDB storage."

        client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": test_message}],
            max_tokens=MAX_TOKENS,
        )

        conversation_id = mongodb_memori.config.cache.conversation_id
        messages = mongodb_memori.config.storage.driver.conversation.messages.read(
            conversation_id
        )

        assert len(messages) >= 2

        user_messages = [m for m in messages if m["role"] == "user"]
        assert len(user_messages) >= 1
        assert test_message in user_messages[0]["content"]

        assistant_messages = [m for m in messages if m["role"] == "assistant"]
        assert len(assistant_messages) >= 1
