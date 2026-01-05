import os

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool

from memori import Memori
from memori.llm._embeddings import embed_texts
from tests.benchmarks.fixtures.sample_data import (
    generate_facts_with_size,
    generate_sample_queries,
)


@pytest.fixture(scope="module")
def postgres_db_connection():
    postgres_uri = os.environ.get(
        "BENCHMARK_POSTGRES_URL",
        "postgresql://memori:memori@localhost:5432/memori_test",
    )

    from sqlalchemy import text

    connect_args = {}
    sslrootcert = os.environ.get("BENCHMARK_POSTGRES_SSLROOTCERT")
    if sslrootcert:
        connect_args["sslrootcert"] = sslrootcert
        if "sslmode" not in postgres_uri:
            separator = "&" if "?" in postgres_uri else "?"
            postgres_uri = f"{postgres_uri}{separator}sslmode=require"

    engine = create_engine(
        postgres_uri,
        poolclass=NullPool,
        connect_args=connect_args,
    )

    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception as e:
        pytest.skip(f"PostgreSQL not available: {e}")

    Session = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    yield Session
    engine.dispose()


@pytest.fixture(scope="module")
def mysql_db_connection():
    mysql_uri = os.environ.get(
        "BENCHMARK_MYSQL_URL",
        "mysql+pymysql://memori:memori@localhost:3306/memori_test",
    )

    from sqlalchemy import text

    engine = create_engine(
        mysql_uri,
        poolclass=NullPool,
    )

    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception as e:
        pytest.skip(f"MySQL not available: {e}")

    Session = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    yield Session
    engine.dispose()


@pytest.fixture(params=["postgres", "mysql"], ids=["postgres", "mysql"], scope="module")
def db_connection(request):
    db_type = request.param
    if db_type == "postgres":
        return request.getfixturevalue("postgres_db_connection")
    elif db_type == "mysql":
        return request.getfixturevalue("mysql_db_connection")
    pytest.skip(f"Unsupported benchmark database type: {db_type}")


@pytest.fixture(scope="module")
def memori_instance(db_connection, request):
    mem = Memori(conn=db_connection)
    mem.config.storage.build()

    try:
        bind = getattr(db_connection, "kw", {}).get("bind", None)
        mem._benchmark_db_type = bind.dialect.name if bind else "unknown"
    except Exception:
        mem._benchmark_db_type = "unknown"

    return mem


@pytest.fixture(scope="session")
def sample_queries():
    return generate_sample_queries()


@pytest.fixture(scope="session")
def fact_content_size():
    return "small"


@pytest.fixture(
    params=[5, 50, 100, 300, 600, 1000], ids=lambda x: f"n{x}", scope="module"
)
def entity_with_n_facts(memori_instance, fact_content_size, request):
    fact_count = request.param
    entity_id = f"bench-{fact_count}-{fact_content_size}"

    memori_instance.attribution(entity_id=entity_id, process_id="bench-proc")

    facts = generate_facts_with_size(fact_count, fact_content_size)
    fact_embeddings = embed_texts(
        facts,
        model=memori_instance.config.embeddings.model,
        fallback_dimension=memori_instance.config.embeddings.fallback_dimension,
    )

    entity_db_id = memori_instance.config.storage.driver.entity.create(entity_id)
    memori_instance.config.storage.driver.entity_fact.create(
        entity_db_id, facts, fact_embeddings
    )

    return {
        "entity_id": entity_id,
        "entity_db_id": entity_db_id,
        "fact_count": fact_count,
        "content_size": fact_content_size,
        "db_type": memori_instance._benchmark_db_type,
        "facts": facts,
    }
