from memori.storage.adapters.sqlalchemy._adapter import Adapter as SqlAlchemyAdapter


def test_commit(session):
    adapter = SqlAlchemyAdapter(lambda: session)
    adapter.commit()


def test_execute(session):
    adapter = SqlAlchemyAdapter(lambda: session)

    assert adapter.execute("select 1 from dual").mappings().fetchone() == {"1": 1}


def test_flush(session):
    adapter = SqlAlchemyAdapter(lambda: session)
    adapter.flush()


def test_get_dialect(session):
    adapter = SqlAlchemyAdapter(lambda: session)
    assert adapter.get_dialect() == "mysql"


def test_get_dialect_oceanbase(mocker):
    class FakeDialect:
        __module__ = "pyobvector.schema.dialect"
        name = "mysql"

    mock_bind = mocker.Mock()
    mock_bind.dialect = FakeDialect()
    session = mocker.Mock()
    session.get_bind.return_value = mock_bind

    adapter = SqlAlchemyAdapter(lambda: session)

    assert adapter.get_dialect() == "oceanbase"


def test_rollback(session):
    adapter = SqlAlchemyAdapter(lambda: session)
    adapter.rollback()


# PostgreSQL tests
def test_commit_postgres(postgres_session):
    adapter = SqlAlchemyAdapter(lambda: postgres_session)
    adapter.commit()


def test_execute_postgres(postgres_session):
    adapter = SqlAlchemyAdapter(lambda: postgres_session)

    assert adapter.execute("select 1 as one").mappings().fetchone() == {"one": 1}


def test_flush_postgres(postgres_session):
    adapter = SqlAlchemyAdapter(lambda: postgres_session)
    adapter.flush()


def test_get_dialect_postgres(postgres_session):
    adapter = SqlAlchemyAdapter(lambda: postgres_session)
    assert adapter.get_dialect() == "postgresql"


def test_rollback_postgres(postgres_session):
    adapter = SqlAlchemyAdapter(lambda: postgres_session)
    adapter.rollback()

