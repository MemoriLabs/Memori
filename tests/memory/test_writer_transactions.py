import sqlite3
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine
from sqlalchemy.exc import IntegrityError, OperationalError
from sqlalchemy.orm import Session

from memori._config import Config
from memori.memory._writer import Writer
from memori.storage.adapters.dbapi._adapter import Adapter as DbapiAdapter
from memori.storage.adapters.sqlalchemy._adapter import Adapter as SqlalchemyAdapter
from memori.storage.drivers.sqlite._driver import Driver
from memori.storage.migrations._sqlite import migrations


def payload(*contents):
    return {
        "messages": [
            {"role": "user", "type": None, "text": content} for content in contents
        ]
    }


@pytest.fixture(params=["dbapi", "sqlalchemy"])
def storage(request, tmp_path):
    path = tmp_path / "memory.sqlite"
    engine = None
    if request.param == "dbapi":
        connection = sqlite3.connect(path)
        adapter = DbapiAdapter(lambda: connection)
    else:
        engine = create_engine(f"sqlite:///{path}")
        connection = Session(engine)
        adapter = SqlalchemyAdapter(lambda: connection)
    for step in migrations[1]:
        adapter.execute(step["operation"])
    adapter.commit()
    config = Config()
    config.entity_id = "transaction-test-user"
    config.session_id = "transaction-test-session"
    config.storage = SimpleNamespace(adapter=adapter, driver=Driver(adapter))
    writer = Writer(config)
    writer.execute(payload("committed-before"))
    try:
        yield writer, config, path
    finally:
        connection.rollback()
        connection.close()
        if engine is not None:
            engine.dispose()
        config.thread_pool_executor.shutdown()


def committed_contents(path):
    with sqlite3.connect(path) as reader:
        return [
            row[0]
            for row in reader.execute(
                "SELECT content FROM memori_conversation_message ORDER BY id"
            )
        ]


def test_failed_batch_cannot_leak_into_next_write(storage):
    writer, config, path = storage
    config.storage.adapter.execute(
        "CREATE TRIGGER reject_message BEFORE INSERT ON memori_conversation_message "
        "WHEN NEW.content = 'rejected' "
        "BEGIN SELECT RAISE(ABORT, 'message rejected'); END"
    )
    config.storage.adapter.commit()
    with pytest.raises(
        (sqlite3.IntegrityError, IntegrityError), match="message rejected"
    ):
        writer.execute(payload("failed-prefix", "rejected"))
    writer.execute(payload("next-turn"))
    assert committed_contents(path) == ["committed-before", "next-turn"]


@pytest.mark.parametrize("attempts", [1, 3])
def test_exhausted_restart_rolls_back_last_attempt(storage, monkeypatch, attempts):
    writer, config, path = storage
    messages = config.storage.driver.conversation.message
    original_create = messages.create
    failure = OperationalError("INSERT", {}, RuntimeError("restart transaction"))
    failures = 0

    def create(*args):
        nonlocal failures
        if args[-1] == "reject":
            failures += 1
            raise failure
        return original_create(*args)

    monkeypatch.setattr(messages, "create", create)
    monkeypatch.setattr("memori.memory._writer.time.sleep", lambda _: None)
    with pytest.raises(OperationalError) as raised:
        writer.execute(payload("failed-prefix", "reject"), max_retries=attempts)
    assert raised.value is failure
    assert failures == attempts
    writer.execute(payload("next-turn"))
    assert committed_contents(path) == ["committed-before", "next-turn"]


@pytest.mark.parametrize("boundary", ["flush", "commit"])
def test_write_boundary_failure_is_rolled_back(storage, monkeypatch, boundary):
    writer, config, path = storage
    adapter = config.storage.adapter
    original = getattr(adapter, boundary)
    failure = RuntimeError(f"{boundary} failed")

    def fail():
        raise failure

    monkeypatch.setattr(adapter, boundary, fail)
    with pytest.raises(RuntimeError) as raised:
        writer.execute(payload("failed-prefix", "failed-suffix"))
    assert raised.value is failure
    monkeypatch.setattr(adapter, boundary, original)
    writer.execute(payload("next-turn"))
    assert committed_contents(path) == ["committed-before", "next-turn"]


def test_intermediate_restart_preserves_complete_batch_once(storage, monkeypatch):
    writer, config, path = storage
    messages = config.storage.driver.conversation.message
    original = messages.create
    fail_once = True

    def create(*args):
        nonlocal fail_once
        if args[-1] == "answer" and fail_once:
            fail_once = False
            raise OperationalError("INSERT", {}, RuntimeError("restart transaction"))
        return original(*args)

    monkeypatch.setattr(messages, "create", create)
    monkeypatch.setattr("memori.memory._writer.time.sleep", lambda _: None)
    writer.execute(payload("question", "answer"))
    assert committed_contents(path) == ["committed-before", "question", "answer"]


def test_rollback_failure_preserves_write_error_without_retry(
    storage, monkeypatch, caplog
):
    writer, config, _ = storage
    failure = OperationalError("INSERT", {}, RuntimeError("restart transaction"))
    attempts = 0

    def fail(*args):
        nonlocal attempts
        attempts += 1
        raise failure

    def rollback():
        raise RuntimeError("rollback unavailable")

    monkeypatch.setattr(config.storage.driver.conversation.message, "create", fail)
    monkeypatch.setattr(config.storage.adapter, "rollback", rollback)
    with pytest.raises(OperationalError) as raised:
        writer.execute(payload("question"))
    assert raised.value is failure
    assert attempts == 1
    assert "rollback unavailable" in caplog.text
