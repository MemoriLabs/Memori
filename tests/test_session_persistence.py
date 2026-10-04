import sqlite3

import pytest

from memori import Memori
from memori.llm._base import BaseInvoke
from memori.llm._constants import OPENAI_LLM_PROVIDER
from memori.llm.pipelines.conversation_injection import inject_conversation_messages
from memori.memory._writer import Writer


@pytest.fixture
def sqlite_memori(tmp_path, mocker):
    # Conversation persistence does not need background fact augmentation.
    mocker.patch("memori.memory.augmentation.Manager.start", return_value=None)
    database = tmp_path / "sessions.sqlite"
    connection = sqlite3.connect(database)
    mem = Memori(conn=lambda: connection, use_rust_core=False)
    mem.attribution(entity_id="garden-planner")
    mem.config.storage.build()
    mem.config.llm.provider = OPENAI_LLM_PROVIDER
    try:
        yield mem, database
    finally:
        mem.close()


def write_message(mem, text):
    Writer(mem.config).execute(
        {"messages": [{"role": "user", "type": None, "text": text}]}
    )


@pytest.mark.parametrize("transition", ["different", "existing", "same", "new"])
def test_session_transition_preserves_persisted_conversations(
    sqlite_memori, transition
):
    mem, database = sqlite_memori
    session_a = "11111111-1111-4111-8111-111111111111"
    session_b = "22222222-2222-4222-8222-222222222222"
    mem.set_session(session_a)
    write_message(mem, "Plan the garden")
    expected = [(session_a, "Plan the garden")]

    if transition == "existing":
        mem.new_session()
        write_message(mem, "Choose a watering schedule")
        expected.append((str(mem.config.session_id), "Choose a watering schedule"))

    with sqlite3.connect(database) as oracle:
        original_ids = dict(
            oracle.execute("SELECT uuid, id FROM memori_session").fetchall()
        )
        original_conversations = dict(
            oracle.execute("SELECT session_id, id FROM memori_conversation").fetchall()
        )

    if transition == "different":
        mem.set_session(session_b)
    elif transition == "new":
        mem.new_session()
    else:
        mem.set_session(session_a)

    target = str(mem.config.session_id)
    expected_history = [text for session, text in expected if session == target]
    invoke = BaseInvoke(mem.config, None)
    request = inject_conversation_messages(
        invoke, {"messages": [{"role": "user", "content": "Choose the flowers"}]}
    )
    write_message(mem, "Choose the flowers")
    expected.append((target, "Choose the flowers"))

    # Read committed state through a separate connection, independently of the cache.
    with sqlite3.connect(database) as oracle:
        messages = oracle.execute(
            """SELECT s.uuid, m.content
               FROM memori_conversation_message m
               JOIN memori_conversation c ON m.conversation_id = c.id
               JOIN memori_session s ON c.session_id = s.id
               ORDER BY m.id"""
        ).fetchall()
        sessions = dict(
            oracle.execute("SELECT uuid, id FROM memori_session").fetchall()
        )
        conversations = dict(
            oracle.execute("SELECT session_id, id FROM memori_conversation").fetchall()
        )
        session_count = oracle.execute(
            "SELECT COUNT(*) FROM memori_session"
        ).fetchone()[0]
        conversation_count = oracle.execute(
            "SELECT COUNT(*) FROM memori_conversation"
        ).fetchone()[0]
        entity_count = oracle.execute("SELECT COUNT(*) FROM memori_entity").fetchone()[
            0
        ]

    assert messages == expected
    assert [message["content"] for message in request["messages"]] == [
        *expected_history,
        "Choose the flowers",
    ]
    assert (
        session_count == conversation_count == len({session for session, _ in expected})
    )
    assert entity_count == 1
    for session, session_id in original_ids.items():
        assert sessions[session] == session_id
        assert conversations[session_id] == original_conversations[session_id]
