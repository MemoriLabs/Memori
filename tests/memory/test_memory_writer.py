from memori.memory._writer import Writer


def test_execute(config, mocker):
    Writer(config).execute(
        {
            "messages": [
                {"role": "user", "type": None, "text": "abc"},
                {"role": "assistant", "type": "text", "text": "def"},
                {"role": "assistant", "type": "text", "text": "ghi"},
            ]
        }
    )

    assert config.cache.session_id is not None
    assert config.cache.conversation_id is not None

    assert config.storage.driver.session.create.called
    assert config.storage.driver.conversation.create.called
    assert config.storage.driver.conversation.message.create.call_count == 3

    calls = config.storage.driver.conversation.message.create.call_args_list
    assert calls[0][0][1] == "user"
    assert calls[0][0][3] == "abc"
    assert calls[1][0][1] == "assistant"
    assert calls[1][0][3] == "def"
    assert calls[2][0][1] == "assistant"
    assert calls[2][0][3] == "ghi"


def test_execute_with_entity_and_process(config, mocker):
    config.entity_id = "123"
    config.process_id = "456"

    Writer(config).execute(
        {
            "messages": [
                {"role": "user", "type": None, "text": "abc"},
                {"role": "assistant", "type": "text", "text": "def"},
                {"role": "assistant", "type": "text", "text": "ghi"},
            ]
        }
    )

    assert config.cache.entity_id is not None
    assert config.cache.process_id is not None
    assert config.cache.session_id is not None
    assert config.cache.conversation_id is not None

    assert config.storage.driver.entity.create.called
    assert config.storage.driver.entity.create.call_args[0][0] == "123"

    assert config.storage.driver.process.create.called
    assert config.storage.driver.process.create.call_args[0][0] == "456"

    assert config.storage.driver.session.create.called
    session_call_args = config.storage.driver.session.create.call_args[0]
    assert session_call_args[1] == config.cache.entity_id
    assert session_call_args[2] == config.cache.process_id

    assert config.storage.driver.conversation.message.create.call_count == 3


def test_execute_includes_system_messages(config, mocker):
    Writer(config).execute(
        {
            "messages": [
                {
                    "role": "system",
                    "type": None,
                    "text": "You are a helpful assistant",
                },
                {"role": "user", "type": None, "text": "Hello"},
                {"role": "assistant", "type": "text", "text": "Hi there!"},
            ]
        }
    )

    assert config.storage.driver.conversation.message.create.call_count == 3

    calls = config.storage.driver.conversation.message.create.call_args_list
    assert calls[0][0][1] == "system"
    assert calls[0][0][3] == "You are a helpful assistant"
    assert calls[1][0][1] == "user"
    assert calls[1][0][3] == "Hello"
    assert calls[2][0][1] == "assistant"
    assert calls[2][0][3] == "Hi there!"


def test_execute_writes_response_type(config, mocker):
    Writer(config).execute(
        {
            "messages": [
                {"role": "user", "type": None, "text": "hello"},
                {"role": "assistant", "type": "text", "text": "ok"},
            ]
        }
    )

    calls = config.storage.driver.conversation.message.create.call_args_list
    assert calls[0][0][2] is None
    assert calls[1][0][2] == "text"


def test_execute_multiple_turns_ingests_all_messages(config, mocker):
    """Multiple conversation turns reuse the same conversation_id and write all messages."""
    conversation_id = 123
    config.storage.driver.conversation.create.return_value = conversation_id
    config.cache.conversation_id = None

    # First turn
    Writer(config).execute(
        {
            "messages": [
                {"role": "user", "type": None, "text": "Hello"},
                {"role": "assistant", "type": "text", "text": "Hi there!"},
            ]
        }
    )

    assert config.cache.conversation_id == conversation_id
    assert config.storage.driver.conversation.message.create.call_count == 2
    calls1 = config.storage.driver.conversation.message.create.call_args_list
    assert calls1[0][0][3] == "Hello"
    assert calls1[1][0][3] == "Hi there!"

    # Second turn: same conversation_id, new messages
    config.storage.driver.conversation.message.create.reset_mock()
    Writer(config).execute(
        {
            "messages": [
                {"role": "user", "type": None, "text": "What's the weather?"},
                {"role": "assistant", "type": "text", "text": "I don't have access."},
            ]
        }
    )

    assert config.cache.conversation_id == conversation_id
    assert config.storage.driver.conversation.message.create.call_count == 2
    calls2 = config.storage.driver.conversation.message.create.call_args_list
    assert calls2[0][0][3] == "What's the weather?"
    assert calls2[1][0][3] == "I don't have access."
