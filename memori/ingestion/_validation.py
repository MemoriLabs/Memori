r"""
 __  __                           _
|  \/  | ___ _ __ ___   ___  _ __(_)
| |\/| |/ _ \ '_ ` _ \ / _ \| '__| |
| |  | |  __/ | | | | | (_) | |  | |
|_|  |_|\___|_| |_| |_|\___/|_|  |_|
                  perfectam memoriam
                       memorilabs.ai
"""

from typing import Any


def validate_message(message: Any, index: int) -> list[str]:
    errors = []

    if not isinstance(message, dict):
        errors.append(
            f"Message {index}: must be a dictionary, got {type(message).__name__}"
        )
        return errors

    if "role" not in message:
        errors.append(f"Message {index}: missing required field 'role'")
    elif message["role"] not in ("user", "assistant", "system"):
        errors.append(
            f"Message {index}: invalid role '{message['role']}', must be 'user', 'assistant', or 'system'"
        )

    if "content" not in message:
        errors.append(f"Message {index}: missing required field 'content'")
    elif not isinstance(message["content"], str):
        errors.append(
            f"Message {index}: 'content' must be a string, got {type(message['content']).__name__}"
        )
    elif len(message["content"].strip()) == 0:
        errors.append(f"Message {index}: 'content' is empty")

    return errors


def validate_conversation(conversation: Any) -> tuple[bool, list[str]]:
    errors = []

    if not isinstance(conversation, dict):
        return False, [
            f"Conversation must be a dictionary, got {type(conversation).__name__}"
        ]

    messages = conversation.get("messages", [])

    if not messages:
        errors.append("Conversation has no messages")
        return False, errors

    if not isinstance(messages, list):
        errors.append(f"'messages' must be a list, got {type(messages).__name__}")
        return False, errors

    for i, msg in enumerate(messages):
        msg_errors = validate_message(msg, i)
        errors.extend(msg_errors)

    return len(errors) == 0, errors


def estimate_conversation_size(messages: list[dict[str, Any]]) -> tuple[int, int]:
    total_chars = sum(len(m.get("content", "")) for m in messages)
    return len(messages), total_chars
