import json
from collections.abc import Iterator
from typing import Any, TypedDict


class ConversationMessage(TypedDict):
    role: str
    type: str | None
    text: str


def _stringify_content(content: Any) -> str:
    if isinstance(content, dict | list):
        return json.dumps(content)
    return str(content)


def parse_payload_conversation_messages(
    payload: dict,
    *,
    adapter: Any | None = None,
    registry: Any | None = None,
) -> Iterator[ConversationMessage]:
    """Yield normalized conversation messages parsed from an LLM payload.

    Normalization rules:
    - Query messages: set `type=None`, stringify `content`
    """
    conversation = payload.get("conversation") if isinstance(payload, dict) else None
    if isinstance(conversation, dict):
        existing = conversation.get("messages")
        if (
            isinstance(existing, list)
            and "query" not in conversation
            and "response" not in conversation
        ):
            for message in existing:
                if not isinstance(message, dict):
                    continue
                role = message.get("role")
                text = message.get("text")
                if role is None or text is None:
                    continue
                yield {
                    "role": str(role),
                    "type": message.get("type"),
                    "text": str(text),
                }
            return

    if adapter is None:
        if registry is None:
            from memori.llm._registry import Registry as LlmRegistry

            registry = LlmRegistry()
        adapter = registry.adapter(
            payload["conversation"]["client"]["provider"],
            payload["conversation"]["client"]["title"],
        )

    for message in adapter.get_formatted_query(payload) or []:
        yield {
            "role": message["role"],
            "type": None,
            "text": _stringify_content(message["content"]),
        }

    for response in adapter.get_formatted_response(payload) or []:
        yield {
            "role": response["role"],
            "type": response.get("type"),
            "text": response["text"],
        }
