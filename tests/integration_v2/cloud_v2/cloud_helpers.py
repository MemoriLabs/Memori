import asyncio
import time
from contextlib import contextmanager
from unittest.mock import patch

captured_message_payloads = []
captured_augmentation_payloads = []
captured_recall_payloads = []
simulated_cloud_history = []


def clear_mock_state():
    """Clear all captured payloads."""
    captured_message_payloads.clear()
    captured_augmentation_payloads.clear()
    captured_recall_payloads.clear()
    simulated_cloud_history.clear()


def mocked_post(self, route, json=None, status_code=False):
    if "cloud/conversation/messages" in route:
        if json:
            captured_message_payloads.append(json)
            for msg in json.get("messages", []):
                if msg not in simulated_cloud_history:
                    simulated_cloud_history.append(msg)
        return 201 if status_code else {}

    elif "cloud/augmentation" in route:
        if json:
            captured_augmentation_payloads.append(json)
        return 200 if status_code else {}

    elif "cloud/recall" in route:
        if json:
            captured_recall_payloads.append(json)
        return {"messages": list(simulated_cloud_history)} if not status_code else 200

    return 200 if status_code else {}


@contextmanager
def inject_recall_fact(fact_content="The user's favorite word is 'MEMORI_42'."):
    """Temporarily overrides the universal mock to inject a specific recall fact."""
    original_mock = mocked_post

    def mock_with_fact(self, route, json=None, status_code=False):
        if "cloud/recall" in route:
            return (
                {
                    "messages": [],
                    "facts": [{"content": fact_content, "rank_score": 0.99}],
                }
                if not status_code
                else 200
            )
        return original_mock(self, route, json, status_code)

    with patch("memori._network.Api.post", new=mock_with_fact):
        yield


def _is_payload_ready(expected_length):
    if not captured_message_payloads:
        return False
    return len(captured_message_payloads[-1].get("messages", [])) >= expected_length


def _get_timeout_error(expected_length):
    if captured_message_payloads:
        messages = captured_message_payloads[-1].get("messages", [])
        return TimeoutError(
            f"Timed out. Expected {expected_length} messages, got {len(messages)}"
        )
    return TimeoutError("Timed out. No payloads captured at all.")


def wait_for_payload(expected_length=2, timeout=3.0):
    start = time.time()
    while time.time() - start < timeout:
        if _is_payload_ready(expected_length):
            return
        time.sleep(0.05)
    raise _get_timeout_error(expected_length)


async def async_wait_for_payload(expected_length=2, timeout=3.0):
    start = time.time()
    while time.time() - start < timeout:
        if _is_payload_ready(expected_length):
            return
        await asyncio.sleep(0.05)
    raise _get_timeout_error(expected_length)


def assert_payload_is_valid(
    expected_content,
    entity_id,
    process_id,
    expected_provider,
    expected_history_length=2,
):
    """Automatically grabs the latest payloads and validates them."""
    msg_payload = captured_message_payloads[-1] if captured_message_payloads else None
    aug_payload = (
        captured_augmentation_payloads[-1] if captured_augmentation_payloads else None
    )
    recall_payload = captured_recall_payloads[-1] if captured_recall_payloads else None

    assert msg_payload is not None, "Missing messages payload"
    messages = msg_payload.get("messages", [])
    assert len(messages) >= expected_history_length, (
        f"Expected {expected_history_length} messages, got {len(messages)}."
    )

    user_message = next(
        (m for m in reversed(messages) if m.get("role") == "user"), None
    )
    assert user_message is not None, "Failed to capture user query in message payload"

    assistant_message = next(
        (m for m in reversed(messages) if m.get("role") in ("assistant", "model")), None
    )
    assert assistant_message is not None, (
        "Failed to capture LLM response in message payload"
    )
    assert expected_content.lower() in assistant_message.get("text", "").lower(), (
        f"Content mismatch: expected '{expected_content}'"
    )

    msg_attr = msg_payload.get("attribution", {})
    assert msg_attr.get("entity", {}).get("id") == entity_id, (
        "Entity ID mismatch in messages"
    )
    assert msg_attr.get("process", {}).get("id") == process_id, (
        "Process ID mismatch in messages"
    )

    if aug_payload:
        provider = (
            aug_payload.get("meta", {}).get("llm", {}).get("model", {}).get("provider")
        )
        assert provider == expected_provider, (
            f"Provider mismatch. Expected '{expected_provider}', Got: {provider}"
        )

    if recall_payload:
        assert (
            recall_payload.get("attribution", {}).get("entity", {}).get("id")
            == entity_id
        ), "Entity ID mismatch in recall"
