import os

import pytest
from anthropic import Anthropic, AsyncAnthropic

from .cloud_helpers import (
    assert_payload_is_valid,
    async_wait_for_payload,
    inject_recall_fact,
    wait_for_payload,
)

pytestmark = pytest.mark.skipif(
    os.getenv("TEST_ENV") != "integration"
    or not os.getenv("ANTHROPIC_API_KEY")
    or not os.getenv("MEMORI_API_KEY"),
    reason="Skipping cloud integration tests. Set TEST_ENV=integration and ensure API keys are present.",
)

MODEL = "claude-haiku-4-5"
TEST_PROMPT = "Say 'hello' in one word."
PROVIDER = "anthropic"


def test_sync(memori_setup):
    memori_client, e_id, p_id = memori_setup
    anthropic_client = Anthropic()
    memori_client.llm.register(client=anthropic_client)

    response = anthropic_client.messages.create(
        model=MODEL, max_tokens=100, messages=[{"role": "user", "content": TEST_PROMPT}]
    )
    wait_for_payload()
    assert_payload_is_valid(response.content[0].text, e_id, p_id, PROVIDER)


@pytest.mark.asyncio
async def test_async(memori_setup):
    memori_client, e_id, p_id = memori_setup
    anthropic_client = AsyncAnthropic()
    memori_client.llm.register(client=anthropic_client)

    response = await anthropic_client.messages.create(
        model=MODEL, max_tokens=100, messages=[{"role": "user", "content": TEST_PROMPT}]
    )
    await async_wait_for_payload()
    assert_payload_is_valid(response.content[0].text, e_id, p_id, PROVIDER)
    await anthropic_client.close()


def test_multi_turn(memori_setup):
    memori_client, e_id, p_id = memori_setup
    anthropic_client = Anthropic()
    memori_client.llm.register(client=anthropic_client)

    anthropic_client.messages.create(
        model=MODEL,
        max_tokens=100,
        messages=[{"role": "user", "content": "My favorite color is blue."}],
    )
    wait_for_payload(expected_length=2)

    response2 = anthropic_client.messages.create(
        model=MODEL,
        max_tokens=100,
        messages=[{"role": "user", "content": "What is my favorite color?"}],
    )
    wait_for_payload(expected_length=2)

    content = response2.content[0].text
    assert "blue" in content.lower(), "LLM forgot context - history injection failed"
    assert_payload_is_valid(content, e_id, p_id, PROVIDER, expected_history_length=2)


def test_recall_fact_injection(memori_setup):
    memori_client, e_id, p_id = memori_setup
    anthropic_client = Anthropic()
    memori_client.llm.register(client=anthropic_client)

    with inject_recall_fact("The user's favorite word is 'MEMORI_42'."):
        response = anthropic_client.messages.create(
            model=MODEL,
            max_tokens=100,
            messages=[{"role": "user", "content": "What is my favorite word?"}],
        )

    wait_for_payload()
    content = response.content[0].text
    assert "MEMORI_42" in content, f"Wrapper failed to inject fact! LLM said: {content}"
    assert_payload_is_valid("MEMORI_42", e_id, p_id, PROVIDER)
