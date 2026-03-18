import os

import pytest
from openai import AsyncOpenAI, OpenAI

from .cloud_helpers import (
    assert_payload_is_valid,
    async_wait_for_payload,
    inject_recall_fact,
    wait_for_payload,
)

pytestmark = pytest.mark.skipif(
    os.getenv("TEST_ENV") != "integration"
    or not os.getenv("OPENAI_API_KEY")
    or not os.getenv("MEMORI_API_KEY"),
    reason="Skipping cloud integration tests. Set TEST_ENV=integration and ensure API keys are present.",
)

MODEL = "gpt-4o-mini"
TEST_PROMPT = "Say 'hello' in one word."
PROVIDER = "openai"


def test_sync(memori_setup):
    memori_client, e_id, p_id = memori_setup
    openai_client = OpenAI()
    memori_client.llm.register(client=openai_client)

    response = openai_client.chat.completions.create(
        model=MODEL, messages=[{"role": "user", "content": TEST_PROMPT}]
    )
    wait_for_payload()
    assert_payload_is_valid(response.choices[0].message.content, e_id, p_id, PROVIDER)


def test_streaming(memori_setup):
    memori_client, e_id, p_id = memori_setup
    openai_client = OpenAI()
    memori_client.llm.register(client=openai_client)

    stream = openai_client.chat.completions.create(
        model=MODEL, messages=[{"role": "user", "content": TEST_PROMPT}], stream=True
    )
    full_content = "".join(
        chunk.choices[0].delta.content
        for chunk in stream
        if chunk.choices and chunk.choices[0].delta.content
    )
    wait_for_payload()
    assert_payload_is_valid(full_content, e_id, p_id, PROVIDER)


@pytest.mark.asyncio
async def test_async(memori_setup):
    memori_client, e_id, p_id = memori_setup
    openai_client = AsyncOpenAI()
    memori_client.llm.register(client=openai_client)

    response = await openai_client.chat.completions.create(
        model=MODEL, messages=[{"role": "user", "content": TEST_PROMPT}]
    )
    await async_wait_for_payload()
    assert_payload_is_valid(response.choices[0].message.content, e_id, p_id, PROVIDER)
    await openai_client.close()


@pytest.mark.asyncio
async def test_async_streaming(memori_setup):
    memori_client, e_id, p_id = memori_setup
    openai_client = AsyncOpenAI()
    memori_client.llm.register(client=openai_client)

    stream = await openai_client.chat.completions.create(
        model=MODEL, messages=[{"role": "user", "content": TEST_PROMPT}], stream=True
    )
    content_parts = []
    async for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            content_parts.append(chunk.choices[0].delta.content)

    await async_wait_for_payload()
    assert_payload_is_valid("".join(content_parts), e_id, p_id, PROVIDER)
    await openai_client.close()


def test_multi_turn(memori_setup):
    memori_client, e_id, p_id = memori_setup
    openai_client = OpenAI()
    memori_client.llm.register(client=openai_client)

    openai_client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "My favorite color is blue."}],
    )
    wait_for_payload(expected_length=2)

    response2 = openai_client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "What is my favorite color?"}],
    )
    wait_for_payload(expected_length=2)

    content = response2.choices[0].message.content
    assert "blue" in content.lower(), "LLM forgot context - history injection failed"
    assert_payload_is_valid(content, e_id, p_id, PROVIDER, expected_history_length=2)


def test_recall_fact_injection(memori_setup):
    memori_client, e_id, p_id = memori_setup
    openai_client = OpenAI()
    memori_client.llm.register(client=openai_client)

    with inject_recall_fact("The user's favorite word is 'MEMORI_42'."):
        response = openai_client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": "What is my favorite word?"}],
        )

    wait_for_payload()
    content = response.choices[0].message.content
    assert "MEMORI_42" in content, f"Wrapper failed to inject fact! LLM said: {content}"
    assert_payload_is_valid("MEMORI_42", e_id, p_id, PROVIDER)
