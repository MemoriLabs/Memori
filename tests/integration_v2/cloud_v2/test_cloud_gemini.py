import os

import pytest
from google import genai

from .cloud_helpers import (
    assert_payload_is_valid,
    async_wait_for_payload,
    inject_recall_fact,
    wait_for_payload,
)

pytestmark = pytest.mark.skipif(
    os.getenv("TEST_ENV") != "integration"
    or not os.getenv("GOOGLE_API_KEY")
    or not os.getenv("MEMORI_API_KEY"),
    reason="Skipping cloud integration tests. Set TEST_ENV=integration and ensure API keys are present.",
)

MODEL = "gemini-2.5-flash"
TEST_PROMPT = "Say 'hello' in one word."
PROVIDER = "google"


def test_sync(memori_setup):
    memori_client, e_id, p_id = memori_setup
    gemini_client = genai.Client()
    memori_client.llm.register(client=gemini_client)

    response = gemini_client.models.generate_content(model=MODEL, contents=TEST_PROMPT)
    wait_for_payload()
    assert_payload_is_valid(response.text, e_id, p_id, PROVIDER)


def test_streaming(memori_setup):
    memori_client, e_id, p_id = memori_setup
    gemini_client = genai.Client()
    memori_client.llm.register(client=gemini_client)

    stream = gemini_client.models.generate_content_stream(
        model=MODEL, contents=TEST_PROMPT
    )
    full_content = "".join(chunk.text for chunk in stream)
    wait_for_payload()
    assert_payload_is_valid(full_content, e_id, p_id, PROVIDER)


@pytest.mark.asyncio
async def test_async(memori_setup):
    memori_client, e_id, p_id = memori_setup
    gemini_client = genai.Client()
    memori_client.llm.register(client=gemini_client)

    response = await gemini_client.aio.models.generate_content(
        model=MODEL, contents=TEST_PROMPT
    )
    await async_wait_for_payload()
    assert_payload_is_valid(response.text, e_id, p_id, PROVIDER)


@pytest.mark.asyncio
async def test_async_streaming(memori_setup):
    memori_client, e_id, p_id = memori_setup
    gemini_client = genai.Client()
    memori_client.llm.register(client=gemini_client)

    stream = await gemini_client.aio.models.generate_content_stream(
        model=MODEL, contents=TEST_PROMPT
    )
    content_parts = []
    async for chunk in stream:
        content_parts.append(chunk.text)

    await async_wait_for_payload()
    assert_payload_is_valid("".join(content_parts), e_id, p_id, PROVIDER)


def test_multi_turn(memori_setup):
    memori_client, e_id, p_id = memori_setup
    gemini_client = genai.Client()
    memori_client.llm.register(client=gemini_client)

    gemini_client.models.generate_content(
        model=MODEL, contents="My favorite color is blue."
    )
    wait_for_payload(expected_length=2)

    response2 = gemini_client.models.generate_content(
        model=MODEL, contents="What is my favorite color?"
    )
    wait_for_payload(expected_length=2)

    content = response2.text
    assert "blue" in content.lower(), "LLM forgot context - history injection failed"
    assert_payload_is_valid(content, e_id, p_id, PROVIDER, expected_history_length=2)


def test_recall_fact_injection(memori_setup):
    memori_client, e_id, p_id = memori_setup
    gemini_client = genai.Client()
    memori_client.llm.register(client=gemini_client)

    with inject_recall_fact("The user's favorite word is 'MEMORI_42'."):
        response = gemini_client.models.generate_content(
            model=MODEL, contents="What is my favorite word?"
        )

    wait_for_payload()
    content = response.text
    assert "MEMORI_42" in content, f"Wrapper failed to inject fact! LLM said: {content}"
    assert_payload_is_valid("MEMORI_42", e_id, p_id, PROVIDER)
