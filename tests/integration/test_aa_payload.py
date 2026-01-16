import asyncio
import os
import time

import pytest

from tests.integration.conftest import requires_openai

os.environ.setdefault("MEMORI_TEST_MODE", "1")

MODEL = "gpt-4o-mini"
MAX_TOKENS = 50
TEST_PROMPT = "Say 'hello' in one word."


class TestSyncAAIntegration:
    @requires_openai
    @pytest.mark.integration
    def test_sync_call_triggers_aa_pipeline(self, memori_instance, openai_api_key):
        from openai import OpenAI

        mem = memori_instance

        client = OpenAI(api_key=openai_api_key)
        mem.llm.register(client)
        mem.attribution(entity_id="test-user", process_id="test-process")

        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": TEST_PROMPT}],
            max_tokens=MAX_TOKENS,
        )

        assert response is not None
        assert response.choices[0].message.content is not None

        mem.config.augmentation.wait(timeout=5.0)

    @requires_openai
    @pytest.mark.integration
    def test_sync_streaming_triggers_aa_pipeline(self, memori_instance, openai_api_key):
        from openai import OpenAI

        mem = memori_instance

        client = OpenAI(api_key=openai_api_key)
        mem.llm.register(client)
        mem.attribution(entity_id="stream-user", process_id="stream-process")

        stream = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": TEST_PROMPT}],
            max_tokens=MAX_TOKENS,
            stream=True,
        )

        chunks = []
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                chunks.append(chunk.choices[0].delta.content)

        assert len(chunks) > 0

        mem.config.augmentation.wait(timeout=5.0)

    @requires_openai
    @pytest.mark.integration
    def test_multi_turn_conversation_triggers_aa(self, memori_instance, openai_api_key):
        from openai import OpenAI

        mem = memori_instance

        client = OpenAI(api_key=openai_api_key)
        mem.llm.register(client)
        mem.attribution(entity_id="multi-turn-user", process_id="multi-turn-proc")

        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "My name is Alice."},
            {"role": "assistant", "content": "Nice to meet you, Alice!"},
            {"role": "user", "content": "What is my name?"},
        ]

        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            max_tokens=MAX_TOKENS,
        )

        assert response is not None
        mem.config.augmentation.wait(timeout=5.0)


class TestAsyncAAIntegration:
    @requires_openai
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_async_call_triggers_aa_pipeline(
        self, memori_instance, openai_api_key
    ):
        from openai import AsyncOpenAI

        mem = memori_instance

        client = AsyncOpenAI(api_key=openai_api_key)
        mem.llm.register(client)
        mem.attribution(entity_id="async-user", process_id="async-process")

        response = await client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": TEST_PROMPT}],
            max_tokens=MAX_TOKENS,
        )

        assert response is not None
        assert response.choices[0].message.content is not None

        await asyncio.sleep(0.5)
        mem.config.augmentation.wait(timeout=5.0)

    @requires_openai
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_async_streaming_triggers_aa_pipeline(
        self, memori_instance, openai_api_key
    ):
        from openai import AsyncOpenAI

        mem = memori_instance

        client = AsyncOpenAI(api_key=openai_api_key)
        mem.llm.register(client)
        mem.attribution(entity_id="async-stream-user", process_id="async-stream-proc")

        stream = await client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": TEST_PROMPT}],
            max_tokens=MAX_TOKENS,
            stream=True,
        )

        chunks = []
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                chunks.append(chunk.choices[0].delta.content)

        assert len(chunks) > 0

        await asyncio.sleep(0.5)
        mem.config.augmentation.wait(timeout=5.0)


class TestAAEdgeCases:
    @requires_openai
    @pytest.mark.integration
    def test_no_aa_without_attribution(self, memori_instance, openai_api_key):
        from openai import OpenAI

        mem = memori_instance

        client = OpenAI(api_key=openai_api_key)
        mem.llm.register(client)

        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": TEST_PROMPT}],
            max_tokens=MAX_TOKENS,
        )

        assert response is not None
        time.sleep(0.5)

    @requires_openai
    @pytest.mark.integration
    def test_aa_with_entity_only(self, memori_instance, openai_api_key):
        from openai import OpenAI

        mem = memori_instance

        client = OpenAI(api_key=openai_api_key)
        mem.llm.register(client)
        mem.attribution(entity_id="entity-only-user")

        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": TEST_PROMPT}],
            max_tokens=MAX_TOKENS,
        )

        assert response is not None
        mem.config.augmentation.wait(timeout=5.0)

    @requires_openai
    @pytest.mark.integration
    def test_multiple_calls_same_session(self, memori_instance, openai_api_key):
        from openai import OpenAI

        mem = memori_instance

        client = OpenAI(api_key=openai_api_key)
        mem.llm.register(client)
        mem.attribution(entity_id="multi-call-user", process_id="multi-call-proc")

        for i in range(3):
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": f"Say the number {i}"}],
                max_tokens=MAX_TOKENS,
            )
            assert response is not None

        mem.config.augmentation.wait(timeout=10.0)


class TestTestModeConfiguration:
    def test_memori_test_mode_is_enabled(self):
        assert os.environ.get("MEMORI_TEST_MODE") == "1"

    @requires_openai
    @pytest.mark.integration
    def test_memori_instance_in_test_mode(self, memori_instance):
        assert os.environ.get("MEMORI_TEST_MODE") is not None
        assert memori_instance is not None
        assert memori_instance.config is not None
