"""Synchronous streaming registration must preserve the SDK iterator contract."""

import inspect
import json
from contextlib import nullcontext

import httpx
import pytest
from openai import OpenAI

from memori._config import Config
from memori.llm.clients import OpenAi


@pytest.mark.parametrize("use_context_manager", [False, True])
def test_registered_sync_stream_records_after_consumption(mocker, use_context_manager):
    requests = []
    responses = []
    chunks = [
        {"index": 0, "delta": {"role": "assistant", "content": "Hello"}},
        {"index": 0, "delta": {"content": " world"}},
    ]
    body = (
        "".join(
            "data: "
            + json.dumps(
                {
                    "id": "chat-test",
                    "object": "chat.completion.chunk",
                    "created": 0,
                    "model": "test-model",
                    "choices": [chunk],
                }
            )
            + "\n\n"
            for chunk in chunks
        )
        + "data: [DONE]\n\n"
    )

    def respond(request):
        requests.append(json.loads(request.content))
        response = httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=httpx.ByteStream(body.encode()),
        )
        responses.append(response)
        return response

    config = Config()
    config.cloud = False
    record = mocker.patch("memori.memory._manager.Manager.execute")
    with OpenAI(
        api_key="test-key",
        http_client=httpx.Client(transport=httpx.MockTransport(respond)),
    ) as client:
        OpenAi(config).register(client, stream=True)
        stream = client.chat.completions.create(
            model="test-model",
            messages=[{"role": "user", "content": "Say hello"}],
            stream=True,
        )
        if inspect.iscoroutine(stream):
            stream.close()
        assert not inspect.isawaitable(stream)
        assert not responses[0].is_closed
        record.assert_not_called()
        with stream if use_context_manager else nullcontext(stream) as iterator:
            first = next(iterator)
            record.assert_not_called()
            assert first.choices[0].delta.content == "Hello"
            assert [chunk.choices[0].delta.content for chunk in iterator] == [" world"]
        record.assert_called_once()
        payload = record.call_args.args[0]
        assert payload["messages"] == [
            {"role": "user", "text": "Say hello", "type": None},
            {"role": "assistant", "text": "Hello world", "type": "text"},
        ]
        assert requests[0]["stream_options"] == {"include_usage": True}
        assert responses[0].is_closed


def test_registered_sync_stream_propagates_request_error(mocker):
    from openai import APIConnectionError

    def fail(request):
        raise httpx.ConnectError("offline fixture", request=request)

    config = Config()
    config.cloud = False
    record = mocker.patch("memori.memory._manager.Manager.execute")
    with OpenAI(
        api_key="test-key",
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(fail)),
    ) as client:
        OpenAi(config).register(client, stream=True)
        with pytest.raises(APIConnectionError):
            result = client.chat.completions.create(
                model="test-model", messages=[], stream=True
            )
            if inspect.iscoroutine(result):
                result.close()
        record.assert_not_called()
