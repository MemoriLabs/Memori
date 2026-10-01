from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from openai.types.chat import ChatCompletion

from memori._config import Config
from memori.llm.invoke.invoke import Invoke, InvokeAsync
from memori.llm.pipelines.post_invoke import format_augmentation_input


@pytest.mark.parametrize("stream_kwargs", [{}, {"stream": None}, {"stream": False}])
@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.asyncio
async def test_nonstreamed_turn_preserves_assistant_in_memory_and_augmentation(
    stream_kwargs, is_async
):
    config = Config()
    config.storage = None
    config.augmentation = None
    response = ChatCompletion(
        id="chatcmpl-local",
        created=0,
        model="local-model",
        object="chat.completion",
        choices=[
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": "Use a vector index."},
            }
        ],
    )
    method = (
        AsyncMock(return_value=response)
        if is_async
        else MagicMock(return_value=response)
    )
    wrapper = (InvokeAsync if is_async else Invoke)(config, method)
    wrapper.set_client(None, "openai", "test")
    kwargs = {
        "model": "local-model",
        "messages": [{"role": "user", "content": "How should I index documents?"}],
        **stream_kwargs,
    }
    with patch("memori.memory._manager.Manager.execute") as ingest:
        result = (
            await wrapper.invoke(**kwargs) if is_async else wrapper.invoke(**kwargs)
        )

    assert result is response
    method.assert_called_once_with(**kwargs)
    if is_async:
        method.assert_awaited_once()
    ingest.assert_called_once()
    payload = ingest.call_args.args[0]
    assert payload["messages"] == [
        {"role": "user", "type": None, "text": "How should I index documents?"},
        {"role": "assistant", "type": "text", "text": "Use a vector index."},
    ]
    augmentation = format_augmentation_input(wrapper, payload)
    assert [(message.role, message.content) for message in augmentation.messages] == [
        ("user", "How should I index documents?"),
        ("assistant", "Use a vector index."),
    ]
