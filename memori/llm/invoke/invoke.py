import inspect
import logging
import time
from collections.abc import AsyncIterator, Iterator

from botocore.eventstream import EventStream
from grpc.experimental.aio import UnaryStreamCall

from memori._logging import truncate
from memori._utils import merge_chunk
from memori.llm._base import BaseInvoke
from memori.llm._utils import client_is_bedrock
from memori.llm.invoke.iterable import Iterable as MemoriIterable
from memori.llm.invoke.iterator import AsyncIterator as MemoriAsyncIterator
from memori.llm.invoke.iterator import Iterator as MemoriIterator
from memori.llm.invoke.streaming import StreamingBody as MemoriStreamingBody
from memori.llm.pipelines.conversation_injection import inject_conversation_messages
from memori.llm.pipelines.post_invoke import handle_post_response
from memori.llm.pipelines.recall_injection import inject_recalled_facts

logger = logging.getLogger(__name__)


class Invoke(BaseInvoke):
    def invoke(self, **kwargs):
        start = time.time()

        kwargs = inject_conversation_messages(
            self,
            inject_recalled_facts(self, self.configure_for_streaming_usage(kwargs)),
        )

        logger.debug(
            "Sending request to LLM - provider: %s, model: %s",
            self.config.llm.provider,
            truncate(str(kwargs.get("model", "unknown")), 100),
        )
        raw_response = self._method(**kwargs)

        if isinstance(raw_response, Iterator) or inspect.isgenerator(raw_response):
            return (
                MemoriIterator(self.config, raw_response)
                .configure_invoke(self)
                .configure_request(kwargs, start)
            )
        elif client_is_bedrock(
            self.config.framework.provider, self.config.llm.provider
        ):
            if isinstance(raw_response["body"], EventStream):
                raw_response["body"] = (
                    MemoriIterable(self.config, raw_response["body"])
                    .configure_invoke(self)
                    .configure_request(kwargs, start)
                )
            else:
                raw_response["body"] = (
                    MemoriStreamingBody(self.config, raw_response["body"])
                    .configure_invoke(self)
                    .configure_request(kwargs, start)
                )

            return raw_response
        else:
            handle_post_response(self, kwargs, start, raw_response)
            return raw_response


class InvokeAsync(BaseInvoke):
    async def invoke(self, **kwargs):
        start = time.time()

        kwargs = inject_conversation_messages(
            self,
            inject_recalled_facts(self, self.configure_for_streaming_usage(kwargs)),
        )

        logger.debug(
            "Sending async request to LLM - provider: %s, model: %s",
            self.config.llm.provider,
            truncate(str(kwargs.get("model", "unknown")), 100),
        )
        raw_response = await self._method(**kwargs)
        if (
            isinstance(raw_response, AsyncIterator)
            or hasattr(raw_response, "__aiter__")
            or isinstance(raw_response, UnaryStreamCall)
        ):
            return (
                MemoriAsyncIterator(self.config, raw_response)
                .configure_invoke(self)
                .configure_request(kwargs, start)
            )
        else:
            handle_post_response(self, kwargs, start, raw_response)
            return raw_response


class InvokeAsyncIterator(BaseInvoke):
    async def invoke(self, **kwargs):
        start = time.time()

        kwargs = inject_conversation_messages(
            self,
            inject_recalled_facts(self, self.configure_for_streaming_usage(kwargs)),
        )

        raw_response = await self._method(**kwargs)
        if (
            isinstance(raw_response, AsyncIterator)
            or hasattr(raw_response, "__aiter__")
            or isinstance(raw_response, UnaryStreamCall)
        ):
            return (
                MemoriAsyncIterator(self.config, raw_response)
                .configure_invoke(self)
                .configure_request(kwargs, start)
            )
        else:
            handle_post_response(self, kwargs, start, raw_response)
            return raw_response


class InvokeAsyncStream(BaseInvoke):
    async def invoke(self, **kwargs):
        start = time.time()

        kwargs = inject_conversation_messages(
            self,
            inject_recalled_facts(self, self.configure_for_streaming_usage(kwargs)),
        )

        stream = await self._method(**kwargs)

        raw_response = {}
        async for chunk in stream:
            raw_response = merge_chunk(raw_response, chunk.__dict__)
            yield chunk

        handle_post_response(self, kwargs, start, raw_response)


class InvokeStream(BaseInvoke):
    async def invoke(self, **kwargs):
        start = time.time()

        kwargs = inject_conversation_messages(
            self,
            inject_recalled_facts(self, self.configure_for_streaming_usage(kwargs)),
        )

        raw_response = await self._method(**kwargs)

        handle_post_response(self, kwargs, start, raw_response)
        return raw_response
