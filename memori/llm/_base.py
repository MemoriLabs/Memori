r"""
 __  __                           _
|  \/  | ___ _ __ ___   ___  _ __(_)
| |\/| |/ _ \ '_ ` _ \ / _ \| '__| |
| |  | |  __/ | | | | | (_) | |  | |
|_|  |_|\___|_| |_| |_|\___/|_|  |_|
                  perfectam memoriam
                       memorilabs.ai
"""

import copy
import inspect
import json

from google.protobuf import json_format

from memori._config import Config
from memori._utils import merge_chunk
from memori.llm._utils import agno_is_openai, llm_is_openai, llm_is_xai
from memori.llm.helpers.serialization import convert_to_json


class BaseClient:
    def __init__(self, config: Config):
        self.config = config
        self.stream = False

    def register(self, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement register()")

    def _wrap_method(
        self,
        obj,
        method_name,
        backup_obj,
        backup_attr,
        provider,
        llm_provider,
        version,
        stream=False,
    ):
        """Helper to wrap a method with the appropriate Invoke wrapper.

        Automatically detects async context and chooses the correct wrapper class.

        Args:
            obj: The object containing the method to wrap (e.g., client.chat.completions)
            method_name: Name of the method to wrap (e.g., 'create')
            backup_obj: The object where backup is stored (e.g., client.chat)
            backup_attr: Name of backup attribute where original is stored (e.g., '_completions_create')
            provider: Framework provider name
            llm_provider: LLM provider name
            version: Provider SDK version
            stream: Whether to use streaming wrappers
        """
        from memori.llm.invoke.invoke import (
            Invoke,
            InvokeAsync,
            InvokeAsyncStream,
            InvokeStream,
        )

        original = getattr(backup_obj, backup_attr)

        is_async = inspect.iscoroutinefunction(original) or type(
            obj
        ).__name__.startswith("Async")

        if is_async:
            wrapper_class = InvokeAsyncStream if stream else InvokeAsync
        else:
            wrapper_class = InvokeStream if stream else Invoke

        setattr(
            obj,
            method_name,
            wrapper_class(self.config, original)
            .set_client(provider, llm_provider, version)
            .invoke,
        )


class BaseInvoke:
    def __init__(self, config: Config, method):
        self.config = config
        self._method = method
        self._uses_protobuf = False
        self._injected_message_count = 0
        self._cloud_conversation_messages: list[dict[str, str]] = []
        self._cloud_summaries: list[dict[str, object]] = []

    def _ensure_cached_conversation_id(self) -> bool:
        if self.config.storage is None or self.config.storage.driver is None:
            return False

        if self.config.session_id is None:
            return False

        driver = self.config.storage.driver

        if self.config.cache.session_id is None:
            if not hasattr(driver.session, "read"):
                return False
            self.config.cache.session_id = driver.session.read(
                str(self.config.session_id)
            )

        if self.config.cache.session_id is None:
            return False

        if self.config.cache.conversation_id is None:
            if not hasattr(driver.conversation, "read_id_by_session_id"):
                return False
            self.config.cache.conversation_id = (
                driver.conversation.read_id_by_session_id(self.config.cache.session_id)
            )

        return self.config.cache.conversation_id is not None

    def configure_for_streaming_usage(self, kwargs: dict) -> dict:
        if (
            llm_is_openai(self.config.framework.provider, self.config.llm.provider)
            or llm_is_xai(self.config.framework.provider, self.config.llm.provider)
            or agno_is_openai(self.config.framework.provider, self.config.llm.provider)
        ):
            is_responses_api = (
                "input" in kwargs or "instructions" in kwargs
            ) and "messages" not in kwargs

            if kwargs.get("stream", None) and not is_responses_api:
                stream_options = kwargs.get("stream_options", None)
                if stream_options is None or not isinstance(stream_options, dict):
                    kwargs["stream_options"] = {}

                kwargs["stream_options"]["include_usage"] = True

        return kwargs

    def set_client(self, framework_provider, llm_provider, provider_sdk_version):
        self.config.framework.provider = framework_provider
        self.config.llm.provider = llm_provider
        self.config.llm.provider_sdk_version = provider_sdk_version
        return self

    def uses_protobuf(self):
        self._uses_protobuf = True
        return self


class BaseIterator:
    def __init__(self, config: Config, source_iterator):
        self.config = config
        self.source_iterator = source_iterator
        self.iterator = None
        self.raw_response: dict | list | None = None

    def configure_invoke(self, invoke: BaseInvoke):
        self.invoke = invoke
        return self

    def configure_request(self, kwargs, time_start):
        self._kwargs = kwargs
        self._time_start = time_start
        return self

    def process_chunk(self, chunk):
        if hasattr(chunk, "type") and chunk.type == "response.completed":
            if hasattr(chunk, "response"):
                response = chunk.response
                if hasattr(response, "model_dump"):
                    self.raw_response = response.model_dump()
                else:
                    self.raw_response = convert_to_json(response)
            return self

        if self.invoke._uses_protobuf is True:
            formatted_chunk = copy.deepcopy(chunk)
            if isinstance(self.raw_response, list):
                if "_pb" in formatted_chunk.__dict__:
                    # Old google-generativeai format (protobuf)
                    self.raw_response.append(
                        json.loads(
                            json_format.MessageToJson(formatted_chunk.__dict__["_pb"])
                        )
                    )
                elif "candidates" in formatted_chunk.__dict__:
                    # New google-genai format (dict with candidates)
                    chunk_data = {}
                    if (
                        hasattr(formatted_chunk, "candidates")
                        and formatted_chunk.candidates
                    ):
                        candidates = []
                        for candidate in formatted_chunk.candidates:
                            candidate_data = {}
                            if hasattr(candidate, "content") and candidate.content:
                                content_data = {}
                                if (
                                    hasattr(candidate.content, "parts")
                                    and candidate.content.parts
                                ):
                                    parts = []
                                    for part in candidate.content.parts:
                                        if hasattr(part, "text"):
                                            parts.append({"text": part.text})
                                    content_data["parts"] = parts
                                if hasattr(candidate.content, "role"):
                                    content_data["role"] = candidate.content.role
                                candidate_data["content"] = content_data
                            candidates.append(candidate_data)
                        chunk_data["candidates"] = candidates
                    self.raw_response.append(chunk_data)
        else:
            if isinstance(self.raw_response, dict):
                self.raw_response = merge_chunk(self.raw_response, chunk.__dict__)

        return self

    def set_raw_response(self):
        if self.raw_response is not None:
            return self

        self.raw_response = {}
        if self.invoke._uses_protobuf:
            self.raw_response = []

        return self


class BaseLlmAdaptor:
    def _exclude_injected_messages(self, messages, payload):
        injected_count = (
            payload.get("conversation", {})
            .get("query", {})
            .get("_memori_injected_count", 0)
        )
        return messages[injected_count:]

    def get_formatted_query(self, payload):
        raise NotImplementedError

    def get_formatted_response(self, payload):
        raise NotImplementedError


class BaseProvider:
    def __init__(self, entity):
        self.client = None
        self.entity = entity
        self.config = entity.config
