import json
import logging

from google.protobuf import json_format

from memori.llm._utils import (
    agno_is_anthropic,
    agno_is_google,
    agno_is_openai,
    agno_is_xai,
    llm_is_anthropic,
    llm_is_bedrock,
    llm_is_google,
    llm_is_openai,
    llm_is_xai,
)

logger = logging.getLogger(__name__)


def _normalize_input_history(kwargs: dict, messages: list[dict[str, str]]) -> dict:
    history_items: list[dict[str, str]] = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "system":
            continue
        history_items.append({"role": role, "content": content})

    existing_input = kwargs.get("input")
    if existing_input is None:
        existing_input = []
    elif isinstance(existing_input, str):
        existing_input = [{"role": "user", "content": existing_input}]

    kwargs["input"] = history_items + existing_input
    return kwargs


def _normalize_google_contents(existing_contents):
    if isinstance(existing_contents, str):
        return [{"parts": [{"text": existing_contents}], "role": "user"}]
    if isinstance(existing_contents, list):
        normalized = []
        for item in existing_contents:
            if isinstance(item, str):
                normalized.append({"parts": [{"text": item}], "role": "user"})
            else:
                normalized.append(item)
        return normalized
    return existing_contents


def _inject_messages_by_provider(
    config, kwargs: dict, messages: list[dict[str, str]]
) -> dict:
    if ("input" in kwargs or "instructions" in kwargs) and "messages" not in kwargs:
        return _normalize_input_history(kwargs, messages)

    if (
        llm_is_openai(config.framework.provider, config.llm.provider)
        or agno_is_openai(config.framework.provider, config.llm.provider)
        or agno_is_xai(config.framework.provider, config.llm.provider)
    ):
        kwargs["messages"] = messages + kwargs["messages"]
    elif (
        llm_is_anthropic(config.framework.provider, config.llm.provider)
        or llm_is_bedrock(config.framework.provider, config.llm.provider)
        or agno_is_anthropic(config.framework.provider, config.llm.provider)
    ):
        filtered_messages = [m for m in messages if m.get("role") != "system"]
        kwargs["messages"] = filtered_messages + kwargs["messages"]
    elif llm_is_xai(config.framework.provider, config.llm.provider):
        from xai_sdk.chat import assistant, user

        xai_messages = []
        for message in messages:
            role = message.get("role", "")
            content = message.get("content", "")
            if role == "user":
                xai_messages.append(user(content))
            elif role == "assistant":
                xai_messages.append(assistant(content))

        kwargs["messages"] = xai_messages + kwargs["messages"]
    elif llm_is_google(
        config.framework.provider, config.llm.provider
    ) or agno_is_google(config.framework.provider, config.llm.provider):
        contents = []
        for message in messages:
            role = message["role"]
            if role == "assistant":
                role = "model"
            contents.append({"parts": [{"text": message["content"]}], "role": role})

        if "request" in kwargs:
            formatted_kwargs = json.loads(
                json_format.MessageToJson(kwargs["request"].__dict__["_pb"])
            )
            formatted_kwargs["contents"] = contents + formatted_kwargs["contents"]
            json_format.ParseDict(formatted_kwargs, kwargs["request"].__dict__["_pb"])
        else:
            existing_contents = _normalize_google_contents(kwargs.get("contents", []))
            kwargs["contents"] = contents + existing_contents
    else:
        raise NotImplementedError

    return kwargs


def inject_conversation_messages(invoke, kwargs: dict) -> dict:
    if invoke.config.cloud is True:
        messages = invoke._cloud_conversation_messages
        invoke._injected_message_count = len(messages)

        if not messages:
            return kwargs

        logger.debug(
            "Injecting %d cloud conversation messages from history",
            len(messages),
        )
        return _inject_messages_by_provider(invoke.config, kwargs, messages)

    if invoke.config.storage is None or invoke.config.storage.driver is None:
        return kwargs

    if invoke.config.cache.conversation_id is None:
        if not invoke._ensure_cached_conversation_id():
            if invoke.config.cache.session_id is None:
                if invoke.config.entity_id is not None:
                    entity_id = invoke.config.storage.driver.entity.create(
                        invoke.config.entity_id
                    )
                    if entity_id is not None:
                        invoke.config.cache.entity_id = entity_id
                if invoke.config.process_id is not None:
                    process_id = invoke.config.storage.driver.process.create(
                        invoke.config.process_id
                    )
                    if process_id is not None:
                        invoke.config.cache.process_id = process_id

                session_id = invoke.config.storage.driver.session.create(
                    invoke.config.session_id,
                    invoke.config.cache.entity_id,
                    invoke.config.cache.process_id,
                )
                if session_id is not None:
                    invoke.config.cache.session_id = session_id

            if invoke.config.cache.session_id is not None:
                existing_conv = invoke.config.storage.driver.conversation.create(
                    invoke.config.cache.session_id,
                    invoke.config.session_timeout_minutes,
                )
                if existing_conv is not None:
                    invoke.config.cache.conversation_id = existing_conv

            if (
                invoke.config.cache.conversation_id is None
                and not invoke._ensure_cached_conversation_id()
            ):
                return kwargs

    messages = invoke.config.storage.driver.conversation.messages.read(
        invoke.config.cache.conversation_id
    )
    if not messages:
        return kwargs

    invoke._injected_message_count = len(messages)
    logger.debug("Injecting %d conversation messages from history", len(messages))
    return _inject_messages_by_provider(invoke.config, kwargs, messages)
