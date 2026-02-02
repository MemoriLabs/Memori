r"""
 __  __                           _
|  \/  | ___ _ __ ___   ___  _ __(_)
| |\/| |/ _ \ '_ ` _ \ / _ \| '__| |
| |  | |  __/ | | | | | (_) | |  | |
|_|  |_|\___|_| |_| |_|\___/|_|  |_|
                  perfectam memoriam
                       memorilabs.ai
"""

from collections.abc import Callable
from typing import Any

from memori._exceptions import UnsupportedLLMProviderError
from memori.llm._base import BaseClient, BaseLlmAdaptor


class Registry:
    _clients: dict[Callable[[Any], bool], type[BaseClient]] = {}
    _adapters: dict[Callable[[str | None, str | None], bool], type[BaseLlmAdaptor]] = {}

    @classmethod
    def register_client(cls, matcher: Callable[[Any], bool]):
        def decorator(client_class: type[BaseClient]):
            cls._clients[matcher] = client_class
            return client_class

        return decorator

    @classmethod
    def register_adapter(cls, matcher: Callable[[str | None, str | None], bool]):
        def decorator(adapter_class: type[BaseLlmAdaptor]):
            cls._adapters[matcher] = adapter_class
            return adapter_class

        return decorator

    def client(self, client_obj: Any, config) -> BaseClient:
        for matcher, client_class in self._clients.items():
            if matcher(client_obj):
                return client_class(config)

        module = type(client_obj).__module__
        if module.startswith("langchain"):
            class_name = type(client_obj).__name__
            param_hint = class_name.lower()
            raise RuntimeError(
                f"LangChain models require named parameters. "
                f"Use: llm.register({param_hint}=client) instead of llm.register(client)"
            )

        provider = f"{type(client_obj).__module__}.{type(client_obj).__name__}"
        raise UnsupportedLLMProviderError(provider)

    def adapter(self, provider: str | None, title: str | None) -> BaseLlmAdaptor:
        for matcher, adapter_class in self._adapters.items():
            if matcher(provider, title):
                return adapter_class()

        provider_str = f"framework={provider}, llm={title}"
        raise UnsupportedLLMProviderError(provider_str)


def register_llm(
    memori,
    client=None,
    openai_chat=None,
    claude=None,
    gemini=None,
    xai=None,
    chatbedrock=None,
    chatgooglegenai=None,
    chatopenai=None,
    chatvertexai=None,
):
    """Register LLM clients or framework models.

    For direct LLM clients:
        llm.register(client)

    For Agno models:
        llm.register(openai_chat=model)
        llm.register(claude=model)
        llm.register(gemini=model)
        llm.register(xai=model)

    For LangChain models:
        llm.register(chatbedrock=model)
        llm.register(chatgooglegenai=model)
        llm.register(chatopenai=model)
        llm.register(chatvertexai=model)
    """
    agno_args = [openai_chat, claude, gemini, xai]
    langchain_args = [chatbedrock, chatgooglegenai, chatopenai, chatvertexai]

    has_agno = any(arg is not None for arg in agno_args)
    has_langchain = any(arg is not None for arg in langchain_args)

    if client is not None and (has_agno or has_langchain):
        raise RuntimeError(
            "Cannot mix direct client registration with framework registration"
        )

    if has_agno and has_langchain:
        raise RuntimeError(
            "Cannot register both Agno and LangChain clients in the same call"
        )

    def _first_provider(*pairs: tuple[object | None, str]) -> str | None:
        return next((provider for value, provider in pairs if value is not None), None)

    if has_agno:
        from memori.llm._constants import (
            AGNO_ANTHROPIC_LLM_PROVIDER,
            AGNO_FRAMEWORK_PROVIDER,
            AGNO_GOOGLE_LLM_PROVIDER,
            AGNO_OPENAI_LLM_PROVIDER,
            AGNO_XAI_LLM_PROVIDER,
        )

        memori.config.framework.provider = AGNO_FRAMEWORK_PROVIDER
        memori.config.llm.provider = _first_provider(
            (openai_chat, AGNO_OPENAI_LLM_PROVIDER),
            (claude, AGNO_ANTHROPIC_LLM_PROVIDER),
            (gemini, AGNO_GOOGLE_LLM_PROVIDER),
            (xai, AGNO_XAI_LLM_PROVIDER),
        )

        memori.agno.register(
            openai_chat=openai_chat,
            claude=claude,
            gemini=gemini,
            xai=xai,
        )
    elif has_langchain:
        from memori.llm._constants import (
            LANGCHAIN_CHATBEDROCK_LLM_PROVIDER,
            LANGCHAIN_CHATGOOGLEGENAI_LLM_PROVIDER,
            LANGCHAIN_CHATVERTEXAI_LLM_PROVIDER,
            LANGCHAIN_FRAMEWORK_PROVIDER,
            LANGCHAIN_OPENAI_LLM_PROVIDER,
        )

        memori.config.framework.provider = LANGCHAIN_FRAMEWORK_PROVIDER
        memori.config.llm.provider = _first_provider(
            (chatbedrock, LANGCHAIN_CHATBEDROCK_LLM_PROVIDER),
            (chatgooglegenai, LANGCHAIN_CHATGOOGLEGENAI_LLM_PROVIDER),
            (chatopenai, LANGCHAIN_OPENAI_LLM_PROVIDER),
            (chatvertexai, LANGCHAIN_CHATVERTEXAI_LLM_PROVIDER),
        )

        memori.langchain.register(
            chatbedrock=chatbedrock,
            chatgooglegenai=chatgooglegenai,
            chatopenai=chatopenai,
            chatvertexai=chatvertexai,
        )
    elif client is not None:
        client_handler = Registry().client(client, memori.config)
        client_handler.register(client)
    else:
        raise RuntimeError("No client or framework model provided to register")

    provider = getattr(memori.config.llm, "provider", None)
    if provider is None or (isinstance(provider, str) and provider.strip() == ""):
        raise UnsupportedLLMProviderError(
            "unknown (provider could not be determined during registration)"
        )

    return memori
