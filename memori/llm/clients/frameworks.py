from memori.llm._base import BaseClient
from memori.llm._constants import (
    AGNO_FRAMEWORK_PROVIDER,
    LANGCHAIN_CHATBEDROCK_LLM_PROVIDER,
    LANGCHAIN_CHATGOOGLEGENAI_LLM_PROVIDER,
    LANGCHAIN_CHATVERTEXAI_LLM_PROVIDER,
    LANGCHAIN_FRAMEWORK_PROVIDER,
    LANGCHAIN_OPENAI_LLM_PROVIDER,
    LLAMAINDEX_ANTHROPIC_LLM_PROVIDER,
    LLAMAINDEX_FRAMEWORK_PROVIDER,
    LLAMAINDEX_GOOGLE_LLM_PROVIDER,
    LLAMAINDEX_OPENAI_LLM_PROVIDER,
)
from memori.llm._registry import Registry
from memori.llm._utils import client_is_llamaindex
from memori.llm.clients.direct import Anthropic, Google, OpenAi, XAi
from memori.llm.invoke.invoke import Invoke, InvokeAsync, InvokeAsyncIterator


class LangChain(BaseClient):
    def _wrap_langchain_google_method(
        self, backup_obj, target_obj, backup_attr, method_name, invoke_cls
    ):
        setattr(backup_obj, backup_attr, getattr(target_obj, method_name))
        setattr(
            target_obj,
            method_name,
            invoke_cls(self.config, getattr(backup_obj, backup_attr))
            .set_client(
                LANGCHAIN_FRAMEWORK_PROVIDER,
                LANGCHAIN_CHATGOOGLEGENAI_LLM_PROVIDER,
                None,
            )
            .uses_protobuf()
            .invoke,
        )

    def _wrap_langchain_google_new_sdk(self, chatgooglegenai):
        self._wrap_langchain_google_method(
            chatgooglegenai.client.models,
            chatgooglegenai.client.models,
            "_generate_content",
            "generate_content",
            Invoke,
        )

        if (
            chatgooglegenai.async_client is not None
            and hasattr(chatgooglegenai.async_client, "models")
            and hasattr(chatgooglegenai.async_client.models, "generate_content")
        ):
            self._wrap_langchain_google_method(
                chatgooglegenai.async_client.models,
                chatgooglegenai.async_client.models,
                "_generate_content",
                "generate_content",
                InvokeAsync,
            )

            if hasattr(chatgooglegenai.async_client.models, "generate_content_stream"):
                self._wrap_langchain_google_method(
                    chatgooglegenai.async_client.models,
                    chatgooglegenai.async_client.models,
                    "_stream_generate_content",
                    "generate_content_stream",
                    InvokeAsyncIterator,
                )

        if hasattr(chatgooglegenai.client.models, "generate_content_stream"):
            self._wrap_langchain_google_method(
                chatgooglegenai.client.models,
                chatgooglegenai.client.models,
                "_stream_generate_content",
                "generate_content_stream",
                Invoke,
            )

    def _wrap_langchain_google_old_sdk(self, chatgooglegenai):
        self._wrap_langchain_google_method(
            chatgooglegenai.client,
            chatgooglegenai.client,
            "_generate_content",
            "generate_content",
            Invoke,
        )

        if chatgooglegenai.async_client is not None:
            self._wrap_langchain_google_method(
                chatgooglegenai.async_client,
                chatgooglegenai.async_client,
                "_stream_generate_content",
                "stream_generate_content",
                InvokeAsyncIterator,
            )

    def _wrap_langchain_openai_client(self, client, invoke_cls):
        endpoints = [
            (
                client.beta,
                client.beta.chat.completions,
                "_chat_completions_create",
                "create",
            ),
            (
                client.beta,
                client.beta.chat.completions,
                "_chat_completions_parse",
                "parse",
            ),
            (client, client.chat.completions, "_chat_completions_create", "create"),
            (client, client.chat.completions, "_chat_completions_parse", "parse"),
        ]

        for backup_obj, target_obj, backup_attr, method_name in endpoints:
            setattr(backup_obj, backup_attr, getattr(target_obj, method_name))
            setattr(
                target_obj,
                method_name,
                invoke_cls(self.config, getattr(backup_obj, backup_attr))
                .set_client(
                    LANGCHAIN_FRAMEWORK_PROVIDER,
                    LANGCHAIN_OPENAI_LLM_PROVIDER,
                    None,
                )
                .invoke,
            )

    def register(
        self, chatbedrock=None, chatgooglegenai=None, chatopenai=None, chatvertexai=None
    ):
        if (
            chatbedrock is None
            and chatgooglegenai is None
            and chatopenai is None
            and chatvertexai is None
        ):
            raise RuntimeError("LangChain::register called without client")

        if chatbedrock is not None:
            if not hasattr(chatbedrock, "client"):
                raise RuntimeError("client provided is not instance of ChatBedrock")

            if not hasattr(chatbedrock.client, "_memori_installed"):
                chatbedrock.client._invoke_model = chatbedrock.client.invoke_model
                chatbedrock.client.invoke_model = (
                    Invoke(self.config, chatbedrock.client._invoke_model)
                    .set_client(
                        LANGCHAIN_FRAMEWORK_PROVIDER,
                        LANGCHAIN_CHATBEDROCK_LLM_PROVIDER,
                        None,
                    )
                    .invoke
                )

                chatbedrock.client._invoke_model_with_response_stream = (
                    chatbedrock.client.invoke_model_with_response_stream
                )
                chatbedrock.client.invoke_model_with_response_stream = (
                    Invoke(
                        self.config,
                        chatbedrock.client._invoke_model_with_response_stream,
                    )
                    .set_client(
                        LANGCHAIN_FRAMEWORK_PROVIDER,
                        LANGCHAIN_CHATBEDROCK_LLM_PROVIDER,
                        None,
                    )
                    .invoke
                )

                chatbedrock.client._memori_installed = True

        if chatgooglegenai is not None:
            if not hasattr(chatgooglegenai, "client"):
                raise RuntimeError(
                    "client provided is not instance of ChatGoogleGenerativeAI"
                )

            if not hasattr(chatgooglegenai.client, "_memori_installed"):
                if hasattr(chatgooglegenai.client, "models") and hasattr(
                    chatgooglegenai.client.models, "generate_content"
                ):
                    self._wrap_langchain_google_new_sdk(chatgooglegenai)
                else:
                    self._wrap_langchain_google_old_sdk(chatgooglegenai)

                chatgooglegenai.client._memori_installed = True

        if chatopenai is not None:
            if not hasattr(chatopenai, "client") or not hasattr(
                chatopenai, "async_client"
            ):
                raise RuntimeError("client provided is not instance of ChatOpenAI")

            for client in filter(
                None,
                [getattr(chatopenai, "http_client", None), chatopenai.client._client],
            ):
                if not hasattr(client, "_memori_installed"):
                    self._wrap_langchain_openai_client(client, Invoke)
                    client._memori_installed = True

            for client in filter(
                None,
                [
                    getattr(chatopenai, "async_http_client", None),
                    chatopenai.async_client._client,
                ],
            ):
                if not hasattr(client, "_memori_installed"):
                    self._wrap_langchain_openai_client(client, InvokeAsyncIterator)
                    client._memori_installed = True

        if chatvertexai is not None:
            if not hasattr(chatvertexai, "prediction_client"):
                raise RuntimeError("client provided isnot instance of ChatVertexAI")

            if not hasattr(chatvertexai.prediction_client, "_memori_installed"):
                chatvertexai.prediction_client.actual_generate_content = (
                    chatvertexai.prediction_client.generate_content
                )
                chatvertexai.prediction_client.generate_content = (
                    Invoke(
                        self.config,
                        chatvertexai.prediction_client.actual_generate_content,
                    )
                    .set_client(
                        LANGCHAIN_FRAMEWORK_PROVIDER,
                        LANGCHAIN_CHATVERTEXAI_LLM_PROVIDER,
                        None,
                    )
                    .uses_protobuf()
                    .invoke
                )

                chatvertexai.prediction_client._memori_installed = True

        return self


class Agno(BaseClient):
    def _wrap_agno_client_getters(self, model, wrapper, include_async: bool = True):
        if not hasattr(model, "_memori_original_get_client"):
            model._memori_original_get_client = model.get_client

            def wrapped_get_client():
                client = model._memori_original_get_client()
                wrapper.register(client, _provider=AGNO_FRAMEWORK_PROVIDER)
                return client

            model.get_client = wrapped_get_client

        if (
            include_async
            and hasattr(model, "get_async_client")
            and not hasattr(model, "_memori_original_get_async_client")
        ):
            model._memori_original_get_async_client = model.get_async_client

            def wrapped_get_async_client():
                client = model._memori_original_get_async_client()
                wrapper.register(client, _provider=AGNO_FRAMEWORK_PROVIDER)
                return client

            model.get_async_client = wrapped_get_async_client

    def register(self, openai_chat=None, claude=None, gemini=None, xai=None):
        if openai_chat is None and claude is None and gemini is None and xai is None:
            raise RuntimeError("Agno::register called without model")

        if openai_chat is not None:
            if not self._is_agno_openai_model(openai_chat):
                raise RuntimeError(
                    "model provided is not instance of agno.models.openai.OpenAIChat"
                )
            client = openai_chat.get_client()
            OpenAi(self.config).register(client, _provider=AGNO_FRAMEWORK_PROVIDER)
            self._wrap_agno_client_getters(openai_chat, OpenAi(self.config))

        if claude is not None:
            if not self._is_agno_anthropic_model(claude):
                raise RuntimeError(
                    "model provided is not instance of agno.models.anthropic.Claude"
                )
            client = claude.get_client()
            Anthropic(self.config).register(client, _provider=AGNO_FRAMEWORK_PROVIDER)
            self._wrap_agno_client_getters(claude, Anthropic(self.config))

        if gemini is not None:
            if not self._is_agno_google_model(gemini):
                raise RuntimeError(
                    "model provided is not instance of agno.models.google.Gemini"
                )
            client = gemini.get_client()
            Google(self.config).register(client, _provider=AGNO_FRAMEWORK_PROVIDER)
            self._wrap_agno_client_getters(
                gemini, Google(self.config), include_async=False
            )

        if xai is not None:
            if not self._is_agno_xai_model(xai):
                raise RuntimeError(
                    "model provided is not instance of agno.models.xai.xAI"
                )
            client = xai.get_client()
            XAi(self.config).register(client, _provider=AGNO_FRAMEWORK_PROVIDER)
            self._wrap_agno_client_getters(xai, XAi(self.config))

        return self

    def _is_agno_openai_model(self, model):
        return "agno.models.openai" in str(type(model).__module__)

    def _is_agno_anthropic_model(self, model):
        return "agno.models.anthropic" in str(type(model).__module__)

    def _is_agno_google_model(self, model):
        return "agno.models.google" in str(type(model).__module__)

    def _is_agno_xai_model(self, model):
        return "agno.models.xai" in str(type(model).__module__)


@Registry.register_client(client_is_llamaindex)
class LlamaIndex(BaseClient):

    """Memori integration for LlamaIndex LLM objects.

    Transparently wraps the underlying LLM SDK client (OpenAI, Anthropic, or
    Google) that powers the LlamaIndex LLM, so every LLM call made through
    LlamaIndex is automatically intercepted for memory injection and capture.

    Supported LlamaIndex LLMs:
      - ``llama_index.llms.openai.OpenAI``
      - ``llama_index.llms.anthropic.Anthropic``
      - ``llama_index.llms.gemini.Gemini``

    Example::

        from llama_index.llms.openai import OpenAI
        from memori import Memori

        llm = OpenAI(model="gpt-4o-mini")
        mem = Memori().llm.register(llm)
        mem.attribution(entity_id="user_123", process_id="llamaindex-app")
    """

    # Attribute names checked (in order) to find the raw SDK client inside
    # each LlamaIndex LLM wrapper class.
    _OPENAI_ATTRS = ("_client", "_aclient")
    _ANTHROPIC_ATTRS = ("_client",)
    _GOOGLE_ATTRS = ("_client",)

    def register(self, client) -> "LlamaIndex":
        """Detect the underlying SDK and delegate to the appropriate wrapper."""
        module = type(client).__module__

        if "openai" in module:
            self._register_openai(client)
        elif "anthropic" in module:
            self._register_anthropic(client)
        elif "gemini" in module or "google" in module:
            self._register_google(client)
        else:
            # Attempt generic detection via attribute presence
            if any(hasattr(client, a) for a in self._OPENAI_ATTRS) and hasattr(
                getattr(client, "_client", None), "chat"
            ):
                self._register_openai(client)
            else:
                from memori._exceptions import UnsupportedLLMProviderError

                provider = f"{module}.{type(client).__name__}"
                raise UnsupportedLLMProviderError(
                    f"llamaindex:{provider} — only OpenAI, Anthropic, and "
                    "Gemini LlamaIndex LLMs are supported"
                )

        return self

    def _register_openai(self, llm) -> None:
        """Patch the underlying openai.OpenAI / openai.AsyncOpenAI client."""
        for attr in self._OPENAI_ATTRS:
            inner = getattr(llm, attr, None)
            if inner is not None and hasattr(inner, "chat"):
                if not hasattr(inner, "_memori_installed"):
                    OpenAi(self.config).register(
                        inner, _provider=LLAMAINDEX_FRAMEWORK_PROVIDER
                    )
        self.config.framework.provider = LLAMAINDEX_FRAMEWORK_PROVIDER
        self.config.llm.provider = LLAMAINDEX_OPENAI_LLM_PROVIDER

    def _register_anthropic(self, llm) -> None:
        """Patch the underlying anthropic.Anthropic / AsyncAnthropic client."""
        for attr in self._ANTHROPIC_ATTRS:
            inner = getattr(llm, attr, None)
            if inner is not None and hasattr(inner, "messages"):
                if not hasattr(inner, "_memori_installed"):
                    Anthropic(self.config).register(
                        inner, _provider=LLAMAINDEX_FRAMEWORK_PROVIDER
                    )
        self.config.framework.provider = LLAMAINDEX_FRAMEWORK_PROVIDER
        self.config.llm.provider = LLAMAINDEX_ANTHROPIC_LLM_PROVIDER

    def _register_google(self, llm) -> None:
        """Patch the underlying google.genai.Client."""
        for attr in self._GOOGLE_ATTRS:
            inner = getattr(llm, attr, None)
            if inner is not None and hasattr(inner, "models"):
                if not hasattr(inner, "_memori_installed"):
                    Google(self.config).register(
                        inner, _provider=LLAMAINDEX_FRAMEWORK_PROVIDER
                    )
        self.config.framework.provider = LLAMAINDEX_FRAMEWORK_PROVIDER
        self.config.llm.provider = LLAMAINDEX_GOOGLE_LLM_PROVIDER
