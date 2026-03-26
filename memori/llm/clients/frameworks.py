from memori.llm._base import BaseClient
from memori.llm._constants import (
    AGNO_FRAMEWORK_PROVIDER,
    LANGCHAIN_CHATBEDROCK_LLM_PROVIDER,
    LANGCHAIN_CHATGOOGLEGENAI_LLM_PROVIDER,
    LANGCHAIN_CHATVERTEXAI_LLM_PROVIDER,
    LANGCHAIN_FRAMEWORK_PROVIDER,
    LANGCHAIN_OPENAI_LLM_PROVIDER,
)
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
