r"""
 __  __                           _
|  \/  | ___ _ __ ___   ___  _ __(_)
| |\/| |/ _ \ '_ ` _ \ / _ \| '__| |
| |  | |  __/ | | | | | (_) | |  | |
|_|  |_|\___|_| |_| |_|\___/|_|  |_|
                  perfectam memoriam
                       memorilabs.ai
"""

import os
from collections.abc import Callable
from typing import Any
from uuid import uuid4

from memori._config import Config
from memori._exceptions import (
    MissingMemoriApiKeyError,
    QuotaExceededError,
    UnsupportedLLMProviderError,
    warn_if_legacy_memorisdk_installed,
)
from memori.embeddings import embed_texts
from memori.llm._providers import Agno as LlmProviderAgno
from memori.llm._providers import Anthropic as LlmProviderAnthropic
from memori.llm._providers import Google as LlmProviderGoogle
from memori.llm._providers import LangChain as LlmProviderLangChain
from memori.llm._providers import OpenAi as LlmProviderOpenAi
from memori.llm._providers import PydanticAi as LlmProviderPydanticAi
from memori.llm._providers import XAi as LlmProviderXAi
from memori.memory.augmentation import Manager as AugmentationManager
from memori.memory.recall import Recall
from memori.storage import Manager as StorageManager

__all__ = ["Memori", "QuotaExceededError", "UnsupportedLLMProviderError"]

warn_if_legacy_memorisdk_installed()


class LlmRegistry:
    def __init__(self, memori):
        self.memori = memori

    def register(
        self,
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
        from memori.llm._registry import register_llm

        return register_llm(
            self.memori,
            client=client,
            openai_chat=openai_chat,
            claude=claude,
            gemini=gemini,
            xai=xai,
            chatbedrock=chatbedrock,
            chatgooglegenai=chatgooglegenai,
            chatopenai=chatopenai,
            chatvertexai=chatvertexai,
        )


class Memori:
    def __init__(
        self,
        conn: Callable[[], Any] | Any | None = None,
        debug_truncate: bool = True,
    ):
        from memori._logging import set_truncate_enabled

        self.config = Config()
        self.config.api_key = os.environ.get("MEMORI_API_KEY", None)
        self.config.session_id = uuid4()
        self.config.debug_truncate = debug_truncate
        set_truncate_enabled(debug_truncate)

        if conn is None:
            conn = self._get_default_connection()
        else:
            self.config.hosted = False

        self.config.storage = StorageManager(self.config).start(conn)
        self.config.augmentation = AugmentationManager(self.config).start(conn)

        self.augmentation = self.config.augmentation
        self.llm = LlmRegistry(self)
        self.agno = LlmProviderAgno(self)
        self.anthropic = LlmProviderAnthropic(self)
        self.google = LlmProviderGoogle(self)
        self.langchain = LlmProviderLangChain(self)
        self.openai = LlmProviderOpenAi(self)
        self.pydantic_ai = LlmProviderPydanticAi(self)
        self.xai = LlmProviderXAi(self)

    def _get_default_connection(self) -> Callable[[], Any] | None:
        connection_string = os.environ.get("MEMORI_COCKROACHDB_CONNECTION_STRING", None)
        if connection_string:
            try:
                import psycopg
            except ImportError as e:
                raise ImportError(
                    "psycopg is required for CockroachDB support. "
                    "Install it with: pip install 'memori[cockroachdb]'"
                ) from e

            self.config.hosted = False
            return lambda: psycopg.connect(connection_string)

        self.config.hosted = True
        api_key = os.environ.get("MEMORI_API_KEY", None)
        if api_key is None or api_key == "":
            raise MissingMemoriApiKeyError()
        return None

    def attribution(self, entity_id=None, process_id=None):
        if entity_id is not None:
            entity_id = str(entity_id)

            if len(entity_id) > 100:
                raise RuntimeError("entity_id cannot be greater than 100 characters")

        if process_id is not None:
            process_id = str(process_id)

            if len(process_id) > 100:
                raise RuntimeError("process_id cannot be greater than 100 characters")

        self.config.entity_id = entity_id
        self.config.process_id = process_id

        return self

    def new_session(self):
        self.config.session_id = uuid4()
        self.config.reset_cache()
        return self

    def set_session(self, id):
        self.config.session_id = id
        return self

    def recall(self, query: str, limit: int = 5):
        return Recall(self.config).search_facts(query, limit)

    def close(self) -> None:
        """Close the underlying storage connection/session, if any.

        This is especially important for long-running processes (e.g. web servers)
        where you want to explicitly release database connections.
        """
        storage = getattr(self.config, "storage", None)
        adapter = getattr(storage, "adapter", None) if storage is not None else None
        if adapter is None:
            return
        try:
            adapter.close()
        except Exception:  # nosec B110
            pass

    def __enter__(self) -> "Memori":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def embed_texts(self, texts: str | list[str], *, async_: bool = False) -> Any:
        embeddings_cfg = self.config.embeddings
        return embed_texts(
            texts,
            model=embeddings_cfg.model,
            async_=async_,
        )
