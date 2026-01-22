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

import psycopg

from memori._config import Config
from memori._exceptions import (
    QuotaExceededError,
    warn_if_legacy_memorisdk_installed,
)
from memori.ingestion import SeedConfig, SeedData, SeedResult, SeedType
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

__all__ = [
    "Memori",
    "QuotaExceededError",
    "SeedType",
    "SeedData",
    "SeedResult",
    "SeedConfig",
]

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
        self.config.enterprise = os.environ.get("MEMORI_ENTERPRISE", "0") == "1"
        self.config.session_id = uuid4()
        self.config.debug_truncate = debug_truncate
        set_truncate_enabled(debug_truncate)

        if conn is None:
            conn = self._get_default_connection()

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

    def _get_default_connection(self) -> Callable[[], Any]:
        connection_string = os.environ.get("MEMORI_COCKROACHDB_CONNECTION_STRING")
        if connection_string:
            return lambda: psycopg.connect(connection_string)

        raise RuntimeError(
            "No connection factory provided. Either pass 'conn' parameter or set "
            "MEMORI_COCKROACHDB_CONNECTION_STRING environment variable."
        )

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

    async def seed(
        self,
        seed_data: "SeedData",
        seed_config: "SeedConfig | None" = None,
        batch_size: int = 10,
        on_progress=None,
    ) -> "SeedResult":
        from memori.ingestion import SeedType, ingest_conversations

        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if not seed_data.entity_id:
            raise ValueError("entity_id is required")
        if len(str(seed_data.entity_id)) > 100:
            raise ValueError("entity_id cannot be greater than 100 characters")
        if seed_data.process_id and len(str(seed_data.process_id)) > 100:
            raise ValueError("process_id cannot be greater than 100 characters")

        if seed_data.seed_type == SeedType.CONVERSATION:
            return await ingest_conversations(
                config=self.config,
                driver=self.config.storage.driver,
                entity_id=seed_data.entity_id,
                conversations=seed_data.data,
                process_id=seed_data.process_id,
                batch_size=batch_size,
                ingestion_config=seed_config,
                on_progress=on_progress,
            )
        else:
            raise ValueError(f"Unsupported seed type: {seed_data.seed_type}")

    def seed_sync(
        self,
        seed_data: "SeedData",
        seed_config: "SeedConfig | None" = None,
        batch_size: int = 10,
        on_progress=None,
    ) -> "SeedResult":
        import asyncio

        return asyncio.run(
            self.seed(
                seed_data=seed_data,
                seed_config=seed_config,
                batch_size=batch_size,
                on_progress=on_progress,
            )
        )
