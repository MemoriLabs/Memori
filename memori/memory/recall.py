r"""
 __  __                           _
|  \/  | ___ _ __ ___   ___  _ __(_)
| |\/| |/ _ \ '_ ` _ \ / _ \| '__| |
| |  | |  __/ | | | | | (_) | |  | |
|_|  |_|\___|_| |_| |_|\___/|_|  |_|
                 perfectam memoriam
                      memorilabs.ai
"""

import logging
import time
from collections.abc import Mapping
from typing import TypeGuard, cast

from sqlalchemy.exc import OperationalError

from memori._config import Config
from memori._logging import truncate
from memori._network import Api
from memori.embeddings import embed_texts
from memori.search import search_facts as search_facts_api
from memori.search._types import FactSearchResult

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 0.05


def _is_str_object_mapping(value: object) -> TypeGuard[Mapping[str, object]]:
    if not isinstance(value, Mapping):
        return False
    return all(isinstance(k, str) for k in value.keys())


class Recall:
    def __init__(self, config: Config) -> None:
        self.config = config

    def _resolve_entity_id(self, entity_id: int | None) -> int | None:
        if entity_id is not None:
            return entity_id

        if self.config.entity_id is None:
            logger.debug("Recall aborted - no entity_id configured")
            return None

        entity_id = self.config.storage.driver.entity.create(self.config.entity_id)
        logger.debug("Entity ID resolved: %s", entity_id)
        if entity_id is None:
            logger.debug("Recall aborted - entity_id is None after resolution")
        return entity_id

    def _resolve_limit(self, limit: int | None) -> int:
        return self.config.recall_facts_limit if limit is None else limit

    def _embed_query(self, query: str) -> list[float]:
        logger.debug("Generating query embedding")
        embeddings_config = self.config.embeddings
        return embed_texts(
            query,
            model=embeddings_config.model,
        )[0]

    def _search_with_retries(
        self, *, entity_id: int, query: str, query_embedding: list[float], limit: int
    ) -> list[FactSearchResult]:
        facts: list[FactSearchResult] = []
        for attempt in range(MAX_RETRIES):
            try:
                logger.debug(
                    f"Executing search_facts - entity_id: {entity_id}, limit: {limit}, embeddings_limit: {self.config.recall_embeddings_limit}"
                )
                facts = search_facts_api(
                    self.config.storage.driver.entity_fact,
                    entity_id,
                    query_embedding,
                    limit,
                    self.config.recall_embeddings_limit,
                    query_text=query,
                )
                logger.debug("Recall complete - found %d facts", len(facts))
                break
            except OperationalError as e:
                if "restart transaction" in str(e) and attempt < MAX_RETRIES - 1:
                    logger.debug(
                        "Retry attempt %d due to OperationalError", attempt + 1
                    )
                    time.sleep(RETRY_BACKOFF_BASE * (2**attempt))
                    continue
                raise

        return facts

    def _search_with_retries_cloud(
        self, *, query: str, limit: int
    ) -> list[FactSearchResult | Mapping[str, object] | str]:
        data = self._cloud_recall(query, limit=limit)
        facts, _messages = self._parse_cloud_recall_response(data)
        return facts

    def _cloud_recall(self, query: str, *, limit: int | None = None) -> object:
        if self.config.entity_id is None:
            logger.debug("Cloud recall aborted - no entity_id configured")
            return []

        api = Api(self.config)
        resolved_limit = self._resolve_limit(limit)
        payload = {
            "attribution": {
                "entity": {"id": str(self.config.entity_id)},
                "process": {"id": self.config.process_id},
            },
            "query": query,
            "session": {"id": str(self.config.session_id)},
            "limit": resolved_limit,
        }
        return api.post("cloud/recall", payload)

    @staticmethod
    def _parse_cloud_recall_response(
        data: object,
    ) -> tuple[
        list[FactSearchResult | Mapping[str, object] | str], list[dict[str, str]]
    ]:
        if isinstance(data, list):
            facts_from_list: list[FactSearchResult | Mapping[str, object] | str] = []
            for item in data:
                if isinstance(item, str):
                    facts_from_list.append(item)
                elif _is_str_object_mapping(item):
                    facts_from_list.append(item)
            return facts_from_list, []

        if not isinstance(data, dict):
            return [], []

        data_map = cast(Mapping[str, object], data)

        def _extract_list(*keys: str) -> list[object] | None:
            for k in keys:
                v = data_map.get(k)
                if isinstance(v, list):
                    return cast(list[object], v)
            return None

        facts_raw = _extract_list("facts", "results", "memories", "data") or []
        facts: list[FactSearchResult | Mapping[str, object] | str] = []
        for item in facts_raw:
            if isinstance(item, str):
                facts.append(item)
            elif _is_str_object_mapping(item):
                facts.append(item)

        messages_raw: list[object] = (
            _extract_list("messages", "conversation_messages", "history") or []
        )
        if not messages_raw:
            convo = data_map.get("conversation")
            if _is_str_object_mapping(convo):
                nested = convo.get("messages")
                if isinstance(nested, list):
                    messages_raw = cast(list[object], nested)

        messages: list[dict[str, str]] = []
        for msg in messages_raw:
            if not _is_str_object_mapping(msg):
                continue
            role = msg.get("role")
            content = msg.get("content")
            if content is None:
                content = msg.get("text")
            if not isinstance(role, str) or not isinstance(content, str):
                continue
            messages.append({"role": role, "content": content})

        return facts, messages

    def search_facts(
        self,
        query: str,
        limit: int | None = None,
        entity_id: int | None = None,
        cloud: bool = False,
    ) -> list[FactSearchResult | Mapping[str, object] | str]:
        logger.debug(
            "Recall started - query: %s (%d chars), limit: %s",
            truncate(query, 50),
            len(query),
            limit,
        )

        if self.config.cloud:
            if self.config.entity_id is None:
                logger.debug("Recall aborted - no entity_id configured")
                return []

            logger.debug(
                "Recall started - query: %s (%d chars), limit: %s, cloud: true",
                truncate(query, 50),
                len(query),
                limit,
            )
            resolved_limit = self._resolve_limit(limit)
            return self._search_with_retries_cloud(query=query, limit=resolved_limit)

        if self.config.storage is None or self.config.storage.driver is None:
            logger.debug("Recall aborted - storage not configured")
            return []

        entity_id = self._resolve_entity_id(entity_id)
        if entity_id is None:
            return []

        limit = self._resolve_limit(limit)
        query_embedding = self._embed_query(query)
        return cast(
            list[FactSearchResult | Mapping[str, object] | str],
            self._search_with_retries(
                entity_id=entity_id,
                query=query,
                query_embedding=query_embedding,
                limit=limit,
            ),
        )
