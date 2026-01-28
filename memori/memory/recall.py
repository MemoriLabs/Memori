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

from sqlalchemy.exc import OperationalError

from memori._config import Config
from memori._logging import truncate
from memori._network import Api, ApiSubdomain
from memori.embeddings import embed_texts
from memori.search import search_facts as search_facts_api
from memori.search._types import FactSearchResult

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 0.05


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

    def _search_with_retries_hosted(self, query: str) -> list[FactSearchResult]:
        api = Api(self.config, ApiSubdomain.HOSTED)
        payload = {
            "attribution": {
                "entity": {"id": str(self.config.entity_id)},
                "process": {"id": self.config.process_id},
            },
            "query": query,
        }

        return api.post("recall", payload)

    def search_facts(
        self,
        query: str,
        limit: int | None = None,
        entity_id: int | None = None,
        hosted: bool = False,
    ) -> list[FactSearchResult]:
        logger.debug(
            "Recall started - query: %s (%d chars), limit: %s",
            truncate(query, 50),
            len(query),
            limit,
        )

        if hosted:
            logger.debug(
                "Recall started - query: %s (%d chars), limit: %s, hosted: true",
                truncate(query, 50),
                len(query),
                limit,
            )
            print(hosted)
            return self._search_with_retries_hosted(query)

        if self.config.storage is None or self.config.storage.driver is None:
            logger.debug("Recall aborted - storage not configured")
            return []

        entity_id = self._resolve_entity_id(entity_id)
        if entity_id is None:
            return []

        limit = self._resolve_limit(limit)
        query_embedding = self._embed_query(query)
        return self._search_with_retries(
            entity_id=entity_id,
            query=query,
            query_embedding=query_embedding,
            limit=limit,
        )
