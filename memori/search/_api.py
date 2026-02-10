from __future__ import annotations

from typing import Any

from memori.search._core import (
    search_entity_facts_core,
)
from memori.search._faiss import find_similar_embeddings
from memori.search._lexical import dense_lexical_weights, lexical_scores_for_ids
from memori.search._types import FactCandidate, FactSearchResult


def search_facts(
    entity_fact_driver: Any | None = None,
    entity_id: int | None = None,
    query_embedding: list[float] | None = None,
    limit: int = 5,
    embeddings_limit: int = 1000,
    *,
    query_text: str | None = None,
    candidates: list[FactCandidate] | None = None,
) -> list[FactSearchResult]:
    """
    Unified search entrypoint.

    - DB-backed mode: provide entity_fact_driver, entity_id, query_embedding, embeddings_limit
    - Pre-scored mode: provide candidates (list[FactCandidate])
    """
    if candidates is not None:
        return search_entity_facts_core(
            entity_fact_driver=None,
            entity_id=0,
            query_embedding=[],
            limit=limit,
            embeddings_limit=0,
            query_text=query_text,
            fact_candidates=candidates,
            find_similar_embeddings=find_similar_embeddings,
            lexical_scores_for_ids=lexical_scores_for_ids,
            dense_lexical_weights=dense_lexical_weights,
        )

    if entity_fact_driver is None:
        raise ValueError("entity_fact_driver is required when candidates is not set")
    if entity_id is None:
        raise ValueError("entity_id is required when candidates is not set")
    if query_embedding is None:
        raise ValueError("query_embedding is required when candidates is not set")

    return search_entity_facts_core(
        entity_fact_driver,
        entity_id,
        query_embedding,
        limit,
        embeddings_limit,
        query_text=query_text,
        find_similar_embeddings=find_similar_embeddings,
        lexical_scores_for_ids=lexical_scores_for_ids,
        dense_lexical_weights=dense_lexical_weights,
    )
