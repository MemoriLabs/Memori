from __future__ import annotations

from typing import Any

from memori.search._core import search_entity_facts_core
from memori.search._faiss import find_similar_embeddings
from memori.search._lexical import dense_lexical_weights, lexical_scores_for_ids


def search_entity_facts(
    entity_fact_driver: Any,
    entity_id: int,
    query_embedding: list[float],
    limit: int,
    embeddings_limit: int,
    *,
    query_text: str | None = None,
) -> list[dict]:
    """
    Public entrypoint for searching entity facts.
    """
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
