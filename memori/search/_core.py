from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)


def _get_embeddings_rows(
    entity_fact_driver: Any, *, entity_id: int, embeddings_limit: int
) -> list[dict]:
    logger.debug(
        "Executing memori_entity_fact query - entity_id: %s, embeddings_limit: %s",
        entity_id,
        embeddings_limit,
    )
    results = entity_fact_driver.get_embeddings(entity_id, embeddings_limit)
    if not results:
        logger.debug("No embeddings found in database for entity_id: %s", entity_id)
        return []
    logger.debug("Retrieved %d embeddings from database", len(results))
    return results


def _candidate_limit(
    *, limit: int, total_embeddings: int, query_text: str | None
) -> int:
    if query_text:
        return max(limit, min(total_embeddings, max(limit * 10, 50)))
    return int(limit)


def _fetch_content_maps(
    entity_fact_driver: Any, *, candidate_ids: list[int]
) -> tuple[dict[int, dict], dict[int, str]]:
    logger.debug("Fetching content for %d fact IDs", len(candidate_ids))
    content_results = entity_fact_driver.get_facts_by_ids(candidate_ids)
    fact_rows: dict[int, dict] = {
        int(row["id"]): row
        for row in content_results
        if isinstance(row, dict) and row.get("id") is not None
    }
    content_map: dict[int, str] = {}
    for fid, row in fact_rows.items():
        content = row.get("content")
        if isinstance(content, str):
            content_map[fid] = content
    return fact_rows, content_map


def _rank_candidates(
    *,
    candidate_ids: list[int],
    similarities_map: dict[int, float],
    query_text: str | None,
    content_map: dict[int, str],
    lexical_scores_for_ids: Callable[..., dict[int, float]],
    dense_lexical_weights: Callable[..., tuple[float, float]],
) -> tuple[list[int], dict[int, float], dict[int, float]]:
    lex_scores: dict[int, float] = {}

    if query_text:
        lex_scores = lexical_scores_for_ids(
            query_text=query_text, ids=candidate_ids, content_map=content_map
        )
        w_cos, w_lex = dense_lexical_weights(query_text=query_text)
        rank_score_map = {
            fid: (w_cos * float(similarities_map.get(fid, 0.0)))
            + (w_lex * float(lex_scores.get(fid, 0.0)))
            for fid in candidate_ids
        }

        def key(fid: int) -> tuple[float, float]:
            return (
                float(rank_score_map.get(fid, 0.0)),
                float(similarities_map.get(fid, 0.0)),
            )

        base_order = sorted(candidate_ids, key=key, reverse=True)
        return base_order, rank_score_map, lex_scores

    rank_score_map = {
        fid: float(similarities_map.get(fid, 0.0)) for fid in candidate_ids
    }
    return list(candidate_ids), rank_score_map, lex_scores


def _build_fact_rows(
    *,
    ordered_ids: list[int],
    fact_rows: dict[int, dict],
    content_map: dict[int, str],
    similarities_map: dict[int, float],
    query_text: str | None,
    lex_scores: dict[int, float],
    rank_score_map: dict[int, float],
) -> list[dict]:
    facts_with_similarity: list[dict] = []
    for fact_id in ordered_ids:
        fact_row = fact_rows.get(int(fact_id), {})
        content = content_map.get(int(fact_id))
        if content is None:
            continue
        row: dict[str, object] = {
            "id": fact_id,
            "content": content,
            "similarity": float(similarities_map.get(fact_id, 0.0)),
        }
        if "date_created" in fact_row and fact_row.get("date_created") is not None:
            row["date_created"] = fact_row.get("date_created")
        facts_with_similarity.append(row)

    if query_text and facts_with_similarity:
        for r in facts_with_similarity:
            fid = int(r["id"])
            r["lexical_score"] = float(lex_scores.get(fid, 0.0))
            r["rank_score"] = float(rank_score_map.get(fid, float(r["similarity"])))

    return facts_with_similarity


def search_entity_facts_core(
    entity_fact_driver: Any,
    entity_id: int,
    query_embedding: list[float],
    limit: int,
    embeddings_limit: int,
    *,
    query_text: str | None,
    find_similar_embeddings: Callable[
        [list[tuple[int, Any]], list[float], int], list[tuple[int, float]]
    ],
    lexical_scores_for_ids: Callable[..., dict[int, float]],
    dense_lexical_weights: Callable[..., tuple[float, float]],
) -> list[dict]:
    results = _get_embeddings_rows(
        entity_fact_driver, entity_id=entity_id, embeddings_limit=embeddings_limit
    )
    if not results:
        return []

    embeddings = [(row["id"], row["content_embedding"]) for row in results]
    cand_limit = _candidate_limit(
        limit=limit, total_embeddings=len(embeddings), query_text=query_text
    )
    similar = find_similar_embeddings(embeddings, query_embedding, cand_limit)
    if not similar:
        logger.debug("No similar embeddings found")
        return []

    candidate_ids = [fact_id for fact_id, _ in similar]
    similarities_map = dict(similar)

    fact_rows, content_map = _fetch_content_maps(
        entity_fact_driver, candidate_ids=candidate_ids
    )
    base_order, rank_score_map, lex_scores = _rank_candidates(
        candidate_ids=candidate_ids,
        similarities_map=similarities_map,
        query_text=query_text,
        content_map=content_map,
        lexical_scores_for_ids=lexical_scores_for_ids,
        dense_lexical_weights=dense_lexical_weights,
    )

    ordered_ids = base_order[:limit]

    facts_with_similarity = _build_fact_rows(
        ordered_ids=ordered_ids,
        fact_rows=fact_rows,
        content_map=content_map,
        similarities_map=similarities_map,
        query_text=query_text,
        lex_scores=lex_scores,
        rank_score_map=rank_score_map,
    )
    logger.debug(
        "Returning %d facts with similarity scores", len(facts_with_similarity)
    )
    return facts_with_similarity
