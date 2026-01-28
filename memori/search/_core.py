from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from typing import Any, cast

from memori.search._types import FactCandidate, FactId, FactSearchResult

logger = logging.getLogger(__name__)


def _candidate_pool_from_candidates(
    candidates: list[FactCandidate], *, limit: int, query_text: str | None
) -> tuple[
    list[int], dict[int, float], dict[int, str], dict[int, FactId], dict[int, dict]
]:
    if not candidates:
        return [], {}, {}, {}, {}

    idx_to_original_id = {i: r.id for i, r in enumerate(candidates)}
    content_map = {i: r.content for i, r in enumerate(candidates)}
    similarities_map = {i: float(r.score) for i, r in enumerate(candidates)}
    date_created_map = {i: r.date_created for i, r in enumerate(candidates)}

    cand_limit = _candidate_limit(
        limit=limit, total_embeddings=len(candidates), query_text=query_text
    )
    candidate_ids = sorted(
        similarities_map,
        key=lambda i: float(similarities_map.get(i, 0.0)),
        reverse=True,
    )[:cand_limit]

    # Mimic DB shape just enough for _build_fact_rows.
    fact_rows: dict[int, dict] = {
        i: {
            "id": idx_to_original_id.get(i),
            "date_created": date_created_map.get(i, ""),
        }
        for i in candidate_ids
    }

    return candidate_ids, similarities_map, content_map, idx_to_original_id, fact_rows


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
    entity_fact_driver: Any, *, candidate_ids: list[FactId]
) -> tuple[dict[FactId, dict], dict[FactId, str]]:
    logger.debug("Fetching content for %d fact IDs", len(candidate_ids))
    content_results = entity_fact_driver.get_facts_by_ids(candidate_ids)

    fact_rows: dict[FactId, dict] = {}
    for row in content_results or []:
        if not isinstance(row, Mapping):
            continue
        rid: FactId = row.get("id")
        if rid is None:
            continue
        fact_rows[rid] = dict(row)

    content_map: dict[FactId, str] = {}
    for fid, row in fact_rows.items():
        content = row.get("content")
        if isinstance(content, str):
            content_map[fid] = content
    return fact_rows, content_map


def _rank_candidates(
    *,
    candidate_ids: list[FactId],
    similarities_map: dict[FactId, float],
    query_text: str | None,
    content_map: dict[FactId, str],
    lexical_scores_for_ids: Callable[..., dict[FactId, float]],
    dense_lexical_weights: Callable[..., tuple[float, float]],
) -> tuple[list[FactId], dict[FactId, float], dict[FactId, float]]:
    lex_scores: dict[FactId, float] = {}

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

        def key(fid: FactId) -> tuple[float, float]:
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
    ordered_ids: list[FactId],
    fact_rows: dict[FactId, dict],
    content_map: dict[FactId, str],
    similarities_map: dict[FactId, float],
    rank_score_map: dict[FactId, float],
) -> list[FactSearchResult]:
    facts_with_similarity: list[FactSearchResult] = []
    for fact_id in ordered_ids:
        fact_row = fact_rows.get(fact_id, {})
        content = content_map.get(fact_id)
        if content is None:
            continue
        date_created = fact_row.get("date_created")
        similarity = float(similarities_map.get(fact_id, 0.0))
        rank_score = float(rank_score_map.get(fact_id, similarity))
        facts_with_similarity.append(
            FactSearchResult(
                id=fact_id,
                content=content,
                similarity=similarity,
                rank_score=rank_score,
                date_created=str(date_created) if date_created is not None else "",
            )
        )

    return facts_with_similarity


def search_entity_facts_core(
    entity_fact_driver: Any,
    entity_id: int,
    query_embedding: list[float],
    limit: int,
    embeddings_limit: int,
    *,
    query_text: str | None,
    fact_candidates: list[FactCandidate] | None = None,
    find_similar_embeddings: Callable[
        [list[tuple[FactId, Any]], list[float], int], list[tuple[FactId, float]]
    ],
    lexical_scores_for_ids: Callable[..., dict[FactId, float]],
    dense_lexical_weights: Callable[..., tuple[float, float]],
) -> list[FactSearchResult]:
    idx_to_original_id: dict[int, FactId] = {}
    if fact_candidates is not None:
        (
            candidate_ids,
            similarities_map,
            content_map,
            idx_to_original_id,
            fact_rows,
        ) = _candidate_pool_from_candidates(
            fact_candidates, limit=limit, query_text=query_text
        )
        if not candidate_ids:
            return []
    else:
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

    # Cast to FactId types - in hosted path these are int indices,
    # in DB path these are already FactId. Both are valid FactId values.
    base_order, rank_score_map, lex_scores = _rank_candidates(
        candidate_ids=cast(list[FactId], candidate_ids),
        similarities_map=cast(dict[FactId, float], similarities_map),
        query_text=query_text,
        content_map=cast(dict[FactId, str], content_map),
        lexical_scores_for_ids=lexical_scores_for_ids,
        dense_lexical_weights=dense_lexical_weights,
    )

    ordered_ids = base_order[:limit]

    facts_with_similarity = _build_fact_rows(
        ordered_ids=ordered_ids,
        fact_rows=cast(dict[FactId, dict], fact_rows),
        content_map=cast(dict[FactId, str], content_map),
        similarities_map=cast(dict[FactId, float], similarities_map),
        rank_score_map=rank_score_map,
    )

    if fact_candidates is not None:
        # Remap back to original hosted IDs.
        remapped: list[FactSearchResult] = []
        for row in facts_with_similarity:
            rid = row.id
            # In hosted path, rid is always int (internal index)
            if isinstance(rid, int) and rid in idx_to_original_id:
                remapped.append(
                    FactSearchResult(
                        id=idx_to_original_id[rid],
                        content=row.content,
                        similarity=row.similarity,
                        rank_score=row.rank_score,
                        date_created=row.date_created,
                    )
                )
            else:
                remapped.append(row)
        facts_with_similarity = remapped

    logger.debug(
        "Returning %d facts with similarity scores", len(facts_with_similarity)
    )

    return facts_with_similarity
