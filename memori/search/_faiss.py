from __future__ import annotations

import logging
from typing import Any

import faiss
import numpy as np

from memori.search._parsing import parse_embedding

logger = logging.getLogger(__name__)


def _query_dim(query_embedding: list[float]) -> int:
    return len(query_embedding)


def _parse_valid_embeddings(
    embeddings: list[tuple[int, Any]], *, query_dim: int
) -> tuple[list[np.ndarray], list[int]]:
    embeddings_list: list[np.ndarray] = []
    id_list: list[int] = []

    for fact_id, raw in embeddings:
        try:
            parsed = parse_embedding(raw)
        except Exception:
            continue

        if parsed.ndim != 1 or parsed.shape[0] != query_dim:
            continue

        embeddings_list.append(parsed)
        id_list.append(fact_id)

    return embeddings_list, id_list


def _stack_embeddings(embeddings_list: list[np.ndarray]) -> np.ndarray | None:
    try:
        return np.stack(embeddings_list, axis=0)
    except ValueError:
        return None


def _faiss_search(
    *,
    embeddings_array: np.ndarray,
    query_embedding: list[float],
    id_list: list[int],
    limit: int,
) -> list[tuple[int, float]]:
    faiss.normalize_L2(embeddings_array)
    query_array = np.asarray([query_embedding], dtype=np.float32)

    if embeddings_array.shape[1] != query_array.shape[1]:
        logger.debug(
            "Embedding dimension mismatch: db=%d, query=%d",
            embeddings_array.shape[1],
            query_array.shape[1],
        )
        return []

    faiss.normalize_L2(query_array)

    index = faiss.IndexFlatIP(embeddings_array.shape[1])
    index.add(embeddings_array)  # type: ignore[call-arg]

    k = min(limit, len(embeddings_array))
    similarities, indices = index.search(query_array, k)  # type: ignore[call-arg]

    results: list[tuple[int, float]] = []
    for result_idx, embedding_idx in enumerate(indices[0]):
        if 0 <= embedding_idx < len(id_list):
            results.append((id_list[embedding_idx], float(similarities[0][result_idx])))

    return results


def find_similar_embeddings(
    embeddings: list[tuple[int, Any]],
    query_embedding: list[float],
    limit: int = 5,
) -> list[tuple[int, float]]:
    """Find most similar embeddings using FAISS cosine similarity."""
    if not embeddings:
        logger.debug("find_similar_embeddings called with empty embeddings")
        return []

    query_dim = _query_dim(query_embedding)
    if query_dim == 0:
        return []

    embeddings_list, id_list = _parse_valid_embeddings(embeddings, query_dim=query_dim)

    if not embeddings_list:
        logger.debug("No valid embeddings after parsing")
        return []

    logger.debug("Building FAISS index with %d embeddings", len(embeddings_list))
    embeddings_array = _stack_embeddings(embeddings_list)
    if embeddings_array is None:
        return []

    results = _faiss_search(
        embeddings_array=embeddings_array,
        query_embedding=query_embedding,
        id_list=id_list,
        limit=limit,
    )

    if results:
        scores = [round(score, 3) for _, score in results]
        logger.debug(
            "FAISS similarity search complete - top %d matches: %s",
            len(results),
            scores,
        )

    return results
