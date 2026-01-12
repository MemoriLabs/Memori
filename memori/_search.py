r"""
 __  __                           _
|  \/  | ___ _ __ ___   ___  _ __(_)
| |\/| |/ _ \ '_ ` _ \ / _ \| '__| |
| |  | |  __/ | | | | | (_) | |  | |
|_|  |_|\___|_| |_| |_|\___/|_|  |_|
                 perfectam memoriam
                      memorilabs.ai
"""

import json
import logging
import math
import re
from typing import Any

import faiss
import numpy as np

logger = logging.getLogger(__name__)


def parse_embedding(raw) -> np.ndarray:
    """Parse embedding from database format to numpy array.

    Handles multiple storage formats:
    - Binary (BYTEA/BLOB/BinData): Most common, used by all databases
    - JSON string: Legacy format
    - Native array: Fallback
    """
    if isinstance(raw, bytes | memoryview):
        return np.frombuffer(raw, dtype="<f4")
    elif isinstance(raw, str):
        # Legacy JSON format
        return np.array(json.loads(raw), dtype=np.float32)
    else:
        # Try to extract bytes from bson.Binary or other wrappers
        if hasattr(raw, "__bytes__"):
            return np.frombuffer(bytes(raw), dtype="<f4")
        # Fallback to native array (MongoDB array format)
        return np.asarray(raw, dtype=np.float32)


def find_similar_embeddings(
    embeddings: list[tuple[int, Any]],
    query_embedding: list[float],
    limit: int = 5,
) -> list[tuple[int, float]]:
    """Find most similar embeddings using FAISS cosine similarity.

    Args:
        embeddings: List of (id, embedding_raw) tuples
        query_embedding: Query embedding as list of floats
        limit: Number of results to return

    Returns:
        List of (id, similarity_score) tuples, sorted by similarity desc
    """
    if not embeddings:
        logger.debug("find_similar_embeddings called with empty embeddings")
        return []

    query_dim = len(query_embedding)
    if query_dim == 0:
        return []

    embeddings_list = []
    id_list = []

    for fact_id, raw in embeddings:
        try:
            parsed = parse_embedding(raw)
            if parsed.ndim != 1 or parsed.shape[0] != query_dim:
                continue
            embeddings_list.append(parsed)
            id_list.append(fact_id)
        except Exception:
            continue

    if not embeddings_list:
        logger.debug("No valid embeddings after parsing")
        return []

    logger.debug("Building FAISS index with %d embeddings", len(embeddings_list))
    try:
        embeddings_array = np.stack(embeddings_list, axis=0)
    except ValueError:
        return []

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

    results = []
    for result_idx, embedding_idx in enumerate(indices[0]):
        if embedding_idx >= 0 and embedding_idx < len(id_list):
            results.append((id_list[embedding_idx], float(similarities[0][result_idx])))

    if results:
        scores = [round(score, 3) for _, score in results]
        logger.debug(
            "FAISS similarity search complete - top %d matches: %s",
            len(results),
            scores,
        )

    return results


def search_entity_facts(
    entity_fact_driver,
    entity_id: int,
    query_embedding: list[float],
    limit: int,
    embeddings_limit: int,
    *,
    query_text: str | None = None,
) -> list[dict]:
    """Search entity facts by embedding similarity.

    Args:
        entity_fact_driver: Driver instance with get_embeddings and get_facts_by_ids methods
        entity_id: Entity ID to search within
        query_embedding: Query embedding as list of floats
        limit: Number of results to return
        embeddings_limit: Number of embeddings to retrieve from database

    Returns:
        List of dicts with keys: id, content, similarity
    """
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
    embeddings = [(row["id"], row["content_embedding"]) for row in results]

    # When query_text is provided, we retrieve a larger semantic candidate pool and
    # perform a lightweight lexical rerank to improve exact-evidence retrieval.
    candidate_limit = int(limit)
    if query_text:
        candidate_limit = max(limit, min(len(embeddings), max(limit * 10, 50)))

    similar = find_similar_embeddings(embeddings, query_embedding, candidate_limit)

    if not similar:
        logger.debug("No similar embeddings found")
        return []

    candidate_ids = [fact_id for fact_id, _ in similar]
    similarities_map = dict(similar)

    logger.debug("Fetching content for %d fact IDs", len(candidate_ids))
    content_results = entity_fact_driver.get_facts_by_ids(candidate_ids)
    content_map = {row["id"]: row["content"] for row in content_results}

    if query_text:
        reranked = _rerank_by_lexical_overlap(
            query_text=query_text,
            candidate_ids=candidate_ids,
            content_map=content_map,
            similarities_map=similarities_map,
        )
        ordered_ids = reranked[:limit]
    else:
        ordered_ids = candidate_ids[:limit]

    facts_with_similarity: list[dict] = []
    for fact_id in ordered_ids:
        content = content_map.get(fact_id)
        if content is None:
            continue
        row = {
            "id": fact_id,
            "content": content,
            "similarity": float(similarities_map.get(fact_id, 0.0)),
        }
        facts_with_similarity.append(row)

    if query_text and facts_with_similarity:
        # Populate rank_score/lexical_score (non-breaking additional keys).
        # This keeps "similarity" as cosine similarity while ranking may differ.
        scores = _lexical_scores_for_ids(
            query_text=query_text,
            ids=[r["id"] for r in facts_with_similarity],
            content_map=content_map,
        )
        for r in facts_with_similarity:
            lex = float(scores.get(r["id"], 0.0))
            cos = float(r["similarity"])
            r["lexical_score"] = lex
            r["rank_score"] = (0.85 * cos) + (0.15 * lex)

    logger.debug(
        "Returning %d facts with similarity scores", len(facts_with_similarity)
    )
    return facts_with_similarity


_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "did",
    "do",
    "does",
    "for",
    "from",
    "had",
    "has",
    "have",
    "how",
    "i",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "then",
    "there",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
    "you",
    "your",
}


def _tokenize(text: str) -> list[str]:
    tokens = [t for t in _TOKEN_RE.findall((text or "").lower()) if t]
    return [t for t in tokens if t not in _STOPWORDS]


def _lexical_scores_for_ids(
    *, query_text: str, ids: list[int], content_map: dict[int, str]
) -> dict[int, float]:
    """
    Compute a simple IDF-weighted token overlap score in [0, 1] for each doc.
    """
    q_tokens = _tokenize(query_text)
    if not q_tokens:
        return dict.fromkeys(ids, 0.0)

    docs: dict[int, set[str]] = {}
    for i in ids:
        content = content_map.get(i, "")
        docs[i] = set(_tokenize(content))

    # IDF over candidate docs.
    n = float(len(ids)) or 1.0
    df: dict[str, int] = {}
    for t in set(q_tokens):
        df[t] = sum(1 for i in ids if t in docs.get(i, set()))
    idf = {t: (math.log((n + 1.0) / (float(df[t]) + 1.0)) + 1.0) for t in df}

    denom = sum(idf.get(t, 0.0) for t in q_tokens) or 1.0
    out: dict[int, float] = {}
    for i in ids:
        doc_tokens = docs.get(i, set())
        num = sum(idf.get(t, 0.0) for t in q_tokens if t in doc_tokens)
        out[i] = float(num / denom)
    return out


def _rerank_by_lexical_overlap(
    *,
    query_text: str,
    candidate_ids: list[int],
    content_map: dict[int, str],
    similarities_map: dict[int, float],
) -> list[int]:
    scores = _lexical_scores_for_ids(
        query_text=query_text, ids=candidate_ids, content_map=content_map
    )

    def key(fid: int) -> tuple[float, float]:
        cos = float(similarities_map.get(fid, 0.0))
        lex = float(scores.get(fid, 0.0))
        return ((0.85 * cos) + (0.15 * lex), cos)

    return sorted(candidate_ids, key=key, reverse=True)
