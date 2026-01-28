from __future__ import annotations

import math
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from memori.search import find_similar_embeddings


@dataclass(frozen=True, slots=True)
class FactAttribution:
    fact_id: int
    dia_id: str
    score: float


class ProvenanceStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self.path), check_same_thread=False)

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS bench_locomo_fact_provenance(
                    run_id TEXT NOT NULL,
                    sample_id TEXT NOT NULL,
                    fact_id INTEGER NOT NULL,
                    dia_id TEXT NOT NULL,
                    score REAL NOT NULL,
                    PRIMARY KEY (run_id, sample_id, fact_id, dia_id)
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_bench_locomo_fact_prov_lookup
                ON bench_locomo_fact_provenance(run_id, sample_id, fact_id, score DESC)
                """
            )
            conn.commit()

    def upsert_many(
        self, rows: list[FactAttribution], *, run_id: str, sample_id: str
    ) -> None:
        if not rows:
            return
        with self._connect() as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO bench_locomo_fact_provenance(
                    run_id, sample_id, fact_id, dia_id, score
                ) VALUES (?, ?, ?, ?, ?)
                """,
                [
                    (run_id, sample_id, r.fact_id, r.dia_id, float(r.score))
                    for r in rows
                ],
            )
            conn.commit()

    def best_dia_ids_for_fact(
        self, *, run_id: str, sample_id: str, fact_id: int, limit: int = 1
    ) -> list[str]:
        if limit <= 0:
            return []
        with self._connect() as conn:
            cur = conn.execute(
                """
                SELECT dia_id
                  FROM bench_locomo_fact_provenance
                 WHERE run_id = ? AND sample_id = ? AND fact_id = ?
                 ORDER BY score DESC
                 LIMIT ?
                """,
                (run_id, sample_id, fact_id, limit),
            )
            return [r[0] for r in cur.fetchall() if r and r[0]]

    def has_any(self, *, run_id: str, sample_id: str) -> bool:
        with self._connect() as conn:
            cur = conn.execute(
                """
                SELECT 1
                  FROM bench_locomo_fact_provenance
                 WHERE run_id = ? AND sample_id = ?
                 LIMIT 1
                """,
                (run_id, sample_id),
            )
            return cur.fetchone() is not None

    def delete_sample(self, *, run_id: str, sample_id: str) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                DELETE FROM bench_locomo_fact_provenance
                 WHERE run_id = ? AND sample_id = ?
                """,
                (run_id, sample_id),
            )
            conn.commit()


def attribute_facts_to_turn_ids(
    *,
    turn_ids: list[str],
    turn_embeddings: list[list[float]],
    turn_texts: list[str] | None = None,
    fact_ids: list[int],
    fact_embeddings: list[list[float]],
    fact_texts: list[str] | None = None,
    top_n: int = 1,
    min_score: float | None = None,
) -> dict[int, list[tuple[str, float]]]:
    """
    Map each fact to the most similar LoCoMo turn_id(s).

    This is intentionally heuristic: it enables benchmark-only provenance without
    changing Memori's product schema.
    """
    if top_n <= 0:
        return {}
    if len(turn_ids) != len(turn_embeddings):
        raise ValueError("turn_ids and turn_embeddings must be the same length")
    if len(fact_ids) != len(fact_embeddings):
        raise ValueError("fact_ids and fact_embeddings must be the same length")
    if turn_texts is not None and len(turn_texts) != len(turn_ids):
        raise ValueError("turn_texts and turn_ids must be the same length")
    if fact_texts is not None and len(fact_texts) != len(fact_ids):
        raise ValueError("fact_texts and fact_ids must be the same length")

    embeddings = list(enumerate(turn_embeddings))
    out: dict[int, list[tuple[str, float]]] = {}
    for i, (fact_id, qemb) in enumerate(zip(fact_ids, fact_embeddings, strict=True)):
        # Use a larger semantic pool before reranking to improve coverage.
        semantic_pool = max(top_n, min(len(turn_ids), max(top_n * 10, 50)))
        similar = find_similar_embeddings(embeddings, qemb, limit=semantic_pool)

        fact_text = (fact_texts[i] if fact_texts is not None else "") or ""
        lexical_scores = (
            _lexical_scores(query_text=fact_text, docs=turn_texts)
            if turn_texts is not None
            else None
        )

        mapped: list[tuple[str, float]] = []
        scored: list[tuple[int, float]] = []
        for raw_idx, score in similar:
            # idx is always int here (from enumerate), cast for type checker
            idx = cast(int, raw_idx)
            if idx < 0 or idx >= len(turn_ids):
                continue
            lex = float(lexical_scores[idx]) if lexical_scores is not None else 0.0
            combined = (0.8 * float(score)) + (0.2 * lex)
            scored.append((idx, combined))

        scored.sort(key=lambda t: t[1], reverse=True)
        for idx, score in scored[:top_n]:
            if min_score is not None and score < min_score:
                continue
            mapped.append((turn_ids[idx], float(score)))

        # If the threshold filtered everything, prefer some attribution over none.
        if not mapped and scored:
            idx_best, score_best = scored[0]
            mapped.append((turn_ids[idx_best], float(score_best)))

        out[int(fact_id)] = mapped
    return out


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


def _lexical_scores(*, query_text: str, docs: list[str]) -> list[float]:
    q_tokens = _tokenize(query_text)
    if not q_tokens or not docs:
        return [0.0 for _ in docs]

    doc_tokens = [set(_tokenize(d)) for d in docs]
    n = float(len(docs)) or 1.0
    df: dict[str, int] = {}
    for t in set(q_tokens):
        df[t] = sum(1 for toks in doc_tokens if t in toks)
    idf = {t: (math.log((n + 1.0) / (float(df[t]) + 1.0)) + 1.0) for t in df}
    denom = sum(idf.get(t, 0.0) for t in q_tokens) or 1.0

    out: list[float] = []
    for toks in doc_tokens:
        num = sum(idf.get(t, 0.0) for t in q_tokens if t in toks)
        out.append(float(num / denom))
    return out
