from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

from memori._search import find_similar_embeddings


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


def attribute_facts_to_turn_ids(
    *,
    turn_ids: list[str],
    turn_embeddings: list[list[float]],
    fact_ids: list[int],
    fact_embeddings: list[list[float]],
    top_n: int = 1,
    min_score: float | None = None,
) -> dict[int, list[tuple[str, float]]]:
    """
    Map each fact to the most similar LoCoMo turn_id(s) using cosine similarity.

    This is intentionally heuristic: it enables benchmark-only provenance without
    changing Memori's product schema.
    """
    if top_n <= 0:
        return {}
    if len(turn_ids) != len(turn_embeddings):
        raise ValueError("turn_ids and turn_embeddings must be the same length")
    if len(fact_ids) != len(fact_embeddings):
        raise ValueError("fact_ids and fact_embeddings must be the same length")

    embeddings = list(enumerate(turn_embeddings))
    out: dict[int, list[tuple[str, float]]] = {}
    for fact_id, qemb in zip(fact_ids, fact_embeddings, strict=True):
        similar = find_similar_embeddings(embeddings, qemb, limit=top_n)
        mapped: list[tuple[str, float]] = []
        for idx, score in similar:
            if idx < 0 or idx >= len(turn_ids):
                continue
            if min_score is not None and score < min_score:
                continue
            mapped.append((turn_ids[idx], float(score)))
        out[int(fact_id)] = mapped
    return out
