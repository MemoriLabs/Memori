r"""
 __  __                           _
|  \/  | ___ _ __ ___   ___  _ __(_)
| |\/| |/ _ \ '_ ` _ \ / _ \| '__| |
| |  | |  __/ | | | | | (_) | |  | |
|_|  |_|\___|_| |_| |_|\___/|_|  |_|
                 perfectam memoriam
                      memorilabs.ai
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass

from memori._config import Config
from memori.memory.recall import Recall


def _pack_embedding(vec: list[float]) -> bytes:
    return struct.pack(f"<{len(vec)}f", *vec)


@dataclass(frozen=True)
class _Case:
    query: str
    expected_top_ids: set[int]


class _FakeEntityFactDriver:
    def __init__(self, *, facts: dict[int, str], embeddings: dict[int, list[float]]):
        self._facts = dict(facts)
        self._embeddings = dict(embeddings)

    def get_embeddings(self, entity_id: int, limit: int = 1000):
        _ = entity_id
        rows = []
        for fid, vec in self._embeddings.items():
            rows.append({"id": fid, "content_embedding": _pack_embedding(vec)})
        return rows[:limit]

    def get_facts_by_ids(self, fact_ids: list[int]):
        rows = []
        for fid in fact_ids:
            content = self._facts.get(fid)
            if content is not None:
                rows.append({"id": fid, "content": content})
        return rows


class _FakeEntityDriver:
    def create(self, entity_id: str) -> int:
        _ = entity_id
        return 1


class _FakeStorageDriver:
    def __init__(self, entity_fact: _FakeEntityFactDriver):
        self.entity = _FakeEntityDriver()
        self.entity_fact = entity_fact


class _FakeStorage:
    def __init__(self, driver: _FakeStorageDriver):
        self.driver = driver


def _recall_at_k(*, cases: list[_Case], results_by_query: dict[str, list[int]], k: int):
    hits = 0
    for c in cases:
        got = results_by_query.get(c.query, [])[:k]
        if any(fid in c.expected_top_ids for fid in got):
            hits += 1
    return hits / (len(cases) or 1)


def _mrr_at_k(*, cases: list[_Case], results_by_query: dict[str, list[int]], k: int):
    rr_sum = 0.0
    for c in cases:
        got = results_by_query.get(c.query, [])[:k]
        rr = 0.0
        for idx, fid in enumerate(got, start=1):
            if fid in c.expected_top_ids:
                rr = 1.0 / float(idx)
                break
        rr_sum += rr
    return rr_sum / (len(cases) or 1)


def _ndcg_at_k(*, cases: list[_Case], results_by_query: dict[str, list[int]], k: int):
    def dcg(ids: list[int], rel: set[int]) -> float:
        total = 0.0
        for i, fid in enumerate(ids[:k], start=1):
            gain = 1.0 if fid in rel else 0.0
            total += gain / math.log2(i + 1.0)
        return total

    ndcg_sum = 0.0
    for c in cases:
        got = results_by_query.get(c.query, [])[:k]
        ideal = list(c.expected_top_ids)[:k]
        denom = dcg(ideal, c.expected_top_ids)
        ndcg_sum += 0.0 if denom == 0.0 else (dcg(got, c.expected_top_ids) / denom)
    return ndcg_sum / (len(cases) or 1)


def test_recall_eval_harness_reports_expected_metrics(mocker):
    facts = {
        1: "Favorite color is blue.",
        2: "Lives in New York City.",
        3: "Likes pizza.",
        4: "Prefers decaf coffee.",
        5: "Dog is named Miso.",
    }

    # 3D toy embedding space: each fact sits near an axis.
    embeddings = {
        1: [1.0, 0.0, 0.0],
        2: [0.0, 1.0, 0.0],
        3: [0.0, 0.0, 1.0],
        4: [0.8, 0.2, 0.0],
        5: [0.0, 0.7, 0.3],
    }

    query_to_embedding = {
        "What's my favorite color?": [0.95, 0.05, 0.0],
        "Where do I live?": [0.05, 0.95, 0.0],
        "What food do I like?": [0.05, 0.05, 0.9],
    }

    def _embed_side_effect(text, *, model):
        _ = model
        if isinstance(text, list):
            return [query_to_embedding[t] for t in text]
        return [query_to_embedding[text]]

    mocker.patch("memori.memory.recall.embed_texts", side_effect=_embed_side_effect)

    cfg = Config()
    cfg.storage = _FakeStorage(
        _FakeStorageDriver(_FakeEntityFactDriver(facts=facts, embeddings=embeddings))
    )
    cfg.entity_id = "entity-1"

    recall = Recall(cfg)

    cases = [
        _Case(query="What's my favorite color?", expected_top_ids={1}),
        _Case(query="Where do I live?", expected_top_ids={2}),
        _Case(query="What food do I like?", expected_top_ids={3}),
    ]

    results_by_query: dict[str, list[int]] = {}
    for c in cases:
        rows = recall.search_facts(query=c.query, limit=3, entity_id=1)
        results_by_query[c.query] = [int(r.id) for r in rows]

    recall_at_1 = _recall_at_k(cases=cases, results_by_query=results_by_query, k=1)
    mrr_at_3 = _mrr_at_k(cases=cases, results_by_query=results_by_query, k=3)
    ndcg_at_3 = _ndcg_at_k(cases=cases, results_by_query=results_by_query, k=3)

    assert recall_at_1 == 1.0
    assert mrr_at_3 == 1.0
    assert ndcg_at_3 == 1.0
