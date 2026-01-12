from __future__ import annotations

from pathlib import Path
from typing import cast

from benchmarks.locomo._run_impl import _format_top_k
from benchmarks.locomo.provenance import attribute_facts_to_turn_ids


def test_attribute_facts_to_turn_ids_maps_best_match():
    turn_ids = ["D1:3", "D1:12"]
    turn_embeddings = [
        [1.0, 0.0],  # D1:3
        [0.0, 1.0],  # D1:12
    ]

    fact_ids = [101, 102]
    fact_embeddings = [
        [0.9, 0.1],  # should map to D1:3
        [0.1, 0.9],  # should map to D1:12
    ]

    mapping = attribute_facts_to_turn_ids(
        turn_ids=turn_ids,
        turn_embeddings=turn_embeddings,
        fact_ids=fact_ids,
        fact_embeddings=fact_embeddings,
        top_n=1,
        min_score=0.0,
    )

    assert mapping[101][0][0] == "D1:3"
    assert mapping[102][0][0] == "D1:12"


def test_attribute_facts_to_turn_ids_can_use_lexical_signal(monkeypatch):
    # Force semantic similarity to prefer the wrong turn, then lexical should fix it.
    monkeypatch.setattr(
        "benchmarks.locomo.provenance.find_similar_embeddings",
        lambda _embs, _q, limit=5: [(0, 0.9), (1, 0.8)][:limit],
    )

    turn_ids = ["D1:1", "D1:2"]
    turn_embeddings = [[1.0, 0.0], [0.0, 1.0]]
    turn_texts = ["Alice loves skiing", "Alice loves painting blue skies"]

    fact_ids = [201]
    fact_embeddings = [[1.0, 0.0]]
    fact_texts = ["blue painting"]

    mapping = attribute_facts_to_turn_ids(
        turn_ids=turn_ids,
        turn_embeddings=turn_embeddings,
        turn_texts=turn_texts,
        fact_ids=fact_ids,
        fact_embeddings=fact_embeddings,
        fact_texts=fact_texts,
        top_n=1,
        min_score=0.0,
    )
    assert mapping[201][0][0] == "D1:2"


def test_provenance_store_delete_sample(tmp_path: Path):
    from benchmarks.locomo.provenance import FactAttribution, ProvenanceStore

    store = ProvenanceStore(tmp_path / "prov.sqlite")
    store.upsert_many(
        [FactAttribution(fact_id=1, dia_id="D1:1", score=0.5)],
        run_id="r1",
        sample_id="s1",
    )
    assert store.has_any(run_id="r1", sample_id="s1") is True

    store.delete_sample(run_id="r1", sample_id="s1")
    assert store.has_any(run_id="r1", sample_id="s1") is False


def test_group_scoring_any_match():
    from benchmarks.locomo.scoring import hit_at_k_groups, mrr_groups

    relevant = {"D7:19"}
    retrieved = [
        {"D7:20", "D7:19"},  # rank 1 contains the relevant ID
        {"D1:1"},
    ]
    assert hit_at_k_groups(relevant, retrieved, 1) == 1.0
    assert mrr_groups(relevant, retrieved) == 1.0


def test_format_top_k_includes_turn_ids_for_aa_mode(tmp_path: Path):
    # This validates the intended contract: AA-mode facts can map to multiple dia_ids.
    from benchmarks.locomo.provenance import FactAttribution, ProvenanceStore

    prov = ProvenanceStore(tmp_path / "prov.sqlite")
    prov.upsert_many(
        [
            FactAttribution(fact_id=1, dia_id="D7:19", score=1.0),
            FactAttribution(fact_id=1, dia_id="D7:20", score=0.9),
        ],
        run_id="r1",
        sample_id="s1",
    )

    retrieved_ids: list[str] = []
    retrieved_groups: list[set[str]] = []
    top_k = _format_top_k(
        results=[{"id": 1, "content": "some fact", "similarity": 0.5}],
        prov_store=prov,
        run_id="r1",
        sample_id="s1",
        retrieved_ids_out=retrieved_ids,
        retrieved_groups_out=retrieved_groups,
        provenance_limit=3,
    )
    assert top_k[0]["turn_id"] in {"D7:19", "D7:20"}
    turn_ids = cast(list[str], top_k[0]["turn_ids"])
    assert set(turn_ids) == {"D7:19", "D7:20"}
    assert retrieved_groups[0] == {"D7:19", "D7:20"}
