from __future__ import annotations

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
