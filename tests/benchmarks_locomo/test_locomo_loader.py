from __future__ import annotations

from pathlib import Path

from benchmarks.locomo.loader import load_locomo_json


def test_load_locomo_tiny_fixture():
    path = Path(__file__).parent / "fixtures" / "locomo_tiny.json"
    samples = load_locomo_json(path)

    assert len(samples) == 1
    sample = samples[0]
    assert sample.sample_id == "sample-001"
    assert len(sample.sessions) == 1
    assert len(sample.sessions[0].turns) == 2
    assert len(sample.qa) == 2
