from __future__ import annotations

import json
from pathlib import Path

from benchmarks.locomo.loader import load_locomo_json


def _write_locomo_tiny(path: Path) -> Path:
    data = [
        {
            "sample_id": "sample-001",
            "conversation": [
                {
                    "session_id": "session-1",
                    "dialogue": [
                        {
                            "turn_id": "t0",
                            "speaker": "user",
                            "text": "My favorite color is blue.",
                        },
                        {"turn_id": "t1", "speaker": "assistant", "text": "Got it."},
                    ],
                }
            ],
            "qa": [
                {
                    "question_id": "q0",
                    "question": "What is my favorite color?",
                    "answer": "blue",
                    "evidence": 0,
                },
                {
                    "question_id": "q1",
                    "question": "Which color do I like best?",
                    "answer": "blue",
                    "evidence": 0,
                },
            ],
        }
    ]
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_load_locomo_tiny_fixture(tmp_path: Path):
    path = _write_locomo_tiny(tmp_path / "locomo_tiny.json")
    samples = load_locomo_json(path)

    assert len(samples) == 1
    sample = samples[0]
    assert sample.sample_id == "sample-001"
    assert len(sample.sessions) == 1
    assert len(sample.sessions[0].turns) == 2
    assert len(sample.qa) == 2
