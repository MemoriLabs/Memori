from __future__ import annotations

import json
from pathlib import Path

from benchmarks.locomo.preprocess import preprocess_locomo_json


def test_preprocess_strips_multimodal_fields(tmp_path: Path) -> None:
    src = tmp_path / "in.json"
    dst = tmp_path / "out.json"
    src.write_text(
        json.dumps(
            [
                {
                    "sample_id": "conv-1",
                    "conversation": {
                        "speaker_a": "A",
                        "speaker_b": "B",
                        "session_1": [
                            {"speaker": "A", "dia_id": "D1:1", "text": "hi"},
                            {
                                "speaker": "B",
                                "dia_id": "D1:2",
                                "text": "look",
                                "img_url": ["https://example.com/x.jpg"],
                            },
                        ],
                    },
                    "qa": [],
                }
            ]
        ),
        encoding="utf-8",
    )

    stats = preprocess_locomo_json(src, dst)
    assert stats["samples_in"] == 1
    assert stats["samples_out"] == 1
    assert stats["removed_multimodal_turns"] == 1

    out = json.loads(dst.read_text(encoding="utf-8"))
    turns = out[0]["conversation"]["session_1"]
    assert "img_url" not in turns[1]


def test_preprocess_rewrites_speaker_b_to_assistant(tmp_path: Path) -> None:
    src = tmp_path / "in.json"
    dst = tmp_path / "out.json"
    src.write_text(
        json.dumps(
            [
                {
                    "sample_id": "conv-1",
                    "conversation": {
                        "speaker_a": "A",
                        "speaker_b": "B",
                        "session_1": [
                            {"speaker": "A", "dia_id": "D1:1", "text": "hi"},
                            {"speaker": "B", "dia_id": "D1:2", "text": "hello"},
                        ],
                    },
                    "qa": [],
                }
            ]
        ),
        encoding="utf-8",
    )

    preprocess_locomo_json(src, dst)
    out = json.loads(dst.read_text(encoding="utf-8"))
    turns = out[0]["conversation"]["session_1"]
    assert turns[0]["speaker"] == "user"
    assert turns[1]["speaker"] == "assistant"


def test_preprocess_keeps_speakers_when_speaker_b_missing(tmp_path: Path) -> None:
    src = tmp_path / "in.json"
    dst = tmp_path / "out.json"
    src.write_text(
        json.dumps(
            [
                {
                    "sample_id": "conv-1",
                    "conversation": {
                        "speaker_a": "A",
                        "session_1": [
                            {"speaker": "A", "dia_id": "D1:1", "text": "hi"},
                            {"speaker": "B", "dia_id": "D1:2", "text": "hello"},
                        ],
                    },
                    "qa": [],
                }
            ]
        ),
        encoding="utf-8",
    )

    preprocess_locomo_json(src, dst)
    out = json.loads(dst.read_text(encoding="utf-8"))
    turns = out[0]["conversation"]["session_1"]
    assert turns[0]["speaker"] == "A"
    assert turns[1]["speaker"] == "B"
