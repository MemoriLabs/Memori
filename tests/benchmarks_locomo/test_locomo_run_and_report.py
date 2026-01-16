from __future__ import annotations

import json
from pathlib import Path

from benchmarks.locomo.report import main as report_main
from benchmarks.locomo.run import main as run_main


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


def _fake_embed_texts(
    texts: str | list[str], model: str, fallback_dimension: int
) -> list[list[int | float]]:
    if isinstance(texts, str):
        items = [texts]
    else:
        items = list(texts)

    def v(s: str) -> list[float]:
        s = s.lower()
        return [
            1.0 if "favorite" in s else 0.0,
            1.0 if "color" in s else 0.0,
            1.0 if "blue" in s else 0.0,
        ]

    return [v(x) for x in items]


def _seed_reuse_db(
    *, sqlite_db: Path, provenance_db: Path, run_id: str, sample_id: str
) -> None:
    import sqlite3

    from benchmarks.locomo.provenance import FactAttribution, ProvenanceStore
    from memori import Memori

    def _conn():
        return sqlite3.connect(str(sqlite_db), check_same_thread=False)

    mem = Memori(conn=_conn)
    mem.config.storage.build()

    entity_external_id = f"locomo:{run_id}:{sample_id}"
    entity_db_id = mem.config.storage.driver.entity.create(entity_external_id)

    contents = [
        "My favorite color is blue.",
        "Got it.",
    ]
    embeddings = _fake_embed_texts(
        contents,
        model=mem.config.embeddings.model,
        fallback_dimension=mem.config.embeddings.fallback_dimension,
    )
    mem.config.storage.driver.entity_fact.create(entity_db_id, contents, embeddings)

    with sqlite3.connect(str(sqlite_db), check_same_thread=False) as conn2:
        rows = conn2.execute(
            "SELECT id, content FROM memori_entity_fact WHERE entity_id = ? ORDER BY id ASC",
            (int(entity_db_id),),
        ).fetchall()
    fact_id_by_content = {r[1]: int(r[0]) for r in rows if r and r[0] and r[1]}

    prov = ProvenanceStore(provenance_db)
    prov.upsert_many(
        [
            FactAttribution(
                fact_id=fact_id_by_content["My favorite color is blue."],
                dia_id="t0",
                score=1.0,
            ),
            FactAttribution(
                fact_id=fact_id_by_content["Got it."],
                dia_id="t1",
                score=1.0,
            ),
        ],
        run_id=run_id,
        sample_id=sample_id,
    )


def test_run_writes_predictions_and_summary(tmp_path: Path, monkeypatch):
    dataset = _write_locomo_tiny(tmp_path / "locomo_tiny.json")
    out_dir = tmp_path / "run"
    sqlite_db = tmp_path / "seeded.sqlite"
    provenance_db = tmp_path / "seeded_prov.sqlite"
    run_id = "test-run"
    _seed_reuse_db(
        sqlite_db=sqlite_db,
        provenance_db=provenance_db,
        run_id=run_id,
        sample_id="sample-001",
    )

    # Prevent embedding model downloads during tests.
    import benchmarks.locomo._run_impl as run_impl_mod
    import memori.memory.recall as recall_mod

    # run_mod.embed_texts = _fake_embed_texts
    monkeypatch.setattr(run_impl_mod, "embed_texts", _fake_embed_texts)
    monkeypatch.setattr(recall_mod, "embed_texts", _fake_embed_texts)

    rc = run_main(
        [
            "--dataset",
            str(dataset),
            "--out",
            str(out_dir),
            "--sqlite-db",
            str(sqlite_db),
            "--provenance-db",
            str(provenance_db),
            "--reuse-db",
            "--run-id",
            run_id,
        ]
    )
    assert rc == 0

    predictions = out_dir / "predictions.jsonl"
    summary = out_dir / "summary.json"
    assert predictions.exists()
    assert summary.exists()

    lines = predictions.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    row0 = json.loads(lines[0])
    assert row0["sample_id"] == "sample-001"
    assert row0["retrieval"]["status"] == "ok"
    assert row0["retrieval"]["metrics"]["hit@1"] == 1.0
    assert row0["retrieval"]["metrics"]["mrr"] == 1.0

    summary_obj = json.loads(summary.read_text(encoding="utf-8"))
    assert summary_obj["sample_count"] == 1
    assert summary_obj["question_count"] == 2
    assert summary_obj["metrics_overall"]["hit@1"] == 1.0
    assert summary_obj["metrics_overall"]["mrr"] == 1.0


def test_run_skips_questions_with_evidence_in_removed_sessions(
    tmp_path: Path, monkeypatch
):
    dataset = tmp_path / "locomo_two_sessions.json"
    data = [
        {
            "sample_id": "sample-001",
            "conversation": [
                {
                    "session_id": "s1",
                    "dialogue": [
                        {"turn_id": "D1:1", "speaker": "user", "text": "A"},
                        {"turn_id": "D1:2", "speaker": "assistant", "text": "B"},
                    ],
                },
                {
                    "session_id": "s2",
                    "dialogue": [
                        {"turn_id": "D2:1", "speaker": "user", "text": "C"},
                        {"turn_id": "D2:2", "speaker": "assistant", "text": "D"},
                    ],
                },
            ],
            "qa": [
                {
                    "question_id": "q_in_s1",
                    "question": "What was the first user message?",
                    "answer": "A",
                    "evidence": ["D1:1"],
                },
                {
                    "question_id": "q_in_s2",
                    "question": "What was the second session user message?",
                    "answer": "C",
                    "evidence": ["D2:1"],
                },
            ],
        }
    ]
    dataset.write_text(json.dumps(data), encoding="utf-8")

    # Prevent embedding model downloads during tests.
    import benchmarks.locomo._run_impl as run_impl_mod
    import memori.memory.recall as recall_mod

    monkeypatch.setattr(run_impl_mod, "embed_texts", _fake_embed_texts)
    monkeypatch.setattr(recall_mod, "embed_texts", _fake_embed_texts)

    sqlite_db = tmp_path / "seeded.sqlite"
    provenance_db = tmp_path / "seeded_prov.sqlite"
    run_id = "test-run"
    _seed_reuse_db(
        sqlite_db=sqlite_db,
        provenance_db=provenance_db,
        run_id=run_id,
        sample_id="sample-001",
    )

    out_dir = tmp_path / "run"
    rc = run_main(
        [
            "--dataset",
            str(dataset),
            "--out",
            str(out_dir),
            "--sqlite-db",
            str(sqlite_db),
            "--provenance-db",
            str(provenance_db),
            "--reuse-db",
            "--run-id",
            run_id,
            "--max-sessions",
            "1",
        ]
    )
    assert rc == 0

    lines = (out_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
    # q_in_s2 should be skipped because its evidence is in session 2 (removed).
    assert len(lines) == 1
    row0 = json.loads(lines[0])
    assert row0["question_id"] == "q_in_s1"


def test_report_aggregates_predictions(tmp_path: Path, monkeypatch):
    dataset = _write_locomo_tiny(tmp_path / "locomo_tiny.json")
    out_dir = tmp_path / "run"
    sqlite_db = tmp_path / "seeded.sqlite"
    provenance_db = tmp_path / "seeded_prov.sqlite"
    run_id = "test-run"
    _seed_reuse_db(
        sqlite_db=sqlite_db,
        provenance_db=provenance_db,
        run_id=run_id,
        sample_id="sample-001",
    )

    import benchmarks.locomo._run_impl as run_impl_mod
    import memori.memory.recall as recall_mod

    # run_mod.embed_texts = _fake_embed_texts
    monkeypatch.setattr(run_impl_mod, "embed_texts", _fake_embed_texts)
    monkeypatch.setattr(recall_mod, "embed_texts", _fake_embed_texts)

    run_main(
        [
            "--dataset",
            str(dataset),
            "--out",
            str(out_dir),
            "--sqlite-db",
            str(sqlite_db),
            "--provenance-db",
            str(provenance_db),
            "--reuse-db",
            "--run-id",
            run_id,
        ]
    )

    summary_out = tmp_path / "summary.json"
    rc = report_main(
        [
            "--predictions",
            str(out_dir / "predictions.jsonl"),
            "--out",
            str(summary_out),
        ]
    )
    assert rc == 0
    summary_obj = json.loads(summary_out.read_text(encoding="utf-8"))
    assert summary_obj["question_count"] == 2
    assert summary_obj["metrics_overall"]["hit@1"] == 1.0


def test_run_can_reuse_existing_sqlite_db_without_ingestion(
    tmp_path: Path, monkeypatch
):
    dataset = _write_locomo_tiny(tmp_path / "locomo_tiny.json")
    sqlite_db = tmp_path / "shared.sqlite"
    provenance_db = tmp_path / "shared_prov.sqlite"
    out_dir_2 = tmp_path / "run2"

    import benchmarks.locomo._run_impl as run_impl_mod
    import memori.memory.recall as recall_mod

    run_id = "test-run"
    _seed_reuse_db(
        sqlite_db=sqlite_db,
        provenance_db=provenance_db,
        run_id=run_id,
        sample_id="sample-001",
    )

    # Second run must not ingest; make ingestion embedding calls fail if they happen.
    def _should_not_be_called(
        texts: str | list[str], model: str, fallback_dimension: int
    ) -> list[list[int | float]]:
        raise AssertionError(
            "embed_texts should not be called during --reuse-db scoring"
        )

    # run_mod.embed_texts = _should_not_be_called
    monkeypatch.setattr(run_impl_mod, "embed_texts", _should_not_be_called)
    monkeypatch.setattr(recall_mod, "embed_texts", _fake_embed_texts)

    rc2 = run_main(
        [
            "--dataset",
            str(dataset),
            "--out",
            str(out_dir_2),
            "--sqlite-db",
            str(sqlite_db),
            "--provenance-db",
            str(provenance_db),
            "--reuse-db",
            "--run-id",
            run_id,
        ]
    )
    assert rc2 == 0
    summary_obj = json.loads((out_dir_2 / "summary.json").read_text(encoding="utf-8"))
    assert summary_obj["metrics_overall"]["hit@1"] == 1.0
