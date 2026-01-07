from __future__ import annotations

import json
from pathlib import Path

from benchmarks.locomo.report import main as report_main
from benchmarks.locomo.run import main as run_main


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


def test_run_writes_predictions_and_summary(tmp_path: Path, monkeypatch):
    dataset = Path(__file__).parent / "fixtures" / "locomo_tiny.json"
    out_dir = tmp_path / "run"

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
            "--ingest",
            "turn_facts",
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


def test_report_aggregates_predictions(tmp_path: Path, monkeypatch):
    dataset = Path(__file__).parent / "fixtures" / "locomo_tiny.json"
    out_dir = tmp_path / "run"

    import benchmarks.locomo._run_impl as run_impl_mod
    import memori.memory.recall as recall_mod

    # run_mod.embed_texts = _fake_embed_texts
    monkeypatch.setattr(run_impl_mod, "embed_texts", _fake_embed_texts)
    monkeypatch.setattr(recall_mod, "embed_texts", _fake_embed_texts)

    run_main(
        ["--dataset", str(dataset), "--out", str(out_dir), "--ingest", "turn_facts"]
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
    dataset = Path(__file__).parent / "fixtures" / "locomo_tiny.json"
    sqlite_db = tmp_path / "shared.sqlite"
    out_dir_1 = tmp_path / "run1"
    out_dir_2 = tmp_path / "run2"

    import benchmarks.locomo._run_impl as run_impl_mod
    import memori.memory.recall as recall_mod

    # First run ingests deterministically.
    # run_mod.embed_texts = _fake_embed_texts
    monkeypatch.setattr(run_impl_mod, "embed_texts", _fake_embed_texts)
    monkeypatch.setattr(recall_mod, "embed_texts", _fake_embed_texts)

    rc = run_main(
        [
            "--dataset",
            str(dataset),
            "--out",
            str(out_dir_1),
            "--sqlite-db",
            str(sqlite_db),
            "--ingest",
            "turn_facts",
        ]
    )
    assert rc == 0
    assert sqlite_db.exists()

    # Second run must not ingest; make ingestion embedding calls fail if they happen.
    def _should_not_be_called(
        texts: str | list[str], model: str, fallback_dimension: int
    ) -> list[list[int | float]]:
        raise AssertionError(
            "ingestion embed_texts should not be called when --reuse-db is set"
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
            "--ingest",
            "turn_facts",
            "--reuse-db",
        ]
    )
    assert rc2 == 0
    summary_obj = json.loads((out_dir_2 / "summary.json").read_text(encoding="utf-8"))
    assert summary_obj["metrics_overall"]["hit@1"] == 1.0
