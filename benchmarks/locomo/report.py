from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from benchmarks.locomo._run_impl import CATEGORY_LABELS


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Aggregate LoCoMo predictions.jsonl into summary.json"
    )
    parser.add_argument(
        "--predictions",
        required=True,
        help="Path to predictions.jsonl created by benchmarks/locomo/run.py",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Path to write summary.json",
    )
    args = parser.parse_args(argv)

    categories: dict[str, int] = {}
    sample_ids: set[str] = set()
    total = 0

    run_id: str | None = None
    timestamp_utc: str | None = None
    sums = {"hit@1": 0.0, "hit@3": 0.0, "hit@5": 0.0, "mrr": 0.0}
    sums_by_cat: dict[str, dict[str, float]] = {}
    counts_by_cat: dict[str, int] = {}

    for row in _read_jsonl(Path(args.predictions)):
        total += 1
        sample_id = str(row.get("sample_id", "")).strip()
        if sample_id:
            sample_ids.add(sample_id)

        cat = str(row.get("category") or "unknown")
        categories[cat] = categories.get(cat, 0) + 1

        metrics = (
            (row.get("retrieval") or {}).get("metrics")
            if isinstance(row.get("retrieval"), dict)
            else None
        )
        if isinstance(metrics, dict):
            for key in sums:
                if key in metrics:
                    sums[key] += float(metrics[key])
            sums_by_cat.setdefault(
                cat, {"hit@1": 0.0, "hit@3": 0.0, "hit@5": 0.0, "mrr": 0.0}
            )
            for key in sums_by_cat[cat]:
                if key in metrics:
                    sums_by_cat[cat][key] += float(metrics[key])
            counts_by_cat[cat] = counts_by_cat.get(cat, 0) + 1

        run_id = run_id or _coerce_str(row.get("run_id"))
        timestamp_utc = timestamp_utc or _coerce_str(row.get("timestamp_utc"))

    denom = float(total) if total else 1.0
    metrics_overall = {k: (sums[k] / denom) for k in sums}
    metrics_by_category: dict[str, dict[str, float]] = {}
    for cat, sums_cat in sums_by_cat.items():
        denom_cat = float(counts_by_cat.get(cat, 0)) or 1.0
        metrics_by_category[cat] = {k: (sums_cat[k] / denom_cat) for k in sums_cat}

    questions_by_category_labeled = {
        CATEGORY_LABELS.get(cat, cat): count for cat, count in categories.items()
    }
    metrics_by_category_labeled = {
        CATEGORY_LABELS.get(cat, cat): vals for cat, vals in metrics_by_category.items()
    }

    out = {
        "run_id": run_id,
        "timestamp_utc": timestamp_utc,
        "sample_count": len(sample_ids),
        "question_count": total,
        "category_labels": dict(CATEGORY_LABELS),
        "questions_by_category": dict(sorted(categories.items(), key=lambda kv: kv[0])),
        "questions_by_category_labeled": dict(
            sorted(questions_by_category_labeled.items(), key=lambda kv: kv[0])
        ),
        "metrics_overall": metrics_overall,
        "metrics_by_category": dict(
            sorted(metrics_by_category.items(), key=lambda kv: kv[0])
        ),
        "metrics_by_category_labeled": dict(
            sorted(metrics_by_category_labeled.items(), key=lambda kv: kv[0])
        ),
    }
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    return 0


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
    return rows


def _coerce_str(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        v = value.strip()
        return v or None
    return str(value).strip() or None


if __name__ == "__main__":
    raise SystemExit(main())
