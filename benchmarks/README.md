## Benchmarks

This directory contains **benchmark harnesses** that are intentionally **not** part of the
default `pytest` unit test suite.

### Performance / latency (pytest-benchmark)

Performance benchmarks (including **end-to-end recall latency**) live in `benchmarks/perf/`.

Run locally (example):

```bash
uv run pytest -m benchmark --benchmark-only benchmarks/perf/test_recall_benchmarks.py -v
```

For EC2 / VPC-adjacent database benchmarking, see `benchmarks/perf/README.md` and the helper
scripts in `benchmarks/perf/`.

### LoCoMo (retrieval evaluation)

LoCoMo is a benchmark dataset by Snap Research for long conversation memory.

In Memori, we treat LoCoMo as a **retrieval evaluation** problem: given a question, does
Memori retrieve the right supporting context (evidence)?

#### Dataset

This repo includes a local copy of LoCoMo in `benchmarks/locomo10.json`.

Upstream: `https://github.com/snap-research/locomo`

#### What gets written (artifacts)

Each run writes:

- `predictions.jsonl`: one row per QA question (retrieved top-k + hit@k/MRR metrics)
- `summary.json`: aggregated metrics (overall + by category)
- `locomo.sqlite`: SQLite DB used by Memori storage during the run
- `locomo_provenance.sqlite`: (AA mode only) benchmark-only mapping of `fact_id → dia_id` for scoring

#### Modes (ingestion)

There are two supported ingestion modes for LoCoMo:

- **`--ingest advanced_augmentation` (default)**:
  - Stores turns as `conversation_message`s and runs Memori **Advanced Augmentation** to produce
    derived `entity_fact`s (closest to real usage).
  - Because LoCoMo evidence is turn-level, we write a **benchmark-only provenance DB**
    (`locomo_provenance.sqlite`) that heuristically maps each derived fact back to the most similar
    LoCoMo `dia_id` turn(s), then score hit@k/MRR against evidence.
  - **Requires**: `MEMORI_API_KEY` (and `MEMORI_TEST_MODE=1` if you want staging).
  - **Note**: may be non-deterministic (API + model changes).

- **`--ingest turn_facts`**:
  - Stores each LoCoMo dialogue turn directly as an `entity_fact` (tagged with the LoCoMo `dia_id`
    like `D1:3`).
  - Retrieval returns turns, and we score against LoCoMo `qa[*].evidence` (also `dia_id`s).
  - **Best for**: measuring Memori retrieval quality in isolation (deterministic, offline).

#### Quickstart (turn_facts baseline, offline)

- Create a results directory:

```bash
mkdir -p results/locomo
```

- Run the LoCoMo harness on a local JSON file:

```bash
uv run python benchmarks/locomo/run.py \
  --dataset benchmarks/locomo10.json \
  --out results/locomo/turn_facts_run \
  --ingest turn_facts
```

#### Quickstart (advanced_augmentation, seeds + scores)

Prerequisite:

- `MEMORI_API_KEY` set (Advanced Augmentation API access)
- Use `MEMORI_TEST_MODE=1` to target staging

Run:

```bash
export MEMORI_API_KEY="..."
export MEMORI_TEST_MODE=1
# Optional: increase AA request timeout (default is 30s)
export MEMORI_AUGMENTATION_TIMEOUT_SECONDS=120

uv run python benchmarks/locomo/run.py \
  --dataset benchmarks/locomo10.json \
  --out results/locomo/aa_run \
  --ingest advanced_augmentation \
  --aa-batch per_sample
```

#### Score-only (reuse an existing DB, no AA calls)

If you already seeded a SQLite DB (and, for AA runs, a provenance DB), you can skip ingestion and
run retrieval+scoring directly from the existing DB:

```bash
uv run python benchmarks/locomo/run.py \
  --dataset benchmarks/locomo10.json \
  --out results/locomo/score_only \
  --sqlite-db results/locomo/aa_run/locomo.sqlite \
  --provenance-db results/locomo/aa_run/locomo_provenance.sqlite \
  --ingest advanced_augmentation \
  --reuse-db
```

If the DB contains multiple prior LoCoMo runs, pass `--run-id` to choose which one to score.

#### Useful knobs (AA mode)

- **Batching**:
  - `--aa-batch per_sample` (one AA request per sample; biggest payload)
  - `--aa-batch per_session` (one AA request per session; smaller payload)
  - `--aa-batch per_chunk --aa-chunk-size 16` (splits the full conversation into chunks; smallest payload)

- **Dry-run** (inspect payload; no network call):
  - `--aa-dry-run` writes `aa_payload_preview.json` and prints the payload + URL.

- **Metadata** (only if your AA endpoint requires it; defaults are provided):
  - `--meta-llm-provider`
  - `--meta-llm-version`
  - `--meta-llm-sdk-version`
  - `--meta-framework-provider`
  - `--meta-platform-provider`

- **Timeout**:
  - AA HTTP timeout is configured via `MEMORI_AUGMENTATION_TIMEOUT_SECONDS`.
