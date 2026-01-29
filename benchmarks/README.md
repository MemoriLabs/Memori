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

LoCoMo is not vendored in this repo. Download the dataset JSON locally, then point the
harness at that file path.

Upstream: `https://github.com/snap-research/locomo`

#### Preprocess (recommended for Memori)

The upstream LoCoMo format is a **third-person** dialogue between two speakers, and some
conversations include multimodal fields (e.g., image URLs + captions) that Memori does not
currently handle well.

To make evaluation more representative of Memori usage, we provide a small preprocessing step
that:

- Skips any conversation that contains multimodal turn fields (`img_url`, `blip_caption`, `query`)
- Rewrites speakers so `conversation.speaker_b` becomes `assistant` and the other speaker becomes `user`

Run:

```bash
uv run python benchmarks/locomo/preprocess.py \
  --in benchmarks/locomo10.json \
  --out benchmarks/locomo10_memori.json
```

#### What gets written (artifacts)

Each run writes:

- `predictions.jsonl`: one row per QA question (retrieved top-k + hit@k/MRR metrics)
- `summary.json`: aggregated metrics (overall + by category)
- `locomo.sqlite`: SQLite DB used by Memori storage during the run
- `locomo_provenance.sqlite`: (AA mode only) benchmark-only mapping of `fact_id → dia_id` for scoring

#### Modes (ingestion)

LoCoMo ingestion always uses **Advanced Augmentation**:

- Stores turns as `conversation_message`s and runs Memori **Advanced Augmentation** to produce
  derived `entity_fact`s (closest to real usage).
- Because LoCoMo evidence is turn-level, we write a **benchmark-only provenance DB**
  (`locomo_provenance.sqlite`) that maps each derived fact back to the LoCoMo `dia_id` turn(s),
  then score hit@k/MRR against evidence.
- **Requires**: `MEMORI_API_KEY`.
- **Note**: may be non-deterministic (API + model changes).

#### Quickstart (advanced_augmentation, seeds + scores)

Prerequisite:

- `MEMORI_API_KEY` set (Advanced Augmentation API access)
- LoCoMo harness forces staging routing (`MEMORI_TEST_MODE=1`)

Run:

```bash
export MEMORI_API_KEY="..."
# Optional: increase AA request timeout (default is 30s)
export MEMORI_AUGMENTATION_TIMEOUT_SECONDS=120

uv run python benchmarks/locomo/run.py \
  --dataset benchmarks/locomo10.json \
  --out results/locomo/aa_run \
  --aa-batch per_pair
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
  --reuse-db
```

If the DB contains multiple prior LoCoMo runs, pass `--run-id` to choose which one to score.

#### Useful knobs (AA mode)

- **Batching**:
  - `--aa-batch per_pair` (one AA request per user+assistant pair)

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
