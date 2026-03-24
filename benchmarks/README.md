# Memori Benchmarks — LoCoMo Evaluation

This directory contains two Jupyter notebooks that evaluate **Memori's Advanced Augmentation** retrieval pipeline against the [LoCoMo](https://github.com/snap-research/locomo) long-conversation benchmark.

The goal is to measure how well Memori's memory layer can answer questions about past conversations, compared to a naive "dump the entire chat history" baseline.

## What the notebooks do

| Notebook | Purpose |
|----------|---------|
| [`01_load_indexes.ipynb`](01_load_indexes.ipynb) | Downloads augmented memories, embeds them with [EmbeddingGemma-300M](https://huggingface.co/google/embeddinggemma-300m), and builds a FAISS search index per conversation. |
| [`02_run_benchmark.ipynb`](02_run_benchmark.ipynb) | Loads the LoCoMo benchmark questions, retrieves relevant memories via hybrid search (FAISS + BM25), generates answers with an LLM, judges correctness, and reports accuracy metrics. |

Run **`01`** first to build the indexes (skip if `indexes_gemma/` already exists), then run **`02`** to evaluate.

## Prerequisites

- **Python 3.10+** (see [`pyproject.toml`](pyproject.toml) for full dependency list).
- A **Hugging Face** account with access to the gated embedding model (see below).
- An **OpenAI API key** (only needed for notebook `02`).

## Quick start

### 1. Install dependencies

From this directory, using [uv](https://docs.astral.sh/uv/):

```bash
cd benchmarks
uv sync                       # core dependencies
uv sync --extra notebook      # adds JupyterLab + widgets (optional)
```

Or with pip:

```bash
pip install -e ".[notebook]"
```

Make sure to select the resulting environment as your Jupyter kernel.

### 2. Set up environment variables

Copy the example file and fill in your tokens:

```bash
cp .env.example .env
```

| Variable | Required by | Description |
|----------|-------------|-------------|
| `HF_TOKEN` | `01` and `02` | Hugging Face read token for the gated embedding model. |
| `OPENAI_API_KEY` | `02` only | OpenAI API key for answer generation and judging. |

Both notebooks call `load_dotenv()` to pick these up automatically.

### 3. Authorize the embedding model on Hugging Face

The embedding model ([`google/embeddinggemma-300m`](https://huggingface.co/google/embeddinggemma-300m)) is **gated** — you must accept its license before downloading:

1. [Create a Hugging Face account](https://huggingface.co/join) if you don't have one.
2. Visit the [model page](https://huggingface.co/google/embeddinggemma-300m), sign in, and **agree to the terms**.
3. Generate a **Read** token at [Settings > Access Tokens](https://huggingface.co/settings/tokens).
4. Add it to `.env` as `HF_TOKEN=hf_...`, or run `huggingface-cli login` in the same environment.

> The first run downloads the model (~1.2 GB), which may take a few minutes.

### 4. Run the notebooks

Open each notebook in Jupyter and run all cells top to bottom:

1. **`01_load_indexes.ipynb`** — builds `indexes_gemma/` from augmented memories.
2. **`02_run_benchmark.ipynb`** — runs the full evaluation pipeline.

## Notebook details

### `01_load_indexes.ipynb`

- **Input:** `advanced_augmented_memories.json` (downloaded automatically if missing).
- **Output:** `indexes_gemma/<conv_id>/` directories, each containing `faiss.index` and `metadata.json`.
- Re-run this notebook whenever the memory data or embedding model changes.

### `02_run_benchmark.ipynb`

- Requires the `indexes_gemma/` directory from notebook `01`.
- Run the configuration cell first, then sections 1 through 8 in order.
- **Benchmark data** is fetched from a URL by default (`locomo10.json`). Set `LOCOMO_LOCAL_PATH` in the config cell to use a local copy.
- **Key config** (all in the first code cell): `INDEX_DIR` (default `./indexes_gemma`), `RESULTS_DIR` (default `./results_gemma`), `OPENAI_MODEL` (default `gpt-4.1-mini`).

## Troubleshooting

| Problem | Solution |
|---------|----------|
| **401 / 403** when loading the embedding model | Make sure you accepted the license on the [model page](https://huggingface.co/google/embeddinggemma-300m) and that `HF_TOKEN` is set (or you ran `huggingface-cli login`). |
| `FileNotFoundError` for indexes | Run notebook `01` first, or check that `INDEX_DIR` in notebook `02` points to the right directory. |
| `OPENAI_API_KEY` errors in notebook `02` | Set the key in your `.env` file. |
| Very slow first run | This is normal — the model download and FAISS index build are one-time costs. CPU is supported but slower than GPU. |

## Verify your environment

Quick check that tokens are configured (doesn't print secrets):

```python
import os
from dotenv import load_dotenv

load_dotenv()
hf = bool(os.getenv("HF_TOKEN", "").strip())
openai = bool(os.getenv("OPENAI_API_KEY", "").strip())
print(f"HF_TOKEN set: {hf}  |  OPENAI_API_KEY set: {openai}")
```
