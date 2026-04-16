# engine-orchestrator

Rust core for the Memori engine, shared by the Python and Node SDKs.

## Layout

| Path                  | Purpose                                                   |
| --------------------- | --------------------------------------------------------- |
| `src/`                | `engine-orchestrator` crate — the public Rust engine      |
| `bindings/python/`    | PyO3 extension module (`memori_python`)                   |
| `bindings/node/`      | napi-rs native addon                                      |
| `docs/`               | Architecture notes                                        |
| `tests/`              | Cross-module contract tests                               |

Dependencies are strictly one-way:

```
engine-orchestrator  →  bindings/{python,node}  →  Memori SDKs
```

Bindings must call through the engine crate; they may not reuse internal modules
directly.

## Public API

The top-level [`EngineOrchestrator`] owns:

- a synchronous embedding pipeline (`fastembed` + `sentence-transformers`),
- two bounded async worker runtimes (`WorkerRuntime<PostprocessJob>` and
  `WorkerRuntime<AugmentationJob>`),
- an optional host-provided [`storage::StorageBridge`] used by the retrieval
  pipeline.

See [`docs/architecture.md`](docs/architecture.md) for the responsibility split.

## Development

```
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
cargo doc --workspace --no-deps
```

Or use the workspace aliases defined in `.cargo/config.toml`:

```
cargo check-all
cargo lint
cargo test-all
cargo doc-all
```

### Python bindings

The `memori_python` extension is built and bundled into the `memori` Python
wheel by `setup.py` (via `setuptools-rust`) when you run `python -m build` at
the repository root.

For local iteration you can build in-place with maturin:

```
uv venv
. .venv/bin/activate
uv tool run maturin develop --manifest-path bindings/python/Cargo.toml
```

### Node bindings

```
cd bindings/node
napi build --release
```

## Configuration

Runtime knobs read from the environment (all optional):

| Variable                        | Purpose                                                     | Default                              |
| ------------------------------- | ----------------------------------------------------------- | ------------------------------------ |
| `MEMORI_API_URL_BASE`           | Override the Memori API base URL                            | `https://api.memorilabs.ai`          |
| `MEMORI_X_API_KEY`              | Override the `X-Memori-API-Key` header                      | built-in public key                  |
| `MEMORI_API_KEY`                | Optional bearer token for per-tenant quota                  | unset                                |
| `MEMORI_TEST_MODE`              | When `1`, point at `staging-*.memorilabs.ai`                | `0`                                  |
| `MEMORI_RECALL_LEX_WEIGHT`      | Lexical weight for hybrid re-rank (clamped `[0.05, 0.40]`)  | `0.15`                               |
| `MEMORI_RECALL_LEX_WEIGHT_SHORT`| Lexical weight for short queries (≤ 2 tokens)               | `0.30`                               |

## License

Apache-2.0 — see the repository root `LICENSE`.
