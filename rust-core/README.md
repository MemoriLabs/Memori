# Rust Engine Workspace

Sync-first Rust workspace for the Memori engine consumed by Python and Node SDKs.

## Layout

- `src/`: the root Rust engine crate with sync request-path helpers and non-blocking postprocess submission.
- `bindings/python`: native Python module via PyO3.
- `bindings/node`: native Node module via napi-rs.
- `examples/`: small Python and Node integration demos.

## Architecture

Dependency direction is one-way:

root engine crate -> `bindings/python` / `bindings/node`

The expected runtime call path for current SDKs is:

Python/Node SDK -> native binding package -> root Rust engine crate

Boundary rules:

- The root engine crate owns the current Rust-side engine behavior and background postprocess handoff.
- `bindings/python` and `bindings/node` expose thin native-language APIs.
- The background worker runtime now lives inside the root engine crate as an internal module.

## Quality Gates

Run with Cargo directly:

- `cargo fmt --all -- --check`
- `cargo clippy --workspace --all-targets -- -D warnings`
- `cargo test --workspace`
- `cargo doc --workspace --no-deps`

Or use workspace aliases:

- `cargo check-all`
- `cargo lint`
- `cargo test-all`
- `cargo doc-all`

## Quick Start

1. Install Rust toolchain (stable).
2. Run `cargo check-all`.
3. Run `cargo lint`.
4. Run `cargo test-all`.
5. Open `docs/architecture.md` before modifying public boundaries.

## Examples

### Python

Build and load the Python module into your active virtual environment:

1. `uv venv`
2. `source .venv/bin/activate`
3. `uv tool run maturin develop --manifest-path bindings/python/Cargo.toml`
4. `python examples/python/run_example.py`

Expected output:

- `hello world`
- Three accepted postprocess job IDs printed immediately.
- Background progress lines on stderr from `[orchestrator postprocess worker]`.

## Makefile Shortcuts

From the repository root:

- `make run-python` to build/install Python bindings and run the Python example.
- `make run-node` to install Node deps, build the addon, and run the Node example.
- `make quality` to run fmt, check, lint, test, and docs for the full workspace.

### Node

1. `cd examples/node && npm install`
2. `npm run build:addon`
3. `npm run run`

Expected output:

- `hello world`
- Three accepted postprocess job IDs printed immediately.
- Background progress lines on stderr from `[orchestrator postprocess worker]`.
