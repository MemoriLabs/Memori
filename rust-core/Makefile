SHELL := /bin/bash

.PHONY: help setup-python build-python run-python run-python-recall run-python-aa setup-node build-node run-node run-node-aa run-embedding run-search run-network check lint test docs quality

help:
	@echo "Available targets:"
	@echo "  setup-python  Create venv and install Python bindings via maturin"
	@echo "  build-python  Build/install Python bindings into .venv"
	@echo "  run-python    Run Python example"
	@echo "  run-python-recall Run Python recall example"
	@echo "  run-python-aa Run Python advanced augmentation example"
	@echo "  setup-node    Install Node example dependencies"
	@echo "  build-node    Build Node native addon"
	@echo "  run-node      Run Node example"
	@echo "  run-node-aa   Run Node advanced augmentation example"
	@echo "  run-embedding Run Rust embedding performance analysis"
	@echo "  run-search    Run Rust search performance analysis"
	@echo "  run-network   Run Rust network integration tests"
	@echo "  check         cargo check workspace"
	@echo "  lint          cargo clippy with warnings as errors"
	@echo "  test          cargo test workspace"
	@echo "  docs          cargo doc workspace"
	@echo "  quality       fmt + check + lint + test + docs"

setup-python:
	uv venv .venv
	. .venv/bin/activate && uv tool run maturin develop --manifest-path bindings/python/Cargo.toml

build-python: setup-python

run-python: build-python
	.venv/bin/python examples/python/run_example.py

run-python-recall: build-python
	.venv/bin/python examples/python/recall_example.py

run-python-aa: build-python
	.venv/bin/python examples/python/advanced_augmentation_example.py

setup-node:
	cd examples/node && npm install

build-node: setup-node
	cd examples/node && npm run build:addon

run-node: build-node
	cd examples/node && npm run run

run-node-aa: build-node
	cd examples/node && npx tsx advanced_augmentation_example.ts

run-embedding:
	cargo run --release --example embed_integration

run-search:
	cargo run --release --example search_integration

run-network:
	cargo run --release --example network_integration

check:
	cargo check --workspace

lint:
	cargo clippy --workspace --all-targets -- -D warnings

test:
	cargo test --workspace

docs:
	cargo doc --workspace --no-deps

quality:
	cargo fmt --all
	cargo check --workspace
	cargo clippy --workspace --all-targets -- -D warnings
	cargo test --workspace
	cargo doc --workspace --no-deps
