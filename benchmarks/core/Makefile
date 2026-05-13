SHELL := /bin/bash

.PHONY: help check lint test fmt fmt-check docs quality

help:
	@echo "Available targets:"
	@echo "  check      cargo check workspace"
	@echo "  fmt        cargo fmt --all"
	@echo "  fmt-check  cargo fmt --all -- --check"
	@echo "  lint       cargo clippy --workspace --all-targets -- -D warnings"
	@echo "  test       cargo test --workspace"
	@echo "  docs       cargo doc --workspace --no-deps"
	@echo "  quality    fmt-check + check + lint + test + docs"

check:
	cargo check --workspace

fmt:
	cargo fmt --all

fmt-check:
	cargo fmt --all -- --check

lint:
	cargo clippy --workspace --all-targets -- -D warnings

test:
	cargo test --workspace

docs:
	cargo doc --workspace --no-deps

quality: fmt-check check lint test docs
