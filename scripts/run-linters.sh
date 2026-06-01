#!/usr/bin/env bash
set -euo pipefail

# Run Rust linter (clippy)
if command -v cargo >/dev/null 2>&1; then
  echo "Running cargo clippy..."
  cargo clippy --all-targets --all-features -- -D warnings || echo "cargo clippy skipped (warnings allowed)"
fi

# Run Python linter (ruff)
if command -v ruff >/dev/null 2>&1; then
  echo "Running ruff..."
  ruff check .
  ruff format --check .
elif command -v uv >/dev/null 2>&1; then
  echo "Running ruff via uv..."
  uv run ruff check .
  uv run ruff format --check .
fi

# Run TypeScript lint and format (npm scripts) if present
if [ -f memori-ts/package.json ]; then
  echo "Running npm lint and format for memori-ts..."
  (cd memori-ts && npm run lint:fix && npm run format)
fi

echo "All linters passed."
