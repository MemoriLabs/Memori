#!/usr/bin/env bash
set -euo pipefail

# Install Python dependencies (uv already installed in CI)
uv sync --dev

# Loop over each example directory and run its main.py with an in‑memory SQLite DB
for d in examples/*; do
  if [ -f "$d/main.py" ]; then
    echo "Running example $d/main.py"
    # Set a temporary DATABASE_URL pointing to SQLite memory
    export DATABASE_URL="sqlite:///:memory:"
    uv run python "$d/main.py" || { echo "Example $d failed"; exit 1; }
  fi
done

echo "All examples ran successfully."
