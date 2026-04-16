.PHONY: help dev-up dev-down dev-shell dev-build dev-clean test lint format clean run-unit run-integration run-integration-provider run-integration-cloud run-rust-core-benchmark run-rust-core-byodb-example

help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Available targets:'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-20s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

dev-up: ## Start development environment (builds and runs containers)
	docker compose up -d --build
	@echo ""
	@echo "✓ Development environment is ready!"
	@echo "  Run 'make dev-shell' to enter the development container"
	@echo "  Run 'make test' to run tests"

dev-down: ## Stop development environment
	docker compose down

dev-shell: ## Open a shell in the development container
	docker compose exec dev /bin/bash

init-db: ## Initialize database schema for integration tests
	docker compose exec -e PYTHONPATH=/app dev python tests/database/init_db.py

init-postgres: ## Initialize PostgreSQL schema
	docker compose exec -e PYTHONPATH=/app dev python tests/build/postgresql.py

init-mysql: ## Initialize MySQL schema
	docker compose exec -e PYTHONPATH=/app dev python tests/build/mysql.py

init-oceanbase: ## Initialize OceanBase schema
	docker compose exec -e PYTHONPATH=/app dev python tests/build/oceanbase.py

init-oracle: ## Initialize Oracle schema
	docker compose exec -e PYTHONPATH=/app dev python tests/build/oracle.py

init-mongodb: ## Initialize MongoDB schema
	docker compose exec -e PYTHONPATH=/app dev python tests/build/mongodb.py

init-sqlite: ## Initialize SQLite schema
	docker compose exec -e PYTHONPATH=/app dev python tests/build/sqlite.py

dev-build: ## Rebuild the development container
	docker compose build --no-cache

dev-clean: ## Complete teardown: stop containers, remove images, prune build cache
	docker compose down -v
	docker builder prune -f
	docker compose rm -f
	@echo "✓ Docker environment cleaned (containers, volumes, and build cache removed)"

test: ## Run tests in the container
	docker compose exec dev pytest

run-unit: ## Run unit tests (no API keys needed)
	@echo "Running unit tests..."
	uv run pytest tests/ --ignore=tests/integration --ignore=tests/benchmarks -v --tb=short

run-integration: ## Run all integration tests (requires API keys)
	@echo "Running all integration tests with MEMORI_TEST_MODE=1..."
	MEMORI_TEST_MODE=1 uv run pytest tests/integration/ -v -m integration --tb=short

run-integration-provider: ## Run specific provider tests (e.g., make run-integration-provider P=openai)
	@echo "Running $(P) integration tests..."
	MEMORI_TEST_MODE=1 uv run pytest tests/integration/providers/test_$(P).py -v -m integration --tb=short

run-integration-cloud: ## Run cloud integration tests (production API, requires MEMORI_API_KEY)
	@echo "Running cloud integration tests..."
	uv run pytest tests/integration/cloud/ -v -m integration --tb=short

run-rust-core-benchmark: ## Build Rust binding and compare recall latency
	@TARGET_DIR="$${CARGO_TARGET_DIR:-$(PWD)/target}"; \
	MANIFEST_PATH="core/Cargo.toml"; \
	if [ ! -f "$$MANIFEST_PATH" ]; then \
		echo "error: missing core/Cargo.toml"; \
		exit 1; \
	fi; \
	if [ "$$(uname)" = "Darwin" ]; then \
		CARGO_TARGET_DIR="$$TARGET_DIR" PYO3_PYTHON="$(PWD)/.venv/bin/python" RUSTFLAGS="-C link-arg=-undefined -C link-arg=dynamic_lookup" cargo build --release -p python-bindings --manifest-path "$$MANIFEST_PATH"; \
		MEMORI_PYTHON_LIB="$$TARGET_DIR/release/libmemori_python.dylib" uv run python examples/rust_core_recall_benchmark.py $(BENCH_ARGS); \
	elif [ "$$(uname)" = "Linux" ]; then \
		CARGO_TARGET_DIR="$$TARGET_DIR" PYO3_PYTHON="$(PWD)/.venv/bin/python" cargo build --release -p python-bindings --manifest-path "$$MANIFEST_PATH"; \
		MEMORI_PYTHON_LIB="$$TARGET_DIR/release/libmemori_python.so" uv run python examples/rust_core_recall_benchmark.py $(BENCH_ARGS); \
	else \
		CARGO_TARGET_DIR="$$TARGET_DIR" PYO3_PYTHON="$(PWD)/.venv/bin/python" cargo build --release -p python-bindings --manifest-path "$$MANIFEST_PATH"; \
		uv run python examples/rust_core_recall_benchmark.py $(BENCH_ARGS); \
	fi

run-rust-core-byodb-example: ## Run SQLite BYODB rust-core smoke example
	@TARGET_DIR="$${CARGO_TARGET_DIR:-$(PWD)/target}"; \
	MANIFEST_PATH="core/Cargo.toml"; \
	if [ ! -f "$$MANIFEST_PATH" ]; then \
		echo "error: missing core/Cargo.toml"; \
		exit 1; \
	fi; \
	if [ "$$(uname)" = "Darwin" ]; then \
		CARGO_TARGET_DIR="$$TARGET_DIR" PYO3_PYTHON="$(PWD)/.venv/bin/python" RUSTFLAGS="-C link-arg=-undefined -C link-arg=dynamic_lookup" cargo build --release -p python-bindings --manifest-path "$$MANIFEST_PATH"; \
		MEMORI_PYTHON_LIB="$$TARGET_DIR/release/libmemori_python.dylib" uv run python examples/sqlite/rust_core_main.py; \
	elif [ "$$(uname)" = "Linux" ]; then \
		CARGO_TARGET_DIR="$$TARGET_DIR" PYO3_PYTHON="$(PWD)/.venv/bin/python" cargo build --release -p python-bindings --manifest-path "$$MANIFEST_PATH"; \
		MEMORI_PYTHON_LIB="$$TARGET_DIR/release/libmemori_python.so" uv run python examples/sqlite/rust_core_main.py; \
	else \
		CARGO_TARGET_DIR="$$TARGET_DIR" PYO3_PYTHON="$(PWD)/.venv/bin/python" cargo build --release -p python-bindings --manifest-path "$$MANIFEST_PATH"; \
		uv run python examples/sqlite/rust_core_main.py; \
	fi

lint: ## Run linting (format check)
	docker compose exec dev uv run ruff check .

security: ## Run security scans (Bandit + pip-audit)
	docker compose exec dev uv run bandit -r memori -ll -ii
	docker compose exec dev uv run pip-audit --require-hashes --disable-pip || true

format: ## Format code
	docker compose exec dev uv run ruff format .

clean: ## Clean up containers, volumes, and Python cache files
	docker compose down -v
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
