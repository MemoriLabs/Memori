# Performance / latency benchmarks

This folder contains **performance benchmarks** for Memori recall, powered by `pytest-benchmark`.

## Run locally

```bash
uv run pytest -m benchmark --benchmark-only benchmarks/perf/test_recall_benchmarks.py -v
```

## Run on EC2 (recommended for realistic DB latency)

Use an EC2 instance in the same VPC as your database (RDS Postgres/MySQL).

- Setup:

```bash
chmod +x benchmarks/perf/setup_ec2_benchmarks.sh
./benchmarks/perf/setup_ec2_benchmarks.sh
```

- Run:

```bash
export BENCHMARK_POSTGRES_URL="CHANGEME"
DB_TYPE=postgres TEST_TYPE=all ./benchmarks/perf/run_benchmarks_ec2.sh
```

### Environment variables

- `DB_TYPE`: `postgres` (default) or `mysql`
- `TEST_TYPE`: `all` (default), `end_to_end`, `db_retrieval`, `semantic_search`, `embedding`
- `BENCHMARK_POSTGRES_URL`: Postgres connection string
- `BENCHMARK_MYSQL_URL`: MySQL connection string

## Outputs

Results are saved to `./results`:

- JSON: `results_{db}_{type}_{timestamp}.json`
- CSV: `report_{db}_{type}_{timestamp}.csv`
