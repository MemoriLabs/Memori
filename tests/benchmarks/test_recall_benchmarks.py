import datetime
import os
from time import perf_counter

import pytest

from memori._search import find_similar_embeddings
from memori.llm._embeddings import embed_texts
from memori.memory.recall import Recall
from tests.benchmarks._results import append_csv_row, results_dir


def _default_benchmark_csv_path() -> str:
    return str(results_dir() / "recall_benchmarks.csv")


def _write_benchmark_row(*, benchmark, row: dict[str, object]) -> None:
    csv_path = (
        os.environ.get("BENCHMARK_RESULTS_CSV_PATH") or _default_benchmark_csv_path()
    )
    stats = getattr(benchmark, "stats", None)
    row_out: dict[str, object] = dict(row)
    row_out["timestamp_utc"] = datetime.datetime.now(datetime.UTC).isoformat()

    for key in (
        "mean",
        "stddev",
        "median",
        "min",
        "max",
        "rounds",
        "iterations",
        "ops",
    ):
        value = getattr(stats, key, None) if stats is not None else None
        if value is not None:
            row_out[key] = value

    header = [
        "timestamp_utc",
        "test",
        "db",
        "fact_count",
        "query_size",
        "retrieval_limit",
        "one_shot_seconds",
        "peak_rss_bytes",
        "mean",
        "stddev",
        "median",
        "min",
        "max",
        "rounds",
        "iterations",
        "ops",
    ]
    append_csv_row(csv_path, header=header, row=row_out)


@pytest.mark.benchmark
class TestQueryEmbeddingBenchmarks:
    def test_benchmark_query_embedding_short(
        self, benchmark, sample_queries, memori_instance
    ):
        query = sample_queries["short"][0]
        cfg = memori_instance.config

        def _embed():
            return embed_texts(
                query,
                model=cfg.embeddings.model,
                fallback_dimension=cfg.embeddings.fallback_dimension,
            )

        start = perf_counter()
        benchmark(_embed)
        one_shot_seconds = perf_counter() - start

        _write_benchmark_row(
            benchmark=benchmark,
            row={
                "test": "query_embedding_short",
                "db": "",
                "fact_count": "",
                "query_size": "short",
                "retrieval_limit": "",
                "one_shot_seconds": one_shot_seconds,
                "peak_rss_bytes": 0,
            },
        )

    def test_benchmark_query_embedding_long(
        self, benchmark, sample_queries, memori_instance
    ):
        query = sample_queries["long"][0]
        cfg = memori_instance.config

        def _embed():
            return embed_texts(
                query,
                model=cfg.embeddings.model,
                fallback_dimension=cfg.embeddings.fallback_dimension,
            )

        start = perf_counter()
        benchmark(_embed)
        one_shot_seconds = perf_counter() - start

        _write_benchmark_row(
            benchmark=benchmark,
            row={
                "test": "query_embedding_long",
                "db": "",
                "fact_count": "",
                "query_size": "long",
                "retrieval_limit": "",
                "one_shot_seconds": one_shot_seconds,
                "peak_rss_bytes": 0,
            },
        )


@pytest.mark.benchmark
class TestDatabaseEmbeddingRetrievalBenchmarks:
    def test_benchmark_db_embedding_retrieval(
        self, benchmark, memori_instance, entity_with_n_facts
    ):
        entity_db_id = entity_with_n_facts["entity_db_id"]
        fact_count = entity_with_n_facts["fact_count"]
        driver = memori_instance.config.storage.driver.entity_fact

        def _retrieve():
            return driver.get_embeddings(entity_db_id, limit=fact_count)

        benchmark(_retrieve)

        _write_benchmark_row(
            benchmark=benchmark,
            row={
                "test": "db_embedding_retrieval",
                "db": entity_with_n_facts["db_type"],
                "fact_count": fact_count,
                "query_size": "",
                "retrieval_limit": "",
                "one_shot_seconds": 0,
                "peak_rss_bytes": 0,
            },
        )


@pytest.mark.benchmark
class TestSemanticSearchBenchmarks:
    def test_benchmark_semantic_search(
        self, benchmark, memori_instance, entity_with_n_facts, sample_queries
    ):
        entity_db_id = entity_with_n_facts["entity_db_id"]
        fact_count = entity_with_n_facts["fact_count"]
        driver = memori_instance.config.storage.driver.entity_fact

        db_results = driver.get_embeddings(entity_db_id, limit=fact_count)
        embeddings = [(row["id"], row["content_embedding"]) for row in db_results]

        query = sample_queries["short"][0]
        query_emb = embed_texts(
            query,
            model=memori_instance.config.embeddings.model,
            fallback_dimension=memori_instance.config.embeddings.fallback_dimension,
        )[0]

        def _search():
            return find_similar_embeddings(embeddings, query_emb, limit=5)

        benchmark(_search)

        _write_benchmark_row(
            benchmark=benchmark,
            row={
                "test": "semantic_search_faiss",
                "db": entity_with_n_facts["db_type"],
                "fact_count": fact_count,
                "query_size": "short",
                "retrieval_limit": "",
                "one_shot_seconds": 0,
                "peak_rss_bytes": 0,
            },
        )


@pytest.mark.benchmark
class TestDatabaseFactContentRetrievalBenchmarks:
    @pytest.mark.parametrize("retrieval_limit", [5, 10], ids=["limit5", "limit10"])
    def test_benchmark_db_fact_content_retrieval(
        self, benchmark, memori_instance, entity_with_n_facts, retrieval_limit
    ):
        entity_db_id = entity_with_n_facts["entity_db_id"]
        driver = memori_instance.config.storage.driver.entity_fact

        seed_rows = driver.get_embeddings(entity_db_id, limit=retrieval_limit)
        fact_ids = [row["id"] for row in seed_rows]

        def _retrieve():
            return driver.get_facts_by_ids(fact_ids)

        benchmark(_retrieve)

        _write_benchmark_row(
            benchmark=benchmark,
            row={
                "test": "db_fact_content_retrieval",
                "db": entity_with_n_facts["db_type"],
                "fact_count": entity_with_n_facts["fact_count"],
                "query_size": "",
                "retrieval_limit": retrieval_limit,
                "one_shot_seconds": 0,
                "peak_rss_bytes": 0,
            },
        )


@pytest.mark.benchmark
class TestEndToEndRecallBenchmarks:
    @pytest.mark.parametrize(
        "query_size",
        ["short", "medium", "long"],
        ids=["short_query", "medium_query", "long_query"],
    )
    def test_benchmark_end_to_end_recall(
        self,
        benchmark,
        memori_instance,
        entity_with_n_facts,
        sample_queries,
        query_size,
    ):
        entity_db_id = entity_with_n_facts["entity_db_id"]
        query = sample_queries[query_size][0]
        recall = Recall(memori_instance.config)

        def _recall():
            return recall.search_facts(query=query, limit=5, entity_id=entity_db_id)

        start = perf_counter()
        result = benchmark(_recall)
        one_shot_seconds = perf_counter() - start

        assert isinstance(result, list)

        _write_benchmark_row(
            benchmark=benchmark,
            row={
                "test": "end_to_end_recall",
                "db": entity_with_n_facts["db_type"],
                "fact_count": entity_with_n_facts["fact_count"],
                "query_size": query_size,
                "retrieval_limit": "",
                "one_shot_seconds": one_shot_seconds,
                "peak_rss_bytes": 0,
            },
        )
