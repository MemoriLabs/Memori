"""Cloud (Memori API) performance / latency benchmarks.

These benchmarks exercise the cloud recall path (network + API latency) rather than
local DB-backed recall.
"""

from __future__ import annotations

import os
from uuid import uuid4

import pytest

from memori._config import Config
from memori.llm._base import BaseInvoke
from memori.llm._constants import OPENAI_LLM_PROVIDER
from memori.memory._manager import Manager
from memori.memory.recall import Recall


def _make_cloud_cfg(*, entity_id: str, process_id: str) -> Config:
    cfg = Config()
    cfg.cloud = True
    cfg.entity_id = entity_id
    cfg.process_id = process_id
    if cfg.session_id is None:
        cfg.session_id = uuid4()
    # Pick a provider so injection rules are deterministic.
    cfg.llm.provider = OPENAI_LLM_PROVIDER
    cfg.framework.provider = "bench"
    return cfg


def _seed_cloud_messages(
    cfg: Config, *, n_messages: int, entity_id: str, process_id: str
) -> None:
    cfg.cloud = True
    cfg.entity_id = entity_id
    cfg.process_id = process_id
    if cfg.session_id is None:
        cfg.session_id = uuid4()

    messages: list[dict[str, object]] = []
    for i in range(n_messages):
        messages.append({"role": "user", "type": None, "text": f"I like item_{i}."})
        messages.append({"role": "assistant", "type": None, "text": "Ok."})

    Manager(cfg).execute(
        {
            "attribution": {
                "entity": {"id": str(entity_id)},
                "process": {"id": str(process_id)},
            },
            "messages": messages,
            "session": {"id": str(cfg.session_id)},
        }
    )


@pytest.mark.benchmark
@pytest.mark.parametrize("n_messages", [20, 100], ids=["n20", "n100"])
def test_benchmark_cloud_recall_latency(benchmark, n_messages: int) -> None:
    """Benchmark end-to-end cloud recall latency.

    Includes: HTTP request + cloud service time. Excludes: local DB retrieval.
    """
    os.environ["MEMORI_TEST_MODE"] = "1"
    api_key = os.environ.get("MEMORI_API_KEY")
    if not api_key:
        pytest.skip("Set MEMORI_API_KEY to benchmark cloud recall.")

    cfg = Config()

    entity_id = os.environ.get("BENCHMARK_CLOUD_ENTITY_ID", "bench-cloud-entity")
    process_id = os.environ.get("BENCHMARK_CLOUD_PROCESS_ID", "bench-cloud-process")

    _seed_cloud_messages(
        cfg,
        n_messages=n_messages,
        entity_id=entity_id,
        process_id=process_id,
    )

    recall = Recall(cfg)

    def _call():
        return recall.search_facts(query="What do I like?", limit=5)

    result = benchmark(_call)
    assert isinstance(result, list)


@pytest.mark.benchmark
@pytest.mark.parametrize("n_history_pairs", [20, 100], ids=["n20", "n100"])
def test_benchmark_cloud_pre_llm_overhead(benchmark, n_history_pairs: int) -> None:
    """Benchmark Memori overhead up to the LLM call (cloud mode).

    This measures the "added latency" before calling an LLM:
    - (optional) fetch cloud conversation history
    - cloud recall request (server-side embeddings + retrieval)
    - prompt/message injection logic

    It intentionally does NOT call an LLM provider.
    """
    if not os.environ.get("MEMORI_API_KEY"):
        pytest.skip("Set MEMORI_API_KEY to benchmark cloud pre-LLM overhead.")

    entity_id = os.environ.get("BENCHMARK_CLOUD_ENTITY_ID", "bench-cloud-entity")
    process_id = os.environ.get("BENCHMARK_CLOUD_PROCESS_ID", "bench-cloud-process")
    cfg = _make_cloud_cfg(entity_id=entity_id, process_id=process_id)

    # Seed a stable session for cloud history fetch.
    _seed_cloud_messages(
        cfg,
        n_messages=n_history_pairs,
        entity_id=entity_id,
        process_id=process_id,
    )

    query = "What do I like?"

    def _prepare():
        invoke = BaseInvoke(cfg, lambda **_kwargs: None)
        kwargs = {"messages": [{"role": "user", "content": query}]}
        kwargs = invoke.inject_conversation_messages(kwargs)
        kwargs = invoke.inject_recalled_facts(kwargs)
        return kwargs

    result = benchmark(_prepare)
    assert isinstance(result, dict)


@pytest.mark.benchmark
@pytest.mark.parametrize("n_history_pairs", [20, 100], ids=["n20", "n100"])
def test_benchmark_cloud_network_history_get(benchmark, n_history_pairs: int) -> None:
    """cloud conversation history fetched via recall (POST), no injection."""
    if not os.environ.get("MEMORI_API_KEY"):
        pytest.skip("Set MEMORI_API_KEY to benchmark cloud history GET.")

    entity_id = os.environ.get("BENCHMARK_CLOUD_ENTITY_ID", "bench-cloud-entity")
    process_id = os.environ.get("BENCHMARK_CLOUD_PROCESS_ID", "bench-cloud-process")
    cfg = _make_cloud_cfg(entity_id=entity_id, process_id=process_id)

    _seed_cloud_messages(
        cfg,
        n_messages=n_history_pairs,
        entity_id=entity_id,
        process_id=process_id,
    )

    def _call():
        recall = Recall(cfg)
        data = recall._cloud_recall(query="History fetch benchmark")
        _facts, messages = recall._parse_cloud_recall_response(data)
        return messages

    result = benchmark(_call)
    assert isinstance(result, list)


@pytest.mark.benchmark
@pytest.mark.parametrize("n_history_pairs", [20, 100], ids=["n20", "n100"])
def test_benchmark_cloud_network_recall_post(benchmark, n_history_pairs: int) -> None:
    """ONLY cloud recall call (POST /recall), no injection.

    We seed N history pairs first so cloud has a consistent amount of data.
    """
    if not os.environ.get("MEMORI_API_KEY"):
        pytest.skip("Set MEMORI_API_KEY to benchmark cloud recall POST.")

    entity_id = os.environ.get("BENCHMARK_CLOUD_ENTITY_ID", "bench-cloud-entity")
    process_id = os.environ.get("BENCHMARK_cloud_PROCESS_ID", "bench-cloud-process")
    cfg = _make_cloud_cfg(entity_id=entity_id, process_id=process_id)

    _seed_cloud_messages(
        cfg,
        n_messages=n_history_pairs,
        entity_id=entity_id,
        process_id=process_id,
    )

    recall = Recall(cfg)

    def _call():
        return recall.search_facts(query="What do I like?", limit=5)

    result = benchmark(_call)
    assert isinstance(result, list)


@pytest.mark.benchmark
@pytest.mark.parametrize("n_history_pairs", [20, 100], ids=["n20", "n100"])
def test_benchmark_cloud_network_only_history_plus_recall(
    benchmark, n_history_pairs: int
) -> None:
    """ONLY the cloud recall call (history + facts), no injection."""
    if not os.environ.get("MEMORI_API_KEY"):
        pytest.skip("Set MEMORI_API_KEY to benchmark cloud network-only overhead.")

    entity_id = os.environ.get("BENCHMARK_cloud_ENTITY_ID", "bench-cloud-entity")
    process_id = os.environ.get("BENCHMARK_CLOUD_PROCESS_ID", "bench-cloud-process")
    cfg = _make_cloud_cfg(entity_id=entity_id, process_id=process_id)

    _seed_cloud_messages(
        cfg,
        n_messages=n_history_pairs,
        entity_id=entity_id,
        process_id=process_id,
    )

    recall = Recall(cfg)

    def _call():
        data = recall._cloud_recall(query="What do I like?")
        facts, _messages = recall._parse_cloud_recall_response(data)
        return facts

    result = benchmark(_call)
    assert isinstance(result, list)


def _make_invoke_with_stubbed_history(
    cfg: Config, *, n_history_pairs: int
) -> BaseInvoke:
    history: list[dict[str, str]] = []
    for i in range(n_history_pairs):
        history.append({"role": "user", "content": f"I like item_{i}."})
        history.append({"role": "assistant", "content": "Ok."})
    invoke = BaseInvoke(cfg, lambda **_kwargs: None)
    invoke._cloud_conversation_messages = history
    return invoke


@pytest.mark.benchmark
@pytest.mark.parametrize("n_history_pairs", [20, 100], ids=["n20", "n100"])
def test_benchmark_cloud_injection_only_history(
    benchmark, n_history_pairs: int
) -> None:
    """ONLY Python-side history injection (no network)."""
    entity_id = os.environ.get("BENCHMARK_CLOUD_ENTITY_ID", "bench-cloud-entity")
    process_id = os.environ.get("BENCHMARK_CLOUD_PROCESS_ID", "bench-cloud-process")
    cfg = _make_cloud_cfg(entity_id=entity_id, process_id=process_id)

    query = "What do I like?"
    invoke = _make_invoke_with_stubbed_history(cfg, n_history_pairs=n_history_pairs)

    def _call():
        kwargs = {"messages": [{"role": "user", "content": query}]}
        return invoke.inject_conversation_messages(kwargs)

    result = benchmark(_call)
    assert isinstance(result, dict)


@pytest.mark.benchmark
@pytest.mark.parametrize("n_recalled_facts", [5, 20], ids=["n5", "n20"])
def test_benchmark_cloud_injection_only_recalled_facts(
    benchmark, n_recalled_facts: int, mocker
) -> None:
    """ONLY Python-side recalled-facts injection (no network)."""
    entity_id = os.environ.get("BENCHMARK_CLOUD_ENTITY_ID", "bench-cloud-entity")
    process_id = os.environ.get("BENCHMARK_CLOUD_PROCESS_ID", "bench-cloud-process")
    cfg = _make_cloud_cfg(entity_id=entity_id, process_id=process_id)

    fake_facts: list[dict[str, object]] = [
        {"content": f"User likes item_{i}", "rank_score": 1.0, "date_created": None}
        for i in range(n_recalled_facts)
    ]
    mocker.patch("memori.memory.recall.Recall.search_facts", return_value=fake_facts)

    invoke = BaseInvoke(cfg, lambda **_kwargs: None)
    query = "What do I like?"

    def _call():
        kwargs = {"messages": [{"role": "user", "content": query}]}
        return invoke.inject_recalled_facts(kwargs)

    result = benchmark(_call)
    assert isinstance(result, dict)
