from __future__ import annotations

import datetime
import json
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from benchmarks.locomo._types import LoCoMoSample
from benchmarks.locomo.loader import load_locomo_json
from benchmarks.locomo.provenance import (
    FactAttribution,
    ProvenanceStore,
    attribute_facts_to_turn_ids,
)
from benchmarks.locomo.retrieval import (
    build_turn_facts,
    evidence_to_turn_ids,
)
from benchmarks.locomo.scoring import hit_at_k_groups, mrr_groups
from memori import Memori
from memori.llm._embeddings import embed_texts
from memori.memory.augmentation.input import AugmentationInput
from memori.memory.recall import Recall

CATEGORY_LABELS: dict[str, str] = {
    "1": "factual",
    "2": "temporal",
    "3": "reasoning",
    "4": "implicit",
    "5": "adversarial",
    "unknown": "unknown",
}


@dataclass(frozen=True, slots=True)
class RunConfig:
    dataset: str
    out: str
    sqlite_db: str = ""
    provenance_db: str = ""
    reuse_db: bool = False
    run_id: str = ""
    k: int = 5
    aa_timeout: float = 180.0
    aa_batch: str = "per_pair"
    aa_dry_run: bool = False
    aa_max_requests: int = 0
    meta_llm_provider: str = "openai"
    meta_llm_version: str = "gpt-4.1-mini"
    meta_llm_sdk_version: str = "unknown"
    meta_framework_provider: str = "memori"
    meta_platform_provider: str = "benchmark"
    aa_provenance_top_n: int = 1
    aa_provenance_min_score: float = 0.25
    aa_provenance_mode: str = "similarity"
    rebuild_provenance: bool = False
    allow_prod_aa: bool = False
    max_samples: int = 0
    max_questions: int = 0
    only_sample_id: str = ""
    max_sessions: int = 0
    verbose: bool = False
    log_every_questions: int = 0
    seed_only: bool = False


@dataclass(frozen=True, slots=True)
class PairRequest:
    messages: list[dict[str, str]]
    pair_turn_ids: tuple[str, str]


def run_locomo(cfg: RunConfig) -> dict:
    out_dir = Path(cfg.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = out_dir / "predictions.jsonl"
    summary_path = out_dir / "summary.json"

    sqlite_path = (
        Path(cfg.sqlite_db).expanduser()
        if cfg.sqlite_db
        else (out_dir / "locomo.sqlite")
    )
    provenance_path = (
        Path(cfg.provenance_db).expanduser()
        if cfg.provenance_db
        else (out_dir / "locomo_provenance.sqlite")
    )

    samples = load_locomo_json(cfg.dataset)
    if cfg.only_sample_id:
        samples = [s for s in samples if s.sample_id == cfg.only_sample_id]
    if cfg.max_samples and cfg.max_samples > 0:
        samples = samples[: cfg.max_samples]

    run_id = _resolve_run_id(
        cfg=cfg,
        out_dir=out_dir,
        sqlite_path=sqlite_path,
        provenance_path=provenance_path,
    )
    ts = datetime.datetime.now(datetime.timezone.utc).isoformat()

    mem = _init_memori(sqlite_path)
    recall = Recall(mem.config)
    prov_store = ProvenanceStore(provenance_path)

    if (
        not cfg.reuse_db
        and not cfg.aa_dry_run
        and not cfg.allow_prod_aa
        and os.environ.get("MEMORI_TEST_MODE") != "1"
    ):
        raise RuntimeError(
            "Advanced Augmentation LoCoMo runs must target staging. "
            "Set MEMORI_TEST_MODE=1 (recommended) or pass --allow-prod-aa to override."
        )

    if cfg.verbose:
        from memori._network import Api

        url = Api(mem.config).url("sdk/augmentation")
        print(
            f"[locomo][aa] resolved_api_url={url} MEMORI_TEST_MODE={os.environ.get('MEMORI_TEST_MODE')!r}"
        )

    totals = _Totals()

    with predictions_path.open("w", encoding="utf-8") as f:
        if cfg.verbose:
            print(
                f"[locomo] start ingest=advanced_augmentation aa_batch={cfg.aa_batch} "
                f"samples={len(samples)} k={cfg.k} seed_only={cfg.seed_only}"
            )
        for sample in samples:
            if cfg.max_sessions and cfg.max_sessions > 0:
                sample = LoCoMoSample(
                    sample_id=sample.sample_id,
                    sessions=sample.sessions[: cfg.max_sessions],
                    qa=sample.qa,
                )
            entity_external_id = f"locomo:{run_id}:{sample.sample_id}"
            if cfg.reuse_db:
                entity_db_id = _get_entity_id_sqlite(
                    sqlite_path=sqlite_path,
                    entity_external_id=entity_external_id,
                )
                if entity_db_id is None:
                    raise RuntimeError(
                        f"--reuse-db was set but entity was not found in sqlite DB: external_id={entity_external_id}. "
                        "Run ingestion first (omit --reuse-db) or pass the correct --run-id."
                    )
            else:
                mem.attribution(
                    entity_id=entity_external_id, process_id="locomo-benchmark"
                )
                entity_db_id = mem.config.storage.driver.entity.create(
                    entity_external_id
                )

            turn_facts, turn_index = build_turn_facts(sample)

            if cfg.reuse_db:
                fact_count = _count_entity_facts_sqlite(
                    sqlite_path=sqlite_path, entity_db_id=entity_db_id
                )
                if fact_count <= 0:
                    raise RuntimeError(
                        f"--reuse-db was set but no facts were found for entity external_id={entity_external_id}. "
                        "Run ingestion first (omit --reuse-db) or pass the correct --run-id."
                    )
                mem.config.recall_embeddings_limit = fact_count
                if cfg.rebuild_provenance and not cfg.seed_only:
                    _build_aa_provenance(
                        mem=mem,
                        prov_store=prov_store,
                        run_id=run_id,
                        sample_id=sample.sample_id,
                        entity_db_id=entity_db_id,
                        turn_facts=turn_facts,
                        top_n=cfg.aa_provenance_top_n,
                        min_score=cfg.aa_provenance_min_score,
                    )
                elif not cfg.seed_only and not prov_store.has_any(
                    run_id=run_id, sample_id=sample.sample_id
                ):
                    raise RuntimeError(
                        f"--reuse-db was set but no provenance rows were found for run_id={run_id} sample_id={sample.sample_id}. "
                        "Seed AA ingestion once (omit --reuse-db) to generate locomo_provenance.sqlite, "
                        "or point --provenance-db at an existing provenance DB, "
                        "or pass --rebuild-provenance to recompute provenance offline."
                    )
                if cfg.verbose:
                    print(
                        f"[locomo][reuse] sample={sample.sample_id} facts={fact_count}"
                    )
            else:
                _configure_aa_meta(mem, cfg)
                if cfg.aa_dry_run:
                    _write_aa_payload_preview(
                        out_dir=out_dir,
                        mem=mem,
                        sample=sample,
                        cfg=cfg,
                        entity_external_id=entity_external_id,
                        process_id="locomo-benchmark",
                    )
                    # Exit before any network call.
                    return {
                        "run_id": run_id,
                        "timestamp_utc": ts,
                        "dataset_path": str(Path(cfg.dataset).resolve()),
                        "sqlite_db_path": str(sqlite_path),
                        "provenance_db_path": str(provenance_path),
                        "ingest": "advanced_augmentation",
                        "sample_count": len(samples),
                        "question_count": 0,
                        "questions_by_category": {},
                        "metrics_overall": {},
                        "metrics_by_category": {},
                        "phase": 2,
                        "note": "AA dry-run: printed/wrote payload preview and exited before network calls.",
                    }
                conv_id = _ingest_with_advanced_augmentation(
                    mem=mem,
                    prov_store=prov_store,
                    run_id=run_id,
                    entity_external_id=entity_external_id,
                    entity_db_id=entity_db_id,
                    sample=sample,
                    cfg=cfg,
                )
                if conv_id is not None:
                    summary_val = _read_conversation_summary(mem, conv_id) or ""
                    print(
                        f"[locomo][aa] sample={sample.sample_id} "
                        f"conversation_id={conv_id} summary_is_set={bool(summary_val)}"
                    )
                    if summary_val:
                        print(f"[locomo][aa] summary={summary_val}")

                if cfg.seed_only:
                    continue

                if cfg.verbose:
                    fact_count = len(
                        mem.config.storage.driver.entity_fact.get_embeddings(
                            entity_db_id, limit=100000
                        )
                    )
                    print(
                        f"[locomo][seed] sample={sample.sample_id} mode=advanced_augmentation "
                        f"turns={len(turn_facts)} facts_written={fact_count}"
                    )
                if cfg.aa_provenance_mode == "similarity":
                    _build_aa_provenance(
                        mem=mem,
                        prov_store=prov_store,
                        run_id=run_id,
                        sample_id=sample.sample_id,
                        entity_db_id=entity_db_id,
                        turn_facts=turn_facts,
                        top_n=cfg.aa_provenance_top_n,
                        min_score=cfg.aa_provenance_min_score,
                    )
                # Keep recall bounded to the facts created for this entity.
                entity_fact_driver = mem.config.storage.driver.entity_fact
                mem.config.recall_embeddings_limit = (
                    len(entity_fact_driver.get_embeddings(entity_db_id, limit=100000))
                    or 1
                )

            qa = list(sample.qa)
            if cfg.max_questions and cfg.max_questions > 0:
                qa = qa[: cfg.max_questions]

            available_turn_ids = {t.turn_id for t in turn_facts}

            for q in qa:
                if (
                    cfg.verbose
                    and cfg.log_every_questions
                    and totals.question_count % cfg.log_every_questions == 0
                ):
                    print(
                        f"[locomo][score] questions={totals.question_count} "
                        f"sample={sample.sample_id}"
                    )

                relevant = evidence_to_turn_ids(q.evidence, turn_index=turn_index)
                if not relevant:
                    if cfg.verbose:
                        print(
                            f"[locomo][skip] sample={sample.sample_id} "
                            f"question_id={q.question_id} reason=no_evidence"
                        )
                    continue
                missing = sorted(
                    tid for tid in relevant if tid not in available_turn_ids
                )
                if missing:
                    if cfg.verbose:
                        preview = ", ".join(missing[:5])
                        more = (
                            "" if len(missing) <= 5 else f" (+{len(missing) - 5} more)"
                        )
                        print(
                            f"[locomo][skip] sample={sample.sample_id} "
                            f"question_id={q.question_id} reason=missing_evidence "
                            f"missing={preview}{more}"
                        )
                    continue

                totals.count_question(q.category)
                results = recall.search_facts(
                    query=q.question,
                    limit=max(cfg.k, 1),
                    entity_id=entity_db_id,
                )

                # Scoring in AA mode needs a mapping from fact_id -> dia_id (LoCoMo turn id).
                # A single fact can plausibly map to multiple turns; we score using any-match
                # semantics per retrieved rank.
                retrieved_ids: list[str] = []
                retrieved_groups: list[set[str]] = []
                top_k = _format_top_k(
                    results=results[: max(cfg.k, 1)],
                    prov_store=prov_store,
                    run_id=run_id,
                    sample_id=sample.sample_id,
                    retrieved_ids_out=retrieved_ids,
                    retrieved_groups_out=retrieved_groups,
                    provenance_limit=max(int(cfg.aa_provenance_top_n), 1),
                )

                metrics = {
                    "hit@1": hit_at_k_groups(relevant, retrieved_groups, 1),
                    "hit@3": hit_at_k_groups(relevant, retrieved_groups, 3),
                    "hit@5": hit_at_k_groups(relevant, retrieved_groups, 5),
                    "mrr": mrr_groups(relevant, retrieved_groups),
                }
                totals.add_metrics(category=q.category or "unknown", metrics=metrics)

                row = {
                    "run_id": run_id,
                    "timestamp_utc": ts,
                    "sample_id": sample.sample_id,
                    "question_id": q.question_id,
                    "category": q.category,
                    "question": q.question,
                    "answers": list(q.answers),
                    "evidence": q.evidence,
                    "retrieval": {
                        "status": "ok",
                        "ingest": "advanced_augmentation",
                        "k": cfg.k,
                        "relevant_turn_ids": sorted(relevant),
                        "top_k": top_k,
                        "metrics": metrics,
                    },
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = totals.to_summary(
        run_id=run_id,
        timestamp_utc=ts,
        dataset_path=str(Path(cfg.dataset).resolve()),
        sqlite_db_path=str(sqlite_path),
        provenance_db_path=str(provenance_path),
        ingest="advanced_augmentation",
        sample_count=len(samples),
    )
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _resolve_run_id(
    *, cfg: RunConfig, out_dir: Path, sqlite_path: Path, provenance_path: Path
) -> str:
    if cfg.run_id:
        return cfg.run_id
    if not cfg.reuse_db:
        return str(uuid4())

    summary_path = out_dir / "summary.json"
    if summary_path.exists():
        try:
            obj = json.loads(summary_path.read_text(encoding="utf-8"))
            rid = obj.get("run_id")
            if isinstance(rid, str) and rid:
                return rid
        except Exception:
            pass

    run_ids: set[str] = set()
    if provenance_path.exists():
        run_ids |= _distinct_run_ids_from_provenance_sqlite(provenance_path)

    if sqlite_path.exists():
        run_ids |= _distinct_run_ids_from_memori_sqlite(sqlite_path)

    if len(run_ids) == 1:
        return next(iter(run_ids))
    if not run_ids:
        raise RuntimeError(
            "--reuse-db was set but no prior LoCoMo run_id was found in the DB(s). "
            "Pass --run-id or run ingestion once without --reuse-db."
        )
    raise RuntimeError(
        "--reuse-db was set but multiple run_ids were found in the DB(s). "
        f"Pass --run-id to choose one. found={sorted(run_ids)}"
    )


def _distinct_run_ids_from_provenance_sqlite(path: Path) -> set[str]:
    with sqlite3.connect(str(path), check_same_thread=False) as conn:
        try:
            cur = conn.execute(
                "SELECT DISTINCT run_id FROM bench_locomo_fact_provenance"
            )
        except sqlite3.OperationalError:
            return set()
        return {r[0] for r in cur.fetchall() if r and isinstance(r[0], str) and r[0]}


def _distinct_run_ids_from_memori_sqlite(path: Path) -> set[str]:
    with sqlite3.connect(str(path), check_same_thread=False) as conn:
        try:
            cur = conn.execute(
                "SELECT external_id FROM memori_entity WHERE external_id LIKE 'locomo:%'"
            )
        except sqlite3.OperationalError:
            return set()
        out: set[str] = set()
        for row in cur.fetchall():
            if not row:
                continue
            external_id = row[0]
            if not isinstance(external_id, str) or not external_id.startswith(
                "locomo:"
            ):
                continue
            parts = external_id.split(":")
            if len(parts) >= 3 and parts[1]:
                out.add(parts[1])
        return out


def _get_entity_id_sqlite(*, sqlite_path: Path, entity_external_id: str) -> int | None:
    with sqlite3.connect(str(sqlite_path), check_same_thread=False) as conn:
        cur = conn.execute(
            "SELECT id FROM memori_entity WHERE external_id = ?",
            (entity_external_id,),
        )
        row = cur.fetchone()
        if not row:
            return None
        try:
            return int(row[0])
        except (TypeError, ValueError):
            return None


def _count_entity_facts_sqlite(*, sqlite_path: Path, entity_db_id: int) -> int:
    with sqlite3.connect(str(sqlite_path), check_same_thread=False) as conn:
        cur = conn.execute(
            "SELECT COUNT(*) FROM memori_entity_fact WHERE entity_id = ?",
            (int(entity_db_id),),
        )
        row = cur.fetchone()
        if not row:
            return 0
        try:
            return int(row[0])
        except (TypeError, ValueError):
            return 0


def _init_memori(sqlite_path: Path) -> Memori:
    def _conn():
        return sqlite3.connect(str(sqlite_path), check_same_thread=False)

    mem = Memori(conn=_conn)
    mem.config.storage.build()
    return mem


def _configure_aa_meta(mem: Memori, cfg: RunConfig) -> None:
    mem.config.framework.provider = cfg.meta_framework_provider
    mem.config.platform.provider = cfg.meta_platform_provider
    mem.config.llm.provider = cfg.meta_llm_provider
    mem.config.llm.version = cfg.meta_llm_version
    mem.config.llm.provider_sdk_version = cfg.meta_llm_sdk_version


def _ingest_with_advanced_augmentation(
    *,
    mem: Memori,
    prov_store: ProvenanceStore,
    run_id: str,
    entity_external_id: str,
    entity_db_id: int,
    sample,
    cfg: RunConfig,
) -> int | None:
    if cfg.aa_batch == "per_pair":
        return _aa_enqueue_pairs_sequential(
            mem=mem,
            prov_store=prov_store,
            run_id=run_id,
            entity_external_id=entity_external_id,
            entity_db_id=entity_db_id,
            sample_id=sample.sample_id,
            sample=sample,
            timeout=cfg.aa_timeout,
            max_requests=int(cfg.aa_max_requests or 0),
            verbose=bool(cfg.verbose),
        )

    raise ValueError(f"unknown aa batch: {cfg.aa_batch}")


def _aa_enqueue_pairs_sequential(
    *,
    mem: Memori,
    prov_store: ProvenanceStore,
    run_id: str,
    entity_external_id: str,
    entity_db_id: int,
    sample_id: str,
    sample,
    timeout: float,
    max_requests: int = 0,
    verbose: bool = False,
) -> int | None:
    all_msgs, all_turn_ids = _build_aa_messages_and_turn_ids_for_sample(sample)
    conv_id = _create_conversation_and_persist_messages(mem, all_msgs)

    reqs = _build_per_pair_requests(all_msgs, all_turn_ids)
    max_n = int(max_requests or 0)
    if max_n > 0:
        reqs = reqs[:max_n]

    entity_fact_driver = mem.config.storage.driver.entity_fact
    known_fact_ids = {
        int(r["id"])
        for r in entity_fact_driver.get_embeddings(int(entity_db_id), limit=100000)
    }

    for idx, req in enumerate(reqs):
        _enqueue_aa(mem, conv_id, entity_external_id, req.messages)
        mem.augmentation.wait(timeout=timeout)

        after = {
            int(r["id"])
            for r in entity_fact_driver.get_embeddings(int(entity_db_id), limit=100000)
        }
        new_fact_ids = sorted(after - known_fact_ids)
        known_fact_ids = after

        if new_fact_ids:
            rows: list[FactAttribution] = []
            for fid in new_fact_ids:
                for dia_id in req.pair_turn_ids:
                    if dia_id:
                        rows.append(
                            FactAttribution(fact_id=fid, dia_id=dia_id, score=1.0)
                        )
            prov_store.upsert_many(rows, run_id=run_id, sample_id=sample_id)

        if verbose:
            summary_val = _read_conversation_summary(mem, int(conv_id)) or ""
            print(
                f"[locomo][aa][pair] idx={idx} new_facts={len(new_fact_ids)} "
                f"pair_turn_ids={req.pair_turn_ids} summary_is_set={bool(summary_val)}"
            )

    return int(conv_id)


def _read_conversation_summary(mem: Memori, conversation_id: int) -> str | None:
    try:
        obj = mem.config.storage.driver.conversation.read(int(conversation_id))
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None
    summary = obj.get("summary")
    return summary if isinstance(summary, str) and summary.strip() else None


def _create_conversation_and_persist_messages(
    mem: Memori, msgs: list[dict[str, str]]
) -> int:
    mem.new_session()
    conv_id = mem.config.storage.driver.conversation.create(
        str(mem.config.session_id), mem.config.session_timeout_minutes
    )
    for m in msgs:
        mem.config.storage.driver.conversation.message.create(
            conv_id, m["role"], "text", m["content"]
        )
    if mem.config.storage.adapter is not None:
        mem.config.storage.adapter.commit()
    return int(conv_id)


def _enqueue_aa(
    mem: Memori, conv_id: int, entity_external_id: str, msgs: list[dict[str, str]]
) -> None:
    mem.augmentation.enqueue(
        AugmentationInput(
            conversation_id=str(conv_id),
            entity_id=entity_external_id,
            process_id="locomo-benchmark",
            conversation_messages=msgs,
            system_prompt=None,
        )
    )


def _build_aa_messages_for_sample(sample) -> list[dict[str, str]]:
    msgs, _turn_ids = _build_aa_messages_and_turn_ids_for_sample(sample)
    return msgs


def _build_aa_messages_for_session(sample, session) -> list[dict[str, str]]:
    speaker_to_role = _speaker_to_role(sample)
    msgs, _turn_ids = _build_aa_messages_and_turn_ids_for_turns(
        sample, session, speaker_to_role
    )
    return msgs


def _build_aa_messages_for_turns(
    sample, session, speaker_to_role: dict[str, str]
) -> list[dict[str, str]]:
    msgs, _turn_ids = _build_aa_messages_and_turn_ids_for_turns(
        sample, session, speaker_to_role
    )
    return msgs


def _build_aa_messages_and_turn_ids_for_sample(
    sample,
) -> tuple[list[dict[str, str]], list[str]]:
    speaker_to_role = _speaker_to_role(sample)
    msgs: list[dict[str, str]] = []
    turn_ids: list[str] = []
    for session in sample.sessions:
        m, t = _build_aa_messages_and_turn_ids_for_turns(
            sample, session, speaker_to_role
        )
        msgs.extend(m)
        turn_ids.extend(t)
    return msgs, turn_ids


def _build_aa_messages_and_turn_ids_for_turns(
    sample, session, speaker_to_role: dict[str, str]
) -> tuple[list[dict[str, str]], list[str]]:
    msgs: list[dict[str, str]] = []
    turn_ids: list[str] = []
    session_id = session.session_id or ""
    for t_idx, turn in enumerate(session.turns):
        speaker = (turn.speaker or "").strip()
        role = speaker_to_role.get(speaker, "assistant")
        turn_id = (
            turn.turn_id or ""
        ).strip() or f"{sample.sample_id}:{session_id}:{t_idx}"
        content = _format_turn_content(
            turn_id=turn_id,
            speaker=speaker,
            text=turn.text,
            session_date_time=session.date_time,
        )
        msgs.append({"role": role, "content": content})
        turn_ids.append(turn_id)
    return msgs, turn_ids


def _format_turn_content(
    *, turn_id: str, speaker: str, text: str, session_date_time: str | None
) -> str:
    # Mirror real-world Memori payloads: content is the plain message text only.
    # Keep turn_id/speaker/session_time out of the message body (benchmark-only metadata).
    return str(text).strip()


def _speaker_to_role(sample) -> dict[str, str]:
    """
    Heuristic mapping: first unique speaker => user, second => assistant, others => assistant.
    """
    out: dict[str, str] = {}
    ordered: list[str] = []
    for session in sample.sessions:
        for turn in session.turns:
            speaker = (turn.speaker or "").strip()
            if not speaker or speaker in out:
                continue
            ordered.append(speaker)
            out[speaker] = "user" if len(ordered) == 1 else "assistant"
    return out


def _build_aa_provenance(
    *,
    mem: Memori,
    prov_store: ProvenanceStore,
    run_id: str,
    sample_id: str,
    entity_db_id: int,
    turn_facts,
    top_n: int,
    min_score: float,
) -> None:
    turn_ids = [t.turn_id for t in turn_facts]
    turn_texts = [t.content for t in turn_facts]
    turn_embs = embed_texts(
        turn_texts,
        model=mem.config.embeddings.model,
        fallback_dimension=mem.config.embeddings.fallback_dimension,
    )

    entity_fact_driver = mem.config.storage.driver.entity_fact
    rows = entity_fact_driver.get_embeddings(entity_db_id, limit=100000)
    fact_ids = [int(r["id"]) for r in rows]
    facts = entity_fact_driver.get_facts_by_ids(fact_ids)
    content_by_id = {int(r["id"]): r["content"] for r in facts}

    fact_ids_aligned = [i for i in fact_ids if i in content_by_id]
    fact_texts = [content_by_id[i] for i in fact_ids_aligned]
    fact_embs = embed_texts(
        fact_texts,
        model=mem.config.embeddings.model,
        fallback_dimension=mem.config.embeddings.fallback_dimension,
    )

    # Clear any prior mappings for this run/sample to avoid mixing old/new strategies.
    prov_store.delete_sample(run_id=run_id, sample_id=sample_id)

    mapping = attribute_facts_to_turn_ids(
        turn_ids=turn_ids,
        turn_embeddings=turn_embs,
        turn_texts=turn_texts,
        fact_ids=fact_ids_aligned,
        fact_embeddings=fact_embs,
        fact_texts=fact_texts,
        top_n=top_n,
        min_score=min_score,
    )

    prov_rows: list[FactAttribution] = []
    for fid, mapped in mapping.items():
        for dia_id, score in mapped:
            prov_rows.append(FactAttribution(fact_id=fid, dia_id=dia_id, score=score))
    prov_store.upsert_many(prov_rows, run_id=run_id, sample_id=sample_id)


def _write_aa_payload_preview(
    *,
    out_dir: Path,
    mem: Memori,
    sample,
    cfg: RunConfig,
    entity_external_id: str,
    process_id: str,
) -> None:
    """
    Build and print the exact AA request payload (no network call).

    This is useful for debugging staging/prod routing, payload structure, and metadata.
    """
    import hashlib

    from memori._network import Api
    from memori.memory.augmentation.augmentations.memori._augmentation import (
        AdvancedAugmentation,
    )

    dialect = (
        mem.config.storage.adapter.get_dialect()
        if mem.config.storage and mem.config.storage.adapter
        else "unknown"
    )

    aug = AdvancedAugmentation(config=mem.config, enabled=True)
    url = Api(mem.config).url("sdk/augmentation")

    all_msgs, all_turn_ids = _build_aa_messages_and_turn_ids_for_sample(sample)
    requests: list[dict[str, object]] = []
    if cfg.aa_batch != "per_pair":
        raise ValueError(f"unknown aa batch: {cfg.aa_batch}")
    for i, req in enumerate(_build_per_pair_requests(all_msgs, all_turn_ids)):
        payload = aug._build_api_payload(  # noqa: SLF001 - benchmark-only debug path
            req.messages,
            "",
            None,
            dialect,
            entity_external_id,
            process_id,
        )
        requests.append(
            {
                "request_index": i,
                "pair_turn_ids": list(req.pair_turn_ids),
                "messages": req.messages,
                "payload": payload,
            }
        )

    api_key = mem.config.api_key or ""
    api_key_preview = ""
    api_key_sha256 = ""
    if api_key:
        api_key_preview = (
            f"{api_key[:4]}…{api_key[-4:]}" if len(api_key) >= 8 else "set"
        )
        api_key_sha256 = hashlib.sha256(api_key.encode("utf-8")).hexdigest()

    preview = {
        "url": url,
        "ingest": "advanced_augmentation",
        "aa_batch": cfg.aa_batch,
        "entity_id": entity_external_id,
        "process_id": process_id,
        "memori_api_key": {
            "is_set": bool(api_key),
            "length": len(api_key),
            "preview": api_key_preview,
            "sha256": api_key_sha256,
        },
        "requests": requests,
    }

    out_path = out_dir / "aa_payload_preview.json"
    out_path.write_text(json.dumps(preview, indent=2), encoding="utf-8")

    print(f"[locomo][aa-dry-run] url={url}")
    print(f"[locomo][aa-dry-run] wrote={out_path}")
    # Avoid flooding stdout for large datasets.
    max_print = 5
    preview_print = dict(preview)
    preview_print["requests_total"] = len(requests)
    preview_print["requests_printed"] = min(max_print, len(requests))
    preview_print["requests"] = requests[:max_print]
    print(json.dumps(preview_print, indent=2))


def _build_per_pair_requests(
    msgs: list[dict[str, str]],
    turn_ids: list[str],
) -> list[PairRequest]:
    """
    Build per-pair AA requests that mirror SDK behavior.

    Each request includes *all* prior messages (oldest -> newest) plus the next
    strict user->assistant pair at the end.

    Any messages that can't be paired as a strict consecutive user+assistant pair
    are skipped as pair boundaries, but still remain in the context for later requests.
    """
    if len(msgs) != len(turn_ids):
        raise ValueError("msgs and turn_ids must be aligned")

    out: list[PairRequest] = []
    i = 0
    while i < len(msgs) - 1:
        role0 = (msgs[i].get("role") or "").strip()
        role1 = (msgs[i + 1].get("role") or "").strip()
        if role0 == "user" and role1 == "assistant":
            tid0 = (turn_ids[i] or "").strip()
            tid1 = (turn_ids[i + 1] or "").strip()
            out.append(
                PairRequest(messages=list(msgs[: i + 2]), pair_turn_ids=(tid0, tid1))
            )
            i += 2
            continue
        i += 1
    return out


def _format_top_k(
    *,
    results: list[dict],
    prov_store: ProvenanceStore,
    run_id: str,
    sample_id: str,
    retrieved_ids_out: list[str],
    retrieved_groups_out: list[set[str]],
    provenance_limit: int = 1,
) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for r in results:
        content = r.get("content", "")
        fact_id = r.get("id")

        turn_ids: list[str] = []
        if isinstance(fact_id, int):
            dia_ids = prov_store.best_dia_ids_for_fact(
                run_id=run_id,
                sample_id=sample_id,
                fact_id=fact_id,
                limit=max(int(provenance_limit), 1),
            )
            turn_ids = [d for d in dia_ids if d]

        # For backward compatibility, keep a single primary turn_id field (first).
        turn_id = turn_ids[0] if turn_ids else ""
        if turn_id:
            retrieved_ids_out.append(turn_id)

        retrieved_groups_out.append(set(turn_ids))

        out.append(
            {
                "turn_id": turn_id,
                "turn_ids": turn_ids,
                "fact_id": fact_id,
                "similarity": r.get("similarity"),
                "content": content,
            }
        )
    return out


class _Totals:
    def __init__(self) -> None:
        self.question_count = 0
        self.questions_by_category: dict[str, int] = {}
        self.sums = {"hit@1": 0.0, "hit@3": 0.0, "hit@5": 0.0, "mrr": 0.0}
        self.sums_by_cat: dict[str, dict[str, float]] = {}
        self.counts_by_cat: dict[str, int] = {}

    def count_question(self, category: str | None) -> None:
        self.question_count += 1
        cat = category or "unknown"
        self.questions_by_category[cat] = self.questions_by_category.get(cat, 0) + 1

    def add_metrics(self, *, category: str, metrics: dict[str, float]) -> None:
        for key in self.sums:
            self.sums[key] += float(metrics[key])
        self.sums_by_cat.setdefault(
            category, {"hit@1": 0.0, "hit@3": 0.0, "hit@5": 0.0, "mrr": 0.0}
        )
        for key in self.sums_by_cat[category]:
            self.sums_by_cat[category][key] += float(metrics[key])
        self.counts_by_cat[category] = self.counts_by_cat.get(category, 0) + 1

    def to_summary(
        self,
        *,
        run_id: str,
        timestamp_utc: str,
        dataset_path: str,
        sqlite_db_path: str,
        provenance_db_path: str,
        ingest: str,
        sample_count: int,
    ) -> dict:
        denom = float(self.question_count) if self.question_count else 1.0
        metrics_overall = {k: (self.sums[k] / denom) for k in self.sums}

        metrics_by_category: dict[str, dict[str, float]] = {}
        for cat, sums_cat in self.sums_by_cat.items():
            denom_cat = float(self.counts_by_cat.get(cat, 0)) or 1.0
            metrics_by_category[cat] = {k: (sums_cat[k] / denom_cat) for k in sums_cat}

        questions_by_category_labeled = {
            CATEGORY_LABELS.get(cat, cat): count
            for cat, count in self.questions_by_category.items()
        }
        metrics_by_category_labeled = {
            CATEGORY_LABELS.get(cat, cat): vals
            for cat, vals in metrics_by_category.items()
        }

        return {
            "run_id": run_id,
            "timestamp_utc": timestamp_utc,
            "dataset_path": dataset_path,
            "sqlite_db_path": sqlite_db_path,
            "provenance_db_path": provenance_db_path,
            "ingest": ingest,
            "sample_count": sample_count,
            "question_count": self.question_count,
            "category_labels": dict(CATEGORY_LABELS),
            "questions_by_category": dict(
                sorted(self.questions_by_category.items(), key=lambda kv: kv[0])
            ),
            "questions_by_category_labeled": dict(
                sorted(questions_by_category_labeled.items(), key=lambda kv: kv[0])
            ),
            "metrics_overall": metrics_overall,
            "metrics_by_category": dict(
                sorted(metrics_by_category.items(), key=lambda kv: kv[0])
            ),
            "metrics_by_category_labeled": dict(
                sorted(metrics_by_category_labeled.items(), key=lambda kv: kv[0])
            ),
            "phase": 2,
        }
