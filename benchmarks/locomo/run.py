import argparse

from benchmarks.locomo._run_impl import RunConfig, run_locomo


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="LoCoMo benchmark harness (Phase 2)")
    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to a LoCoMo JSON file (downloaded locally).",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output directory for artifacts (predictions.jsonl, summary.json).",
    )
    parser.add_argument(
        "--sqlite-db",
        default="",
        help="SQLite DB file path used for the run (default: <out>/locomo.sqlite).",
    )
    parser.add_argument(
        "--provenance-db",
        default="",
        help="Benchmark-only provenance DB (default: <out>/locomo_provenance.sqlite).",
    )
    parser.add_argument(
        "--ingest",
        choices=["turn_facts", "advanced_augmentation"],
        default="advanced_augmentation",
        help="How to ingest LoCoMo before retrieval scoring.",
    )
    parser.add_argument(
        "--reuse-db",
        action="store_true",
        help="Skip ingestion and reuse the existing SQLite/provenance DB for retrieval+scoring.",
    )
    parser.add_argument(
        "--run-id",
        default="",
        help="Run namespace used for entity external IDs/provenance (required when --reuse-db and multiple runs exist in the DB).",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Retrieval top-k to store and score (default: 5).",
    )
    parser.add_argument(
        "--aa-timeout",
        type=float,
        default=180.0,
        help="Timeout (seconds) to wait for Advanced Augmentation to finish per sample.",
    )
    parser.add_argument(
        "--aa-batch",
        choices=["per_sample", "per_session", "per_chunk"],
        default="per_sample",
        help="How to batch messages when calling Advanced Augmentation (default: per_sample).",
    )
    parser.add_argument(
        "--aa-chunk-size",
        type=int,
        default=16,
        help="When --aa-batch=per_chunk, how many messages to send per AA request (default: 16).",
    )
    parser.add_argument(
        "--aa-dry-run",
        action="store_true",
        help="Print/write AA request payload and exit before making any network calls.",
    )
    parser.add_argument(
        "--meta-llm-provider",
        default="openai",
        help="Metadata only: LLM provider to report to Advanced Augmentation (default: openai).",
    )
    parser.add_argument(
        "--meta-llm-version",
        default="gpt-4.1-mini",
        help="Metadata only: LLM model version to report (default: gpt-4.1-mini).",
    )
    parser.add_argument(
        "--meta-llm-sdk-version",
        default="unknown",
        help="Metadata only: LLM SDK version to report (default: unknown).",
    )
    parser.add_argument(
        "--meta-framework-provider",
        default="memori",
        help="Metadata only: framework provider to report (default: memori).",
    )
    parser.add_argument(
        "--meta-platform-provider",
        default="benchmark",
        help="Metadata only: platform provider to report (default: benchmark).",
    )
    parser.add_argument(
        "--aa-provenance-top-n",
        type=int,
        default=1,
        help="How many turn_ids to attribute to each augmented fact (default: 1).",
    )
    parser.add_argument(
        "--aa-provenance-min-score",
        type=float,
        default=0.25,
        help="Min cosine similarity to accept a fact->turn attribution (default: 0.25).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Limit number of samples (0 = no limit).",
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=0,
        help="Limit questions per sample (0 = no limit).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print progress/logging about seeding and scoring.",
    )
    parser.add_argument(
        "--log-every-questions",
        type=int,
        default=0,
        help="When --verbose, log progress every N questions (0 = disabled).",
    )
    args = parser.parse_args(argv)
    run_locomo(
        RunConfig(
            dataset=args.dataset,
            out=args.out,
            sqlite_db=args.sqlite_db,
            provenance_db=args.provenance_db,
            ingest=args.ingest,
            reuse_db=args.reuse_db,
            run_id=args.run_id,
            k=args.k,
            aa_timeout=args.aa_timeout,
            aa_batch=args.aa_batch,
            aa_chunk_size=args.aa_chunk_size,
            aa_dry_run=args.aa_dry_run,
            meta_llm_provider=args.meta_llm_provider,
            meta_llm_version=args.meta_llm_version,
            meta_llm_sdk_version=args.meta_llm_sdk_version,
            meta_framework_provider=args.meta_framework_provider,
            meta_platform_provider=args.meta_platform_provider,
            aa_provenance_top_n=args.aa_provenance_top_n,
            aa_provenance_min_score=args.aa_provenance_min_score,
            max_samples=args.max_samples,
            max_questions=args.max_questions,
            verbose=args.verbose,
            log_every_questions=args.log_every_questions,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
