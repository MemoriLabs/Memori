import argparse
import os

from benchmarks.locomo._run_impl import RunConfig, run_locomo


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="LoCoMo benchmark harness (Phase 2)")
    parser.add_argument(
        "--allow-prod-aa",
        action="store_true",
        help="(Deprecated) Allow Advanced Augmentation calls without MEMORI_TEST_MODE=1 (production).",
    )
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
        choices=["per_pair"],
        default="per_pair",
        help="How to batch messages when calling Advanced Augmentation (default: per_pair).",
    )
    parser.add_argument(
        "--aa-dry-run",
        action="store_true",
        help="Print/write AA request payload and exit before making any network calls.",
    )
    parser.add_argument(
        "--aa-max-requests",
        type=int,
        default=0,
        help="Limit the number of AA requests to enqueue during ingestion (0 = no limit).",
    )
    parser.add_argument(
        "--seed-only",
        action="store_true",
        help="Run ingestion only (no retrieval/scoring). Useful for AA payload/summary validation.",
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
        "--aa-provenance-mode",
        choices=["similarity"],
        default="similarity",
        help=(
            "How to attribute AA facts back to LoCoMo turn IDs for scoring. "
            "'similarity' maps facts to turns post-hoc using embedding/text similarity (default). "
        ),
    )
    parser.add_argument(
        "--rebuild-provenance",
        action="store_true",
        help="When --reuse-db: recompute provenance offline from the SQLite DB (no AA calls).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Limit number of samples (0 = no limit).",
    )
    parser.add_argument(
        "--only-sample-id",
        default="",
        help="Run only a single sample_id (exact match).",
    )
    parser.add_argument(
        "--max-sessions",
        type=int,
        default=0,
        help="Limit number of sessions per sample (0 = no limit).",
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
    # LoCoMo benchmarks should never hit production AA. Force staging routing.
    os.environ["MEMORI_TEST_MODE"] = "1"
    run_locomo(
        RunConfig(
            dataset=args.dataset,
            out=args.out,
            sqlite_db=args.sqlite_db,
            provenance_db=args.provenance_db,
            reuse_db=args.reuse_db,
            run_id=args.run_id,
            k=args.k,
            aa_timeout=args.aa_timeout,
            aa_batch=args.aa_batch,
            aa_dry_run=args.aa_dry_run,
            aa_max_requests=args.aa_max_requests,
            meta_llm_provider=args.meta_llm_provider,
            meta_llm_version=args.meta_llm_version,
            meta_llm_sdk_version=args.meta_llm_sdk_version,
            meta_framework_provider=args.meta_framework_provider,
            meta_platform_provider=args.meta_platform_provider,
            aa_provenance_top_n=args.aa_provenance_top_n,
            aa_provenance_min_score=args.aa_provenance_min_score,
            aa_provenance_mode=args.aa_provenance_mode,
            rebuild_provenance=args.rebuild_provenance,
            allow_prod_aa=args.allow_prod_aa,
            max_samples=args.max_samples,
            only_sample_id=args.only_sample_id,
            max_sessions=args.max_sessions,
            max_questions=args.max_questions,
            seed_only=args.seed_only,
            verbose=args.verbose,
            log_every_questions=args.log_every_questions,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
