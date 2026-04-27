# Changelog

All notable changes to the Memori Python SDK will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- Conversation injection no longer corrupts OpenAI-compatible message
  sequences for tool-using conversations. Recalled history previously
  replayed `role="tool"` rows (with no `tool_call_id`) and empty
  assistant rows (whose `tool_calls` were never persisted), causing
  upstream providers to reject the request with `400: An assistant
  message with 'tool_calls' must be followed by tool messages
  responding to each 'tool_call_id'`. The OpenAI/Anthropic/Bedrock
  injection paths now sanitise recalled history before prepending it.
  Legacy Gemini-era `role="model"` rows are normalised to
  `role="assistant"` for the same reason. The injected-message counter
  now tracks the post-sanitisation count so the post-response payload
  does not slice into the current user message before persistence and
  augmentation. (#434)

## [3.3.0rc1] - 2026-04-16

### Added

- **Experimental Rust-backed retrieval and augmentation path.** Opt-in via
  `MEMORI_USE_RUST_CORE=1` in BYODB mode. Provides a native hybrid-search
  recall pipeline (dense + lexical re-ranking) and lower-overhead background
  augmentation dispatch via a `tokio`-based worker runtime. The pure-Python
  path remains the default and is unchanged.
- **Prebuilt platform wheels.** Released as `cp310-abi3` wheels covering
  Python 3.10 through 3.14 on `manylinux_2_28_{x86_64,aarch64}`,
  `macosx_{x86_64,arm64}`, and `win_amd64`. First-time users of the Rust core
  will download a ~25 MB ONNX embedding model from Hugging Face on first use,
  cached under `~/.fastembed_cache/`.
- **Source distribution now includes the `core/` Rust crate.** Users on
  unsupported platforms (or who opt out of the wheel) can build from source
  provided a Rust toolchain is available.
- Continuous integration for the Rust core (`core-ci.yml`) covering
  `cargo fmt`, `clippy`, unit tests, and cross-platform wheel build smoke via
  `cibuildwheel`.
- `dry_run` and `publish_memorisdk` inputs on the PyPI publish workflow for
  release dress rehearsals without touching the index.

### Changed

- Internal crate directory `rust-core/` renamed to `core/`. No public import
  path is affected; the Rust crate name (`engine-orchestrator`) and the Python
  extension name (`memori_python`) are unchanged.
- Debug payload logging in the augmentation pipeline now routes through the
  standard `logging` module at `DEBUG` level. The previous stdout-based
  behavior gated by `MEMORI_DEBUG_AA_PAYLOAD=1` has been removed; enable
  debug-level logging on the `memori._rust_core` and `engine_orchestrator`
  loggers instead.
- PyPI publish pipeline rewritten around `cibuildwheel` v3.4.0 for
  PyPI-compliant wheel tags across all supported platforms. Pure-Python
  fallback is still available via the sdist.

### Fixed

- Fixed multi-turn conversation ingestion for AzureOpenAI and OpenAI clients.
  Previously, only the first conversation turn was being recorded. Now
  `conversation_id` is resolved early in the request lifecycle, ensuring all
  conversation turns are properly ingested into the same conversation.
  (Fixes #83)

[3.3.0rc1]: https://github.com/MemoriLabs/Memori/releases/tag/v3.3.0rc1
[3.0.0]: https://github.com/MemoriLabs/Memori/releases/tag/v3.0.0
