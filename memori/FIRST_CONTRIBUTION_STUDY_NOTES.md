# Memori Repo Study Notes (First Contribution)

## 1) What this project does

Memori is a memory layer for LLM applications. It captures conversation turns, stores durable memory, and retrieves relevant context for future prompts so agents do not forget across sessions.

In this repo, the Python SDK (`memori/`) is the main runtime path for:

- registering/wrapping LLM clients,
- injecting recall + prior conversation history before model calls,
- persisting new conversation data after model calls,
- running advanced augmentation in the background.

Key entrypoint: `memori/__init__.py` (`class Memori`).

---

## 2) Architecture flow (repo-specific)

### Core components

- `memori/__init__.py`
  - Builds `Config`, storage manager, augmentation manager, optional Rust adapter.
  - Exposes `Memori().llm.register(...)`, `attribution(...)`, `recall(...)`, session controls.
- `memori/llm/_registry.py` + `memori/llm/clients/direct.py`
  - Detects provider/framework and wraps SDK methods (OpenAI, Anthropic, Google, xAI, etc.).
- `memori/llm/invoke/invoke.py`
  - Main request pipeline execution:
    - pre-invoke injection,
    - provider request,
    - post-invoke persistence/augmentation.
- `memori/llm/pipelines/recall_injection.py`
  - Builds and injects `<memori_context>` with recalled facts/summaries.
- `memori/llm/pipelines/conversation_injection.py`
  - Injects prior conversation history with provider-specific normalization.
- `memori/llm/pipelines/post_invoke.py`
  - Formats payload, writes messages, and triggers augmentation handler.
- `memori/memory/recall.py`
  - Recall logic for cloud and BYODB; relevance filtering; response normalization.
- `memori/memory/_writer.py` and `memori/memory/_manager.py`
  - Conversation persistence (DB/API) with retries/transaction handling.

### Runtime modes

- Cloud mode (default if `MEMORI_API_KEY` exists and no DB connection is passed).
- BYODB mode (when a DB connection is passed).
- Optional Rust core acceleration for BYODB retrieval (`use_rust_core`, env flags).

---

## 3) Request flow (end-to-end)

1. App creates `Memori`, registers an LLM client, and sets attribution (`entity_id`, optional `process_id`).
2. Wrapped LLM call enters invoke pipeline (`memori/llm/invoke/invoke.py`).
3. `inject_recalled_facts(...)`:
   - extracts user query,
   - runs recall (cloud/local/rust fallback paths),
   - filters by relevance threshold,
   - formats fact/summary lines,
   - injects context into provider-specific request fields.
4. `inject_conversation_messages(...)`:
   - fetches prior messages (cloud response or local storage),
   - sanitizes/normalizes history per provider,
   - prepends into request payload.
5. Actual provider SDK method executes.
6. `handle_post_response(...)`:
   - normalizes request/response payload,
   - persists conversation turn,
   - triggers augmentation pipeline.

---

## 4) How Memori injection works

Memori performs **pre-request prompt augmentation**.

- Recalled facts are rendered as bullet lines (`- ...`).
- If available, summaries are added under `## Summaries`.
- The full block is wrapped in `<memori_context> ... </memori_context>`.
- Injection target depends on provider/API shape:
  - `system` field for Anthropic/Bedrock,
  - Google system instruction helper,
  - `instructions` for OpenAI Responses-style payloads,
  - otherwise prepended system message in `messages`.

Conversation history injection is separate and provider-aware (including OpenAI-compatible sanitization for malformed tool-call replay edge cases).

---

## 5) My first contribution

Commit: `6e1fb27`

Files:

- Added `tests/llm/pipelines/test_recall_injection.py`
- Updated `tests/test_utils.py`

What I added:

- unit tests for `format_recalled_fact_lines(...)`
- unit tests for `format_recalled_summary_lines(...)`
- unit tests for `format_date_created(...)`

Tested behaviors:

- accepts mixed fact input types (string/dict/object),
- skips empty or invalid content safely,
- formats timestamps consistently in output text,
- deduplicates repeated summaries,
- preserves expected ordering (primary summaries first),
- handles `None` and ISO-`Z` date strings correctly.

---

## 6) Why these tests matter

These tests protect user-visible prompt quality at a high-leverage point: recall-context rendering.

They prevent regressions such as:

- empty/noisy bullets being injected into prompts,
- duplicate summaries increasing token usage,
- unstable ordering causing inconsistent outputs,
- timestamp formatting drift across refactors,
- type-handling bugs when recall payloads vary by source/path.

Impact: more stable prompt injection, lower context noise, and more predictable model behavior.

---

## 7) Likely interview / hackathon questions (concise answers)

### Q1: Why monkey-patch provider SDK methods?

So existing app calls remain unchanged while Memori transparently adds memory behavior (injection + persistence) around each call.

### Q2: What is attribution in Memori?

`entity_id` + optional `process_id` to scope memory correctly to a user/agent/workflow and avoid cross-entity contamination.

### Q3: Difference between recall injection and conversation injection?

- Recall injection: distilled relevant facts/summaries.
- Conversation injection: prior raw chat history for continuity.

### Q4: Why relevance threshold filtering?

To avoid low-signal memory entering prompts, reducing token waste and hallucination-prone context.

### Q5: Where is data persisted?

After invocation in `post_invoke`, then through `MemoryManager` (cloud API path or local writer transaction path).

### Q6: Why are these tests high value for a first contribution?

They lock down output contracts at a user-facing boundary (prompt context text), where small formatting bugs can cause large behavior changes.

---

## 8) 30-second explanation script

"Memori is a memory middleware for LLM apps. In this Python SDK, it wraps provider calls, injects relevant recalled context and conversation history before each request, then stores the new turn and runs augmentation after response. My first contribution added tests around recall-context formatting and date normalization, which protects prompt quality and prevents regressions like duplicate summaries, empty facts, and inconsistent timestamps."
