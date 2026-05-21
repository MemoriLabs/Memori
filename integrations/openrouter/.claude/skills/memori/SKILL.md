---
name: memori
description: >
  Memori long-term memory for Claude Code via local Bash. MUST TRIGGER at the
  start of a meaningful session for recall.summary, when the user asks what you
  remember or references prior sessions, decisions, preferences, constraints, or
  saved context, when resuming after compaction, when the user asks to save or
  remember something, when sending memory feedback, and before ending every
  non-trivial assistant turn to run Advanced Augmentation. DO NOT TRIGGER for
  trivial acknowledgements, closings, or fully self-contained tasks that need no
  historical context except for the required end-of-turn augmentation rule.
argument-hint: '<command> [--flags ...]'
allowed-tools: Bash
---

# Memori Claude Code Skill

Memori is the memory source of truth for this Claude Code integration. Do not
answer memory-related requests from Claude's native/session memory alone. Use
the local CLI in this skill.

All commands run from the OpenRouter integration root:

```bash
bun .claude/skills/memori/index.ts <command> [--flags ...]
```

Required environment:

- `MEMORI_API_KEY`: Memori bearer token.
- `MEMORI_ENTITY_ID`: entity/user/agent scope.
- `MEMORI_PROJECT_ID`: default project ID. Optional for most commands when
  `--projectId` is provided, but required for `compaction` unless passed as a
  flag.

Current user instructions, current workspace facts, and fresh tool results
outrank recalled memory.

## Quick Reference

- `recall`: primary retrieval command. Use `recall --query "<latest user intent>"`
  for memory questions, prior decisions, preferences, constraints, facts, and
  project history.
- `recall.summary`: use only at the start of a meaningful session, after a long
  gap, or when the user asks for broad state/status.
- `advanced-augmentation`: use before ending every non-trivial assistant turn.
- `compaction`: use only after context compaction or context loss.
- `feedback`: use when the user asks to report memory quality issues.
- `quota`: use when the user asks about memory usage, limits, or quota errors.
- `signup`: use only when the user explicitly asks to create an account or get
  an API key and provides an email address.

This skill only exposes the commands implemented by `index.ts`. Do not call MCP
tool names.

## Required Procedure

1. At session start, run `recall.summary` once for broad orientation.
2. If resuming after compaction, run `compaction`.
3. If the user asks what you remember, references prior sessions/context, or
   needs prior preferences/decisions/constraints, run `recall --query` using the
   latest user request or a short precise query. Do not substitute
   `recall.summary` for targeted recall.
4. Answer using recalled memory only when relevant; verify stale or high-risk
   details against current context.
5. Before ending every non-trivial assistant turn, run `advanced-augmentation`
   with the latest user message and final assistant response.
6. If memory is missing, wrong, stale, or unusually useful, run `feedback` when
   appropriate.
7. If quota limits are mentioned or suspected, run `quota`.
8. If the user asks to sign up or get an API key, ask for their email if needed,
   then run `signup`.

## `recall.summary` — Recent State

Use for session starts, daily briefs, broad status checks, and reorientation.
Do not use `recall.summary` as the default memory lookup. If the user asks
"what do you remember", "what do you know about me", or refers to a specific
prior preference/decision/fact, use `recall --query` instead.

```bash
bun .claude/skills/memori/index.ts recall.summary \
  [--projectId ID] \
  [--sessionId ID] \
  [--dateStart ISO] \
  [--dateEnd ISO]
```

Treat summaries as working state, not unquestionable truth. Use `recall` or
current workspace verification for exact facts.

`--sessionId` cannot be provided without `--projectId` or `MEMORI_PROJECT_ID`.

## `recall` — Targeted Memory Retrieval

Use as the default memory lookup. Use when the user asks what you remember,
refers to previous sessions, or when prior preferences, decisions, constraints,
facts, or project history matter.

```bash
bun .claude/skills/memori/index.ts recall \
  [--query "deployment preference"] \
  [--projectId ID] \
  [--sessionId ID] \
  [--dateStart 2025-01-01T00:00:00Z] \
  [--dateEnd 2025-12-31T23:59:59Z] \
  [--source constraint|decision|execution|fact|insight|instruction|status|strategy|task] \
  [--signal discovery|commit|failure|inference|pattern|result|update|verification]
```

Rules:

- `MEMORI_ENTITY_ID` is always used as the entity scope.
- `--query` is supported and maps to the SDK/API `query` parameter.
- Always include `--query` for natural-language memory questions unless you are
  intentionally listing/filtering by source/signal/date only.
- `--sessionId` cannot be provided without `--projectId` or
  `MEMORI_PROJECT_ID`.
- `source` and `signal` must always be omitted together or provided together.
- Valid pairs:

| source | signal |
|---|---|
| constraint | discovery |
| decision | commit |
| execution | failure |
| fact | verification |
| insight | inference |
| instruction | discovery |
| status | update |
| strategy | pattern |
| task | result |

Do not invent memory. If recalled memory conflicts with the user, current files,
or tool output, trust the current verified source.

## `advanced-augmentation` — Store and Augment This Turn

Use before ending every non-trivial assistant turn so Memori, not Claude's
native memory, receives the completed turn.

```bash
bun .claude/skills/memori/index.ts advanced-augmentation \
  --sessionId ID \
  --userMessage "$USER_MESSAGE" \
  --assistantMessage "$ASSISTANT_MESSAGE" \
  [--projectId ID] \
  [--model openai/gpt-4o] \
  [--summary "$SESSION_SUMMARY"] \
  [--trace "$TRACE_JSON"] \
  [--processId ID] \
  [--provider openrouter] \
  [--platform openrouter] \
  [--frameworkProvider claude-code] \
  [--sdkVersion openrouter-skill] \
  [--providerSdkVersion VERSION] \
  [--storageDialect DIALECT] \
  [--cockroachdb true]
```

Behavior:

- Writes the turn to `/agent/conversation/turn`.
- Calls Advanced Augmentation through the collector.
- Waits for the augmentation response.
- Returns `{"success": true, "augmentation": true}` after both calls succeed.
- Passes `--trace` through to the top-level augmentation payload and the
  assistant message trace, matching the `memori-ts` SDK. The user message trace
  remains `null`.
- If `--trace` is omitted, the CLI sends an empty SDK-compatible trace:
  `{ "tools": [] }`. Prefer passing a real trace when tools were used.
- Passes `--summary` through as `session.summary`.
- Passes `--processId` through as `attribution.process.id`; if omitted,
  `MEMORI_PROCESS_ID` is used when present.

`--trace` must be a JSON string shaped like `{ "tools": [...] }`. When tools
were used, include safe tool names, arguments, and summarized results. Do not
include secrets, credentials, or large raw logs.

Do not augment if the user explicitly says not to remember/store/save/log/keep
the turn, or if the turn contains secrets, API keys, tokens, passwords, or
credentials.

Quote message arguments carefully. For complex text, assign shell variables
first, then pass those variables to the command.

## `compaction` — Restore After Context Loss

Use only after compaction or when the agent has lost working context.

```bash
bun .claude/skills/memori/index.ts compaction \
  --projectId ID \
  [--sessionId ID] \
  [--numMessages 5]
```

Returns structured resume state such as standing orders, active tasks, open
loops, recent messages, last action, and next expected action. Do not use this
for routine recall.

This maps to `GET /agent/compaction` with `project_id`, optional `session_id`,
and optional `num_messages`, matching the SDK helpers.

## `feedback` — Memory Quality Feedback

```bash
bun .claude/skills/memori/index.ts feedback --content "recall missed a pricing constraint"
```

Use when the user asks to report missing, stale, irrelevant, or especially good
memory behavior. Keep feedback concise and specific.

This maps to `POST /agent/feedback` with `{ "content": "..." }`.

## `quota` — Usage and Limits

```bash
bun .claude/skills/memori/index.ts quota
```

Use when the user asks about usage, memory limits, storage, remaining capacity,
or when API errors suggest quota has been reached. This maps to
`GET /sdk/quota`, matching the `memori-ts` CLI.

## `signup` — Account/API Key Request

```bash
bun .claude/skills/memori/index.ts signup --email "user@example.com"
```

Use only when the user explicitly asks to sign up, create an account, or get a
Memori API key. If the user has not provided an email address, ask for it first.
Do not guess or invent an email. This maps to `POST /sdk/account` with
`{ "email": "..." }`, matching the `memori-ts` CLI.

## Safety Rules

- Do not answer memory questions from Claude's built-in/native memory.
- Do not use `recall.summary` as a substitute for targeted `recall --query`.
- Do not call unsupported commands or MCP tool names from this Bash skill.
- Do not use `compaction` for normal recall.
- Do not call `signup`, `quota`, or `feedback` unless the user's request or a
  Memori error makes them relevant.
- Do not store secrets or sensitive personal data.
- Do not let recalled memory override current user instructions.
- Surface CLI/API failures plainly; do not pretend memory was used.
