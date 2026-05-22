---
name: memori
description: >
  Ambient Memori long-term memory for Claude Code via local Bash. MUST TRIGGER
  on essentially every non-trivial user turn: run recall before drafting any
  substantive response, whether or not the user mentioned memory, a prior
  session, or past work. The default is to check memory first; do not wait to
  be asked. Always fire before external lookups (WebSearch/WebFetch) and
  before answering any question about preferences, prior decisions,
  constraints, project history, status, past work, or any non-trivial
  coding/research task. Skip only for trivial acknowledgements/closings,
  purely self-contained turns where prior context cannot possibly help, or
  when the user explicitly opts out. Use Advanced Augmentation after drafting
  the final response for every non-trivial turn, as the last memory step. Use
  recall.summary only for broad session summaries/orientation, compaction
  after context loss, feedback for memory quality, quota for limits, and
  signup only when explicitly requested or when you need a Memori API key.
argument-hint: '<command> [--flags ...]'
allowed-tools: Bash
---

# Memori Claude Code Skill

Thin CLI wrapper around Memori Cloud for Claude Code subprocesses. Memori is an
ambient memory enhancement, not an explicit destination tool. Do not wait for
the user to ask to "use Memori" before applying the memory lifecycle below.
Claude Code's primary job remains answering the user, editing code, debugging,
reviewing, and completing the requested work.

Run commands from the Claude Code integration root:

```bash
bun .claude/skills/memori/index.ts <command> [--flags ...]
```

Flags accept `--flag value` or `--flag=value`.

## Setup

Configure credentials in `.claude/settings.local.json` (per-project, not
checked into source control). Claude Code injects everything under `env`
into every Bash subprocess, including this skill:

```json
{
  "env": {
    "MEMORI_API_KEY": "your_memori_api_key",
    "MEMORI_ENTITY_ID": "stable_identifier_for_this_user"
  }
}
```

Required:

- `MEMORI_API_KEY` — Memori Cloud API key.
- `MEMORI_ENTITY_ID` — any stable string that namespaces memory for this
  user/agent. If it is missing on the first command, pick a sensible default
  yourself (the machine hostname, or a generated UUID), write it to `.claude/settings.local.json` so it persists,
  and continue. If no sensible default can be inferred, ask the user.

Auto-resolved (no setup required, override with flags or env if needed):

- `MEMORI_PROJECT_ID` — defaults to `basename($CLAUDE_PROJECT_DIR)`, the
  Claude Code workspace folder name. Override per call with `--projectId`,
  or pin it by adding `MEMORI_PROJECT_ID` to the `env` block.
- `MEMORI_SESSION_ID` — defaults to `$CLAUDE_CODE_SESSION_ID` for writes
  (`advanced-augmentation`) and for `compaction` (which by definition
  restores the current session). **Reads (`recall`, `recall.summary`) do
  not auto-apply session_id** — it is a narrowing filter on the agent
  recall API, so the default keeps retrieval project-scoped across new
  Claude Code sessions and `/clear`. Pass `--sessionId` explicitly when
  you want a single-session recall.

Optional:

- `MEMORI_PROCESS_ID` — process attribution for `advanced-augmentation`.

A colocated `.env` file next to `index.ts` is loaded as a fallback when
`settings.local.json` is not used.

## Commands

```bash
bun .claude/skills/memori/index.ts recall [--projectId ID] [--sessionId ID] [--dateStart ISO] [--dateEnd ISO] [--source SOURCE --signal SIGNAL]
bun .claude/skills/memori/index.ts recall.summary [--projectId ID] [--sessionId ID] [--dateStart ISO] [--dateEnd ISO]
bun .claude/skills/memori/index.ts advanced-augmentation --userMessage "$USER_MESSAGE" --assistantMessage "$ASSISTANT_MESSAGE" [--sessionId ID] [--projectId ID] [--model MODEL] [--summary "$SUMMARY"] [--trace "$TRACE_JSON"] [--processId ID]
bun .claude/skills/memori/index.ts compaction [--projectId ID] [--sessionId ID] [--numMessages 5]
bun .claude/skills/memori/index.ts feedback --content "feedback text"
bun .claude/skills/memori/index.ts quota
bun .claude/skills/memori/index.ts signup --email "user@example.com"
```

## Procedure

1. Run `recall` before drafting any substantive response. This is the default
   for every non-trivial user turn — including general coding tasks,
   debugging, code review, and research — not just memory-question turns. Do
   not wait for the user to mention a previous session, past work, or memory;
   the default is to check memory first.
2. Always run `recall` before any external information lookup (WebSearch,
   WebFetch, or any other tool that fetches facts from outside the repo). The
   user may have prior context on the topic stored in Memori, and memory must
   be consulted before going external. This is mandatory.
3. Use `recall.summary` only for broad summary cases: session orientation,
   daily briefs, broad status, or reorientation after a long gap. Do not use
   it for specific user queries.
4. Use `compaction` only after context compaction or lost working context.
5. Answer or complete the user's actual request. Memory should improve
   context, not replace the answer.
6. After drafting the final response for a non-trivial turn, run
   `advanced-augmentation` with that final response. This is mandatory
   ambient bookkeeping and must be the last memory step of the turn.
7. Use `feedback`, `quota`, and `signup` only when the user request or a
   Memori error makes them relevant.

Skip the memory lifecycle only for trivial acknowledgements/closings, purely
self-contained turns where prior context cannot help, or when the user
explicitly asks not to remember/store the turn.

## Recall

`recall` is targeted retrieval. The endpoint does not support `query`; never
pass `--query`. Use filters, especially source/signal.

For prompts like "what did we discuss last session about X?", run `recall`
before answering even though the topic text cannot be passed as `--query`. Use
the closest source/signal category and then answer from the returned memories.

Source and signal must be provided together:

| Use case | Flags |
|---|---|
| Fact or preference | `--source fact --signal verification` |
| Prior decision | `--source decision --signal commit` |
| Constraint or requirement | `--source constraint --signal discovery` |
| Standing instruction | `--source instruction --signal discovery` |
| Status or progress | `--source status --signal update` |
| Completed task/result | `--source task --signal result` |
| Failure or error | `--source execution --signal failure` |
| Strategy or pattern | `--source strategy --signal pattern` |
| Inferred lesson | `--source insight --signal inference` |

## Advanced Augmentation

Run after drafting the final assistant response for every non-trivial turn:

```bash
bun .claude/skills/memori/index.ts advanced-augmentation \
  --userMessage "$USER_MESSAGE" \
  --assistantMessage "$ASSISTANT_MESSAGE" \
  --trace "$TRACE_JSON"
```

`--sessionId` defaults to `$CLAUDE_CODE_SESSION_ID`; pass it explicitly only
when correlating with a session other than the current Claude Code one.

Trace is JSON shaped like `{ "tools": [...] }`. If omitted, the CLI sends
`{ "tools": [] }`. Each tool entry must include:

- `name`: string tool/function name
- `args`: object of arguments passed to the tool
- `result`: summarized tool result; the key must be present

Valid trace example:

```json
{
  "tools": [
    {
      "name": "ReadFile",
      "args": { "path": "src/app.ts" },
      "result": "Read app entrypoint"
    }
  ]
}
```

Do not include secrets, credentials, or large raw logs in trace fields.

## Output And Errors

On success, commands print JSON to stdout and exit 0. On failure, errors print
to stderr and exit 1. If memory fails, briefly report the memory issue when
relevant and continue with the user's actual request using current context.
