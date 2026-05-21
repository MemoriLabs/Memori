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

- `recall.summary`: use at the start of a meaningful session for recent state.
- `recall`: use for prior decisions, preferences, constraints, facts, and
  memory questions.
- `advanced-augmentation`: use before ending every non-trivial assistant turn.
- `compaction`: use only after context compaction or context loss.
- `feedback`: use when the user asks to report memory quality issues.

This skill only exposes the commands implemented by `index.ts`. Do not call
unsupported commands such as `quota`, `signup`, or MCP tool names.

## Required Procedure

1. At session start, run `recall.summary`.
2. If resuming after compaction, run `compaction`.
3. During a task, run `recall` only when prior context materially helps.
4. Answer using recalled memory only when relevant; verify stale or high-risk
   details against current context.
5. Before ending every non-trivial assistant turn, run `advanced-augmentation`
   with the latest user message and final assistant response.
6. If memory is missing, wrong, stale, or unusually useful, run `feedback` when
   appropriate.

## `recall.summary` — Recent State

Use for session starts, daily briefs, broad status checks, and reorientation.

```bash
bun .claude/skills/memori/index.ts recall.summary \
  [--projectId ID] \
  [--sessionId ID] \
  [--dateStart ISO] \
  [--dateEnd ISO]
```

Treat summaries as working state, not unquestionable truth. Use `recall` or
current workspace verification for exact facts.

## `recall` — Targeted Memory Retrieval

Use when the user asks what you remember, refers to previous sessions, or when
prior preferences, decisions, constraints, facts, or project history matter.

```bash
bun .claude/skills/memori/index.ts recall \
  [--projectId ID] \
  [--sessionId ID] \
  [--dateStart 2025-01-01T00:00:00Z] \
  [--dateEnd 2025-12-31T23:59:59Z] \
  [--source constraint|decision|execution|fact|insight|instruction|status|strategy|task] \
  [--signal discovery|commit|failure|inference|pattern|result|update|verification]
```

Rules:

- `MEMORI_ENTITY_ID` is always used as the entity scope.
- This CLI does not support a natural-language `query` parameter.
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
  [--model openai/gpt-4o]
```

Behavior:

- Writes the turn to `/agent/conversation/turn`.
- Calls Advanced Augmentation through the collector.
- Waits for the augmentation response.
- Returns `{"success": true, "augmentation": true}` after both calls succeed.

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

## `feedback` — Memory Quality Feedback

```bash
bun .claude/skills/memori/index.ts feedback --content "recall missed a pricing constraint"
```

Use when the user asks to report missing, stale, irrelevant, or especially good
memory behavior. Keep feedback concise and specific.

## Safety Rules

- Do not answer memory questions from Claude's built-in/native memory.
- Do not call unsupported commands.
- Do not use `compaction` for normal recall.
- Do not store secrets or sensitive personal data.
- Do not let recalled memory override current user instructions.
- Surface CLI/API failures plainly; do not pretend memory was used.
