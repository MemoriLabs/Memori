---
name: memori
description: >
  Memori long-term memory — recall, summarize, run Advanced Augmentation, and compact memories across sessions.
  TRIGGER when: user asks about past sessions, prior decisions, preferences, constraints,
  or anything you should already know. Also trigger at session start for a daily brief,
  when resuming after context compaction, or when the user asks to send feedback about memory.
  DO NOT TRIGGER when: the task is fully self-contained and needs no historical context.
argument-hint: '<command> [--flags ...]'
allowed-tools: Bash
---

# Memori Skill

Long-term memory for agents. Runs Advanced Augmentation for conversation turns and recalls structured memories across sessions.

## Setup

Required env vars:
- `MEMORI_API_KEY` — your Memori API key
- `MEMORI_ENTITY_ID` — the entity (user/agent) to scope memory to
- `MEMORI_PROJECT_ID` — default project (optional, can be overridden per-call)

## Commands

### `recall` — search memories
```bash
bun .claude/skills/memori/index.ts recall \
  [--projectId ID] \
  [--sessionId ID] \
  [--dateStart 2025-01-01T00:00:00Z] \
  [--dateEnd 2025-12-31T23:59:59Z] \
  [--source constraint|decision|execution|fact|insight|instruction|status|strategy|task] \
  [--signal discovery|commit|failure|inference|pattern|result|update|verification]
```

`source` and `signal` must always be provided together. Valid pairs:
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

### `recall.summary` — summarized view of recent memory
```bash
bun .claude/skills/memori/index.ts recall.summary \
  [--projectId ID] \
  [--sessionId ID] \
  [--dateStart ISO] \
  [--dateEnd ISO]
```
Defaults to last 24 hours when no date range is given.

### `advanced-augmentation` — save and augment a conversation turn
```bash
bun .claude/skills/memori/index.ts advanced-augmentation \
  --sessionId ID \
  --userMessage "..." \
  --assistantMessage "..." \
  [--projectId ID] \
  [--model openai/gpt-4o]
```
Writes the turn to the conversation DB, then calls memory extraction and waits for the augmentation response. Call this at the end of each agent turn.

### `compaction` — restore working state after context reset
```bash
bun .claude/skills/memori/index.ts compaction \
  --projectId ID \
  [--sessionId ID] \
  [--numMessages 5]
```
Returns a structured snapshot: active tasks, open loops, standing orders, environment, and last/next action. **Costs 100 memory credits — do not call on every turn.**

### `feedback` — send feedback to the Memori team
```bash
bun .claude/skills/memori/index.ts feedback --content "recall missed a pricing constraint"
```

## Behavior guidelines

- At session start, call `recall.summary` for a brief of recent activity
- Use `recall` for targeted retrieval — prefer narrow queries, add filters only when needed
- Call `advanced-augmentation` at the end of each turn to persist and augment the conversation
- After context compaction, call `compaction` to restore working state
- Do not recall on every turn — only when prior context is actually needed
- Do not invent memory; treat recent user instructions as higher priority than recalled memory
