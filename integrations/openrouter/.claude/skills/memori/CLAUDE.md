# Memori Memory Skill

Use the local Memori CLI for long-term memory operations. Do not answer
memory-related requests from context alone.

All commands run through:

```bash
bun .claude/skills/memori/index.ts <command> [--flags ...]
```

## Environment

The script requires:

- `MEMORI_API_KEY` - Memori bearer token.
- `MEMORI_ENTITY_ID` - Entity to scope memory to.
- `MEMORI_PROJECT_ID` - Default project ID. Optional for most commands when
  `--projectId` is provided, but required for `compaction` unless passed as a
  flag.

## Session Start

At the start of every session, request the recent memory summary:

```bash
bun .claude/skills/memori/index.ts recall.summary
```

## Available Methods

### `recall`

Search stored memories for the current entity.

Use when the user asks what you remember, what you know about them, what
happened in past sessions, or when prior decisions/preferences/constraints are
needed for the current task.

```bash
bun .claude/skills/memori/index.ts recall \
  [--projectId <projectId>] \
  [--sessionId <sessionId>] \
  [--dateStart <ISO datetime>] \
  [--dateEnd <ISO datetime>] \
  [--source <source>] \
  [--signal <signal>]
```

Notes:

- `entity_id` is always read from `MEMORI_ENTITY_ID`.
- `project_id` uses `--projectId` when provided, otherwise
  `MEMORI_PROJECT_ID`.
- `source` and `signal` must be provided together.
- Valid `source` and `signal` pairs are:

| source | signal |
| --- | --- |
| `constraint` | `discovery` |
| `decision` | `commit` |
| `execution` | `failure` |
| `fact` | `verification` |
| `insight` | `inference` |
| `instruction` | `discovery` |
| `status` | `update` |
| `strategy` | `pattern` |
| `task` | `result` |

### `recall.summary`

Fetch a summarized view of recent memory.

Use at session start and when a compact overview is better than raw memory
records.

```bash
bun .claude/skills/memori/index.ts recall.summary \
  [--projectId <projectId>] \
  [--sessionId <sessionId>] \
  [--dateStart <ISO datetime>] \
  [--dateEnd <ISO datetime>]
```

When no date range is supplied, the API default is used.

### `advanced-augmentation`

Persist a user/assistant turn and trigger asynchronous memory extraction.

Use at the end of every assistant turn so Memori can store the conversation and
run augmentation. Also use it immediately when the user explicitly asks you to
save or remember something.

```bash
bun .claude/skills/memori/index.ts advanced-augmentation \
  --sessionId <sessionId> \
  --userMessage "<what the user said>" \
  --assistantMessage "<what the assistant said>" \
  [--projectId <projectId>] \
  [--model <model name>]
```

Required flags:

- `--sessionId`
- `--userMessage`
- `--assistantMessage`

Behavior:

- Writes the turn to `/agent/conversation/turn`.
- Calls memory augmentation through the collector and waits for the response.
- Returns `{"success": true, "augmentation": true}` once both calls succeed.

### `compaction`

Restore working state after context compaction.

Use only after a context reset/compaction or when you explicitly need the
structured working snapshot. This call may consume more memory credits than
recall.

```bash
bun .claude/skills/memori/index.ts compaction \
  [--projectId <projectId>] \
  [--sessionId <sessionId>] \
  [--numMessages <count>]
```

Requirements:

- `--projectId` or `MEMORI_PROJECT_ID` must be present.

Returns active tasks, open loops, standing orders, environment, and recent
messages.

### `feedback`

Send memory-related feedback to the Memori team.

Use when the user asks to send feedback about recall quality, missing memories,
incorrect memories, or the memory feature itself.

```bash
bun .claude/skills/memori/index.ts feedback --content "<feedback>"
```

Required flags:

- `--content`

## Behavior Guidelines

- Use `recall.summary` once at session start.
- Use `recall` for targeted retrieval when historical context is actually
  needed.
- Always use `advanced-augmentation` at the end of every assistant turn with the
  latest user message and assistant response so augmentation runs consistently.
- Do not invent memory; treat user instructions in the current session as the
  highest priority.
- Do not call `compaction` on every turn.
- Include `--projectId` or `--sessionId` when narrowing scope would improve
  precision.
- When using `advanced-augmentation`, quote user and assistant messages
  carefully so shell parsing does not change their content.
