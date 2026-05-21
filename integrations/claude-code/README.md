# Memori — Claude Code Integration

Ambient long-term memory for [Claude Code](https://claude.com/claude-code) via a local Bash-invoked skill. Claude Code calls a small TypeScript CLI (`bun`) that talks to Memori Cloud to recall prior context and record each turn.

This folder is a reference implementation. The skill is just two files (`SKILL.md` + `index.ts`) — drop them into any `.claude/skills/memori/` directory and it works the same way.

## Skill layout

```
<your-project>/
└── .claude/
    └── skills/
        └── memori/
            ├── SKILL.md        # skill definition + procedure
            └── index.ts        # CLI wrapper around Memori Cloud
```

`.claude/` can live at the project root or globally at `~/.claude/`. Claude Code discovers skills automatically from either location.

## Prerequisites

- [Bun](https://bun.sh) (`curl -fsSL https://bun.sh/install | bash`)
- [Claude Code](https://docs.anthropic.com/en/docs/claude-code/setup)
- A Memori Cloud account with an API key and entity ID

## Install

1. Copy `.claude/skills/memori/` (both `SKILL.md` and `index.ts`) into your target project's `.claude/skills/` directory, or into `~/.claude/skills/` to make it available everywhere.
2. Provide credentials (see below).

## Configuration

Set these environment variables before launching Claude Code:

```bash
export MEMORI_API_KEY=your_memori_api_key
export MEMORI_ENTITY_ID=your_entity_id
export MEMORI_PROJECT_ID=your_default_project_id   # optional
```

Put them in your shell profile, a tool like `direnv`, or a project-local `.env` file. If you use `.env`, invoke the CLI with `bun --env-file=.env ...` (Claude Code itself does not auto-load `.env`).

| Variable | Required | Purpose |
|---|---|---|
| `MEMORI_API_KEY` | yes | Authenticates to Memori Cloud |
| `MEMORI_ENTITY_ID` | yes | Per-user / per-agent memory namespace |
| `MEMORI_PROJECT_ID` | no | Default project scope; can be overridden per call with `--projectId` |

## How the skill is used

`SKILL.md` instructs Claude Code to:

1. Run `recall` before drafting any substantive response or external lookup.
2. Answer the user's actual request.
3. Run `advanced-augmentation` after the final response to record the turn.

You do not invoke the skill manually — it is ambient. Claude Code triggers it automatically based on the directives in `SKILL.md`.

## CLI reference

```bash
bun .claude/skills/memori/index.ts <command> [--flag value ...]
```

(Adjust the path if you installed the skill globally — e.g. `bun ~/.claude/skills/memori/index.ts ...`.)

| Command | Purpose |
|---|---|
| `recall` | Targeted retrieval. Use with `--source` and `--signal` (see source/signal table in `SKILL.md`). |
| `recall.summary` | Broad session summary / orientation. |
| `advanced-augmentation` | Record a user/assistant turn. Required: `--sessionId`, `--userMessage`, `--assistantMessage`. Optional: `--trace`, `--summary`, `--model`, `--projectId`, `--processId`. |
| `compaction` | Replace lost context after Claude Code compaction. Requires `--projectId` (or env). |
| `feedback` | Send free-form feedback. `--content "..."` |
| `quota` | Show remaining quota for the API key. |
| `signup` | Create a new account. `--email user@example.com` |

Flags accept both `--flag value` and `--flag=value`. On success, commands print JSON to stdout and exit 0; on failure they print to stderr and exit 1.

### Trace shape for `advanced-augmentation`

```json
{
  "tools": [
    { "name": "ReadFile", "args": { "path": "src/app.ts" }, "result": "Read app entrypoint" }
  ]
}
```

Each entry requires `name` (string), `args` (object), and `result` (any — key must be present). Never include secrets, credentials, or large raw logs in trace fields.

## Troubleshooting

- **`MEMORI_API_KEY is required`** — credentials not in the environment. Export the variables in your shell or invoke the CLI with `bun --env-file=.env ...`.
- **`--sessionId requires --projectId or MEMORI_PROJECT_ID`** — set `MEMORI_PROJECT_ID` or pass `--projectId`.
- **Claude prompts on every Bash call** — confirm `Bash(bun *)` and `Skill(memori)` are in your `settings.local.json` / `settings.json`.
- **Skill never fires** — confirm Claude Code can see it: `claude` → `/skills` should list `memori`.
- **No memories returned** — first session against a new entity has nothing to recall. Run `test.sh` to seed.

## Reference

- Skill behavior and source/signal taxonomy: [`SKILL.md`](SKILL.md)
- CLI source: [`index.ts`](index.ts)
