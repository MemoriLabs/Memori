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
- A Memori Cloud account with an API key

## Install

1. Copy `.claude/skills/memori/` (both `SKILL.md` and `index.ts`) into your target project's `.claude/skills/` directory, or into `~/.claude/skills/` to make it available everywhere.
2. Provide credentials (see below).

## Configuration

The recommended setup is `.claude/settings.local.json` — Claude Code injects every entry under `env` into the environment of every Bash subprocess, including this skill, and Claude Code auto-gitignores this file when it creates it.

Paste the following into `.claude/settings.local.json`:

```json
{
  "env": {
    "MEMORI_API_KEY": "your_memori_api_key",
    "MEMORI_ENTITY_ID": "stable_identifier_for_this_user"
  }
}
```

That's the full required setup. `MEMORI_PROJECT_ID` auto-resolves from the current Claude Code workspace, and the current Claude Code session ID is supplied to write paths automatically. Read paths (`recall`, `recall.summary`) intentionally stay project-scoped — pass `--sessionId` explicitly when you want a single-session recall.

| Variable | Required | Purpose |
|---|---|---|
| `MEMORI_API_KEY` | yes | Authenticates to Memori Cloud. |
| `MEMORI_ENTITY_ID` | yes | Stable per-user / per-agent memory namespace. Any string. If missing, Claude Code is instructed by `SKILL.md` to pick a sensible default (slugified `git user.email`, hostname, or a UUID) and persist it to `.claude/settings.local.json`. If no default can be inferred, the agent asks the user. |
| `MEMORI_PROJECT_ID` | no | Defaults to `basename($CLAUDE_PROJECT_DIR)` (the Claude Code workspace folder name). Override per call with `--projectId`, or pin it by adding it to the `env` block. |
| `MEMORI_SESSION_ID` | no | Used by `advanced-augmentation` (write) and `compaction` (restore current session). Defaults to `$CLAUDE_CODE_SESSION_ID` (resets on `/clear`). **Not** auto-applied to `recall` / `recall.summary` — those stay project-scoped unless `--sessionId` is passed. Override per call with `--sessionId`, or pin it in the `env` block. |
| `MEMORI_PROCESS_ID` | no | Process attribution for `advanced-augmentation`. |

### Alternative: shell env or `.env` file

If you prefer not to use `settings.local.json`, the skill also reads from:

1. Real environment variables exported in your shell, shell profile, or `direnv`.
2. A `.env` file colocated next to `index.ts` (works regardless of cwd, including global installs at `~/.claude/skills/memori/`). Copy `.env.example` to `.env` and fill in the values.

Precedence (highest wins): per-call `--flag` → real env var → `.env` file. `MEMORI_PROJECT_ID` falls through to `basename($CLAUDE_PROJECT_DIR)` when none of the above are set. `MEMORI_SESSION_ID` falls through to `$CLAUDE_CODE_SESSION_ID` only for `advanced-augmentation` and `compaction`; `recall` / `recall.summary` require an explicit `--sessionId` to scope to a session.

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
| `advanced-augmentation` | Record a user/assistant turn. Required: `--userMessage`, `--assistantMessage`. Optional: `--sessionId` (defaults to `$CLAUDE_CODE_SESSION_ID`), `--trace`, `--summary`, `--model`, `--projectId`, `--processId`. |
| `compaction` | Replace lost context after Claude Code compaction. Uses the same project/session defaults. |
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

- **`MEMORI_API_KEY is required`** — credentials not in the environment. Add `MEMORI_API_KEY` to the `env` block of `.claude/settings.local.json`, export it in your shell, or place it in a `.env` file next to `index.ts`.
- **`MEMORI_ENTITY_ID is required`** — namespace not configured. Add `MEMORI_ENTITY_ID` to the `env` block of `.claude/settings.local.json` with any stable string (e.g. hostname, or a UUID). Claude Code reads `SKILL.md` and is instructed to do this automatically on first failure if it can infer a value.
- **`MEMORI_PROJECT_ID could not be resolved`** — neither `MEMORI_PROJECT_ID` nor `CLAUDE_PROJECT_DIR` was set. Pass `--projectId`, set it in the `env` block, or run from inside a Claude Code workspace.
- **Claude prompts on every Bash call** — confirm `Bash(bun *)` and `Skill(memori)` are in your `settings.local.json` / `settings.json`.
- **Skill never fires** — confirm Claude Code can see it: `claude` → `/skills` should list `memori`.

## Reference

- Skill behavior and source/signal taxonomy: [`SKILL.md`](SKILL.md)
- CLI source: [`index.ts`](index.ts)
