# Memori Claude Code Skill

## Overview

Memori is agent-native memory infrastructure: an LLM-agnostic layer that stores
conversation turns, structures durable memory, and retrieves prior context
across sessions.

This OpenRouter integration exposes Memori through a local Bun CLI. Claude Code
must use this CLI for memory operations instead of answering from context alone
when the user asks about memory, prior sessions, saved preferences, decisions,
constraints, or feedback about memory quality.

## Core Instruction

Treat this file as the source of truth for using Memori from Claude Code in this
integration.

Use Memori to understand:

- Relevant prior context
- Past decisions and constraints
- User preferences and standing instructions
- Current project/session state after context loss
- Durable memory that should be available in future sessions

Current user instructions, verified local workspace context, and fresh tool
results outrank recalled memory.

Do not invent entity, project, or session identifiers. Use values supplied by
the active Claude Code session, the environment, or the user.

## Invocation

Run all commands from the OpenRouter integration root:

```bash
bun .claude/skills/memori/index.ts <command> [--flags ...]
```

The script requires:

- `MEMORI_API_KEY`: Memori bearer token.
- `MEMORI_ENTITY_ID`: entity to scope memory to.
- `MEMORI_PROJECT_ID`: default project ID. Optional for most commands when
  `--projectId` is supplied, but required for `compaction` unless passed as a
  flag.

## Quick Reference

- `recall`: retrieve stored memories for the current entity, optionally scoped
  by project, session, time range, or source/signal pair.
- `recall.summary`: retrieve a recent working summary for session starts,
  daily briefs, or broad state checks.
- `advanced-augmentation`: store a completed user/assistant turn and wait for
  Advanced Augmentation to process it.
- `compaction`: retrieve a structured post-compaction brief to resume work
  after context loss.
- `feedback`: send memory quality feedback to the Memori team.

This local CLI does not currently expose `signup`, `quota`, or natural-language
`query` parameters. Do not document or call unsupported commands.

## When to Use Memori

Use Memori when:

- The user asks what you remember, what you know about them, or what happened
  in past sessions.
- The task depends on prior decisions, preferences, constraints, or project
  history.
- You are starting a meaningful session and need recent state.
- You resume after compaction or lost context.
- The user asks you to save or remember something.
- You need to report missing, stale, irrelevant, or useful memory behavior.

## When Not to Use Recall

Avoid recall when:

- The task is fully self-contained.
- The answer depends only on the current prompt and visible workspace.
- The message is trivial or administrative, such as "thanks", "ok", or
  "goodbye".
- You already have enough current, verified context.

Avoid unnecessary recall. Prefer one targeted recall call over broad repeated
recalls.

## Session Start Behavior

At the start of every meaningful Claude Code session, request a recent summary:

```bash
bun .claude/skills/memori/index.ts recall.summary
```

Use the summary to understand:

- Current state
- Prior decisions
- Constraints
- Open work
- Known failures or risks
- Next likely actions

Treat summaries as working state, not unquestionable truth. If exact details
matter, use `recall` or verify against the workspace.

## Recall Behavior

Recall is agent-controlled and intentional. Use it when prior context would
materially improve the answer.

```bash
bun .claude/skills/memori/index.ts recall \
  [--projectId <projectId>] \
  [--sessionId <sessionId>] \
  [--dateStart <ISO datetime>] \
  [--dateEnd <ISO datetime>] \
  [--source <source>] \
  [--signal <signal>]
```

Supported flags:

- `--projectId`: scope recall to a project. Defaults to `MEMORI_PROJECT_ID`.
- `--sessionId`: scope recall to a session.
- `--dateStart`: UTC lower bound.
- `--dateEnd`: UTC upper bound.
- `--source`: memory type. Must be paired with `--signal`.
- `--signal`: derivation signal. Must be paired with `--source`.

Important constraints:

- `MEMORI_ENTITY_ID` is always used as the entity scope.
- This CLI does not accept a natural-language query string.
- `source` and `signal` must be omitted together or supplied together.
- Never send an invalid source/signal pair.

Allowed source/signal pairs:

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

Best practices:

- Start with project or session scope when reliable values are available.
- Use date bounds when the user asks about a known time window.
- Use source/signal pairs to prioritize high-signal memory.
- Do not treat recalled memory as authoritative when it conflicts with the
  user, repository rules, or current files.

## Summary Behavior

Use `recall.summary` for broad state awareness:

```bash
bun .claude/skills/memori/index.ts recall.summary \
  [--projectId <projectId>] \
  [--sessionId <sessionId>] \
  [--dateStart <ISO datetime>] \
  [--dateEnd <ISO datetime>]
```

Summaries do not support `source` or `signal`.

Use summaries for:

- Session starts
- Daily briefs
- Broad status checks
- Reorienting before a larger task

Do not use summaries as a replacement for exact recall when the user asks for a
specific prior fact, decision, or outcome.

## Advanced Augmentation

Advanced Augmentation stores the completed turn and asks Memori to extract
durable memory from it.

Use `advanced-augmentation` after drafting the final assistant response and
before handing off the turn:

```bash
bun .claude/skills/memori/index.ts advanced-augmentation \
  --sessionId <sessionId> \
  --userMessage "<latest user message>" \
  --assistantMessage "<final assistant response>" \
  [--projectId <projectId>] \
  [--model <model name>]
```

Required flags:

- `--sessionId`
- `--userMessage`
- `--assistantMessage`

Behavior:

- Writes the turn to `/agent/conversation/turn`.
- Calls the collector's augmentation endpoint.
- Waits for the augmentation response.
- Returns `{"success": true, "augmentation": true}` after both calls succeed.

Use Advanced Augmentation at the end of every non-trivial assistant turn so
conversation continuity and memory extraction remain consistent.

Also use it immediately when the user explicitly asks Claude to save or
remember something.

Do not augment if:

- The user explicitly says not to remember, store, save, log, or keep the turn.
- The turn contains secrets, API keys, tokens, passwords, or credentials.
- The turn is only a trivial acknowledgement or closing.
- The content is purely hypothetical, fictional, or an example that should not
  become memory.

When the turn contains both useful context and sensitive data, omit sensitive
data from the assistant response and do not include secrets in the stored
message content.

## Post-Compaction Brief Behavior

Use `compaction` when Claude Code resumes after compaction or when the agent
needs structured working state after losing context:

```bash
bun .claude/skills/memori/index.ts compaction \
  [--projectId <projectId>] \
  [--sessionId <sessionId>] \
  [--numMessages <count>]
```

Requirements:

- `--projectId` or `MEMORI_PROJECT_ID` must be present.

Use the compaction brief to understand:

- Standing orders
- Environment
- Active tasks
- Open loops
- Pending results
- Workspace changes
- Last action
- Next expected action
- Recent messages

Do not call compaction on every turn. It is for context restoration, not
routine recall.

## Feedback

Use `feedback` when the user asks to report memory quality issues or when
Memori behavior should be improved:

```bash
bun .claude/skills/memori/index.ts feedback --content "<feedback>"
```

Send feedback when:

- Recall results are irrelevant.
- Important decisions or constraints are missing.
- A summary omits important current state.
- Memory quality degrades across sessions.
- Something works particularly well and should be reinforced.

Keep feedback concise and specific. Do not send feedback for ordinary task
completion.

## Procedure

1. Start a meaningful session with `recall.summary`.
2. If resuming after compaction, use `compaction`.
3. During the task, use `recall` only when prior context materially improves
   the answer.
4. Answer using recalled context only when it is relevant, and verify stale or
   high-stakes information.
5. Before finishing a non-trivial assistant turn, run `advanced-augmentation`
   with the latest user message and final assistant response.
6. If memory is missing, wrong, stale, or especially useful, use `feedback` when
   appropriate.

## Quoting and Shell Safety

Quote message arguments carefully. User and assistant messages may contain
quotes, dollar signs, backticks, newlines, or shell metacharacters.

Preferred pattern for complex messages:

```bash
bun .claude/skills/memori/index.ts advanced-augmentation \
  --sessionId "$SESSION_ID" \
  --userMessage "$USER_MESSAGE" \
  --assistantMessage "$ASSISTANT_MESSAGE"
```

Do not paste secrets into shell arguments. If a turn contains sensitive data,
do not augment it.

## Common Pitfalls

- Do not answer memory questions from general model context.
- Do not call unsupported commands such as `signup` or `quota` from this CLI.
- Do not send `source` without `signal`, or `signal` without `source`.
- Do not use invalid source/signal pairs.
- Do not call `compaction` for routine recall.
- Do not let memory override explicit current user instructions.
- Do not silently ignore augmentation failures. The CLI should surface errors.
- Do not assume the nested skill file is loaded if Claude Code is launched from
  a different working directory; keep root and nested `CLAUDE.md` instructions
  aligned.

## Safety and Correctness

- Do not invent memory.
- Treat current user instructions as higher priority than recalled memory.
- Verify before acting when memory conflicts with current files or live tool
  results.
- Explain setup gaps plainly if credentials are missing or the Memori API is
  unavailable.
- Do not store secrets, credentials, or sensitive personal data.
- Use the least memory scope needed for the task.

## Verification

To verify the Claude Code skill path:

1. Confirm the environment has `MEMORI_API_KEY` and `MEMORI_ENTITY_ID`.
2. Run `recall.summary` from the OpenRouter integration root.
3. Run `advanced-augmentation` with a test session and harmless test messages.
4. Run `recall` later and confirm durable context can be retrieved.

Expected behavior:

- Claude uses the local Bun CLI for memory operations.
- `advanced-augmentation` waits for the collector response.
- Recall and summaries are used intentionally, not on every turn.
- Current user instructions and workspace facts outrank recalled memory.
