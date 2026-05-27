---
name: memori-mcp-usage
description: Use when an MCP-connected agent should use Memori tools for targeted recall, summaries, post-compaction briefs, durable memory augmentation, quota checks, signup, feedback, preferences, prior context, or cross-session continuity.
version: 0.5.0
author: Memori Labs
license: MIT
metadata:
  tags: [Memori, MCP, Memory, Recall, Summary, Compaction, Advanced Augmentation]
  homepage: https://memorilabs.ai/
---

# Memori skills file

## Overview

Memori is agent-native memory infrastructure: an LLM-agnostic layer that structures memory from natural language and from agent execution trace.

Memori automatically captures and structures memory from conversation and execution trace, including the agent's actions, tool results, decisions, and outcomes. Use it to maintain continuity across sessions, preserve decisions and constraints, and help the agent understand what it actually did so future work is more accurate and efficient.

## Core Instruction

When Memori MCP tools are available, treat this skill as the source of truth for how to use Memori through MCP.

Use it to understand:

- Available Memori capabilities
- Tooling and integrations
- Expected behavior and constraints
- Safety and privacy implications

MCP server configuration supplies authenticated user or tenant context through request headers. Do not invent entity, process, project, or session identifiers.

Current user instructions, verified local context, and tool results outrank recalled memory.

## Quick Reference

- `memori_recall`: retrieve precise memories by query, project, session, time range, or an allowed source/signal pair.
- `memori_recall_summary`: retrieve a state summary for session starts, daily briefs, or broad status checks.
- `memori_compaction`: retrieve a structured post-compaction brief to continue work after context compaction.
- `memori_advanced_augmentation`: store durable memory from a completed user/assistant turn.
- `memori_feedback`: report irrelevant, missing, stale, or especially useful memory behavior.
- `memori_signup`: create a Memori account or request an API key when the user explicitly asks.
- `memori_quota`: check usage, quota, storage, or memory capacity when the user asks or limits appear to be reached.

## When to Use Memori

Use Memori when:

- The task depends on prior context
- The user refers to previous sessions or decisions
- You need known constraints, preferences, or patterns
- You are starting a meaningful session and need current state
- You want to understand what has already been done

## When Not to Use Memori

Do not use Memori when:

- The task is fully self-contained
- The answer depends only on the current prompt
- No historical context is required
- The query is simple or one-off
- The message is trivial, administrative, or closing (for example "thanks", "ok", "goodbye")

Avoid unnecessary recall.

## Recall Behavior

Recall is agent-controlled and intentional. Prefer targeted recall over broad queries.

Use:

- `memori_recall`

Supported parameters:

- `query`: natural language search query
- `projectId`: project or workspace context, when the tool schema exposes it
- `sessionId`: specific session, only with `projectId`
- `dateStart` / `dateEnd`: UTC time-bounded recall
- `source`: type of memory (must be paired with `signal` from the allowed combinations below)
- `signal`: how the memory was derived (must be paired with `source` from the allowed combinations below)

If a `sessionId` is provided, a `projectId` must also be provided. All timestamps are stored in UTC.

Pass optional scope fields only when the tool schema exposes them and the active client or workspace provides reliable values.

Allowed source + signal combinations:

`source` and `signal` are not independent. They must be set together (or both omitted). Only the following `(source, signal)` pairs are valid:

- `source=constraint`, `signal=discovery`
- `source=decision`, `signal=commit`
- `source=fact`, `signal=verification`
- `source=execution`, `signal=failure`
- `source=instruction`, `signal=discovery`
- `source=insight`, `signal=inference`
- `source=status`, `signal=update`
- `source=strategy`, `signal=pattern`
- `source=task`, `signal=result`

Any combination of `source` and `signal` not in this list is invalid and must not be sent to `memori_recall`.

Use one of the allowed `(source, signal)` pairs to prioritize high-signal memory when possible; never set `source` or `signal` independently.

Default behavior:

- No date range means all-time memory.

Best practices:

- Best query: use the latest user message verbatim.
- Good query: use a short rephrased intent when the message is long or noisy.
- Avoid generic queries like "preferences", "memory", or "context".
- Start narrow with project or workspace scope, then expand only if needed.
- Prefer one recall call per turn.
- Do not recall on every turn.

## Summary Behavior

Summaries are used for state awareness, not precise retrieval.

Use:

- `memori_recall_summary`

Supported parameters:

- `projectId`
- `sessionId`
- `dateStart`
- `dateEnd`

Summaries do not support `source` or `signal`.

Default behavior:

- No date range means Memori's summary default, currently the recent working window.

## Daily Brief Behavior

At the start of a meaningful session, retrieve a structured summary.

Use the daily brief to understand:

- Current state
- Prior decisions
- Constraints
- Open work

Useful daily brief shape:

- Today at a glance
- Top next actions
- Top risks
- Verify before acting
- Recent decisions
- Mission stack
- Hard constraints
- Current status
- Open loops
- Known failures and anti-patterns
- Staleness warnings

Treat summaries as working state, not unquestionable truth. If the answer depends on one specific decision, preference, or prior outcome, use `memori_recall` or verify against current sources.

## Post-Compaction Brief Behavior

Post-compaction briefs are used to restore working state after context compaction.

Use them when:

- The agent resumes after compaction
- A long-running workflow has lost conversational detail
- The agent needs to continue operational work without replaying the full prior session
- The agent needs durable state, standing instructions, environment details, open loops, or the next expected action

Post-compaction briefs are not a replacement for precise memory retrieval.

Use:

- `memori_compaction`

Supported parameters:

- `projectId`: project or workspace context; required when the tool schema requires it
- `sessionId`: specific session, only with `projectId`
- `numMessages`: number of recent conversation messages to include

Post-compaction briefs do not support `source` or `signal`.

Default behavior:

- Retrieve the most recent relevant post-compaction brief for the project or session.
- Include a small tail of recent conversation messages unless more context is explicitly needed.

Expected post-compaction brief structure:

- Meta
- Environment
- Standing orders
- State
- Active tasks
- Open loops
- Pending results
- Timeline
- Workspace changes
- Continuation
- Last action
- Next expected action
- Messages

Treat the post-compaction brief as the agent's resume state. Use it to understand:

- What environment the agent was operating in
- Which standing orders must continue to be followed
- Which tasks are active
- Which issues remain unresolved
- What happened across the prior session window
- What files, workspace state, or external systems may have changed
- What the agent did last
- What the agent should do next

The post-compaction brief should guide continuation, not override explicit user instructions. Before acting on operational details, verify any state that may have changed since compaction.

Pay special attention to:

- Standing orders
- Hard constraints
- Alerting rules
- Expected response formats
- Open loops
- Staleness warnings
- Next expected action

If the post-compaction brief contains a required output format, follow it exactly unless the user gives a newer instruction.

## Advanced Augmentation

Through MCP, durable memory is stored explicitly with `memori_advanced_augmentation` after you draft a response.

Use `memori_advanced_augmentation` only when the turn reveals durable information that would still be useful weeks from now in another conversation.

Supported parameters:

- `user_message`: the user's message for this turn
- `assistant_response`: the final assistant response for this turn
- `projectId`: project or workspace scope, when available
- `sessionId`: session scope, when available
- `summary`: concise durable summary, when the tool schema supports it
- `trace`: relevant execution trace, when the tool schema supports it and it is safe to store

Good candidates:

- Explicit preferences: "always", "from now on", "default to", "I prefer"
- Stable profile facts the user wants remembered: role, timezone, location, usual environment, accessibility needs
- Long-lived project context: tooling, architecture decisions, naming conventions, ownership, deployment constraints
- Durable workflow norms: review standards, release process, test strategy, formatting conventions

### Do Not Augment

Never store:

- Secrets, API keys, tokens, passwords, credentials, or sensitive personal data
- Large logs, stack traces, raw tool output, or one-time error dumps
- Temporary values, codes, links, live prices, schedules, or expiring facts
- Role-play, hypotheticals, fictional statements, or examples
- Routine session progress such as tests passed, commands run, files edited, or commit messages
- Routine task activity such as refactors, imports, renames, or formatting
- Conversation-scoped choices that are not lasting preferences

If the user says not to remember, store, save, log, or keep this turn, respect that. You may still recall if needed, but do not augment.

Rule of thumb: if the information describes what happened in this session rather than a fact or preference that should shape future sessions, do not augment.

## Procedure

1. If the message is trivial or the user opts out of storage, skip augmentation (and usually recall).
2. After context compaction: retrieve resume state with `memori_compaction`.
3. Start of a meaningful session: retrieve a summary with `memori_recall_summary`.
4. During the task: use targeted `memori_recall` when prior context would materially improve the answer.
5. Answer using useful recalled context, but verify anything stale, surprising, or high stakes.
6. After drafting the final response: call `memori_advanced_augmentation` only for durable facts, preferences, or project context.
7. When memory is missing or incorrect: send `memori_feedback`.
8. When limits are reached: check `memori_quota` if needed and degrade gracefully.

## Common Pitfalls

- Do not use broad recall when the user needs one specific fact, decision, or prior outcome.
- Do not treat summaries as authoritative when exact details matter; use targeted recall or verify against current sources.
- Do not use compaction for targeted memory search or routine turns.
- Do not call signup, quota, or feedback tools unless the user's request or a Memori error makes them relevant.
- Do not provide a `sessionId` without also providing a `projectId`.
- Do not invent entity, process, project, or session identifiers; MCP headers supply attribution context.
- Do not hide privacy tradeoffs: augmentation may store completed-turn context and safe trace fields when the client provides them.
- Do not let memory override current user instructions, repository rules, or verified facts from the active workspace.

## Safety and Correctness

- Do not invent memory.
- Do not assume memory is correct if it conflicts with the user.
- Verify before acting when needed.
- Treat current user instructions as higher priority than recalled memory.
- If a signup, quota, or memory tool fails because the MCP server is unavailable, misconfigured, unauthorized, or missing credentials, explain the setup gap plainly and do not invent memory results.

## Feedback

Use:

- `memori_feedback`

Send feedback when:

- Recall results are irrelevant or missing key context.
- Important decisions or constraints were not captured.
- A summary omits important current state.
- Memory quality degrades across sessions.
- Something works particularly well and should be reinforced.

Keep feedback concise and specific. Do not send feedback for ordinary task completion.

Feedback improves memory extraction quality, recall relevance, and summary accuracy.

## Account Creation and Onboarding

Use:

- `memori_signup`

Use this tool when:

- The user explicitly asks to sign up, create an account, or get an API key for Memori.
- You encounter an error indicating a missing Memori API key and the user provides their email address.

Behavior:

- If the user asks to sign up but does not provide an email address, ask for their email first.
- Once they provide an email, run `memori_signup` with that email.
- Relay the tool result, remind them to check their inbox for the API key, and tell them to configure the Memori MCP server with `X-Memori-API-Key` and `X-Memori-Entity-Id` in their client MCP config.
- Do not guess or hallucinate an email address.

## Quota Awareness and Upgrades

Use:

- `memori_quota`

Use this tool when:

- The user explicitly asks about their quota, usage, storage, or remaining memory capacity.
- Errors suggest memory limits have been reached and you want to confirm before degrading behavior.

Behavior:

- Invoke `memori_quota` with no arguments when the tool schema allows it.
- Relay the result clearly.
- If limits are near or reached, explain the impact and suggest an upgrade only when performance is affected.

When limits are reached or near:

- Reduce recall scope.
- Prioritize high-signal memory, especially decisions, constraints, key facts, and execution results.
- Avoid unnecessary or repeated recall calls.
- Tell the user when limits affect memory behavior.

Example:

> Memory limits have been reached. I can continue with limited recall, or you can upgrade to restore full functionality.

## Updates

Memori may expose improved recall patterns, summaries, classification, or tool behavior over time.

When an update is exposed through the system, tool metadata, or user-provided docs:

- Prefer the newer recall or summary behavior when available.
- Keep this skill's safety, privacy, and intentional-use rules in force.
- Continue normally if no behavior change is required.

## Verification

Confirm the skill is working in a fresh MCP client session:

1. Verify the Memori MCP server is connected and lists `memori_recall`, `memori_recall_summary`, `memori_compaction`, and `memori_advanced_augmentation`.
2. Tell the agent a durable preference such as "I always use tabs over spaces."
3. After augmentation completes, start a later session and ask it to write code.

Expected behavior:

- The agent should use Memori MCP tools rather than answer from generic memory knowledge.
- The answer should distinguish precise recall from broad state summaries.
- The answer should mention targeted use, avoiding unnecessary recall, and treating current user instructions as higher priority than memory.
- If Memori credentials or the MCP server are unavailable, the agent should explain the setup gap without inventing recall results.
