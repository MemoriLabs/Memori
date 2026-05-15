---
name: memori-memory
description: Use when working with Memori agent-native memory in Hermes for targeted recall, summaries, quota checks, signup, feedback, and trace-backed continuity across sessions.
version: 0.1.0
author: Memori Labs
license: MIT
metadata:
  hermes:
    tags: [Memori, Memory, Recall, Agent Trace, Long-Term Memory]
    homepage: https://memorilabs.ai/
    related_skills: [hermes-agent]
prerequisites:
  pip: [memori]
---

# Memori Memory

## Overview

Memori is agent-native memory infrastructure: an LLM-agnostic layer that structures memory from natural language and from agent execution trace.

Memori automatically captures and structures memory from conversation and execution trace, including the agent's actions, tool results, decisions, and outcomes. Use it to maintain continuity across sessions, preserve decisions and constraints, and help the agent understand what it actually did so future work is more accurate and efficient.

## Core Instruction

When this skill is explicitly loaded, treat it as the source of truth for how to use Memori tools in Hermes.

Use it to understand:

- Available Memori capabilities
- Tooling and integrations
- Expected behavior and constraints
- Safety and privacy implications

## Quick Reference

- `memori_recall`: retrieve precise memories by query, project, session, time range, or an allowed source/signal pair.
- `memori_recall_summary`: retrieve a state summary for session starts, daily briefs, or broad status checks.
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

Avoid unnecessary recall.

## Recall Behavior

Recall is agent-controlled and intentional. Prefer targeted recall over broad queries.

Use:

- `memori_recall`

Supported parameters:

- `query`: natural language search query
- `project_id`: project or workspace context
- `session_id`: specific session, only with `project_id`
- `date_start` / `date_end`: UTC time-bounded recall
- `source`: type of memory (must be paired with `signal` from the allowed combinations below)
- `signal`: how the memory was derived (must be paired with `source` from the allowed combinations below)

If a `session_id` is provided, a `project_id` must also be provided. All timestamps are stored in UTC.

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

- Start narrow with the configured project scope.
- Add time bounds only when needed.
- Use an allowed `(source, signal)` pair to refine results (never set them independently).
- Expand scope only if needed.
- Do not recall on every turn.

## Summary Behavior

Summaries are used for state awareness, not precise retrieval.

Use:

- `memori_recall_summary`

Supported parameters:

- `project_id`
- `session_id`
- `date_start`
- `date_end`

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

Treat summaries as working state, not unquestionable truth.

## Procedure

1. Start of a meaningful session: retrieve a summary.
2. During the task: use targeted recall.
3. When memory is missing or incorrect: send feedback.
4. When limits are reached: degrade gracefully.

## Common Pitfalls

- Do not use broad recall when the user needs one specific fact, decision, or prior outcome.
- Do not treat summaries as authoritative when exact details matter; use targeted recall or verify against current sources.
- Do not call signup, quota, or feedback tools unless the user's request or a Memori error makes them relevant.
- Do not provide a `session_id` without also providing a `project_id`.
- Do not hide privacy tradeoffs: Memori captures completed-turn trace, including tool arguments and final tool result content after Hermes processing.
- Do not let memory override current user instructions, repository rules, or verified facts from the active workspace.

## Safety and Correctness

- Do not invent memory.
- Do not assume memory is correct if it conflicts with the user.
- Verify before acting when needed.
- Treat current user instructions as higher priority than recalled memory.
- Remember that Memori captures completed-turn trace using Hermes' full raw after-processing policy, including tool arguments and final tool result content.

## Feedback

Use:

- `memori_feedback`

Send feedback when:

- Recall results are irrelevant or missing key context.
- Important decisions or constraints were not captured.
- Memory quality degrades across sessions.
- Something works particularly well and should be reinforced.

Feedback improves memory extraction quality, recall relevance, and summary accuracy.

## Account Creation and Onboarding

Use:

- `memori_signup`

Use this tool when:

- The user explicitly asks to sign up, create an account, or get an API key for Memori.
- You encounter an error indicating a missing `MEMORI_API_KEY` and the user provides their email address.

Behavior:

- If the user asks to sign up but does not provide an email address, ask for their email first.
- Once they provide an email, run `memori_signup` with that email.
- Relay the tool result, remind them to check their inbox for the API key, and tell them to configure Memori through `hermes memory setup` or by setting `MEMORI_API_KEY` and `MEMORI_ENTITY_ID`.
- If the Memori SDK is missing, tell them to install it with `pip install memori`.

## Quota Awareness and Upgrades

Use:

- `memori_quota`

Use this tool when:

- The user explicitly asks about their quota, usage, storage, or remaining memory capacity.
- Errors suggest memory limits have been reached and you want to confirm before degrading behavior.

Behavior:

- Invoke `memori_quota` with no arguments.
- Relay the result clearly.
- If limits are near or reached, explain the impact and suggest an upgrade only when performance is affected.

When limits are reached or near:

- Reduce recall scope.
- Prioritize high-signal memory, especially decisions, constraints, key facts, and execution results.
- Avoid unnecessary or repeated recall calls.
- Tell the user when limits affect memory behavior.

Example:

> Memory limits have been reached. I can continue with limited recall, or you can upgrade to restore full functionality.

## Verification

Confirm the skill is working by loading it in a fresh Hermes session and checking that the agent follows the Memori-specific guidance:

```bash
hermes --toolsets skills -q "Use the memori-memory skill to explain when to use Memori recall versus Memori summaries."
```

Expected behavior:

- The agent should load the Memori skill rather than answer from generic memory knowledge.
- The answer should distinguish precise recall from broad state summaries.
- The answer should mention targeted use, avoiding unnecessary recall, and treating current user instructions as higher priority than memory.
- If Memori credentials or the `memori` package are unavailable, the agent should explain the setup gap without inventing recall results.
