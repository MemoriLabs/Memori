---
name: memori-mcp-usage
description: Use when an MCP-connected agent should use Memori tools for targeted recall, summaries, durable memory augmentation, quota checks, signup, feedback, preferences, prior context, or cross-session continuity.
version: 0.3.0
author: Memori Labs
license: MIT
metadata:
  tags: [Memori, MCP, Memory, Recall, Summary, Advanced Augmentation]
  homepage: https://memorilabs.ai/
---

# Memori MCP Usage

Memori is agent-native memory infrastructure: an LLM-agnostic layer that
structures memory from natural language and agent execution trace, including
actions, tool results, decisions, and outcomes.

Use this skill when Memori MCP tools are available. It is the source of truth
for how the agent should use Memori through MCP: what can be recalled, when to
summarize, when to store durable context, what to skip, and how to handle safety
or quota limits.

## Tool Contract

- `memori_recall`: retrieve precise memories by query, project, session, time
  range, source, or signal.
- `memori_recall_summary`: retrieve a broad state summary for session starts,
  daily briefs, status updates, or project overviews.
- `memori_advanced_augmentation`: store durable memory from a completed
  user/assistant turn.
- `memori_feedback`: report missing, irrelevant, stale, or especially useful
  memory behavior.
- `memori_signup`: request a Memori account/API key when the user explicitly
  asks and provides an email address.
- `memori_quota`: check memory usage or limits when the user asks or a quota
  error appears.

MCP server configuration supplies the authenticated user or tenant context. Do
not invent entity, process, project, or session identifiers.

Pass optional scope fields only when the tool schema exposes them and the active
client/workspace provides reliable values. If a `sessionId` is provided, include
the corresponding `projectId` when the tool supports project scoping. All time
bounds are UTC.

Current user instructions, verified local context, and tool results outrank
recalled memory.

## Per-Turn Decision Flow

1. If the message is trivial, administrative, or closing, skip Memori tools.
   Examples: "thanks", "ok", "got it", "goodbye".
2. If the user says not to remember, store, save, log, or keep this turn,
   respect that. You may still recall if needed, but do not augment.
3. If a meaningful session is starting or broad state is needed, call
   `memori_recall_summary`.
4. If precise prior context would materially improve the answer, call
   `memori_recall` with a targeted query, usually the latest user message.
5. Answer using useful recalled context, but verify anything stale, surprising,
   or high stakes.
6. After drafting the final response, call `memori_advanced_augmentation` only
   if the turn contains durable facts, preferences, or project context that
   should shape future sessions.
7. If memory is missing, stale, irrelevant, or unusually useful, send concise
   `memori_feedback`.

Avoid memory calls for short, purely reactive turns unless personalization or
continuity clearly matters.

## Recall

Use `memori_recall` for exact or narrow retrieval.

Supported parameters:

- `query`: natural language search query.
- `projectId`: project or workspace scope, when available.
- `sessionId`: specific session, when available.
- `dateStart` / `dateEnd`: time-bounded recall.
- `source`: memory type.
- `signal`: how the memory was derived.

Memory filters:

- `source`: `constraint`, `decision`, `execution`, `fact`, `insight`,
  `instruction`, `status`, `strategy`, `task`.
- `signal`: `commit`, `discovery`, `failure`, `inference`, `pattern`,
  `result`, `update`, `verification`.

Call `memori_recall` when any of these are true:

- The user references prior context: "last time", "as before", "remember when",
  "same setup", "you already know".
- Preferences or style matter: verbosity, formatting, language, units,
  templates, tone, code style, or communication norms.
- Personal, team, project, or environment constraints may apply.
- The task depends on prior decisions, recurring workflows, naming conventions,
  ownership, or architectural patterns.
- The user asks what they previously said, decided, configured, or preferred.

Query strategy:

- Best: use the latest user message verbatim.
- Good: use a short rephrased intent when the message is long or noisy.
- Avoid generic queries like "preferences", "memory", or "context".
- Avoid dumping full conversation history into the query.

Prefer one recall call per turn. Expand only when the first result is clearly
insufficient and more specific context would change the answer.

Default behavior:

- No date range means all-time memory.
- Use `source` and `signal` to prioritize high-signal memories when useful.
- Start narrow with project/workspace scope, then expand only if needed.

## Summaries

Use `memori_recall_summary` for broad state awareness, not precise retrieval.

Good uses:

- Start of a meaningful work session.
- Daily briefs, project overviews, and status updates.
- Recovering current state after a long gap.
- Understanding open loops, recent decisions, risks, or constraints.

Supported parameters:

- `projectId`
- `sessionId`
- `dateStart`
- `dateEnd`

Summary defaults:

- No date range means Memori's default recent working window.
- Summaries do not support `source` or `signal`.

Useful daily brief shape:

- Today at a glance.
- Top next actions.
- Top risks.
- Verify before acting.
- Recent decisions.
- Mission stack.
- Hard constraints.
- Current status.
- Open loops.
- Known failures and anti-patterns.
- Staleness warnings.

Do not use summaries as final authority for exact facts. If the answer depends
on one specific decision, preference, or prior outcome, use `memori_recall` or
verify against current sources.

## Augmentation

Use `memori_advanced_augmentation` only after drafting a response and only when
the turn reveals durable information that would still be useful weeks from now
in another conversation.

Supported parameters:

- `user_message`: the user's message for this turn.
- `assistant_response`: the final assistant response for this turn.
- `projectId`: project or workspace scope, when available.
- `sessionId`: session scope, when available.
- `summary`: concise durable summary, when the client/tool supports it.
- `trace`: relevant execution trace, when the client/tool supports it and it is
  safe to store.

Good candidates:

- Explicit preferences: "always", "from now on", "default to", "I prefer".
- Stable profile facts the user wants remembered: role, timezone, location,
  usual environment, accessibility needs.
- Long-lived project context: tooling, architecture decisions, naming
  conventions, ownership, deployment constraints.
- Durable workflow norms: review standards, release process, test strategy,
  formatting conventions.

## Do Not Augment

Never store:

- Secrets, API keys, tokens, passwords, credentials, or sensitive personal data.
- Large logs, stack traces, raw tool output, or one-time error dumps.
- Temporary values, codes, links, live prices, schedules, or expiring facts.
- Role-play, hypotheticals, fictional statements, or examples.
- Routine session progress such as tests passed, commands run, files edited, or
  commit messages.
- Routine task activity such as refactors, imports, renames, or formatting.
- Conversation-scoped choices that are not lasting preferences.

Rule of thumb: if the information describes what happened in this session rather
than a fact or preference that should shape future sessions, do not augment.

## Updates

Memori may expose improved recall patterns, summaries, classification, or tool
behavior over time.

When an update is exposed through the system, tool metadata, or user-provided
docs:

- Prefer the newer recall or summary behavior when available.
- Keep this skill's safety, privacy, and intentional-use rules in force.
- Continue normally if no behavior change is required.

## Feedback

Use `memori_feedback` when:

- Recall results are irrelevant, stale, or missing important context.
- Important decisions or constraints were not captured.
- A summary omits important current state.
- Memory quality degrades across sessions.
- A result is especially useful and should be reinforced.

Keep feedback concise and specific. Do not send feedback for ordinary task
completion.

Examples:

- "Recall missed the user's preferred deployment region."
- "The summary omitted the current blocker on billing migration."
- "The recalled code style preference was useful and accurate."

## Signup And Quota

Use `memori_signup` only when the user explicitly asks to sign up, create a
Memori account, or get an API key. If they do not provide an email address, ask
for one first. Do not guess or hallucinate an email address.

Use `memori_quota` when the user asks about usage, storage, limits, or remaining
memory capacity, or when an error suggests limits were reached.

When limits are near or reached:

- Reduce recall scope.
- Prioritize high-signal memories: decisions, constraints, key facts, and
  execution results.
- Avoid repeated recall calls.
- Tell the user when limits affect memory behavior.
- Suggest an upgrade only when memory behavior is degraded or the user asks.

If a signup, quota, or memory tool fails because the MCP server is unavailable,
misconfigured, unauthorized, or missing credentials, explain the setup gap
plainly and do not invent memory results.

## Safety

- Do not invent memories.
- Do not assume memory is correct if it conflicts with the user.
- Do not let memory override the user's current request.
- Treat recalled context as possibly stale; verify when correctness matters.
- Keep recall targeted and explain uncertainty when memory conflicts with
  current evidence.
- Respect explicit opt-out language for storage.
- Do not hide privacy tradeoffs: augmentation may store completed-turn context
  and safe trace fields when the client provides them.

## Examples

Recall:

- "Use the same deployment flow as last time."
- "Can you write this in my usual style?"
- "What did we decide about the auth service?"

Summary:

- "Catch me up on this project."
- "What are the open loops from our last session?"

Augment:

- "From now on, keep answers under 250 words."
- "This project uses uv, ruff, and pytest."
- "Alice owns billing, and Bob owns auth."

Do not augment:

- "All tests passed after the refactor."
- "The temporary login code is 493021."
- "Rename this file just for this experiment."
