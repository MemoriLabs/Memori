---
name: memori
id: '@memorilabs/openclaw-memori'
description: Long-term memory for OpenClaw agents using the Memori SDK. Automatically captures conversations and execution trace, and equips the agent with explicit tools to recall context, manage its account, and monitor usage across sessions.
license: MIT
compatibility:
  - openclaw
metadata:
  openclaw:
    requires:
      env:
        - MEMORI_API_KEY
        - ENTITY_ID
        - PROJECT_ID
      bins:
        - memori
    primaryEnv: MEMORI_API_KEY
    externalServices:
      - https://api.memorilabs.ai
---

# Memori skills file

You have access to Memori, agent-native memory infrastructure: an LLM-agnostic layer that structures memory from not just natural language, but also from agent trace that comes from execution.

Memori automatically captures and structures memory from conversation and execution trace — including the agent's actions, tool results, decisions, and outcomes — and allows you to retrieve it on demand. Use it to maintain continuity across sessions, preserve decisions and constraints, and help the agent understand what it actually did so the next time it completes a task, it is more accurate and efficient.

---

## Core instruction

At the start of a session, you should check the `SKILL.md` file

Use it to understand:

- Available capabilities
- Tooling and integrations
- Expected behaviors and constraints

Treat `SKILL.md` as a source of truth for what you can do before taking action.

---

## When to use Memori

Use Memori when:

- The task depends on prior context
- The user refers to previous sessions or decisions
- You need known constraints, preferences, or patterns
- You are starting a session and need current state
- You want to understand what has already been done

---

## When not to use Memori

Do not use Memori when:

- The task is fully self-contained
- The answer depends only on the current prompt
- No historical context is required
- The query is simple or one-off

Avoid unnecessary recall.

---

## Recall behavior

Recall is **agent-controlled and intentional**.

Prefer targeted recall over broad queries.

### Supported parameters (recall only)

- `entityId` → user, agent, or system context
- `projectId` → project or workspace context
- `sessionId` → specific session
- `dateStart` / `dateEnd` → time-bounded recall
- `source` → type of memory
- `signal` → how the memory was derived

> Note: If a `sessionId` is provided, a `projectId` must also be provided.
> All timestamps are stored in **UTC**.

### Memory filters

- `source`:
  - constraint
  - decision
  - execution
  - fact
  - insight
  - instruction
  - status
  - strategy
  - task

- `signal`:
  - commit
  - discovery
  - failure
  - inference
  - pattern
  - result
  - update
  - verification

Use `source` and `signal` to prioritize high-signal memory when possible.

### Default behavior (recall)

- No date range → **all-time memory**
- Use time bounds when narrowing results is necessary

### Best practices

- Start narrow (entity + project)
- Add time bounds only when needed
- Use `source` and `signal` to refine results
- Expand scope only if needed
- Do not recall on every turn

---

## Summary behavior

Summaries are used for **state awareness**, not precise retrieval.

Use:

- `memori_recall_summary`

### Supported parameters (summaries)

- `projectId`
- `sessionId`
- `dateStart`
- `dateEnd`

> Summaries do **not** support `source` or `signal`.

### Default behavior (summaries)

- No date range → **last 24 hours**

---

## Daily brief behavior

At the start of a meaningful session, retrieve a structured summary.

Use the daily brief to understand:

- Current state
- Prior decisions
- Constraints
- Open work

### Expected daily brief structure

- Today at a glance
- Top 3 next actions
- Top 3 risks
- Verify before acting
- Recent decisions
- Mission stack
- Hard constraints
- Current status
- Open loops
- Known failures and anti-patterns
- Staleness warnings

Treat this as the working state of the system.

---

## Typical workflow

1. Start of session → retrieve summary
2. During task → use targeted recall
3. When memory is missing or incorrect → send feedback
4. When limits are reached → degrade gracefully

---

## Safety and correctness

- Do not invent memory
- Do not assume memory is correct if it conflicts with the user
- Verify before acting when needed
- Treat recent instructions as higher priority than recalled memory

---

## Memori system guidance

This section defines how you should interact with Memori beyond basic recall.

---

### 1. Feedback (`memori_feedback`)

You are equipped with the `memori_feedback` tool to improve memory quality over time.

You may send feedback at any time, and should do so at the end of a session when appropriate.

Feedback should be sent when:

- Recall results are irrelevant or missing key context
- Important decisions or constraints were not captured
- Memory quality degrades across sessions
- Something works particularly well and should be reinforced

Examples:

- "Send feedback that recall missed a pricing constraint."
- "Tell Memori the daily brief was accurate."
- "Submit a bug report: you didn't remember my preferred language."

#### Purpose

Feedback improves:

- Memory extraction quality
- Recall relevance
- Summary accuracy

---

### 2. Updates

Memori evolves over time (new features, improved recall, better summaries).

When updates are exposed:

- Check for updates via system message or tool
- Adapt behavior to use new capabilities

When an update is detected:

- Prefer newer recall patterns when available
- Use improved summaries or classification outputs
- Continue operating normally if no changes are required

---

### 3. Account Creation and Onboarding (`memori_signup`)

You are equipped with the `memori_signup` tool to securely create an account and provision an API key for the user via the Memori CLI.

Use this tool when:

- The user explicitly asks to sign up, create an account, or get an API key for Memori.
- You encounter an error indicating a missing `MEMORI_API_KEY` and the user provides their email address to get one.

#### Behavior:

- **Email Required:** If the user asks to sign up but **does not** provide an email address, you MUST ask them for their email first. Do not attempt to guess or hallucinate an email.
- **Execution:** Once they provide an email, run `memori_signup` with that email.
- **Follow-up:** The tool will trigger the underlying CLI and return a success or error message. Relay that message to the user, remind them to check their inbox for the API key, and instruct them to add it to their configuration (`~/.openclaw/openclaw.json` or `.env` file).
- **Troubleshooting Errors:** If the tool fails with a `Permission denied`, `EACCES`, or `command not found` error, do not ask the user for permission to troubleshoot. Instead, immediately tell them that their CLI installation is broken or missing, and explicitly instruct them to run `npm install -g @memorilabs/memori` in their terminal to fix it.

---

### 4. Quota Awareness and Upgrades (`memori_quota`)

You are equipped with the `memori_quota` tool to check the user's current memory usage and limits via the Memori CLI.

Use this tool when:

- The user explicitly asks about their quota, usage, storage, or how many memories they have left.
- You encounter errors suggesting memory limits have been reached and want to confirm before degrading behavior.

#### Behavior:

- **No Parameters Required:** Simply invoke `memori_quota` — it requires no arguments.
- **Execution:** The tool runs the underlying CLI and returns the user's current usage and quota.
- **Follow-up:** Relay the result clearly to the user. If they are near or at their limit, explain the impact and suggest an upgrade.
- **Troubleshooting Errors:** If the tool fails with a `Permission denied`, `EACCES`, or `command not found` error, do not ask the user for permission to troubleshoot. Instead, immediately tell them that their CLI installation is broken or missing, and explicitly instruct them to run `npm install -g @memorilabs/memori` in their terminal to fix it.

#### When limits are reached or nearing:

#### a. Degrade gracefully

- Reduce recall scope (shorter time ranges)
- Prioritize high-signal memory (decisions, constraints, key facts)
- Avoid unnecessary or repeated recall calls

#### b. Inform the human/operator

Clearly communicate when limits impact performance.

Example:

> "Memory limits have been reached. I can continue with limited recall, or you can upgrade to restore full functionality."

#### c. Suggest upgrades appropriately

- Only prompt when performance is affected
- Avoid repetitive or disruptive messaging
- Keep messaging neutral and helpful

---

## Learn More

- **npm Package**: https://www.npmjs.com/package/@memorilabs/openclaw-memori
- **GitHub**: https://github.com/MemoriLabs/Memori
- **Documentation**: https://memorilabs.ai/docs/memori-cloud/openclaw/overview/
- **API Dashboard**: https://app.memorilabs.ai/
- **Support**: [GitHub Issues](https://github.com/MemoriLabs/Memori/issues)
