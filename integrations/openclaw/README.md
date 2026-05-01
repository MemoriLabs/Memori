[![Memori Labs](https://images.memorilabs.ai/banner-dark-large.jpg)](https://memorilabs.ai/)

<p align="center">
  <strong>Memory from what agents do, not just what they say.</strong>
</p>

<p align="center">
  <i>Give OpenClaw persistent, structured memory with Memori. Capture what matters, recall it when relevant, and move from lightweight experimentation to production-ready memory infrastructure.</i>
</p>

<p align="center">
  <a href="https://www.npmjs.com/package/@memorilabs/openclaw-memori">
    <img src="https://img.shields.io/npm/v/@memorilabs/openclaw-memori.svg" alt="NPM version">
  </a>
  <a href="https://www.npmjs.com/package/@memorilabs/openclaw-memori">
    <img src="https://img.shields.io/npm/dm/@memorilabs/openclaw-memori.svg" alt="NPM Downloads">
  </a>
  <a href="https://opensource.org/license/apache-2-0">
    <img src="https://img.shields.io/badge/license-Apache%202.0-blue" alt="License">
  </a>
  <a href="https://discord.gg/abD4eGym6v">
    <img src="https://img.shields.io/discord/1042405378304004156?logo=discord" alt="Discord">
  </a>
</p>

---

# OpenClaw Overview

Memori gives OpenClaw agents a structured, long-term memory system. It automatically captures what happens and lets agents recall it on demand — so context survives across sessions without bloating the prompt.

Instead of relying solely on natural-language memory, Memori structures persistent memory from both conversation and agent trace — the agent's actions, tool results, decisions, and outcomes — so it can recall what actually happened when it matters.

---

## The Problem with OpenClaw's Built-In Memory

OpenClaw ships with a basic memory layer, but it has fundamental limitations that break down at scale:

| Limitation                    | What happens                                                                                                                                                            |
| ----------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Flat markdown files**       | Memory lives in plain text files with no structure or relationships. The agent reads and writes large chunks of text with no way to index, query, or deduplicate facts. |
| **Context compaction**        | As sessions grow longer, important details begin to disappear. Memory gets lost when context is compressed to stay within token limits.                                 |
| **No relationship reasoning** | OpenClaw retrieves semantically similar text, but cannot understand relationships between facts or connect them to each other.                                          |
| **Weak daily briefs**         | Daily summaries are shallow and lossy, missing key decisions, constraints, and patterns from prior sessions.                                                            |
| **Cross-project noise**       | When working across multiple projects, memories are not isolated. Searches return irrelevant results from other contexts, polluting the agent's responses.              |
| **No user isolation**         | Users do not have their own isolated memories. Memories bleed across users and data is mixed, creating privacy and relevance issues.                                    |

---

## What Changes When You Add Memori?

The Memori plugin replaces OpenClaw’s flat-file memory workflow with structured, scoped memory.

### Memory scoping model

Memories are organized using four core identifiers:

- **Project (`project_id`)** → The broader domain (e.g., “sales prospecting”)
- **Process (`process_id`)** → The agent itself (e.g., “bob_the_sales_prospector”)
- **Session (`session_id`)** → A specific run (e.g., “prospecting_2026-04-29”)
- **Entity (`entity_id`)** → The subject of memory (e.g., the user, customer, or system being tracked)

> Note: If a `session_id` is provided, a `project_id` must also be provided.  
> All timestamps are stored in **UTC**.

---

## Core Capabilities

### 1. Structured memory from conversation and agent trace

Memori transforms raw agent sessions (messages + traces) into structured memory primitives and continuously updated summaries.

Instead of replaying full transcripts, Memori turns conversation and agent execution into structured memory that preserves what the agent did, what happened, and what it learned across long-running interactions.

The system focuses on extracting what matters:

- Key facts
- Decisions
- Outcomes
- Patterns

Rather than preserving every detail, Memori prioritizes signal over noise — keeping context lean while retaining meaning. This avoids replaying or summarizing entire conversations on every turn.

#### Dual memory model

Memori stores knowledge in two complementary forms:

- **Structured memory primitives** — precise, queryable records of facts, decisions, constraints, actions, tool results, and outcomes used for targeted retrieval and reasoning
- **Rolling summaries** — continuously updated context used for grounding and situational awareness

Structured memory is stored in a knowledge graph, enabling relationships, deduplication, and precise retrieval.

#### Grounded in agent execution, not just text

Memori incorporates tool calls and execution traces alongside conversation data. This means memory reflects not just what was discussed, but what the agent actually did and what results those actions produced.

By structuring memory from actions, tool results, decisions, and outcomes, the system gives the agent a fuller understanding of prior task execution so the next time it acts, it can be more accurate and efficient.

---

### 2. Advanced Augmentation (automatic)

After each interaction, Memori converts raw session data into structured, reusable memories asynchronously.

- Transforms raw agent sessions into structured memory units
- Captures the agent’s actions, reasoning, tool usage, responses, corrections, and failures
- Organizes into classes to enable efficient retrieval
- Generates embeddings for semantic retrieval
- Updates structured memory and the knowledge graph

This is how structured memory is continuously built and updated over time.

It runs **after the agent responds** and does not impact latency.

---

### 3. Agent-controlled recall

Recall is **explicit and initiated by the agent**.

Memori separates memory creation from memory recall:

- Creation is automatic (advanced augmentation)
- Recall is intentional (agent-controlled)

Agents decide:

- When to recall
- What scope to recall from
- How much history to include

Supported parameters:

- `entity_id`
- `project_id`
- `session_id`
- `date_start`
- `date_end`
- `source`
- `signal`

#### Memory classification schema

**Sources**

- constraint
- decision
- execution
- fact
- insight
- instruction
- status
- strategy
- task

**Signals**

- commit
- discovery
- failure
- inference
- pattern
- result
- update
- verification

#### Default behavior

- If no date range is provided → returns **all-time**

#### Returned context may include:

- Relevant facts
- Prior decisions
- Constraints
- Patterns
- Summaries

---

### 4. Summaries and daily briefs

Memori provides structured summaries that go beyond OpenClaw’s default daily brief.

These summaries are generated from structured memory and execution traces — not just compressed conversation history.

They represent the agent’s **working state**, not just a recap.

#### Daily brief structure

Memori generates a consistent, structured daily brief with the following sections:

- **Today at a glance** — high-level summary of activity and progress
- **Top 3 next actions** — highest priority tasks
- **Top 3 risks** — immediate risks or blockers
- **Verify before acting** — assumptions requiring validation
- **Recent decisions** — key decisions and context
- **Mission stack** — active goals and objectives
- **Hard constraints** — non-negotiable rules
- **Current status** — current state of execution
- **Open loops** — unresolved tasks or dependencies
- **Known failures and anti-patterns** — what to avoid
- **Staleness warnings** — potentially outdated information

This structure ensures the daily brief is:

- Actionable
- Reliable
- Context-aware

Supported parameters:

- `project_id`
- `session_id`
- `date_start`
- `date_end`

#### Default behavior

- If no date range is provided → returns **last 24 hours**

---

### 5. Production-ready observability

Memori Cloud provides full visibility into memory behavior:

- Memory creation and updates
- Recall activity and hit rates
- Session tracking
- Quota usage
- Top entities and subjects
- Retrieval performance metrics

---

## How it Works

Memori separates memory into two systems:

- **Advanced Augmentation (automatic)** — captures and structures memory
- **Agent-controlled recall (on demand)** — retrieves memory when needed

### 1. Advanced Augmentation (`agent_end` hook)

1. Extract the latest user and assistant messages
2. Sanitize the exchange (remove metadata, timestamps, thinking blocks)
3. Send asynchronously to Memori
4. Memori:
   - Extracts durable facts
   - Deduplicates and updates memory
   - Updates structured memory and the knowledge graph

This runs in the background and **never blocks the agent’s response**.

---

### 2. Agent-controlled recall (plugin tools)

Memori does not automatically inject memory. The agent retrieves it explicitly.

Available tools:

- **`memori_recall`**  
  Query structured memory for facts, constraints, decisions, and patterns

- **`memori_recall_summary`**  
  Retrieve summaries and the daily brief

- **`memori_feedback`**  
  Report on memory quality to improve the system

---

### 3. Retrieval model

Memori returns:

- High-signal structured memory
- Rolling summaries and daily briefs
- Context scoped by entity, project, session, and time

This keeps context:

- **Targeted**
- **Compact**
- **Actionable**

## Memori and Agent Interaction

Memori provides agents with a set of tools to interact with memory in real time.

- **Feedback (`memori_feedback`)**  
  Agents can send feedback to improve memory quality, including missing context or incorrect recall.

- **Updates**  
  Memori evolves over time. Agents can detect new capabilities and adapt their behavior accordingly.

- **Quota awareness**  
  Memori enforces usage limits. When limits are reached, agents can adjust recall behavior and inform the user if needed.

These interactions make Memori a two-way system: agents don’t just retrieve memory — they help improve and shape it over time.

See SKILL.md for detailed agent behavior.

---

# OpenClaw Quickstart

Get a structured, long-term memory system running in your OpenClaw gateway in three steps.

## Prerequisites

- [OpenClaw](https://openclaw.ai) `v2026.3.2` or later
- A Memori API key from [app.memorilabs.ai](https://app.memorilabs.ai)
- An Entity ID to scope memory to a specific user, agent, or system
- A Project ID to scope memory to a specific project or workspace

## 1. Install and Enable

```bash
# Install the plugin from npm
openclaw plugins install @memorilabs/openclaw-memori

# Enable it in your workspace
openclaw plugins enable openclaw-memori
```

## 2. Configure

You need three values: your Memori API key, an Entity ID, and a Project ID.

### Option A: Via CLI (Recommended)

```bash
openclaw memori init \
  --api-key "YOUR_MEMORI_API_KEY" \
  --entity-id "your-app-user-id" \
  --project-id "my-project"
```

### Option B: Via `openclaw.json`

Add the following to `~/.openclaw/openclaw.json`:

```json
{
  "plugins": {
    "entries": {
      "openclaw-memori": {
        "enabled": true,
        "config": {
          "apiKey": "your-memori-api-key",
          "entityId": "your-app-user-id",
          "projectId": "my-project"
        }
      }
    }
  }
}
```

### Configuration Options

<Properties>
  <Property name="apiKey" type="string" required>
    Your Memori API key, available from [app.memorilabs.ai](https://app.memorilabs.ai).
  </Property>
  <Property name="entityId" type="string" required>
    A unique identifier for the entity (user, agent, or tenant) to attribute memories to.
  </Property>
  <Property name="projectId" type="string" required>
    A project or workspace ID used to scope all extracted facts and summaries.
  </Property>
</Properties>

## 3. Verify

After configuring, restart the gateway and verify your API connectivity:

```bash
openclaw gateway restart
openclaw memori status --check
```

You should see:

```
Memori Plugin Status
────────────────────────────────────
  API Key:    ****...A3xQ
  Entity ID:  your-app-user-id
  Project ID: my-project

Checking API connectivity... OK
Status: Ready
```

### Test the Full Memory Loop

1. Send a message with a durable preference:

   > "I always use TypeScript and prefer functional patterns."

2. Check the gateway logs to confirm advanced augmentation ran in the background:

   ```
   [Memori] Augmentation successful!
   ```

3. Start a new session (so the agent is a blank slate) and ask:

   > "Write a hello world script in my preferred language."

4. Confirm the agent used `memori_recall` to fetch your preferences:

   ```
   [Memori] memori_recall params: {"projectId":"my-project","query":"preferred programming language"}
   ```

5. Tell the agent to send feedback:
   > "Send feedback to the developers that the recall was perfect." (This will trigger the `memori_feedback` tool).

---

## What Happens Under the Hood

The Memori plugin operates on two parallel tracks:

| Track                       | Mechanism        | What it does                                                                                                                                                                                 |
| --------------------------- | ---------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Agent-Controlled Recall** | Plugin Tools     | Equips the agent with `memori_recall`, `memori_recall_summary`, and `memori_feedback`. The agent retrieves memory explicitly when needed.                                                    |
| **Advanced Augmentation**   | `agent_end` Hook | After the agent responds, the exchange and execution trace are sanitized and sent to Memori in the background to structure memory from conversation, tool activity, decisions, and outcomes. |

Together, these systems continuously structure memory from not just natural language, but also from agent trace and execution. Memori captures the agent's actions, tool results, decisions, and outcomes into durable memory the agent can recall on demand — so the next time it performs a task, it is more accurate and efficient.

Memori does not automatically inject memory into the prompt. Instead, agents retrieve only the context they need, improving accuracy and efficiency while avoiding unnecessary token usage.

<Admonition type="tip" title="Multi-Agent Gateways">
  The plugin is fully stateless and thread-safe. You can run it across multiple agents in the same gateway without any shared state or concurrency issues.
</Admonition>

---

## Contributing

We welcome contributions from the community! Please see our [Contributing Guidelines](https://github.com/MemoriLabs/Memori/blob/main/CONTRIBUTING.md) for details on code style, standards, and submitting pull requests.

To build from source:

```bash
# Clone the repository
git clone https://github.com/memorilabs/openclaw-memori.git
cd openclaw-memori

# Install dependencies and build
npm install
npm run build

# Run formatting, linting, and type checking
npm run check
```

---

## Support

- [**Documentation**](https://memorilabs.ai/docs/memori-cloud/openclaw/quickstart)
- [**Discord**](https://discord.gg/abD4eGym6v)
- [**Issues**](https://github.com/MemoriLabs/memori/issues)

---

## License

Apache 2.0 - see [LICENSE](https://github.com/MemoriLabs/Memori/blob/main/LICENSE)
