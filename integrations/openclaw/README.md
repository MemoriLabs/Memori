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

# Memori for OpenClaw

Memori gives OpenClaw agents a structured, long-term memory system. It automatically captures what happens and lets agents recall it on demand — so context survives across sessions without bloating the prompt.

Instead of relying solely on natural-language memory, Memori structures persistent memory from both conversation and agent trace — the agent's actions, tool results, decisions, and outcomes — so it can recall what actually happened when it matters.

---

## The problem

OpenClaw's default memory works for simple use cases, but breaks at scale:

- Memory is stored as flat markdown files
- Context is lost due to compaction
- Important decisions and constraints disappear
- No relationships between facts
- Memory bleeds across users and projects

---

## What Memori changes

Memori replaces flat memory with structured, scoped memory built from:

- Agent execution (tool calls, results, decisions, outcomes)

Instead of replaying history, agents retrieve exactly what they need.

---

## How it works

Memori runs on two parallel systems:

### 1. Advanced augmentation

After each interaction, Memori converts raw session data into structured, reusable memories asynchronously.

- Transforms raw agent sessions into structured memory units
- Captures the agent's actions, reasoning, tool usage, responses, corrections, and failures
- Organizes into classes to enable efficient retrieval
- Generates embeddings for semantic retrieval
- Updates structured memory and the knowledge graph

This is how structured memory is continuously built and updated over time.

It runs **after the agent responds** and does not impact latency.

---

### 2. Agent-controlled recall

Recall is **explicit and initiated by the agent**.

Memori separates memory creation from memory recall:

- Creation is automatic (advanced augmentation)
- Recall is intentional (agent-controlled)

Agents decide:

- When to recall
- What scope to recall from
- How much history to include

Memori does not automatically inject memory into the prompt. The agent retrieves only the context it needs, keeping token usage efficient.

Available tools:

- **`memori_recall`** — query structured memory for facts, constraints, decisions, and patterns
- **`memori_recall_summary`** — retrieve summaries and the daily brief
- **`memori_feedback`** — report on memory quality to improve the system

---

## Quickstart

### Prerequisites

- [OpenClaw](https://openclaw.ai) `v2026.3.2` or later
- A Memori API key from [app.memorilabs.ai](https://app.memorilabs.ai)
- An Entity ID to scope memory to a specific user, agent, or system
- A Project ID to scope memory to a specific project or workspace

### 1. Install

```bash
openclaw plugins install @memorilabs/openclaw-memori
openclaw plugins enable openclaw-memori
```

### 2. Configure

```bash
openclaw memori init \
  --api-key "YOUR_MEMORI_API_KEY" \
  --entity-id "your-app-user-id" \
  --project-id "my-project"
```

### 3. Verify

```bash
openclaw gateway restart
openclaw memori status --check
```

Expected:

```
Status: Ready
```

### 4. Test the memory loop

1. Tell the agent something durable:

   > "I always use TypeScript and prefer functional patterns."

2. Start a new session and ask:

   > "Write a hello world script in my preferred language."

3. Confirm the agent used `memori_recall` to fetch your preferences:
   ```
   [Memori] memori_recall params: {"projectId":"my-project","query":"preferred programming language"}
   ```

If it works, you now have persistent memory across sessions.

---

## Memory model

Memory is scoped to prevent noise and ensure relevance:

- `entity_id` → user, agent, or system context
- `project_id` → project or workspace context
- `session_id` → specific session (requires `project_id`)
- `date_start` / `date_end` → time-bounded recall (defaults to all-time if omitted)
- `source` → type of memory (recall only)
- `signal` → how the memory was derived (recall only)

All timestamps are stored in **UTC**.

---

## Agent behavior (read this)

Agents should:

- Retrieve a summary at the start of meaningful sessions
- Use targeted recall (not broad queries)
- Avoid recalling on every turn
- Use memory only when context is needed
- Send feedback when memory is missing or incorrect

See SKILL.md for full behavior guidelines.

---

## Typical workflow

1. Start session → retrieve summary
2. During task → targeted recall
3. Missing context → send feedback
4. End of session → memory is captured automatically

---

## Multi-agent ready

The plugin is fully stateless and thread-safe. You can run it across multiple agents in the same gateway without shared state or concurrency issues.

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
