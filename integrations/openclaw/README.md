[![Memori Labs](https://s3.us-east-1.amazonaws.com/images.memorilabs.ai/banner.png)](https://memorilabs.ai/)

<p align="center">
  <strong>The memory fabric for enterprise AI</strong>
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

## Why Memori for OpenClaw?

OpenClaw ships with a simple file-first memory system designed for lightweight experimentation. As deployments scale into production environments, teams often run into memory problems that need more structured, deterministic infrastructure.

Memori provides a drop-in memory layer purpose-built for agentic systems running OpenClaw in production. It works through OpenClaw's plugin lifecycle, so you get persistent, structured memory without changing your agent logic.

## Common Challenges with Default OpenClaw Memory

### 1. Fact conflicts in long-running agents

OpenClaw stores memory as plain markdown files. When facts change or contradict over time, there is no deterministic conflict resolution or lifecycle management.

Memori introduces structured memory with update logic, decay policies, and deterministic fact handling.

### 2. Context loss from token limits

As sessions grow, context must be compacted to fit within model token limits. Important details can be dropped during compression.

Memori stores memory outside the prompt and retrieves the right facts at query time, eliminating compaction loss.

### 3. No relationship reasoning

OpenClaw retrieves semantically similar text but does not model relationships between entities.

Memori builds structured memory graphs that let agents reason across linked facts, not just retrieve similar chunks.

### 4. Cross-project noise

When multiple projects share memory storage, irrelevant context can bleed across workflows.

Memori supports scoped memory namespaces to isolate projects and workflows.

### 5. No user-level isolation

Default memory systems do not provide deterministic isolation across users.

Memori enforces user-scoped memory boundaries for secure multi-user deployments.

## What Changes When You Add Memori?

The Memori plugin replaces OpenClaw's flat-file memory workflow with managed, structured memory that is scoped by `entity_id`, `process_id`, and `session_id` and enriched automatically through OpenClaw's existing hooks.

| Capability | What changes |
| --- | --- |
| **Structured memory storage** | Instead of raw markdown blobs, Memori stores conversations, facts, preferences, and knowledge-graph triples as structured records tied to an entity, process, and session. Facts are extracted as subject-predicate-object relationships, deduplicated over time, and connected into a graph so related memories stay queryable instead of being buried in text files. |
| **Advanced Augmentation** | After each conversation, Memori processes the user and assistant exchange asynchronously in the background, identifies facts, preferences, skills, and attributes, generates embeddings for semantic search, and updates the knowledge graph without blocking the agent's response path. |
| **Intelligent Recall** | Before the agent responds, Memori searches the current entity's stored facts and knowledge graph, ranks memories by semantic relevance and importance, and injects the most useful context into the prompt so durable knowledge survives context-window compression. |
| **Production-ready observability** | Memori Cloud gives you dashboard visibility into memory creation, recalls, cache hit rate, sessions, quota usage, top subjects, per-memory retrieval metrics, and knowledge-graph relationships, so you can inspect what was stored and how recall is behaving in production. |

The plugin still remains drop-in: OpenClaw handles the agent loop, while Memori adds recall, augmentation, sanitization, and observability around it.


## Quickstart

Get persistent memory running in your OpenClaw gateway in three steps.

### Prerequisites

- [OpenClaw](https://openclaw.ai) `v2026.3.2` or later
- A Memori API key from [app.memorilabs.ai](https://app.memorilabs.ai)
- An Entity ID to attribute memories to, such as a user ID, tenant ID, or agent name

### 1. Install and Enable

Run the following commands in your terminal to install and enable the plugin:

```bash
# 1. Install the plugin from npm
openclaw plugins install @memorilabs/openclaw-memori

# 2. Enable it in your workspace
openclaw plugins enable openclaw-memori

# 3. Restart the OpenClaw gateway
openclaw gateway restart
```

### 2. Configure

The plugin needs your Memori API key and an Entity ID to function. You can configure this via the OpenClaw CLI or your `openclaw.json` file.

### Option A: Via OpenClaw CLI (Recommended)

```bash
openclaw config set plugins.entries.openclaw-memori.config.apiKey "YOUR_MEMORI_API_KEY"
openclaw config set plugins.entries.openclaw-memori.config.entityId "your-app-user-id"
```

### Option B: Via `openclaw.json`

Add the following to your `~/.openclaw/openclaw.json` file:

```json
{
  "plugins": {
    "entries": {
      "openclaw-memori": {
        "enabled": true,
        "config": {
          "apiKey": "your-memori-api-key",
          "entityId": "your-app-user-id"
        }
      }
    }
  }
}
```

### Configuration Options

| Option     | Type     | Required | Description                                                                                         |
| ---------- | -------- | -------- | --------------------------------------------------------------------------------------------------- |
| `apiKey`   | `string` | **Yes**  | Your Memori API key.                                                                                |
| `entityId` | `string` | **Yes**  | The unique identifier for the entity (e.g., user, agent, or tenant) to attribute these memories to. |

### 3. Verify

Restart the gateway and inspect the logs:

```bash
openclaw gateway restart
openclaw gateway logs --filter "[Memori]"
```

You should see:

```text
[Memori] === INITIALIZING PLUGIN ===
[Memori] Tracking Entity ID: your-app-user-id
```

To test the full memory loop:

1. Send a message with a durable preference:
   `I always use TypeScript and prefer functional patterns.`
2. Confirm augmentation ran:
   `Augmentation successful!`
3. Start a new session and ask:
   `Write a hello world script.`
4. Confirm recall ran:
   `Successfully injected memory context.`

## How It Works

This plugin integrates with OpenClaw's event lifecycle to provide persistent memory without interfering with the agent's core logic:

1. **`before_prompt_build` (Intelligent Recall):** When a user sends a message, the plugin intercepts the event, queries the Memori API, and safely prepends relevant memories to the agent's system context.
2. **`agent_end` (Advanced Augmentation):** Once the agent finishes generating its response, the plugin captures the final `user` and `assistant` messages, sanitizes them, and sends them to the Memori integration endpoint for long-term storage and entity mapping.

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

- **Documentation**: [https://memorilabs.ai/docs](https://memorilabs.ai/docs)
- **Discord**: [https://discord.gg/abD4eGym6v](https://discord.gg/abD4eGym6v)
- **Issues**: [GitHub Issues](https://github.com/memorilabs/openclaw-memori/issues)

---

## License

Apache 2.0 - see [LICENSE](https://github.com/MemoriLabs/Memori/blob/main/LICENSE)
