---
name: memori
id: '@memorilabs/openclaw-memori'
description: Agent-native memory for OpenClaw that structures memory from agent trace, execution history, decisions, tool calls, and conversations into durable long-term memory primitives.
license: Apache-2.0
homepage: https://github.com/MemoriLabs/Memori
repository: https://github.com/MemoriLabs/Memori.git
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

# Memori - Structured Long-term Memory for OpenClaw

Give your OpenClaw agents persistent, structured memory derived from agent execution, tool usage, workflow history, and conversations. Memori integrates seamlessly in the background via lifecycle hooks and provides agents with the tools to retrieve context when it is relevant.

## Core Workflow

Memori operates on two parallel tracks through standard OpenClaw lifecycle hooks:

### 1. Advanced Augmentation (automatic)

After each interaction, Memori converts raw session data into structured, reusable memories asynchronously.

- Transforms raw agent sessions into structured memory units
- Captures the agent's actions, reasoning, tool usage, responses, corrections, and failures
- Organizes into classes to enable efficient retrieval
- Generates embeddings for semantic retrieval
- Updates structured memory and the knowledge graph

This is how structured memory is continuously built and updated over time. It runs after the agent responds and does not impact latency.

### 2. Agent-Controlled-Intelligent Recall

Recall is explicit and initiated by the agent.

Memori separates memory creation from memory recall:

- Creation is automatic (advanced augmentation)
- Recall is intentional (agent-controlled)

Agents decide:

- When to recall
- What scope to recall from
- How much history to include

To maintain an efficient context window, Memori equips the agent with specific tools to retrieve history when required for the conversation:

1. **`memori_recall`**: Searches the structured memory graph for specific facts, constraints, and prior decisions.
2. **`memori_recall_summary`**: Retrieves structured daily briefs and rolling summaries of prior sessions.
3. **`memori_compaction`**: Retrieves structured post-compaction brief to continue task without interruption.
4. **`memori_feedback`**: Reports on memory quality to improve extraction accuracy.

## Installation

```bash
openclaw plugins install @memorilabs/openclaw-memori
```

## Configuration

Add to your `~/.openclaw/openclaw.json` or use the `openclaw memori init` CLI command to set up your workspace:

```bash
openclaw memori init \
  --api-key "YOUR_MEMORI_API_KEY" \
  --entity-id "your-entity-id" \
  --project-id "your-project-id"
```

Alternatively, configure it directly via JSON:

```json
{
  "plugins": {
    "entries": {
      "openclaw-memori": {
        "enabled": true,
        "config": {
          "apiKey": "${MEMORI_API_KEY}",
          "entityId": "openclaw-user",
          "projectId": "default-project"
        }
      }
    }
  }
}
```

### Configuration Options

- **apiKey** (required): Memori API key — acquire with `memori signup`
- **entityId** (required): Unique identifier for this user's memories
- **projectId** (required): Scopes all memories to a specific project or workspace

## Agentic Tool Guidelines

When this plugin is active, the agent is equipped with tools to manage long-term context. The agent should use its discretion to call these tools when helpful:

- **Contextual Recall**: The agent can run a `memori_recall` search to retrieve relevant details if context is missing regarding user preferences.
- **Summaries**: The agent can utilize the `memori_recall_summary` tool to construct a brief if a user requests a recap.
- **Account Creation**: If a user explicitly asks to create an account, the agent can use the `memori_signup` tool to initiate the process by asking for an email address. Keys are never returned in the chat. The system securely emails the credentials to the user, who must then manually configure them to activate the plugin.
- **Quota Monitoring**: The agent can use the `memori_quota` tool to check the user's current memory usage and storage limits to communicate quota status or gracefully degrade behavior if limits are reached.
- **Date Defaults**: If the agent chooses to search memory, providing specific start/end dates is recommended to keep context windows efficient. Omitting dates will search all available history.

## Verification

Check that the plugin is working and securely connected:

```bash
# Verify plugin is securely connected to the API
openclaw memori status --check

# Check for Memori logs in gateway output
openclaw gateway logs --filter "[Memori]"
```

## Quota Management

Check your current API quota:

```bash
memori quota
```

**Example output:**

```
 __  __                           _
|  \/  | ___ _ __ ___   ___  _ __(_)
| |\/| |/ _ \ '_ ` _ \ / _ \| '__| |
| |  | |  __/ | | | | | (_) | |  | |
|_|  |_|\___|_| |_| |_|\___/|_|  |_|
                  perfectam memoriam
                       memorilabs.ai

+ Maximum # of Memories: 100
+ Current # of Memories: 0

+ You are not currently over quota.
```

Use this to monitor usage and upgrade if needed.

## Performance

- **Automatic deduplication** prevents memory bloat
- **Agent-controlled retrieval** ensures token usage remains targeted, compact, and actionable
- **Semantic ranking** ensures relevant memories surface first

## Privacy, Consent & Data Handling

**Explicit Opt-In Required:** Memori requires the user to explicitly configure an API key (`MEMORI_API_KEY`) and an `entityId`. **No data is captured or transmitted unless these credentials are actively provided by the user.**

- ✅ Conversations are securely transmitted to the Memori backend (`https://api.memorilabs.ai`) only when the plugin is fully configured by the user.
- ✅ Data is encrypted in transit and at rest.
- ✅ Users control their data scope via their specific `projectId` and `entityId`.
- ✅ The backend automatically filters sensitive data (API keys, passwords, secrets) prior to storage.

For details: [Memori Privacy Policy](https://memorilabs.ai/privacy)

## Memory Persistence

Memories persist safely across:

- Session restarts
- Gateway restarts
- System reboots
- OpenClaw upgrades

All storage is handled by the Memori backend and is scoped safely alongside your local `MEMORY.md` file without overwriting it.

## Troubleshooting

**Plugin not loading:**

- Verify `enabled: true` in openclaw.json
- Check API key: `echo $MEMORI_API_KEY`
- Restart gateway: `openclaw gateway restart`

**No memories captured:**

- Check gateway logs for `[Memori]` errors
- Verify API endpoint reachable
- Test API key: `memori quota`

**Memories not recalled:**

- Did the agent utilize the retrieval tool? Check your gateway logs for `memori_recall` tool execution. If it didn't use the tool, you can prompt it to search its memory.
- Ensure `entityId` and `projectId` are consistent across sessions.
- Verify memories exist: `memori quota` shows count > 0.

**Quota exceeded:**

- Run `memori quota` to check usage
- Upgrade at [memorilabs.ai](https://app.memorilabs.ai/)
- Or clear old memories via dashboard

## Learn More

- **npm Package**: https://www.npmjs.com/package/@memorilabs/openclaw-memori
- **GitHub**: https://github.com/MemoriLabs/Memori
- **Documentation**: https://memorilabs.ai/docs/memori-cloud/openclaw/overview/
- **API Dashboard**: https://app.memorilabs.ai/
- **Support**: [GitHub Issues](https://github.com/MemoriLabs/Memori/issues)

## Notes

This skill informs the agent about the Memori plugin. The plugin must be installed separately via npm. Once installed, memory capture happens in the background, and the agent is empowered to explicitly query its memories when needed.
