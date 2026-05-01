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

# Memori - Structured Long-term Memory for OpenClaw

Give your OpenClaw agents a persistent, structured memory system. Memori automatically captures what happens in the background and gives agents the tools to bring it back when relevant.

## Core Workflow

Memori operates on two parallel tracks through OpenClaw lifecycle hooks:

### 1. Automatic Capture (Advanced Augmentation)

After the agent responds, Memori automatically:

1. Captures the conversation turn (user + assistant)
2. Sends it to the Memori backend for intelligent processing
3. Extracts durable facts, deduplicates, and updates the knowledge graph.
   _(No manual save commands needed - capture just works)._

### 2. Agent-Controlled Recall (Intelligent Retrieval)

Memori does **not** blindly stuff the context window. Instead, it equips the agent with explicit tools to retrieve history exactly when it needs it:

1. **`memori_recall`**: Searches the structured memory graph for specific facts, constraints, and prior decisions.
2. **`memori_recall_summary`**: Retrieves structured daily briefs and rolling summaries of prior sessions.
3. **`memori_feedback`**: Reports on memory quality to improve extraction accuracy.
4. **`memori_signup`**: Creates a Memori account and provisions an API key directly from the agent.
5. **`memori_quota`**: Checks current memory usage and limits so the agent can degrade gracefully when approaching quota.

## Installation

```bash
openclaw plugins install @memorilabs/openclaw-memori
```

## Configuration

Add to your `~/.openclaw/openclaw.json` or use the `openclaw memori init` CLI command:

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

- **apiKey** (required): Your Memori API key. If you don't have one, run `memori sign-up <your-email>` in your terminal or ask your OpenClaw agent to sign you up!
- **entityId** (required): Unique identifier for this user's memories
- **projectId** (required): Scopes all memories to a specific project or workspace

## Agent Instructions & Skill Rules

When this plugin is active, the OpenClaw agent is bound by the following strict behavioral rules injected into its system prompt:

- **Manual Recall (IMPORTANT)**: The agent does NOT automatically receive context from past sessions. It is explicitly instructed: **"You must NEVER say 'I don't know' about the user, their preferences, or past events without FIRST running a `memori_recall` search to check if you remember it."**
- **Summaries**: If a user asks "what did we do last time" or "give me a summary", the agent MUST use `memori_recall_summary` before answering. It is forbidden from guessing project status.
- **Feedback**: The agent MUST use the `memori_feedback` tool immediately if the user asks to send feedback, report a bug, or suggests a feature.
- **Sign Up**: The agent MUST use `memori_signup` when the user asks to create an account or get an API key. It will ask for an email address if one is not provided.
- **Quota**: The agent uses `memori_quota` when the user asks about usage limits, or proactively when it encounters errors suggesting limits have been reached.
- **Date Defaults**: If the agent searches memory but omits start/end dates, recall defaults to **all-time memory** and summaries default to the **last 24 hours**.

## Verification

Check that the plugin is working:

```bash
# Verify plugin is securely connected to the API
openclaw memori status --check

# Check for Memori logs in gateway output
openclaw gateway logs --filter "[Memori]"
```

## Sign Up

If you don't already have a Memori API key, you can sign up directly from the command line using the SDK's CLI:

```bash
memori sign-up <your-email@example.com>
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

## Privacy & Data Handling

**Transparent data flow:**

- ✅ Conversations sent to Memori backend (https://api.memorilabs.ai)
- ✅ Data encrypted in transit and at rest
- ✅ You control data via your API key and entityId
- ✅ No third-party sharing
- ⚠️ Only install if you trust Memori with conversation data

Backend automatically filters sensitive data (API keys, passwords, secrets).

For details: [Memori Privacy Policy](https://memorilabs.ai/privacy)

## Memory Persistence

Memories persist across:

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

- Did the agent actually use the tool? Check your gateway logs for `memori_recall` tool execution. If it didn't use the tool, explicitly ask it to search its memory.
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

This skill teaches the agent about the Memori plugin. The plugin must be installed separately via npm. Once installed, memory capture happens automatically in the background, and the agent is empowered to explicitly query its memories when needed.
