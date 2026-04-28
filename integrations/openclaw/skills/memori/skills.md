## Memori — Persistent Structured Memory for OpenClaw

> Drop-in plugin that replaces flat-file memory with structured, searchable, production-grade memory infrastructure. No commands needed — it hooks into the OpenClaw lifecycle automatically.

---

### What This Skill Covers

This skill teaches you (the agent) how Memori works inside your OpenClaw gateway so you can:

1. Understand what's happening behind the scenes with your memory.
2. Troubleshoot memory issues when recall seems wrong or missing.
3. Guide the operator through installation, configuration, and verification.
4. Know the boundaries between Memori and your existing memory system.

---

### Why Memori Over Flat-File Memory

You currently use `MEMORY.md` + `memory/*.md` — markdown files you curate manually. This works but has real limits:

- **Contradictions:** Old facts linger, things conflict → _Memori: newer facts supersede older ones automatically._
- **Compaction loss:** Long sessions get compressed, details vanish → _Memori: memory lives outside the prompt, retrieved at query time._
- **No relationships:** `memory_search` finds similar text, not connected concepts → _Memori: knowledge graph links entities._
- **Cross-project bleed:** Everything shares one namespace → _Memori: scoped by `entityId` + `projectId` + `sessionId`._
- **No observability:** You guess whether memory is working → _Memori: dashboard shows recalls, cache hits, stored facts._

> **Key insight:** Memori does NOT replace your `MEMORY.md` curation habit. It adds a structured layer underneath.

---

### How It Works (Agent Perspective)

Two lifecycle hooks fire automatically — you don't invoke them:

**Before You Respond → Intelligent Recall**

1. Plugin intercepts `before_prompt_build`.
2. Queries Memori API with current conversation context.
3. Ranks stored facts by semantic relevance and importance.
4. Injects the most useful memories into your system context.

_Note: You may see injected context you didn't write. That's Memori surfacing relevant facts from past sessions._

**After You Respond → Advanced Augmentation**

1. Plugin captures the user + assistant exchange.
2. Sends to Memori backend asynchronously (no latency hit).
3. Backend extracts: facts, preferences, skills, attributes, people, relationships, rules, events.
4. Deduplicates, embeds, indexes, and updates the knowledge graph.

_Note: You don't need to manually extract every fact. Memori catches what you miss. But keep writing important things to `MEMORY.md` — belt and suspenders._

**On Demand → Manual Recall**

In addition to the automatic hooks, you can explicitly query Memori memory at any point using two methods:

- `agentRecall(params?)` — fetch raw memory facts, optionally filtered and searched.
- `agentRecallSummary(params?)` — fetch summarized views of stored memory.

Both accept optional filter parameters:

| Parameter   | Description                                                               |
| ----------- | ------------------------------------------------------------------------- |
| `query`     | Semantic search string to find relevant facts (`agentRecall` only)        |
| `projectId` | Scope results to a specific project (defaults to current project context) |
| `sessionId` | Scope results to a specific session — **requires `projectId` to be set**  |
| `limit`     | Maximum number of results to return                                       |

> **Constraint:** `sessionId` cannot be used without `projectId`. Providing a session without a project will be rejected before the request is sent.

---

### Installation Guide (For the Operator)

**Prerequisites:** OpenClaw v2026.3.2+ and a Memori API key from [MemoriLabs](https://app.memorilabs.ai/signup).

```bash
openclaw plugins install @memorilabs/openclaw-memori
openclaw plugins enable openclaw-memori
openclaw config set plugins.entries.openclaw-memori.config.apiKey "YOUR_KEY"
openclaw config set plugins.entries.openclaw-memori.config.entityId "your-entity-id"
openclaw gateway restart
```

- **Config:** `apiKey` (required), `entityId` (required — scopes all memories to this user/tenant).
- **`entityId` tips:** \* Single-operator → "david" or "operator-main"
  - Multi-user → unique user ID
  - Multi-agent → per-agent or shared

---

### Verification Checklist

1. Run `openclaw plugins list` → shows `openclaw-memori` enabled.
2. Run `openclaw gateway logs --filter "[Memori]"` → shows INITIALIZING PLUGIN + Tracking Entity ID.
3. Send a durable fact → logs show "Augmentation successful!".
4. Start a new session and ask about the fact → logs show "Successfully injected memory context".
5. Run `memori quota` → check your usage limits.

---

### Coexistence With Existing Memory

Memori adds to the stack, it doesn't replace it:

- **Session context** (built-in): Current conversation — dies with session.
- **Active memory** (plugin): Cross-session recall — session transcripts.
- **File memory** (`MEMORY.md`): Curated facts, daily logs — files on disk.
- **Structured memory** (Memori): Auto-extracted facts, knowledge graph — Memori cloud.

> **Best practice:** Use `MEMORY.md` for strategic context. Use Memori for granular fact capture.

---

### Troubleshooting

- **Plugin not loading:** Check `enabled: true` in your config, ensure the API key is set, restart the gateway, and check the logs.
- **Not stored:** Check for augmentation errors in the logs, verify the API is reachable, and test with `memori quota`.
- **Not recalled:** Ensure a consistent `entityId`, verify your memory count is > 0, and wait ~30s for new memories to finish indexing.
- **Quota exceeded:** The free tier allows for 100 memories. Upgrade at app.memorilabs.ai.

---

### Privacy

Conversations are sent to `api.memorilabs.ai`. They are encrypted in transit and at rest. The system auto-filters secrets. The operator has full control via the dashboard, and there is no third-party data sharing. Make sure the operator is comfortable with data leaving the container.

---

### Links

- **npm:** [https://www.npmjs.com/package/@memorilabs/openclaw-memori](https://www.npmjs.com/package/@memorilabs/openclaw-memori)
- **GitHub:** [https://github.com/MemoriLabs/Memori](https://github.com/MemoriLabs/Memori)
- **Docs:** [https://memorilabs.ai/docs/memori-cloud/openclaw/overview/](https://memorilabs.ai/docs/memori-cloud/openclaw/overview/)
- **Dashboard:** [https://app.memorilabs.ai/](https://app.memorilabs.ai/)
- **Discord:** [https://discord.gg/abD4eGym6v](https://discord.gg/abD4eGym6v)
