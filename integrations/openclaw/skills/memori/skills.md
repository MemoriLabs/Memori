## Memori — Your Persistent Memory Layer

You have access to Memori, a structured long-term memory backend.

**Automatic augmentation** (`agent_end`): After you respond, the conversation turn is automatically sent to Memori to extract and store facts, preferences, decisions, and relationships for future sessions. You do not need to do this manually.

**Manual Recall (IMPORTANT)**: You do NOT automatically receive context from past sessions. If a user refers to past interactions, previous projects, or established preferences, **you must explicitly use your recall tools** to fetch that context before answering.

---

### Memory Retrieval Tools

Use these to search your memory explicitly:

**`memori_recall`** — Fetch granular memory facts using a search query and optional filters. Use this when you need specific details (e.g., "what database did we choose?").

| Parameter   | Type   | Description                                                                                                              |
| ----------- | ------ | ------------------------------------------------------------------------------------------------------------------------ |
| `query`     | string | **Required.** A natural language semantic search query (e.g. "dogs"). **DO NOT use wildcards like `*`.**                 |
| `limit`     | number | Maximum number of memories to return (default: 10)                                                                       |
| `dateStart` | string | ISO 8601 — memories on or after this time                                                                                |
| `dateEnd`   | string | ISO 8601 — memories on or before this time                                                                               |
| `projectId` | string | CRITICAL: Leave EMPTY to use the current project. ONLY provide a value if the user explicitly names a different project. |
| `sessionId` | string | Scope to a specific session — **requires `projectId`**                                                                   |
| `signal`    | string | Filter by signal type: `system`, `user`, `derived`                                                                       |
| `source`    | string | Filter by source origin                                                                                                  |

**`memori_recall_summary`** — Fetch summarized views of stored memories. Use this when you need high-level context (e.g., "what is this project about?").

| Parameter   | Type   | Description                                                                                                              |
| ----------- | ------ | ------------------------------------------------------------------------------------------------------------------------ |
| `dateStart` | string | ISO 8601 — summaries on or after this time                                                                               |
| `dateEnd`   | string | ISO 8601 — summaries on or before this time                                                                              |
| `projectId` | string | CRITICAL: Leave EMPTY to use the current project. ONLY provide a value if the user explicitly names a different project. |
| `sessionId` | string | Scope to a specific session — **requires `projectId`**                                                                   |

> `sessionId` cannot be used without `projectId`. The backend will reject it.

---

### Memory Scoping

All memories are scoped to the current `entityId` and `projectId`. The current project is applied by default — you only need to pass `projectId` when explicitly overriding it for a cross-project lookup.

---

### Coexistence With File Memory

Memori works alongside local file memory (e.g., `MEMORY.md`), it does not replace it:

| Layer                     | Scope                                 | Lifetime                    |
| ------------------------- | ------------------------------------- | --------------------------- |
| Session context           | Current conversation                  | Dies with session           |
| File memory (`MEMORY.md`) | Curated strategic facts               | Persistent on disk          |
| Memori                    | Auto-extracted facts, knowledge graph | Cloud — survives compaction |
