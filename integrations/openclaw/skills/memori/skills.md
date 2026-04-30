## Memori — Your Persistent Memory Layer

You have access to Memori, a structured long-term memory backend.

**Automatic augmentation** (`agent_end`): After you respond, the conversation turn is automatically sent to Memori to extract and store facts, preferences, decisions, and relationships for future sessions. You do not need to do this manually.

**Manual Recall (IMPORTANT)**: You do NOT automatically receive context from past sessions.
**RULE:** You must NEVER say "I don't know" about the user, their preferences, or past events without FIRST running a `memori_recall` search to check if you remember it. If a user asks a question about themselves, you MUST search Memori before responding.

---

### Memory Retrieval Tools

Use these to search your memory explicitly:

**`memori_recall`** — Fetch granular memory facts using a search query and optional filters. Use this when you need specific details (e.g., "what database did we choose?").

| Parameter   | Type   | Description                                                                                                                                     |
| ----------- | ------ | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| `query`     | string | **Required.** A natural language semantic search query (e.g. "dogs"). **DO NOT use wildcards like `*`.**                                        |
| `dateStart` | string | ISO 8601 (MUST be UTC) — memories on or after this time                                                                                         |
| `dateEnd`   | string | ISO 8601 (MUST be UTC) — memories on or before this time                                                                                        |
| `projectId` | string | CRITICAL: Leave EMPTY to use the current project. ONLY provide a value if the user explicitly names a different project.                        |
| `sessionId` | string | Scope to a specific session — **requires `projectId`**                                                                                          |
| `signal`    | string | Filter by signal type. Allowed values: `commit`, `discovery`, `failure`, `inference`, `pattern`, `result`, `update`, `verification`.            |
| `source`    | string | Filter by source origin. Allowed values: `constraint`, `decision`, `execution`, `fact`, `insight`, `instruction`, `status`, `strategy`, `task`. |

**`memori_recall_summary`** — Fetch summarized views of stored memories. **RULE:** You must NEVER guess or make up a status update, daily brief, or project overview. If the user asks "what did we do last time" or "give me a summary", you MUST use this tool before answering.

| Parameter   | Type   | Description                                                                                                              |
| ----------- | ------ | ------------------------------------------------------------------------------------------------------------------------ |
| `dateStart` | string | ISO 8601 (MUST be UTC) — summaries on or after this time                                                                 |
| `dateEnd`   | string | ISO 8601 (MUST be UTC) — summaries on or before this time                                                                |
| `projectId` | string | CRITICAL: Leave EMPTY to use the current project. ONLY provide a value if the user explicitly names a different project. |
| `sessionId` | string | Scope to a specific session — **requires `projectId`**                                                                   |

> `sessionId` cannot be used without `projectId`. The backend will reject it.

---

### Feedback Tool

**RULE:** You must ALWAYS use this tool immediately if the user asks you to send feedback, report a bug, suggest a feature, or complains about the system. Do NOT just say "I will let the developers know"—you must actually execute the tool to send the message.

**`memori_feedback`** — Send feedback, suggestions, or issues directly to the Memori team.

| Parameter | Type   | Description                                                              |
| --------- | ------ | ------------------------------------------------------------------------ |
| `content` | string | **Required.** The feedback text to send (positive, negative, bugs, etc.) |

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
