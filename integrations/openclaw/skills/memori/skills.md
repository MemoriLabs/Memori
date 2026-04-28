## Memori — Your Persistent Memory Layer

You have access to Memori, a structured long-term memory backend. Two things happen automatically every turn — you do not need to trigger them:

**Automatic recall** (`before_prompt_build`): Before you respond, Memori retrieves memories relevant to the current prompt and injects them as context. You may see `<memori_context>` blocks in your context — that is recalled memory, not something you wrote.

**Automatic augmentation** (`agent_end`): After you respond, the conversation turn is sent to Memori to extract and store facts, preferences, decisions, and relationships for future sessions.

---

### Manual Recall Tools

Use these when automatic recall is not enough — for example, when the user asks about a specific past session, project, or date range.

**`memori_recall`** — Fetch raw memory facts with optional filters:

| Parameter   | Type   | Description                                            |
| ----------- | ------ | ------------------------------------------------------ |
| `dateStart` | string | ISO 8601 — memories on or after this time              |
| `dateEnd`   | string | ISO 8601 — memories on or before this time             |
| `projectId` | string | Override the configured project (defaults to current)  |
| `sessionId` | string | Scope to a specific session — **requires `projectId`** |
| `signal`    | string | Filter by signal type: `system`, `user`, `derived`     |
| `source`    | string | Filter by source origin                                |

**`memori_recall_summary`** — Fetch summarized views of stored memories:

| Parameter   | Type   | Description                                            |
| ----------- | ------ | ------------------------------------------------------ |
| `dateStart` | string | ISO 8601 — summaries on or after this time             |
| `dateEnd`   | string | ISO 8601 — summaries on or before this time            |
| `projectId` | string | Override the configured project (defaults to current)  |
| `sessionId` | string | Scope to a specific session — **requires `projectId`** |

> `sessionId` cannot be used without `projectId`. The backend will reject it.

---

### When to Use Manual Recall

- User asks "what did we work on last week?" → `memori_recall` with `dateStart`/`dateEnd`
- User references a specific past session → `memori_recall` with `sessionId` + `projectId`
- User wants a high-level summary of a project → `memori_recall_summary` with `projectId`
- Automatic recall returned nothing but context suggests prior work exists → `memori_recall` with broader filters

---

### Memory Scoping

All memories are scoped to the configured `entityId` and `projectId`. The current project is applied by default — you only need to pass `projectId` when overriding it for a cross-project lookup.

---

### Coexistence With File Memory

Memori works alongside `MEMORY.md`/`memory/*.md` — it does not replace them:

| Layer                     | Scope                                 | Lifetime                    |
| ------------------------- | ------------------------------------- | --------------------------- |
| Session context           | Current conversation                  | Dies with session           |
| File memory (`MEMORY.md`) | Curated strategic facts               | Persistent on disk          |
| Memori                    | Auto-extracted facts, knowledge graph | Cloud — survives compaction |

Use `MEMORY.md` for strategic context you curate intentionally. Memori catches the granular facts automatically.

---

### Troubleshooting

- **Recall returns nothing**: Verify `entityId` is consistent across sessions. New memories take ~30s to index after being stored.
- **Augmentation errors in logs**: Check API connectivity to `api.memorilabs.ai` and confirm the API key is valid.
- **Quota exceeded**: The free tier allows 100 memories. Upgrade at app.memorilabs.ai.
