# Using Memori with Claude Code, Cursor, and Cline

Memori is a **Python SDK** that adds persistent, structured memory to LLM applications by registering against an LLM client (OpenAI/Anthropic/etc.) and storing memories in your datastore.

Tools like **Claude Code**, **Cursor**, and **Cline** are developer-facing “agent shells” that run inside a CLI or IDE. They are not Python libraries you import, so integration usually means one of the following patterns.

## TL;DR

- You **do not** need to run vLLM to use Memori with Claude models.
- If you control the code that calls the LLM (your agent/service), you can integrate Memori directly by registering the **Anthropic** (or OpenAI-compatible) client.
- If you *don’t* control the tool’s internal LLM calls (common for IDE assistants), you integrate by adding a **local companion service/tool** that the assistant can call (HTTP, MCP, etc.) which uses Memori under the hood.

---

## Pattern A (recommended): Your agent/service uses Claude → integrate Memori directly

If you have a Python app that calls Claude (Anthropic) directly, integrate Memori in-process.

1) Create a DB connection factory (SQLite is fine for dev).
2) Instantiate your Anthropic client.
3) Register it with Memori.
4) Set attribution.

```python
import os
import sqlite3

from memori import Memori
from anthropic import Anthropic


def get_sqlite_connection():
    return sqlite3.connect("memori.db")

client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

mem = Memori(conn=get_sqlite_connection).llm.register(client)
mem.attribution(entity_id="dev_123", process_id="claude_code_agent")
mem.config.storage.build()

# Use Claude as usual (exact call signature depends on your anthropic SDK version)
# client.messages.create(...)

# In short-lived scripts:
mem.augmentation.wait()

facts = mem.recall("my preferences", limit=5)
print(facts)
```

If you’re using a gateway like LiteLLM / OpenAI-compatible endpoints, you can register the OpenAI-compatible client instead.

---

## Pattern B: IDE assistant calls the LLM internally → add a local “Memori companion” tool

In many IDE shells (Cursor/Cline/Claude Code), the tool itself owns the LLM call. In that case you generally **can’t** wrap the internal client from your Python process.

Instead, you expose memory capabilities as a tool the assistant can call:

### Option B1: HTTP tool (works everywhere)

Create a small local HTTP service that exposes:
- `POST /remember` (store a note/fact)
- `GET /recall?q=...` (semantic recall)

The IDE assistant calls the service; the service uses Memori to store/recall.

**Why this helps:** it’s deterministic, debuggable, and keeps memory in your database (not locked inside an IDE).

### Option B2: MCP server (best UX where supported)

Some assistants support the Model Context Protocol (MCP). In that case, an MCP server can provide “memory” tools (remember/recall) backed by Memori.

> If you’d like this officially supported, we recommend adding a `memori-mcp` reference server to the repo (or cookbook) so users can connect it to Cursor/Cline/Claude tools without writing glue code.

---

## Claude Code: what’s possible today

Claude Code is a CLI agent. If it does not expose a way to hook/wrap its internal LLM client, you’ll want **Pattern B** (companion tool/service).

If Claude Code supports MCP (or tool calling), you can connect a Memori-backed tool server and get persistent memory without changing Claude Code itself.

---

## Cursor + Cline: what’s possible today

Cursor and Cline are IDE agents that commonly support external tools/servers.

Recommended approach:
1) Run a local Memori companion service (HTTP or MCP).
2) Configure the IDE agent to call `recall` before generating a response, and `remember` after important outcomes (decisions, preferences, TODOs).
3) Use attribution to scope memory by repo/project/user.

---

## Suggested attribution scheme for coding tools

A pragmatic default:
- `entity_id`: your developer/user id (or machine id)
- `process_id`: the tool + repository name (e.g. `cursor:my-repo`)
- session: per work session/branch/task

This prevents unrelated repos from polluting each other’s memory.

---

## If you want official support

The highest-leverage product/DX improvement here is shipping an official:
- **Memori MCP server** (reference implementation)
- and a short setup guide for Cursor/Cline/Claude tools

That would reduce “time-to-first-success” from *hours of glue* to *~5 minutes*.
