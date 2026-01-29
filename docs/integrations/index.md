# IDE & Agent Tool Integrations

Memori integrates with popular AI-powered coding tools to give your development workflows persistent memory. Whether you're using Claude Code, Cursor, or Cline, Memori helps your AI assistant remember context across sessions.

## Quick Comparison

| Tool | Type | MCP Support | Integration Complexity |
|------|------|-------------|----------------------|
| [Claude Code](./claude-code.md) | CLI Agent | Yes | Medium |
| [Cursor](./cursor.md) | IDE | Yes | Medium |
| [Cline](./cline.md) | VS Code Extension | Yes | Medium |

## How It Works

Memori is a Python SDK that adds memory to LLM applications. Since these IDE tools manage their own LLM calls internally, you integrate Memori through the **Model Context Protocol (MCP)** - an open standard for connecting AI tools to external capabilities.

```
┌─────────────────┐     MCP      ┌─────────────────┐     SQL      ┌──────────┐
│ IDE/Agent Tool  │ ◄──────────► │ Memori MCP      │ ◄──────────► │ Database │
│ (Claude/Cursor) │   tools      │ Server          │              │          │
└─────────────────┘              └─────────────────┘              └──────────┘
```

You run a small MCP server that provides `remember` and `recall` tools backed by Memori. The IDE tool calls these tools as needed.

## Which Guide Should I Follow?

- **Claude Code** → [Claude Code Integration](./claude-code.md)
  - Best for: Terminal-based AI coding workflows
  - Setup: `claude mcp add memori python /path/to/server.py`

- **Cursor** → [Cursor Integration](./cursor.md)
  - Best for: Full IDE experience with AI
  - Setup: Add to Cursor's MCP settings

- **Cline** → [Cline Integration](./cline.md)
  - Best for: VS Code users wanting autonomous coding
  - Setup: Configure in Cline's MCP servers

## Common MCP Server

All three tools can use the same MCP server pattern. Here's a minimal example:

```python
#!/usr/bin/env python3
"""Universal Memori MCP Server"""
import sqlite3
import os
from mcp.server import Server
from memori import Memori

DB_PATH = os.path.expanduser("~/.memori/ide_memory.db")
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

def get_connection():
    return sqlite3.connect(DB_PATH)

memori = Memori(conn=get_connection)
memori.config.storage.build()

server = Server("memori")

@server.tool()
async def remember(content: str, project: str = "default") -> str:
    """Store information in persistent memory."""
    memori.attribution(entity_id="developer", process_id=f"ide:{project}")
    memori.store(content=content)
    return f"Remembered: {content}"

@server.tool()
async def recall(query: str, project: str = "default", limit: int = 5) -> str:
    """Recall relevant information from memory."""
    memori.attribution(entity_id="developer", process_id=f"ide:{project}")
    facts = memori.recall(query, limit=limit)
    if not facts:
        return "No relevant memories found."
    return "\n".join(f"- {fact.content}" for fact in facts)

if __name__ == "__main__":
    import asyncio
    from mcp.server.stdio import stdio_server
    asyncio.run(stdio_server(server))
```

## What Should AI Assistants Remember?

Good candidates for memory:

| Category | Examples |
|----------|----------|
| **Architecture** | "Using repository pattern", "Event-driven microservices" |
| **Conventions** | "camelCase for variables", "Functional components preferred" |
| **Context** | "API at api.example.com", "Uses PostgreSQL 15" |
| **Decisions** | "Chose Redis over Memcached for X reason" |
| **Bugs** | "Auth bug was token expiration", "Fixed by adding retry" |

## Project Isolation

Keep memories separate per project using attribution:

```python
memori.attribution(
    entity_id="your_user_id",
    process_id=f"tool:{project_name}"  # e.g., "cursor:my-app"
)
```

## Requirements

- Python 3.8+
- `pip install memori mcp`
- SQLite (default) or PostgreSQL for production

## Next Steps

Choose your tool's specific guide:
- [Claude Code](./claude-code.md)
- [Cursor](./cursor.md)
- [Cline](./cline.md)

Or learn more about:
- [Database Setup](../features/databases.md) - Production database configuration
- [Advanced Augmentation](../advanced-augmentation.md) - Memory enhancement features
