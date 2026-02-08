# Using Memori with Cursor

[Cursor](https://cursor.sh/) is an AI-powered code editor built on VS Code. This guide shows how to add persistent memory to your Cursor workflows using Memori.

## Overview

Cursor is an IDE that manages its own AI interactions internally. Since you can't directly wrap its internal LLM client, integration works through external tools that Cursor can call.

| Approach | Best For | Complexity |
|----------|----------|------------|
| **MCP Server** | Native integration via Cursor's MCP support | Medium |
| **HTTP Service** | Universal fallback, works with any setup | Medium |

---

## Option 1: MCP Server (Recommended)

Cursor supports the [Model Context Protocol (MCP)](https://modelcontextprotocol.io/), allowing you to connect a Memori-backed server that provides memory tools.

### How It Works

```
┌─────────────┐     MCP      ┌─────────────────┐     SQL      ┌──────────┐
│   Cursor    │ ◄──────────► │ Memori MCP      │ ◄──────────► │ Database │
│   (IDE)     │   tools      │ Server          │              │ (SQLite) │
└─────────────┘              └─────────────────┘              └──────────┘
```

### Step 1: Create a Memori MCP Server

Create a file called `memori_mcp_server.py`:

```python
#!/usr/bin/env python3
"""
Memori MCP Server - Provides memory tools for Cursor
"""
import sqlite3
import os
from mcp.server import Server
from mcp.types import Tool, TextContent
from memori import Memori

# Use a consistent database location
DB_PATH = os.path.expanduser("~/.memori/cursor_memory.db")
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

def get_connection():
    return sqlite3.connect(DB_PATH)

memori = Memori(conn=get_connection)
memori.config.storage.build()

server = Server("memori")

@server.tool()
async def remember(content: str, project: str = "default") -> str:
    """
    Store a fact, decision, or piece of information in memory.
    Use this to remember important context about the codebase.
    """
    memori.attribution(entity_id="cursor_user", process_id=f"cursor:{project}")
    memori.store(content=content)
    return f"Stored in memory: {content}"

@server.tool()
async def recall(query: str, project: str = "default", limit: int = 5) -> str:
    """
    Recall relevant information from memory.
    Use this to retrieve context about the codebase, past decisions, or user preferences.
    """
    memori.attribution(entity_id="cursor_user", process_id=f"cursor:{project}")
    facts = memori.recall(query, limit=limit)
    if not facts:
        return "No relevant memories found for this query."
    return "Relevant memories:\n" + "\n".join(f"- {fact.content}" for fact in facts)

@server.tool()
async def list_memories(project: str = "default", limit: int = 10) -> str:
    """List recent memories for the current project."""
    memori.attribution(entity_id="cursor_user", process_id=f"cursor:{project}")
    facts = memori.facts.list(limit=limit)
    if not facts:
        return "No memories stored yet."
    return "Recent memories:\n" + "\n".join(f"- {fact.content}" for fact in facts)

if __name__ == "__main__":
    import asyncio
    from mcp.server.stdio import stdio_server
    asyncio.run(stdio_server(server))
```

### Step 2: Configure Cursor

Add the MCP server to Cursor's settings. Open Cursor Settings and add to your MCP configuration:

```json
{
  "mcpServers": {
    "memori": {
      "command": "python",
      "args": ["/path/to/memori_mcp_server.py"]
    }
  }
}
```

Or in `~/.cursor/mcp.json`:

```json
{
  "servers": {
    "memori": {
      "command": "python",
      "args": ["/absolute/path/to/memori_mcp_server.py"]
    }
  }
}
```

### Step 3: Use Memory in Cursor

Now Cursor's AI can use memory tools in your conversations:

```
You: Remember that we're using the repository pattern for data access in this project

Cursor: I'll store that architectural decision.
[Using remember tool]
Done! I've noted that this project uses the repository pattern for data access.

You: How should I structure the new UserService?

Cursor: Let me check what I know about this project's patterns.
[Using recall tool]
Based on the project's architecture, you're using the repository pattern.
Here's how I'd structure UserService...
```

---

## Option 2: HTTP Companion Service

If MCP isn't available or you prefer HTTP, run a local REST service.

### Create the HTTP Server

```python
#!/usr/bin/env python3
"""
Memori HTTP Server - REST API for Cursor integration
"""
import sqlite3
import os
from flask import Flask, request, jsonify
from memori import Memori

app = Flask(__name__)

DB_PATH = os.path.expanduser("~/.memori/cursor_memory.db")
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

def get_connection():
    return sqlite3.connect(DB_PATH)

memori = Memori(conn=get_connection)
memori.config.storage.build()

@app.route("/remember", methods=["POST"])
def remember():
    data = request.json
    content = data.get("content")
    project = data.get("project", "default")

    memori.attribution(entity_id="cursor_user", process_id=f"cursor:{project}")
    memori.store(content=content)

    return jsonify({"status": "stored", "content": content})

@app.route("/recall", methods=["GET"])
def recall():
    query = request.args.get("q", "")
    project = request.args.get("project", "default")
    limit = int(request.args.get("limit", 5))

    memori.attribution(entity_id="cursor_user", process_id=f"cursor:{project}")
    facts = memori.recall(query, limit=limit)

    return jsonify({
        "query": query,
        "facts": [{"content": f.content, "confidence": f.confidence} for f in facts]
    })

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8765)
```

Run the server:
```bash
pip install flask
python memori_http_server.py
```

### Use with Cursor

You can reference the service in your Cursor rules or system prompt:

```
When I ask you to remember something, make a POST request to http://localhost:8765/remember
When you need context, query http://localhost:8765/recall?q=<query>
```

---

## Per-Project Memory Isolation

To keep memories separate per project, pass the project name:

```python
# In your MCP server or HTTP calls
memori.attribution(
    entity_id="your_user_id",
    process_id=f"cursor:{project_name}"  # e.g., "cursor:my-react-app"
)
```

This ensures:
- Each project has its own memory space
- Queries only return relevant context
- You can clear one project's memory without affecting others

---

## What to Remember

Good candidates for memory storage:

- **Architectural decisions**: "This project uses the repository pattern"
- **Coding conventions**: "We use camelCase for variables, PascalCase for components"
- **Project-specific context**: "The API is hosted at api.example.com"
- **User preferences**: "User prefers functional components over class components"
- **Bug context**: "The auth bug was caused by token expiration handling"

---

## Troubleshooting

### MCP server not connecting

1. Check the path is absolute in your Cursor config
2. Verify Python and dependencies are installed:
   ```bash
   pip install memori mcp
   ```
3. Test the server manually:
   ```bash
   python /path/to/memori_mcp_server.py
   ```

### Memories not persisting across sessions

Ensure the database path is absolute:
```python
DB_PATH = os.path.expanduser("~/.memori/cursor_memory.db")
```

### Slow recall queries

For large memory stores, consider using PostgreSQL instead of SQLite:
```python
memori = Memori(conn="postgresql://user:pass@localhost/memori")
```

---

## Next Steps

- [Claude Code Integration](./claude-code.md) - Use Memori with Claude Code CLI
- [Cline Integration](./cline.md) - Use Memori with Cline VS Code extension
- [Database Setup](../features/databases.md) - Use PostgreSQL for production
