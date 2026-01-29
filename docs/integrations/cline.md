# Using Memori with Cline

[Cline](https://github.com/cline/cline) is an autonomous coding agent that runs as a VS Code extension. This guide shows how to add persistent memory to your Cline workflows using Memori.

## Overview

Cline is a VS Code extension that manages its own AI interactions. Since you can't directly wrap its internal LLM client, integration works through MCP servers or external tools.

| Approach | Best For | Complexity |
|----------|----------|------------|
| **MCP Server** | Native integration, Cline has excellent MCP support | Medium |
| **HTTP Service** | Fallback option if MCP isn't suitable | Medium |

---

## Option 1: MCP Server (Recommended)

Cline has built-in support for [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) servers. This is the cleanest integration path.

### How It Works

```
┌─────────────┐     MCP      ┌─────────────────┐     SQL      ┌──────────┐
│   Cline     │ ◄──────────► │ Memori MCP      │ ◄──────────► │ Database │
│ (VS Code)   │   tools      │ Server          │              │ (SQLite) │
└─────────────┘              └─────────────────┘              └──────────┘
```

### Step 1: Create a Memori MCP Server

Create a file called `memori_mcp_server.py`:

```python
#!/usr/bin/env python3
"""
Memori MCP Server - Provides memory tools for Cline
"""
import sqlite3
import os
from mcp.server import Server
from mcp.types import Tool, TextContent
from memori import Memori

# Persistent database location
DB_PATH = os.path.expanduser("~/.memori/cline_memory.db")
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

def get_connection():
    return sqlite3.connect(DB_PATH)

memori = Memori(conn=get_connection)
memori.config.storage.build()

server = Server("memori")

@server.tool()
async def remember(
    content: str,
    category: str = "general",
    project: str = "default"
) -> str:
    """
    Store important information in persistent memory.

    Args:
        content: The information to remember (decisions, patterns, context)
        category: Type of memory (architecture, convention, bug, preference)
        project: Project name for scoping memories

    Use this to remember:
    - Architectural decisions
    - Coding conventions
    - Bug fixes and their causes
    - User preferences
    - Project-specific context
    """
    memori.attribution(entity_id="cline_user", process_id=f"cline:{project}")
    memori.store(content=f"[{category}] {content}")
    return f"Remembered ({category}): {content}"

@server.tool()
async def recall(
    query: str,
    project: str = "default",
    limit: int = 5
) -> str:
    """
    Retrieve relevant information from memory.

    Args:
        query: Natural language query describing what you need
        project: Project name to scope the search
        limit: Maximum number of memories to return

    Use this before:
    - Making architectural decisions
    - Writing code that might follow existing patterns
    - Answering questions about the codebase
    """
    memori.attribution(entity_id="cline_user", process_id=f"cline:{project}")
    facts = memori.recall(query, limit=limit)

    if not facts:
        return "No relevant memories found. Consider asking the user for context."

    result = "Relevant memories:\n"
    for fact in facts:
        result += f"- {fact.content}\n"
    return result

@server.tool()
async def forget(query: str, project: str = "default") -> str:
    """
    Remove outdated or incorrect memories matching a query.

    Args:
        query: Description of memories to remove
        project: Project name to scope the deletion
    """
    memori.attribution(entity_id="cline_user", process_id=f"cline:{project}")
    # Note: Implement based on your Memori version's deletion API
    return f"Memory cleanup requested for: {query}"

if __name__ == "__main__":
    import asyncio
    from mcp.server.stdio import stdio_server
    asyncio.run(stdio_server(server))
```

### Step 2: Configure Cline

1. Open VS Code with Cline installed
2. Open Cline settings (click the gear icon in Cline's panel)
3. Navigate to MCP Servers configuration
4. Add your Memori server:

```json
{
  "memori": {
    "command": "python",
    "args": ["/absolute/path/to/memori_mcp_server.py"],
    "env": {}
  }
}
```

Alternatively, create/edit `~/.cline/mcp_servers.json`:

```json
{
  "memori": {
    "command": "python",
    "args": ["/Users/yourname/scripts/memori_mcp_server.py"]
  }
}
```

### Step 3: Restart Cline

After adding the MCP server, restart VS Code or reload the Cline extension.

### Step 4: Use Memory with Cline

Cline will now have access to memory tools:

```
You: Before we start, remember that this project uses TypeScript strict mode
     and we follow the Airbnb style guide.

Cline: I'll store those conventions in memory.
[Using remember tool with category="convention"]
Got it! I've remembered:
- TypeScript strict mode is enabled
- Following Airbnb style guide

You: Create a new utility function for date formatting

Cline: Let me check if there are any relevant patterns I should follow.
[Using recall tool with query="utility functions patterns conventions"]

Based on my memory, this project uses TypeScript strict mode and
follows Airbnb style guide. Here's the utility function...
```

---

## Option 2: HTTP Companion Service

If you prefer HTTP or need more control, run a local REST service.

### Create the HTTP Server

```python
#!/usr/bin/env python3
"""
Memori HTTP Server for Cline
"""
import sqlite3
import os
from flask import Flask, request, jsonify
from memori import Memori

app = Flask(__name__)

DB_PATH = os.path.expanduser("~/.memori/cline_memory.db")
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

def get_connection():
    return sqlite3.connect(DB_PATH)

memori = Memori(conn=get_connection)
memori.config.storage.build()

@app.route("/api/remember", methods=["POST"])
def remember():
    data = request.json
    content = data.get("content")
    project = data.get("project", "default")
    category = data.get("category", "general")

    memori.attribution(entity_id="cline_user", process_id=f"cline:{project}")
    memori.store(content=f"[{category}] {content}")

    return jsonify({"status": "stored", "content": content})

@app.route("/api/recall", methods=["GET"])
def recall():
    query = request.args.get("q", "")
    project = request.args.get("project", "default")
    limit = int(request.args.get("limit", 5))

    memori.attribution(entity_id="cline_user", process_id=f"cline:{project}")
    facts = memori.recall(query, limit=limit)

    return jsonify({
        "query": query,
        "memories": [{"content": f.content} for f in facts]
    })

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    print("Memori HTTP server running on http://localhost:8766")
    app.run(host="127.0.0.1", port=8766)
```

Run the server:
```bash
pip install flask memori
python memori_http_server.py
```

---

## Memory Categories

Organize memories with categories for better recall:

| Category | Use For | Example |
|----------|---------|---------|
| `architecture` | System design decisions | "Using microservices with event-driven communication" |
| `convention` | Coding standards | "Components use PascalCase, utilities use camelCase" |
| `bug` | Past issues and fixes | "Auth timeout was caused by missing token refresh" |
| `preference` | User/team preferences | "User prefers functional components" |
| `context` | Project-specific info | "API endpoint is api.example.com/v2" |

```python
# When storing
memori.store(content="[architecture] Using repository pattern for data access")
memori.store(content="[convention] All API responses follow JSON:API spec")
```

---

## Workspace-Scoped Memory

For multi-project setups, scope memory by workspace:

```python
import os

# Get workspace name from environment or path
workspace = os.path.basename(os.getcwd())

memori.attribution(
    entity_id="developer_id",
    process_id=f"cline:{workspace}"
)
```

This keeps memories separate per project while allowing cross-project queries when needed.

---

## Troubleshooting

### MCP server not appearing in Cline

1. Verify the path in your MCP config is absolute
2. Check Python is in your PATH:
   ```bash
   which python
   ```
3. Test the server runs without errors:
   ```bash
   python /path/to/memori_mcp_server.py
   ```

### "Module not found" errors

Install dependencies in the correct Python environment:
```bash
pip install memori mcp
```

### Cline not using memory tools

Cline decides when to use tools based on context. You can encourage usage by:
- Explicitly asking Cline to remember something
- Asking questions that require past context
- Adding instructions to your Cline custom instructions

### Slow memory operations

For better performance with large memory stores:
```python
# Use PostgreSQL instead of SQLite
memori = Memori(conn="postgresql://user:pass@localhost/memori")
```

---

## Custom Instructions for Cline

Add to your Cline custom instructions to encourage memory usage:

```
When working on this project:
1. Use the `recall` tool at the start of tasks to check for relevant context
2. Use the `remember` tool to store important decisions, patterns, and fixes
3. Categorize memories (architecture, convention, bug, preference, context)
4. Always check memory before making architectural decisions
```

---

## Next Steps

- [Claude Code Integration](./claude-code.md) - Use Memori with Claude Code CLI
- [Cursor Integration](./cursor.md) - Use Memori with Cursor IDE
- [Database Setup](../features/databases.md) - Use PostgreSQL for production
