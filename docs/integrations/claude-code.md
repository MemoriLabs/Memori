# Using Memori with Claude Code

[Claude Code](https://docs.anthropic.com/en/docs/claude-code) is Anthropic's agentic coding tool that runs in your terminal. This guide shows how to add persistent memory to your Claude Code workflows using Memori.

## Overview

Claude Code is a CLI agent that manages its own LLM calls internally. Since you can't directly wrap its internal Anthropic client, there are two integration approaches:

| Approach | Best For | Complexity |
|----------|----------|------------|
| **MCP Server** | Native integration, best UX | Medium |
| **Direct Integration** | Custom agents you build | Low |

---

## Option 1: MCP Server (Recommended)

Claude Code natively supports the [Model Context Protocol (MCP)](https://modelcontextprotocol.io/). You can run a Memori-backed MCP server that provides `remember` and `recall` tools.

### How It Works

```
┌─────────────┐     MCP      ┌─────────────────┐     SQL      ┌──────────┐
│ Claude Code │ ◄──────────► │ Memori MCP      │ ◄──────────► │ Database │
│   (CLI)     │   tools      │ Server          │              │ (SQLite) │
└─────────────┘              └─────────────────┘              └──────────┘
```

### Step 1: Create a Memori MCP Server

Create a file called `memori_mcp_server.py`:

```python
#!/usr/bin/env python3
"""
Memori MCP Server - Provides memory tools for Claude Code
"""
import sqlite3
from mcp.server import Server
from mcp.types import Tool, TextContent
from memori import Memori

# Initialize Memori with SQLite
def get_connection():
    return sqlite3.connect("claude_code_memory.db")

memori = Memori(conn=get_connection)
memori.config.storage.build()

# Create MCP server
server = Server("memori")

@server.tool()
async def remember(content: str, entity_id: str = "default", process_id: str = "claude_code") -> str:
    """Store a fact or piece of information in memory."""
    memori.attribution(entity_id=entity_id, process_id=process_id)
    memori.store(content=content)
    return f"Remembered: {content}"

@server.tool()
async def recall(query: str, entity_id: str = "default", limit: int = 5) -> str:
    """Recall relevant information from memory."""
    memori.attribution(entity_id=entity_id, process_id="claude_code")
    facts = memori.recall(query, limit=limit)
    if not facts:
        return "No relevant memories found."
    return "\n".join(f"- {fact.content}" for fact in facts)

if __name__ == "__main__":
    import asyncio
    from mcp.server.stdio import stdio_server
    asyncio.run(stdio_server(server))
```

### Step 2: Register with Claude Code

Add the MCP server to Claude Code:

```bash
claude mcp add memori python /path/to/memori_mcp_server.py
```

### Step 3: Use Memory in Claude Code

Now Claude Code can use memory tools:

```
You: Remember that this project uses PostgreSQL 15 and Python 3.11

Claude: I'll store that information.
[Calling remember("This project uses PostgreSQL 15 and Python 3.11")]
Done! I've remembered the project configuration.

You: What database does this project use?

Claude: Let me check my memory.
[Calling recall("database project configuration")]
Based on my memory: This project uses PostgreSQL 15.
```

---

## Option 2: Direct Integration (Custom Agents)

If you're building your own agent that uses Claude (not Claude Code itself), you can integrate Memori directly by registering the Anthropic client.

### Example: Custom Claude Agent with Memory

```python
import os
import sqlite3
from anthropic import Anthropic
from memori import Memori

def get_connection():
    return sqlite3.connect("my_agent_memory.db")

# Initialize Anthropic client
client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

# Register with Memori
memori = Memori(conn=get_connection).llm.register(client)
memori.attribution(entity_id="developer_123", process_id="my_claude_agent")
memori.config.storage.build()

# Use Claude as normal - Memori captures facts automatically
response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "My name is Alice and I prefer dark mode."}
    ]
)

print(response.content[0].text)

# Wait for async memory extraction
memori.augmentation.wait()

# Later, recall context
facts = memori.recall("user preferences")
print(f"Recalled: {facts}")
```

---

## Attribution Best Practices

For coding workflows, use this attribution scheme:

```python
memori.attribution(
    entity_id="your_user_id",           # Developer or team ID
    process_id="claude_code:repo_name", # Tool + repository
    session_id="feature_branch"         # Optional: branch or task
)
```

This ensures:
- Memory is scoped per developer
- Different repos don't pollute each other
- You can query by project or branch

---

## Troubleshooting

### MCP server not recognized

Verify the server is registered:
```bash
claude mcp list
```

If not listed, re-add with the full path:
```bash
claude mcp add memori python $(which python) /absolute/path/to/memori_mcp_server.py
```

### Memory not persisting

Ensure the database path is absolute or in a consistent location:
```python
def get_connection():
    return sqlite3.connect("/Users/you/.memori/claude_code_memory.db")
```

### "Memori not found" error

Install Memori in the same Python environment as your MCP server:
```bash
pip install memori
```

---

## Next Steps

- [Cursor Integration](./cursor.md) - Use Memori with Cursor IDE
- [Cline Integration](./cline.md) - Use Memori with Cline VS Code extension
- [Database Setup](../features/databases.md) - Use PostgreSQL for production
