# Using Memori with AI Coding Assistants

A practical guide to integrating Memori's persistent memory with Claude Code, Cursor, and Cline.

## Understanding the Integration Challenge

AI coding assistants like Claude Code, Cursor, and Cline are "agent shells" that **own the LLM call internally**. This means you can't directly wrap their LLM client with Memori's SDK like you would in your own application code.

**The solution**: Run Memori as an MCP (Model Context Protocol) server that exposes `remember` and `recall` tools. The AI assistant calls these tools to store and retrieve memories.

```
┌─────────────────┐     MCP Protocol      ┌─────────────────┐
│  Claude Code    │ ◄──────────────────► │  Memori MCP     │
│  Cursor / Cline │    remember/recall    │  Server         │
└─────────────────┘                       └────────┬────────┘
                                                   │
                                                   ▼
                                          ┌─────────────────┐
                                          │  SQLite/Postgres│
                                          │  Memory Store   │
                                          └─────────────────┘
```

---

## Quick Start: Claude Code

### Step 1: Install the Memori MCP Server

```bash
# Using npm
pip install memori-mcp

# Or using pipx (Python)
pipx install memori-mcp-server
```

### Step 2: Configure Claude Code

Add to your Claude Code MCP configuration (`~/.claude/claude_desktop_config.json` or via settings):

```json
{
  "mcpServers": {
    "memori": {
      "command": "memori-mcp-server",
      "args": ["--db", "sqlite:///~/.memori/claude-code.db"],
      "env": {
        "MEMORI_ATTRIBUTION": "claude-code:${USER}"
      }
    }
  }
}
```

### Step 3: Restart Claude Code

After updating the config, restart Claude Code to load the MCP server.

### Step 4: Test It

In Claude Code, try:

```
Remember that I prefer TypeScript over JavaScript for new projects.
```

Then in a new conversation:

```
What are my coding preferences?
```

Claude Code will use the Memori MCP tools to recall your stored preference.

---

## Quick Start: Cursor

### Step 1: Install the Memori MCP Server

```bash
pip install memori-mcp
```

### Step 2: Configure Cursor

Edit `~/.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "memori": {
      "command": "memori-mcp-server",
      "args": ["--db", "sqlite:///~/.memori/cursor.db"],
      "env": {
        "MEMORI_ATTRIBUTION": "cursor:${USER}"
      }
    }
  }
}
```

### Step 3: Enable MCP in Cursor Settings

1. Open Cursor Settings (`Cmd/Ctrl + ,`)
2. Search for "MCP"
3. Enable "Model Context Protocol"
4. Restart Cursor

### Step 4: Use Memory in Your Workflow

In Cursor's AI chat:

```
Remember: This project uses Prisma for database access and Zod for validation.
```

Later, when working on a new file:

```
What ORM and validation libraries does this project use?
```

---

## Quick Start: Cline

### Step 1: Install the Memori MCP Server

```bash
pip install memori-mcp
```

### Step 2: Configure Cline

In VS Code, open Cline settings and add the MCP server:

```json
{
  "cline.mcpServers": {
    "memori": {
      "command": "memori-mcp-server",
      "args": ["--db", "sqlite:///~/.memori/cline.db"]
    }
  }
}
```

### Step 3: Restart VS Code

Reload the window to initialize the MCP connection.

---

## Available MCP Tools

Once configured, these tools are available to the AI assistant:

### `memori_remember`

Store a memory for later recall.

```typescript
// The AI calls this internally when you ask it to remember something
{
  "tool": "memori_remember",
  "arguments": {
    "content": "User prefers functional programming patterns",
    "type": "preference",        // preference | fact | summary | rule
    "attribution": "user:colin", // optional, defaults to config
    "ttl": null                  // optional, auto-expire in seconds
  }
}
```

### `memori_recall`

Retrieve relevant memories based on a query.

```typescript
{
  "tool": "memori_recall",
  "arguments": {
    "query": "What are the user's coding preferences?",
    "limit": 5,                  // max memories to return
    "type": null,                // filter by memory type
    "min_score": 0.7             // similarity threshold
  }
}
```

### `memori_forget`

Remove a specific memory or memories matching criteria.

```typescript
{
  "tool": "memori_forget",
  "arguments": {
    "memory_id": "mem_abc123",   // specific memory ID
    // OR
    "query": "outdated project config",
    "older_than": "30d"          // forget memories older than 30 days
  }
}
```

### `memori_list`

List all memories for inspection.

```typescript
{
  "tool": "memori_list",
  "arguments": {
    "type": "preference",        // optional filter
    "limit": 20,
    "offset": 0
  }
}
```

---

## Integration Patterns

### Pattern 1: Project-Specific Memory

Store memories scoped to a specific project:

```bash
# In your project's .claude/config.json or equivalent
{
  "mcpServers": {
    "memori": {
      "command": "memori-mcp-server",
      "args": ["--db", "sqlite:///.memori/project.db"],
      "env": {
        "MEMORI_ATTRIBUTION": "project:my-app"
      }
    }
  }
}
```

This keeps memories isolated per project, so architectural decisions for Project A don't leak into Project B.

### Pattern 2: Shared Team Memory

For team knowledge that should persist across projects:

```bash
{
  "mcpServers": {
    "memori": {
      "command": "memori-mcp-server",
      "args": [
        "--db", "postgresql://localhost:5432/team_memory",
        "--attribution", "team:engineering"
      ]
    }
  }
}
```

### Pattern 3: User + Project Hybrid

Combine personal preferences with project-specific knowledge:

```json
{
  "mcpServers": {
    "memori-personal": {
      "command": "memori-mcp-server",
      "args": ["--db", "sqlite:///~/.memori/personal.db"],
      "env": { "MEMORI_ATTRIBUTION": "user:${USER}" }
    },
    "memori-project": {
      "command": "memori-mcp-server",
      "args": ["--db", "sqlite:///.memori/project.db"],
      "env": { "MEMORI_ATTRIBUTION": "project:${PWD##*/}" }
    }
  }
}
```

---

## Practical Examples

### Example 1: Onboarding to a New Codebase

When you join a new project, tell Claude Code to remember key facts:

```
Remember these facts about this codebase:
- Authentication uses JWT tokens stored in HTTP-only cookies
- The API follows REST conventions with /api/v1 prefix
- Database migrations are in /db/migrations using Prisma
- Tests use Vitest with React Testing Library
- The team prefers named exports over default exports
```

Now, when you ask questions later:

```
How should I add a new API endpoint?
```

Claude Code recalls the REST conventions and Prisma setup to give you contextual guidance.

### Example 2: Coding Style Preferences

```
Remember my coding preferences:
- I prefer explicit type annotations over inference
- Use async/await over .then() chains
- Prefer early returns to reduce nesting
- Use descriptive variable names, avoid abbreviations
```

When Claude Code generates code, it can recall these preferences to match your style.

### Example 3: Bug Investigation Context

During debugging:

```
Remember: The payment processing bug was caused by a race condition in
the webhook handler. Fixed by adding idempotency keys in PR #234.
```

Months later:

```
We're seeing duplicate charges again. What was the previous payment bug?
```

### Example 4: Architecture Decisions

```
Remember our architecture decision: We chose PostgreSQL over MongoDB because:
1. Strong consistency requirements for financial data
2. Team expertise with SQL
3. Need for complex joins in reporting queries
Decision made on 2024-03-15, documented in ADR-007.
```

---

## Advanced: Self-Hosted MCP Server

For full control, run the Memori MCP server from source:

### Python Version

```python
# memori_mcp_server.py
from memori import Memori
from mcp import MCPServer, Tool

# Initialize Memori
mem = Memori(
    database_connect="sqlite:///~/.memori/claude-code.db",
    attribution="claude-code:user"
)

# Create MCP server
server = MCPServer(name="memori", version="1.0.0")

@server.tool("memori_remember")
async def remember(content: str, type: str = "fact", ttl: int = None):
    """Store a memory for later recall."""
    memory_id = mem.store(
        content=content,
        type=type,
        ttl=ttl
    )
    return {"status": "stored", "memory_id": memory_id}

@server.tool("memori_recall")
async def recall(query: str, limit: int = 5, min_score: float = 0.7):
    """Retrieve relevant memories."""
    memories = mem.recall(
        query=query,
        limit=limit,
        min_score=min_score
    )
    return {"memories": memories}

@server.tool("memori_forget")
async def forget(memory_id: str = None, query: str = None):
    """Remove memories."""
    if memory_id:
        mem.forget(memory_id=memory_id)
    elif query:
        mem.forget(query=query)
    return {"status": "forgotten"}

if __name__ == "__main__":
    server.run()
```

Run it:

```bash
python memori_mcp_server.py
```

Configure your AI assistant to use it:

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

### Node.js Version

```typescript
// memori-mcp-server.ts
import { MCPServer } from '@modelcontextprotocol/sdk';
import { Memori } from '@memoriwork/sdk';

const mem = new Memori({
  databaseConnect: 'sqlite:///~/.memori/claude-code.db',
  attribution: 'claude-code:user'
});

const server = new MCPServer({
  name: 'memori',
  version: '1.0.0'
});

server.tool('memori_remember', {
  description: 'Store a memory for later recall',
  parameters: {
    content: { type: 'string', required: true },
    type: { type: 'string', enum: ['preference', 'fact', 'summary', 'rule'] },
    ttl: { type: 'number', description: 'Time-to-live in seconds' }
  },
  handler: async ({ content, type, ttl }) => {
    const memoryId = await mem.store({ content, type, ttl });
    return { status: 'stored', memoryId };
  }
});

server.tool('memori_recall', {
  description: 'Retrieve relevant memories',
  parameters: {
    query: { type: 'string', required: true },
    limit: { type: 'number', default: 5 },
    minScore: { type: 'number', default: 0.7 }
  },
  handler: async ({ query, limit, minScore }) => {
    const memories = await mem.recall({ query, limit, minScore });
    return { memories };
  }
});

server.start();
```

---

## FAQ

### Do I need vLLM to use Memori with Claude Code?

**No.** vLLM is for self-hosting open-source LLMs. When using Claude Code, Cursor, or Cline, the LLM is already provided by the tool. You only need Memori's MCP server for memory storage.

### Can I use Memori with Anthropic's Claude API directly?

**Yes.** If you're writing your own code that calls the Anthropic API, you can integrate Memori directly:

```python
from memori import Memori
from anthropic import Anthropic

mem = Memori(database_connect="sqlite:///memory.db")
client = Anthropic()

# Register the client with Memori
mem.llm.register(client)
mem.attribution("user:123")
mem.enable()

# Now Claude calls automatically capture and recall memories
response = client.messages.create(
    model="claude-sonnet-4-20250514",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

### How do I share memories between Claude Code and Cursor?

Point both tools to the same database:

```json
// Both configs use the same DB path
"args": ["--db", "sqlite:///~/.memori/shared.db"]
```

### Are my memories sent to the cloud?

By default, **no**. The MCP server runs locally with a local SQLite database. Your memories stay on your machine.

For team/enterprise features, you can optionally connect to Memori's cloud service.

### How do I debug memory issues?

Enable verbose logging:

```bash
MEMORI_LOG_LEVEL=debug memori-mcp-server --db sqlite:///~/.memori/test.db
```

Or use the `memori_list` tool to inspect stored memories.

---

## Troubleshooting

### "MCP server not found"

1. Verify the server is installed: `which memori-mcp-server`
2. Check the config path is correct
3. Restart the AI assistant after config changes

### "No memories found" when they should exist

1. Check attribution matches: memories are scoped by attribution
2. Verify the database path is consistent
3. Check similarity threshold (`min_score`) isn't too high

### Memories aren't persisting

1. Ensure the database directory exists and is writable
2. Check for errors in the MCP server logs
3. Verify SQLite file permissions

---

## Resources

- **Memori MCP Server**: [github.com/colygon/memori-mcp](https://github.com/colygon/memori-mcp)
- **GitHub Issue Discussion**: [github.com/MemoriLabs/Memori/issues/66](https://github.com/MemoriLabs/Memori/issues/66)
- **MCP Specification**: [modelcontextprotocol.io](https://modelcontextprotocol.io)
- **MCP Security Guide**: [modelcontextprotocol-security.io](https://modelcontextprotocol-security.io)
- **Memori Documentation**: [memorilabs.ai/docs](https://memorilabs.ai/docs)
- **Memori Cookbook**: [github.com/MemoriLabs/memori-cookbook](https://github.com/MemoriLabs/memori-cookbook)

---

## Next Steps

1. **Install** the Memori MCP server for your preferred AI assistant
2. **Configure** with project or personal attribution
3. **Start remembering** key facts about your codebase
4. **Share feedback** on [GitHub Issues](https://github.com/MemoriLabs/Memori/issues)
