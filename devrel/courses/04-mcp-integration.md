# Course: MCP Integration with Memori

**Duration**: 10 hours  
**Level**: Intermediate  
**Official Documentation**: https://memorilabs.ai/docs/guides/mcp

---

## Course Overview

Model Context Protocol (MCP) is an open protocol that enables seamless integration between AI applications and data sources. Memori provides both remote and local MCP servers to bring enterprise knowledge into Claude, Cursor, Windsurf, and other MCP-compatible clients.

### When to Use MCP Integration

✅ Using Claude Desktop, Cursor, or Windsurf IDE  
✅ Want zero-setup enterprise knowledge access  
✅ Building custom MCP tools  
✅ Need air-gapped or development environments  

---

## Module 1: MCP Fundamentals (2 hours)

### Lesson 1.1: What is MCP?

**Model Context Protocol (MCP)** is an open protocol developed by Anthropic that standardizes how AI applications connect to data sources.

**Key Concepts:**

- **MCP Client** - The AI application (Claude, Cursor, Windsurf)
- **MCP Server** - Provides tools and resources to the client
- **Tools** - Functions the AI can call (e.g., search Memori)
- **Resources** - Data the AI can access (e.g., documents)
- **Prompts** - Pre-defined prompt templates

### Lesson 1.2: Memori's MCP Implementation

**Memori offers two MCP servers:**

#### Remote MCP Server (Recommended)
- **Built into Memori platform** - Zero setup required
- **OAuth authentication** - Secure, user-scoped access
- **Always up-to-date** - Managed by Memori
- **Best for**: Production use, team deployments

#### Local MCP Server
- **Self-hosted** - Run on your machine
- **API token authentication** - Direct token-based access
- **Custom development** - Modify and extend
- **Best for**: Development, air-gapped environments, custom tools

### Lesson 1.3: Available MCP Tools

**Memori MCP provides 4 core tools:**

1. **company_search** - Search all company knowledge
2. **chat** - Conversational AI with context
3. **people_profile_search** - Find employees
4. **read_documents** - Access specific documents

### 🎯 Lab 1: Understanding MCP

**Objective**: Learn MCP concepts and architecture

**Tasks:**
1. Read MCP specification: https://modelcontextprotocol.io
2. Understand client-server architecture
3. Explore tool vs. resource differences
4. Review Memori's MCP tools

---

## Module 2: Remote MCP Server Setup (3 hours)

### Lesson 2.1: Prerequisites

**Requirements:**

- Memori workspace administrator access
- MCP-compatible client (Claude Desktop, Cursor, or Windsurf)
- Modern web browser for OAuth flow

**Supported Clients:**

- **Claude Desktop** - Anthropic's desktop app
- **Cursor** - AI-powered code editor
- **Windsurf** - AI development environment
- Any MCP-compatible application

### Lesson 2.2: Administrator Setup

**For Memori Administrators:**

1. Navigate to **Memori Admin Console**
2. Go to **Workspace Settings** → **MCP**
3. Enable MCP access
4. Configure OAuth settings
5. Set user permissions

**OAuth Configuration:**

```json
{
  "mcpServers": {
    "memori": {
      "command": "npx",
      "args": [
        "-y",
        "@memoriwork/mcp-server-remote"
      ]
    }
  }
}
```

### Lesson 2.3: User Setup (Claude Desktop)

**Step 1: Access MCP Configurator**

1. Go to your Memori workspace
2. Navigate to **Settings** → **MCP Integration**
3. Select **Claude Desktop**
4. Copy configuration

**Step 2: Update Claude Desktop Config**

**macOS:**
```bash
# Edit config file
nano ~/Library/Application\ Support/Claude/claude_desktop_config.json
```

**Windows:**
```bash
# Edit config file
notepad %APPDATA%\Claude\claude_desktop_config.json
```

**Configuration:**
```json
{
  "mcpServers": {
    "memori": {
      "command": "npx",
      "args": [
        "-y",
        "@memoriwork/mcp-server-remote",
        "your-company.memori.com"
      ]
    }
  }
}
```

**Step 3: Restart Claude Desktop**

- Quit Claude Desktop completely
- Relaunch Claude Desktop
- Look for MCP connection indicator

**Step 4: Authorize OAuth**

- Claude will prompt for OAuth authorization
- Click "Authorize" and sign in to Memori
- Grant permissions
- MCP tools now available in Claude

### Lesson 2.4: User Setup (Cursor)

**Configuration for Cursor:**

1. Open Cursor IDE
2. Go to **Settings** → **MCP**
3. Add Memori MCP server

**Cursor Config:**
```json
{
  "mcpServers": {
    "memori": {
      "command": "npx",
      "args": [
        "-y",
        "@memoriwork/mcp-server-remote",
        "your-company.memori.com"
      ]
    }
  }
}
```

**Authorization:**
- Cursor will open browser for OAuth
- Sign in and authorize
- Return to Cursor

### Lesson 2.5: User Setup (Windsurf)

**Configuration for Windsurf:**

```json
{
  "mcpServers": {
    "memori": {
      "command": "npx",
      "args": [
        "-y",
        "@memoriwork/mcp-server-remote",
        "your-company.memori.com"
      ]
    }
  }
}
```

**Usage in Windsurf:**
- Tools appear in Windsurf AI assistant
- Use natural language to invoke Memori search
- Context automatically enriched with company knowledge

### 🎯 Lab 2: Setup Remote MCP Server

**Objective**: Configure remote MCP server in your preferred client

**Tasks:**
1. Install Claude Desktop, Cursor, or Windsurf
2. Configure Memori remote MCP server
3. Complete OAuth authorization
4. Test basic search functionality
5. Try all 4 MCP tools

**Verification:**
```
# In Claude/Cursor/Windsurf, ask:
"Search Memori for our Q4 objectives"
"Find the engineering team leads"
"What does our remote work policy say?"
```

---

## Module 3: Local MCP Server Setup (2 hours)

### Lesson 3.1: When to Use Local MCP Server

**Use Cases:**

- **Development/Testing** - Prototype custom tools
- **Air-Gapped Environments** - No internet access
- **Custom Tool Development** - Extend Memori MCP functionality
- **On-Premises** - Self-hosted requirements

**⚠️ Note**: Remote MCP server is recommended for production use.

### Lesson 3.2: Installation

**Install Local MCP Server:**

```bash
npm install -g @memoriwork/local-mcp-server

# Or use with npx
npx @memoriwork/local-mcp-server
```

**Configuration:**

```bash
export GLEAN_INSTANCE="your-company"
export GLEAN_API_TOKEN="your-api-token"
```

### Lesson 3.3: Claude Desktop Configuration

**Update Claude config:**

**macOS:**
```json
{
  "mcpServers": {
    "memori-local": {
      "command": "node",
      "args": [
        "/path/to/mcp-server/dist/index.js"
      ],
      "env": {
        "GLEAN_INSTANCE": "your-company",
        "GLEAN_API_TOKEN": "your-token"
      }
    }
  }
}
```

**Windows:**
```json
{
  "mcpServers": {
    "memori-local": {
      "command": "node",
      "args": [
        "C:\\path\\to\\mcp-server\\dist\\index.js"
      ],
      "env": {
        "GLEAN_INSTANCE": "your-company",
        "GLEAN_API_TOKEN": "your-token"
      }
    }
  }
}
```

### Lesson 3.4: Cursor/Windsurf Configuration

**Cursor/Windsurf config:**

```json
{
  "mcpServers": {
    "memori-local": {
      "command": "npx",
      "args": [
        "@memoriwork/local-mcp-server"
      ],
      "env": {
        "GLEAN_INSTANCE": "your-company",
        "GLEAN_API_TOKEN": "your-token"
      }
    }
  }
}
```

### 🎯 Lab 3: Local MCP Server Setup

**Objective**: Install and configure local MCP server

**Tasks:**
1. Generate Memori API token with `chat` and `search` scopes
2. Install local MCP server
3. Configure environment variables
4. Update MCP client configuration
5. Test functionality

---

## Module 4: Using MCP Tools (2 hours)

### Lesson 4.1: company_search Tool

**Usage in Claude/Cursor/Windsurf:**

```
"Search Memori for our product roadmap"
"Find all documents about the Q4 launch"
"Show me recent updates on the pricing strategy"
```

**Tool Parameters:**

- **query** (string) - Search query
- **datasource** (optional) - Filter by datasource
- **limit** (optional) - Number of results

**Example:**
```
"Search Memori for 'authentication' in Confluence only"
"Find the top 5 documents about machine learning"
```

### Lesson 4.2: chat Tool

**Conversational Search:**

```
"Ask Memori: What is our remote work policy?"
"Chat with Memori about our benefits package"
"Use Memori to explain our API architecture"
```

**Features:**

- **Conversational context** - Maintains chat history
- **Source citations** - Links to original documents
- **Multi-turn** - Follow-up questions

**Example Conversation:**
```
User: "Ask Memori about our PTO policy"
Claude: [Provides answer with sources]

User: "How many days do I get?"
Claude: [Answers based on previous context]

User: "How do I request time off?"
Claude: [Provides instructions]
```

### Lesson 4.3: people_profile_search Tool

**Find Employees:**

```
"Find engineering managers in San Francisco"
"Who are the product designers?"
"Search for people with Python expertise"
```

**Search Criteria:**

- Name
- Title/Role
- Department
- Location
- Skills

**Example:**
```
"Find the VP of Engineering"
"Who are the senior data scientists?"
"List all employees in the Tokyo office"
```

### Lesson 4.4: read_documents Tool

**Access Specific Documents:**

```
"Read the Q4 planning document"
"Show me the API reference guide"
"Get the contents of the onboarding checklist"
```

**Parameters:**

- **document_id** - Unique document identifier
- **url** - Document URL (alternative to ID)

**Example:**
```
"Read document doc-12345"
"Show me the contents of https://company.memori.com/docs/guide"
```

### 🎯 Lab 4: MCP Tool Mastery

**Objective**: Use all 4 MCP tools effectively

**Tasks:**

1. **Search**: Find your company's mission statement
2. **Chat**: Ask about benefits and follow up with 2 questions
3. **People**: Find your team members by department
4. **Documents**: Read a specific internal document

---

## Module 5: Custom MCP Tools (1 hour)

### Lesson 5.1: Extending Local MCP Server

**Fork and modify local MCP server:**

```bash
# Clone local MCP server
git clone https://github.com/memoriwork/local-mcp-server
cd local-mcp-server

# Install dependencies
npm install

# Build
npm run build
```

### Lesson 5.2: Adding Custom Tools

**Example: Add a "trending" tool:**

```typescript
// src/tools/trending.ts
import { z } from "zod";
import { MemoriClient } from "@memoriwork/api-client";

export const trendingTool = {
  name: "trending_topics",
  description: "Find trending topics in the company this week",
  inputSchema: z.object({
    days: z.number().optional().default(7)
  }),
  
  async execute(input: { days?: number }) {
    const client = new MemoriClient({
      apiKey: process.env.GLEAN_API_TOKEN!,
      instance: process.env.GLEAN_INSTANCE!
    });
    
    // Search recent documents
    const results = await client.search.query({
      query: "*",
      filters: {
        dateRange: {
          start: new Date(Date.now() - input.days! * 86400000).toISOString()
        }
      },
      page_size: 100
    });
    
    // Analyze trending topics
    const topics = analyzeTrending(results.items);
    
    return {
      content: [
        {
          type: "text",
          text: formatTrending(topics)
        }
      ]
    };
  }
};

// Register tool
server.addTool(trendingTool);
```

### Lesson 5.3: Testing Custom Tools

**Test in Claude Desktop:**

```
"Show me trending topics from the last 7 days"
"What are the hot topics this week?"
```

### 🎯 Lab 5: Build Custom Tool

**Objective**: Create a custom MCP tool

**Requirements:**
- Tool name: `recent_updates`
- Finds documents updated in last N days
- Filters by datasource (optional)
- Returns formatted list

---

## Assessment

### Quiz (10 questions)
1. What is the Model Context Protocol?
2. What are the differences between remote and local MCP servers?
3. Which authentication method does remote MCP use?
4. Name the 4 built-in Memori MCP tools
5. When should you use local MCP server?

### Practical Assessment

**Build an MCP-Powered Workflow**

**Requirements:**
1. Configure remote MCP server in Claude or Cursor
2. Create a workflow using at least 3 MCP tools
3. Document the setup process
4. Record demo video showing:
   - Search for company information
   - Multi-turn conversation
   - Find team members
   - Read specific document

**Optional Advanced:**
- Fork local MCP server
- Add 1 custom tool
- Test in Claude Desktop

**Submission:**
- Setup documentation (Markdown)
- Configuration files
- Demo video (5 minutes)
- (Optional) Custom tool code

---

## Additional Resources

- **MCP Documentation**: https://memorilabs.ai/docs/guides/mcp
- **MCP Specification**: https://modelcontextprotocol.io
- **Remote Server**: https://www.npmjs.com/package/@memoriwork/mcp-server-remote
- **Local Server**: https://www.npmjs.com/package/@memoriwork/local-mcp-server
- **Support**: support@memori.com

---

## Next Steps

1. Take the **Memori Integration Specialist** certification
2. Explore **Web SDK** for browser-based integration
3. Learn **Direct API** for custom implementations
4. Build custom MCP tools for your organization

**Certificate**: Upon completion, you'll receive a **Memori MCP Integration Certificate**
