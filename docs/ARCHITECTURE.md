# Memori System Architecture Overview

This document provides a comprehensive guide to understanding the Memori system architecture, data flows, and key concepts. It's designed for users and contributors who want to understand how Memori works under the hood.

## Table of Contents

1. [System Overview](#system-overview)
2. [Cloud vs BYODB Modes](#cloud-vs-byodb-modes)
3. [Data Flow Architecture](#data-flow-architecture)
4. [Memory Types & Augmentation](#memory-types--augmentation)
5. [Attribution System](#attribution-system)
6. [Session Management](#session-management)
7. [Recall Pipeline](#recall-pipeline)
8. [Component Architecture](#component-architecture)

---

## System Overview

### What is Memori?

Memori is an **agent-native memory infrastructure** that transforms LLM agent conversations and execution traces into structured, queryable memory. It sits as a middleware layer between your application and LLM providers, transparently capturing and recalling contextual information.

### Key Principles

- **Transparent Integration**: Works with existing LLM clients without code changes
- **LLM-Agnostic**: Supports OpenAI, Anthropic, Google Gemini, xAI, AWS Bedrock, and more
- **Framework-Agnostic**: Integrates with Agno, LangChain, Pydantic AI, or custom frameworks
- **Structured Memory**: Converts unstructured conversations into typed memory (facts, events, relationships, skills, preferences, rules, people)
- **Dual Mode**: Cloud-hosted (Memori Cloud) or self-hosted (BYODB - Bring Your Own Database)

### Architecture Layers

```
┌─────────────────────────────────────────────────┐
│         Your Application / Agent Code            │
└──────────────────┬──────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────┐
│      Memori SDK (Python / TypeScript)            │
│  ┌─────────────────────────────────────────┐    │
│  │ LLM Registry & Client Wrappers          │    │
│  │ - Direct clients (OpenAI, Anthropic)    │    │
│  │ - Framework adapters (LangChain, Agno)  │    │
│  └─────────────────────────────────────────┘    │
│  ┌─────────────────────────────────────────┐    │
│  │ Memory System                           │    │
│  │ - Capture (intercept LLM calls)         │    │
│  │ - Augmentation (extract facts/events)   │    │
│  │ - Recall (retrieve relevant context)    │    │
│  └─────────────────────────────────────────┘    │
│  ┌─────────────────────────────────────────┐    │
│  │ Storage Manager                         │    │
│  │ - Cloud backend or local database       │    │
│  └─────────────────────────────────────────┘    │
└──────────────────┬──────────────────────────────┘
                   │
        ┌──────────┴──────────┐
        │                     │
┌───────▼────────┐    ┌──────▼──────────┐
│ Memori Cloud   │    │   BYODB (Local) │
│ (Hosted)       │    │  Database       │
│                │    │                 │
│ - API Backend  │    │ - PostgreSQL    │
│ - Storage      │    │ - MySQL         │
│ - Dashboard    │    │ - MongoDB       │
│                │    │ - SQLite        │
└────────────────┘    └─────────────────┘
        │                     │
        └──────────┬──────────┘
                   │
┌──────────────────▼──────────────────────────────┐
│     Rust Core (fastembed, retrieval)            │
│  ┌─────────────────────────────────────────┐    │
│  │ Embedding Engine (fastembed)            │    │
│  │ - Fast local embeddings                 │    │
│  │ - Zero external API calls               │    │
│  └─────────────────────────────────────────┘    │
│  ┌─────────────────────────────────────────┐    │
│  │ Retrieval Pipeline                      │    │
│  │ - Hybrid search (vector + lexical)      │    │
│  │ - Ranking & filtering                   │    │
│  │ - Re-ranking with BM25                  │    │
│  └─────────────────────────────────────────┘    │
└────────────────────────────────────────────────┘
```

---

## Cloud vs BYODB Modes

### Memori Cloud

**Best for**: Quick setup, zero infrastructure, managed service

#### Characteristics:
- No local database required
- API key authentication
- Data stored on Memori's infrastructure
- Automatic backup and scaling
- Advanced augmentation features included

#### Flow:
```
Your App → Memori SDK → Memori Cloud API → Memori Servers
         ↓
      (intercept LLM calls)
         ↓
    Send to Cloud → Storage & Augmentation → Return enriched context
```

#### Configuration:
```python
from memori import Memori
from openai import OpenAI

# Cloud mode (default)
# Requires: MEMORI_API_KEY environment variable
mem = Memori()
client = OpenAI()
mem.llm.register(client)
mem.attribution(entity_id="user_123", process_id="my_agent")

# Now all OpenAI calls are automatically captured and stored
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello"}]
)
```

### BYODB (Bring Your Own Database)

**Best for**: Privacy, control, on-premises deployment

#### Characteristics:
- You manage the database
- Supported databases: PostgreSQL, MySQL, SQLite, MongoDB
- Data stays in your infrastructure
- Embeddings computed locally (Rust core)
- Full control over retention policies

#### Flow:
```
Your App → Memori SDK → Storage Manager → Your Database
         ↓                  ↓
      (intercept)      (store locally)
         ↓                  ↓
  Rust Core Embeddings ← Query embedding
         ↓
    Hybrid Search (Vector + Lexical) → Retrieved Facts
```

#### Configuration:

**With Provisioning (recommended):**
```python
from memori import Memori

# Auto-provision SQLite BYODB
mem = Memori.provision(
    provider="sqlite",
    path="./memori.db"
)
mem.attribution(entity_id="user_123", process_id="my_agent")
```

**With Manual Connection:**
```python
import sqlite3
from memori import Memori

# Bring your own connection
conn = lambda: sqlite3.connect("./memori.db")
mem = Memori(conn=conn)
mem.attribution(entity_id="user_123", process_id="my_agent")
```

#### Supported BYODB Providers:

| Provider | Use Case | Notes |
|----------|----------|-------|
| SQLite | Development, testing, single-user | Built-in, no setup |
| PostgreSQL | Production, high-volume, teams | Recommended for scale |
| MySQL | Production, existing MySQL infra | Good compatibility |
| MongoDB | Document-oriented, flexible schema | Supported via pymongo |
| CockroachDB | Distributed, geo-redundant | Special mode available |

---

## Data Flow Architecture

### Memory Capture Flow

```
1. LLM Call Intercepted
   └─ Memori wraps the LLM client
   └─ Captures request + response

2. Parse & Normalize
   └─ Extract messages, tokens, metadata
   └─ Standardize across providers

3. Embedding Generation
   └─ Convert messages to dense vectors
   └─ Using fastembed (local, fast)

4. Advanced Augmentation (Cloud/Rust)
   └─ Extract facts (what was discussed)
   └─ Extract events (what actions happened)
   └─ Extract relationships (connections between entities)
   └─ Extract skills (capabilities discussed)
   └─ Extract preferences (user preferences mentioned)
   └─ Extract rules (constraints/rules learned)
   └─ Extract people (entities mentioned)

5. Storage
   └─ Cloud: POST to Memori API
   └─ BYODB: Write to local database with embedding vectors

6. Background Indexing
   └─ FAISS index updates (Cloud)
   └─ Database index updates (BYODB)
```

### Memory Recall Flow

```
1. Recall Query
   └─ mem.recall("What was the user's preference?", limit=5)

2. Query Embedding
   └─ Convert query string to dense vector
   └─ Using same model as memory embeddings

3. Hybrid Search
   ├─ Dense Search (Vector Similarity)
   │  └─ FAISS index search
   │  └─ Find semantically similar facts
   │
   └─ Lexical Search (BM25)
      └─ Keyword-based search
      └─ Find exact matches

4. Re-ranking
   └─ Combine dense + lexical scores
   └─ Weight: 85% dense, 15% lexical (configurable)
   └─ Filter by relevance threshold

5. Return Results
   └─ Cloud: CloudRecallResponse (facts + messages)
   └─ BYODB: RecallFact objects
   └─ With scores, summaries, metadata
```

### Example Memory Capture & Recall

```python
from memori import Memori
from openai import OpenAI

mem = Memori()
client = OpenAI()
mem.llm.register(client)
mem.attribution(entity_id="alice", process_id="customer_support")

# Message 1: User tells their favorite color
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "My favorite color is blue."}]
)
# ✅ Automatically captured:
#    - Fact: "alice's favorite color is blue"
#    - Stored with embedding
#    - Indexed for retrieval

# Message 2: Recall the preference later
recall_result = mem.recall("What is alice's favorite color?")
# ✅ Returns:
#    - RecallFact with blue color preference
#    - Similarity score (e.g., 0.89)
#    - Memory ID and timestamp

response2 = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": "What color do I like?"},
        # Memori automatically injects recalled facts into context here
    ]
)
# ✅ LLM now "remembers" blue was mentioned before
```

---

## Memory Types & Augmentation

### Seven Memory Types

Memori's Advanced Augmentation extracts and tracks seven distinct memory types:

#### 1. **Facts**
- What we know as absolute truth
- Examples: "User's name is John", "API key format is 256 bits"
- Most precise memory type
- High recall relevance

```python
# Fact captured from:
# "My name is John and I work at Acme Corp."
# → Fact: "entity user_123 has name John"
# → Fact: "entity user_123 works at Acme Corp"
```

#### 2. **Events**
- Actions or occurrences in time
- Examples: "User completed onboarding", "System error occurred"
- Include timestamp context
- Useful for historical context

```python
# Event captured from:
# "I just finished the tutorial"
# → Event: "entity user_123 completed tutorial at 2024-06-27T10:30:00Z"
```

#### 3. **Relationships**
- Connections between entities
- Examples: "Alice manages Bob", "Project X belongs to Team Y"
- Build knowledge graphs
- Enable inference

```python
# Relationship captured from:
# "I manage a team of 5 engineers"
# → Relationship: "entity user_123 manages multiple engineers"
```

#### 4. **Skills**
- Capabilities or expertise
- Examples: "Knows Python", "Can design databases"
- Track capabilities over time
- Useful for task assignment

```python
# Skill captured from:
# "I've been programming in Python for 10 years"
# → Skill: "entity user_123 has skill Python (10 years experience)"
```

#### 5. **Preferences**
- User or process preferences
- Examples: "Prefers email over phone", "Uses dark mode"
- Personalization data
- Non-persistent across sessions

```python
# Preference captured from:
# "I prefer detailed explanations"
# → Preference: "entity user_123 prefers detailed explanations"
```

#### 6. **Rules**
- Constraints or guidelines
- Examples: "Budget limit is $5000", "Only approve requests >$1000"
- Business logic rules
- Safety constraints

```python
# Rule captured from:
# "Never approve expenses above $10,000 without VP sign-off"
# → Rule: "constraint expense >$10,000 requires VP approval"
```

#### 7. **People**
- Information about individuals mentioned
- Examples: "John is the VP of Sales", "Sarah reported the bug"
- Build organization graphs
- Enable delegation

```python
# People captured from:
# "John from Sales reported a critical bug"
# → Person: "John in Sales department"
# → Event: "John reported critical bug"
```

### Augmentation Process

Memori's Rust core orchestrates augmentation:

```
LLM Response Text
       ↓
   [Preprocessing]
   - Tokenization
   - Normalization
       ↓
   [Extraction Phase]
   - NLP-based extraction
   - Identify entities, relationships
       ↓
   [Classification Phase]
   - Classify into 7 memory types
   - Assign confidence scores
       ↓
   [Enrichment Phase]
   - Add context (session, entity, process)
   - Generate summaries
       ↓
   [Storage Phase]
   - Embed each memory
   - Index for retrieval
       ↓
   Structured Memory in Database
```

---

## Attribution System

### Three-Level Attribution

Memori tracks memory at three hierarchical levels:

#### 1. **Entity** (who)
- Think: person, place, or thing
- Examples: "user_123", "alice", "account_456"
- Scope: User or customer
- Required for memory capture
- Max length: 100 characters

```python
mem.attribution(entity_id="user_123")
```

#### 2. **Process** (what)
- Think: agent, LLM interaction, or program
- Examples: "customer_support_agent", "sales_assistant", "data_analyzer"
- Scope: Specific agent or workflow
- Optional but recommended
- Max length: 100 characters

```python
mem.attribution(entity_id="user_123", process_id="support_agent_v2")
```

#### 3. **Session** (when)
- Think: current conversation or interaction
- Scope: Single conversation thread
- Auto-managed (UUID per session)
- Manually controllable for multi-turn flows
- Useful for separating distinct conversations

```python
# Start a new conversation
mem.new_session()  # Creates new UUID

# Or join an existing session
mem.set_session(session_id="conv_789")
```

### Attribution Example

```
Scenario: Alice (user) talks to support agent about her billing issue

Attribution Breakdown:
├─ Entity: "alice"                          (WHO: the user)
├─ Process: "billing_support_agent"        (WHAT: the agent handling this)
└─ Session: "conv_20240627_xyz"            (WHEN: this conversation)

Memory Scopes:
├─ Entity-level memories:
│  └─ "alice's preferred language is Spanish"
│  └─ "alice has account since 2020"
│
├─ Process-level memories:
│  └─ "support_agent_v2 resolved billing issue for alice"
│  └─ "support_agent_v2 suggested plan upgrade"
│
└─ Session-level memories:
   └─ "alice asked about invoice #12345"
   └─ "alice was offered 10% discount"

Recall Behavior:
┌─ recall("What is alice's language preference?") 
│  → Returns entity-level fact
│  → Works across all processes and sessions
│
├─ recall("What was the support agent's last interaction?")
│  → Returns process-level memories
│  → Scoped to the support agent
│
└─ recall("What discount was offered in this session?")
   → Returns current session facts
   → Must be in same session
```

### Why Attribution Matters

1. **Multi-tenant Systems**: Keep memories separate for different users
2. **Multi-agent Systems**: Each agent builds its own context
3. **Multi-turn Conversations**: Distinguish between different conversation threads
4. **Audit Trail**: Know which agent accessed which memories
5. **Privacy**: Enforce data access boundaries

---

## Session Management

### Session Lifecycle

```
1. Session Created
   └─ When: Memori instance initialized
   └─ Default: UUID auto-generated
   └─ Contains: Entity + Process + Conversation turn data

2. Session Active
   └─ All LLM calls captured under this session
   └─ Memories linked to this session_id
   └─ Can persist across multiple LLM calls

3. Session Ended (optional)
   └─ new_session() creates fresh UUID
   └─ Previous session data remains in database
   └─ Useful for multi-conversation flows

4. Session Queried
   └─ Recall with session filter
   └─ Agent can query past sessions
   └─ Historical context available
```

### When to Create New Sessions

#### ✅ Create New Session When:
- User starts a different topic/task
- Conversation context fundamentally changes
- Agent switches to a different workflow
- Long-running process completes
- New conversation thread begins

```python
# Example: Support agent handling multiple tickets
for ticket_id in tickets:
    mem.new_session()  # Fresh session per ticket
    mem.attribution(entity_id=ticket_id, process_id="support_agent")
    
    # Handle this ticket
    response = client.chat.completions.create(...)
    # Memories stored under this ticket's session
```

#### ❌ Keep Same Session When:
- Multi-turn conversation on same topic
- Related LLM calls within same workflow
- Maintaining conversation context
- Agent reasoning across multiple steps

```python
# Example: Multi-turn reasoning
mem.attribution(entity_id="query_id", process_id="analyzer")

# Call 1: Analyze data
response1 = client.chat.completions.create(
    messages=[{"role": "user", "content": "Analyze this dataset"}]
)

# Call 2: Follow-up question (same session)
response2 = client.chat.completions.create(
    messages=[
        {"role": "user", "content": "Analyze this dataset"},
        {"role": "assistant", "content": response1.choices[0].message.content},
        {"role": "user", "content": "What trends do you see?"}
    ]
)
# Both calls in same session, context flows naturally
```

---

## Recall Pipeline

### Recall Process (Step by Step)

#### Step 1: Query Preparation
```python
facts = mem.recall(
    query="What is the user's budget limit?",
    limit=5  # Return top 5 results
)
```

#### Step 2: Entity Resolution
- Look up entity_id from current attribution
- If no entity_id set → error
- If entity_id doesn't exist in DB → create it

#### Step 3: Query Embedding
- Convert query text to dense vector
- Using same embedding model as memories
- Default: fastembed (local, CPU efficient)

```
"What is the user's budget limit?"
         ↓
    [fastembed]
         ↓
[0.123, -0.456, 0.789, ...] (384-dim vector)
```

#### Step 4: Hybrid Search

**Dense Search (Vector Similarity)**
- Query vector vs stored memory vectors
- FAISS index for fast similarity
- Returns top-k by cosine similarity

```
Query Vector [0.123, -0.456, ...]
         ↓
    [FAISS Index]
         ↓
Memory 1: score 0.92 (semantic match: "budget is $5000")
Memory 2: score 0.87 (semantic match: "monthly limit")
Memory 3: score 0.71 (weaker match)
```

**Lexical Search (BM25)**
- Keyword matching against facts
- Handles exact matches
- Good for fact-based queries

```
Query Keywords: ["budget", "limit"]
         ↓
    [BM25 Index]
         ↓
Memory 1: score 2.1 (matches both keywords)
Memory 2: score 1.5 (matches one keyword)
```

#### Step 5: Re-ranking
- Combine dense + lexical scores
- Default weights: 85% dense, 15% lexical
- Configurable via `MEMORI_RECALL_LEX_WEIGHT`

```
Final Score = (0.85 × dense_score) + (0.15 × lexical_score)

Memory 1: (0.85 × 0.92) + (0.15 × 2.1) = 0.782 + 0.315 = 1.097 ✓ Top
Memory 2: (0.85 × 0.87) + (0.15 × 1.5) = 0.740 + 0.225 = 0.965
```

#### Step 6: Filtering
- Filter by relevance threshold
- Default: 0.5 (configurable)
- Remove low-confidence results

```
Memory 1: 1.097 ✓ Pass (≥ 0.5)
Memory 2: 0.965 ✓ Pass (≥ 0.5)
Memory 3: 0.312 ✗ Filtered (< 0.5)
```

#### Step 7: Return Results

**Cloud Mode:**
```python
result = mem.recall("What is the budget?")
# Returns: CloudRecallResponse
# {
#   "facts": [
#       {"id": "...", "content": "budget is $5000", "rank_score": 1.097},
#       {"id": "...", "content": "monthly limit applies", "rank_score": 0.965}
#   ],
#   "messages": [{"role": "user", "content": "..."}, ...]
# }
```

**BYODB Mode:**
```python
result = mem.recall("What is the budget?")
# Returns: List[RecallFact]
# [
#     RecallFact(id=..., content="budget is $5000", rank_score=1.097),
#     RecallFact(id=..., content="monthly limit applies", rank_score=0.965)
# ]
```

### Performance Characteristics

| Operation | Time | Notes |
|-----------|------|-------|
| Embedding generation | ~5-10ms | Local, CPU efficient |
| Vector search (FAISS) | ~1-5ms | Index size dependent |
| Lexical search (BM25) | ~2-10ms | Keyword count dependent |
| Re-ranking | ~1ms | Linear in result count |
| **Total recall** | **~10-30ms** | Cloud: includes network |

---

## Component Architecture

### Python SDK Structure

```
memori/
├── __init__.py                 # Main entry: Memori class
├── _config.py                  # Config management
├── _exceptions.py              # Custom exceptions
│
├── llm/                        # LLM integration layer
│   ├── _registry.py            # LLM registration logic
│   ├── _base.py                # Base classes (BaseClient, BaseLlmAdaptor)
│   ├── _providers/             # Provider implementations
│   │   ├── openai.py           # OpenAI client wrapper
│   │   ├── anthropic.py        # Anthropic Claude wrapper
│   │   ├── google.py           # Google Gemini wrapper
│   │   └── ...                 # Other providers
│   ├── _frameworks/            # Framework adapters
│   │   ├── agno.py             # Agno integration
│   │   ├── langchain.py        # LangChain integration
│   │   └── pydantic_ai.py      # Pydantic AI integration
│   └── _constants.py           # Provider constants
│
├── memory/                     # Memory system
│   ├── capture.py              # Capture LLM calls
│   ├── recall.py               # Recall/search logic
│   ├── augmentation/           # Advanced augmentation
│   │   ├── manager.py          # Background worker
│   │   ├── augmentations/      # Extraction logic
│   │   │   └── memori/         # Memori's augmentation
│   │   └── models.py           # Data models
│   └── models.py               # Memory data structures
│
├── storage/                    # Storage adapters
│   ├── __init__.py             # Manager & registration
│   ├── adapter.py              # Base adapter class
│   ├── adapters/               # Database implementations
│   │   ├── sqlite.py           # SQLite adapter
│   │   ├── postgresql.py       # PostgreSQL adapter
│   │   ├── mysql.py            # MySQL adapter
│   │   └── mongodb.py          # MongoDB adapter
│   └── driver.py               # Data access layer
│
├── search/                     # Search algorithms
│   ├── _search.py              # FAISS + BM25 search
│   └── _types.py               # Search data types
│
├── embeddings/                 # Embedding generation
│   └── _embeddings.py          # fastembed wrapper
│
├── api/                        # Cloud API client
│   └── _network.py             # HTTP client
│
└── native/                     # Rust core bindings
    └── _rust.py                # PyO3 adapter
```

### Rust Core Architecture

```
core/
├── src/
│   ├── lib.rs                  # Engine orchestrator
│   ├── engine/                 # Core engine
│   │   ├── mod.rs              # Engine entry point
│   │   ├── embedding.rs        # fastembed wrapper
│   │   ├── augmentation.rs     # Augmentation pipeline
│   │   └── retrieval.rs        # Retrieval logic
│   │
│   ├── storage/                # Storage bridge interface
│   │   └── mod.rs              # Host-provided storage adapter
│   │
│   └── runtime/                # Background worker pool
│       ├── mod.rs              # Worker pool implementation
│       ├── postprocess.rs      # Postprocess jobs
│       └── augmentation.rs     # Augmentation jobs
│
├── bindings/
│   ├── python/                 # PyO3 bindings
│   │   ├── Cargo.toml
│   │   ├── lib.rs              # Python FFI
│   │   └── src/                # Python implementation
│   │
│   └── node/                   # Node.js N-API bindings
│       ├── Cargo.toml
│       └── src/                # Node FFI
│
└── docs/
    └── architecture.md         # Rust architecture notes
```

### Data Flow Through Components

```
┌─────────────────────────────────────────────┐
│ User Code                                   │
│ client = OpenAI()                           │
│ mem.llm.register(client)                    │
└────────────────┬────────────────────────────┘
                 │
        ┌────────▼─────────┐
        │ LLM Registry     │
        │ _registry.py     │
        │ Matches provider │
        │ Selects wrapper  │
        └────────┬─────────┘
                 │
        ┌────────▼──────────────┐
        │ Provider Wrapper      │
        │ memori/llm/_providers │
        │ Intercepts calls      │
        │ Wraps responses       │
        └────────┬──────────────┘
                 │
        ┌────────▼──────────────────────┐
        │ Capture Module               │
        │ memori/memory/capture.py     │
        │ - Stores conversation        │
        │ - Queues augmentation job    │
        └────────┬─────────────────────┘
                 │
        ┌────────▼──────────────────────┐
        │ Storage Adapter              │
        │ memori/storage/adapters      │
        │ - Writes to database         │
        │ - Creates embeddings entry   │
        └────────┬─────────────────────┘
                 │
        ┌────────▼──────────────────────┐
        │ Augmentation Manager         │
        │ (Background Worker)          │
        │ - Processes messages         │
        │ - Extracts facts/events      │
        │ - Calls Rust core            │
        └────────┬─────────────────────┘
                 │
        ┌────────▼──────────────────────┐
        │ Rust Core Engine             │
        │ core/src/engine              │
        │ - fastembed (embedding)      │
        │ - Augmentation (extraction)  │
        │ - Stores results to DB       │
        └──────────────────────────────┘
```

---

## Environment Variables

### Cloud Mode

| Variable | Required | Default | Purpose |
|----------|----------|---------|----------|
| `MEMORI_API_KEY` | Yes | None | Memori Cloud authentication |
| `MEMORI_API_URL_BASE` | No | `https://api.memorilabs.ai` | Cloud API endpoint |
| `MEMORI_ENTITY_ID` | No | None | Global entity attribution |
| `MEMORI_PROCESS_ID` | No | None | Global process attribution |

### BYODB Mode

| Variable | Required | Default | Purpose |
|----------|----------|---------|----------|
| `MEMORI_STORAGE_PROVIDER` | Yes | None | Database type (sqlite, postgresql, etc) |
| `MEMORI_STORAGE_CONNECTION` | Depends | None | Database connection string |
| `MEMORI_USE_RUST_CORE` | No | Auto-detect | Force Rust core usage |
| `MEMORI_COCKROACHDB_CONNECTION_STRING` | No | None | CockroachDB connection |

### Recall Tuning

| Variable | Default | Range | Purpose |
|----------|---------|-------|----------|
| `MEMORI_RECALL_LEX_WEIGHT` | 0.15 | 0.05-0.40 | Lexical search weight |
| `MEMORI_RECALL_LEX_WEIGHT_SHORT` | 0.30 | 0.05-0.40 | Weight for ≤2 token queries |
| `MEMORI_RECALL_FACTS_LIMIT` | 10 | 1-100 | Default recall result count |
| `MEMORI_RECALL_EMBEDDINGS_LIMIT` | 50 | 1-1000 | Max embeddings to process |
| `MEMORI_RECALL_RELEVANCE_THRESHOLD` | 0.5 | 0.0-1.0 | Minimum relevance score |

### Debug & Testing

| Variable | Default | Purpose |
|----------|---------|----------|
| `MEMORI_TEST_MODE` | 0 | Point at staging API |
| `MEMORI_DEBUG` | 0 | Enable verbose logging |
| `MEMORI_LOG_TRUNCATE` | True | Truncate long content in logs |

---

## Summary

Memori's architecture is built around three core concepts:

1. **Transparent Integration**: Works seamlessly with existing LLM clients
2. **Dual-Mode Deployment**: Cloud for convenience, BYODB for control
3. **Structured Memory**: Transforms conversations into queryable facts, events, and relationships

The system efficiently handles:
- **Capture**: Intercepts and stores LLM interactions
- **Augmentation**: Extracts structured insights from unstructured conversation
- **Retrieval**: Fast hybrid search combining vector and lexical matching
- **Context Injection**: Automatically enriches LLM prompts with relevant memory

Understanding these architectural components helps you:
- Deploy Memori appropriately (Cloud vs BYODB)
- Optimize attribution and session management
- Tune recall parameters for your use case
- Debug integration issues
- Contribute to the project
