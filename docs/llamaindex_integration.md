# Memori × LlamaIndex Integration Guide

> **Persistent long-term memory for LlamaIndex agents and chat engines.**

---

## Overview

By default, LlamaIndex agents forget everything when a session ends.  The Memori × LlamaIndex integration fixes that.  It hooks directly into the LLM client layer, so:

- **Before every LLM call** — relevant memories are retrieved from Memori and injected into the system prompt automatically.
- **After every LLM call** — the conversation turn is persisted for future recall.

No changes to your prompts or agent code are required.

---

## Installation

```bash
pip install memori-llamaindex
```

Install the LlamaIndex LLM package(s) you need:

```bash
pip install "memori-llamaindex[openai]"      # llama-index-llms-openai
pip install "memori-llamaindex[anthropic]"   # llama-index-llms-anthropic
pip install "memori-llamaindex[gemini]"      # llama-index-llms-gemini
```

Set environment variables:

```bash
export MEMORI_API_KEY="your-memori-api-key"
export OPENAI_API_KEY="your-openai-api-key"
```

---

## Quick Start

### Chat Engine (recommended starting point)

```python
from llama_index.core.chat_engine import SimpleChatEngine
from llama_index.llms.openai import OpenAI
from memori import Memori
from memori.integrations.llamaindex import MemoriChatEngine

mem    = Memori()                                   # uses MEMORI_API_KEY
llm    = OpenAI(model="gpt-4o-mini")
engine = SimpleChatEngine.from_defaults(llm=llm)

memory_engine = MemoriChatEngine(
    chat_engine=engine,
    memori=mem,
    entity_id="user_123",          # who is the user?
    process_id="support-bot",      # what is this agent?
)

# Session 1
response = memory_engine.chat("My favourite colour is blue.")

# Session 2 (reset → new Memori session)
memory_engine.reset()
response = memory_engine.chat("What's my favourite colour?")
# → Memori injects the remembered fact and the agent answers correctly.
```

### Agent

```python
from llama_index.core.agent import ReActAgent
from llama_index.llms.openai import OpenAI
from memori import Memori
from memori.integrations.llamaindex import MemoriAgent

llm   = OpenAI(model="gpt-4o-mini")
agent = ReActAgent.from_tools([], llm=llm, verbose=True)
mem   = Memori()

memory_agent = MemoriAgent(
    agent=agent,
    memori=mem,
    entity_id="user_123",
    process_id="react-agent",
)

response = memory_agent.chat("Remember my project deadline is June 30.")
```

### Direct LLM Registration (lowest-level API)

If you prefer not to use the wrappers:

```python
from llama_index.llms.openai import OpenAI
from memori import Memori

llm = OpenAI(model="gpt-4o-mini")
mem = Memori()
mem.llm.register(llm)                      # auto-detects LlamaIndex
mem.attribution(entity_id="user_123", process_id="my-app")

# All subsequent calls through `llm` automatically have memory active.
```

---

## BYODB (Bring Your Own Database) Setup

For local or self-hosted mode:

```python
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from memori import Memori
from memori.integrations.llamaindex import MemoriChatEngine

engine  = create_engine("sqlite:///my_memories.db")
Session = sessionmaker(bind=engine)
mem     = Memori(conn=Session)
mem.config.storage.build()              # initialise schema once

# Then use as normal:
memory_engine = MemoriChatEngine(chat_engine=..., memori=mem, entity_id="u1")
```

---

## Configuration

### MemoriLlamaIndexConfig

```python
from memori.integrations.llamaindex import MemoriChatEngine, MemoriLlamaIndexConfig

config = MemoriLlamaIndexConfig(
    entity_id="user_123",
    process_id="support-bot",
    retrieval_k=5,              # top-K memories recalled per query
    project_id="my-project",   # Memori Cloud project ID
)

engine = MemoriChatEngine(chat_engine=base_engine, memori=mem, config=config)
```

| Parameter | Default | Description |
|---|---|---|
| `entity_id` | *(required)* | Identifies the user/entity |
| `process_id` | `"llamaindex"` | Identifies the agent/process |
| `retrieval_k` | `5` | Max memory facts recalled per query |
| `auto_retrieve_memories` | `True` | Inject memories into prompts automatically |
| `enable_chat_memory` | `True` | Enable memory for chat engines |
| `enable_agent_memory` | `True` | Enable memory for agents |
| `session_id` | `None` | Explicit session ID (auto-generated if not set) |
| `project_id` | `None` | Memori Cloud project ID for `capture_agent_turn` |

---

## Optional: Memori Cloud Callback Handler

For users who want explicit, durable per-turn records in Memori Cloud:

```python
from llama_index.core.callbacks import CallbackManager
from memori.integrations.llamaindex import MemoriCallbackHandler

mem.attribution(entity_id="user_123", process_id="my-agent")
handler = MemoriCallbackHandler(mem, project_id="my-project")
callback_manager = CallbackManager([handler])

# Pass callback_manager to your LlamaIndex engine or agent.
```

> **Note:** The callback handler is optional. Memory injection and recall are already handled transparently by the patched LLM client. The callback handler adds explicit turn records to the Memori Cloud dashboard and enables per-session summaries.

---

## Multi-Session Patterns

### Explicit session management

```python
# Start a new session (old memories are still retrievable)
memory_engine.reset()                 # resets chat history + new Memori session

# Or just reset Memori session while keeping chat history:
mem.new_session()
```

### Sharing memories between agents

Two agents serving the same user share memories automatically when they have the same `entity_id`:

```python
support_agent = MemoriAgent(agent=..., memori=mem, entity_id="user_123", process_id="support")
coding_agent  = MemoriAgent(agent=..., memori=mem, entity_id="user_123", process_id="coder")
# Both recall and store to the same user's memory pool.
```

---

## Supported LlamaIndex LLMs

| LlamaIndex LLM | Package |
|---|---|
| `llama_index.llms.openai.OpenAI` | `llama-index-llms-openai` |
| `llama_index.llms.anthropic.Anthropic` | `llama-index-llms-anthropic` |
| `llama_index.llms.gemini.Gemini` | `llama-index-llms-gemini` |

Other LlamaIndex LLMs that wrap an OpenAI-compatible client should work via the automatic fallback path.

---

## Architecture

```
User Query
    │
    ▼
MemoriChatEngine.chat(message)
    │
    ▼
[patched llm._client.chat.completions.create]
    │
    ├─ inject_recalled_facts()     ← Memori injects memories into system prompt
    ├─ inject_conversation_messages()  ← Memori injects conversation history
    │
    ▼
OpenAI / Anthropic / Gemini API
    │
    ▼
handle_post_response()             ← Memori stores the turn + runs augmentation
    │
    ▼
Response returned to caller
```

Memory injection and storage happen inside the patched SDK client — completely invisible to LlamaIndex's chat engine layer.

---

## Examples

See [`examples/llamaindex/main.py`](../../examples/llamaindex/main.py) for a complete customer support bot demo.

---

## Troubleshooting

**`UnsupportedLLMProviderError`**: Your LlamaIndex LLM is not yet supported.  Try the direct SDK client registration:
```python
mem.llm.register(your_llm._client)
```

**Memories not persisting**: Ensure `mem.augmentation.wait()` is called before your process exits (especially in short-lived scripts).

**Empty recall**: Make sure `mem.attribution(entity_id=...)` is called before any LLM interactions.
