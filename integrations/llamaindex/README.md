# Memori × LlamaIndex Integration

Persistent long-term memory for LlamaIndex agents and chat engines, powered by [Memori](https://memorilabs.ai).

## Why?

By default, LlamaIndex agents forget everything between sessions.  Memori fixes that — it automatically captures structured memory from every conversation turn and makes it available on recall, so your agents remember users, decisions, and context across sessions.

## How it Works

Memori integrates at the **LLM client level**, not at the framework callback level.  When you wrap your chat engine or agent with `MemoriChatEngine` / `MemoriAgent`, Memori patches the underlying OpenAI / Anthropic / Gemini SDK client transparently.

- **Before each LLM call**: relevant memories are injected into the system prompt automatically.
- **After each LLM call**: the conversation turn is stored in Memori for future recall.

No changes to your prompts or agent code are needed.

## Installation

```bash
pip install memori-llamaindex

# With LLM extras (install the LlamaIndex LLM packages you use):
pip install "memori-llamaindex[openai]"
pip install "memori-llamaindex[anthropic]"
pip install "memori-llamaindex[gemini]"
```

## Quick Start

### Chat Engine

```python
import os
from llama_index.core.chat_engine import SimpleChatEngine
from llama_index.llms.openai import OpenAI
from memori import Memori
from memori.integrations.llamaindex import MemoriChatEngine

# Requires MEMORI_API_KEY and OPENAI_API_KEY in environment
llm    = OpenAI(model="gpt-4o-mini")
engine = SimpleChatEngine.from_defaults(llm=llm)
mem    = Memori()

memory_engine = MemoriChatEngine(
    chat_engine=engine,
    memori=mem,
    entity_id="user_123",
    process_id="support-bot",
)

response = memory_engine.chat("My favourite colour is blue.")
print(response)

# In a later session — Memori recalls the user's colour preference.
response = memory_engine.chat("What's my favourite colour?")
print(response)

# Wait for background augmentation to finish before exiting.
mem.augmentation.wait()
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

response = memory_agent.chat("Remember that my project deadline is June 30.")
```

### Direct LLM registration (lowest-level API)

If you'd rather not use the wrappers, register your LlamaIndex LLM directly:

```python
from llama_index.llms.openai import OpenAI
from memori import Memori

llm = OpenAI(model="gpt-4o-mini")
mem = Memori()
mem.llm.register(llm)                                 # auto-detects LlamaIndex
mem.attribution(entity_id="user_123", process_id="my-app")
```

All subsequent calls through this `llm` object will automatically have memory injected and captured.

## Configuration

Use `MemoriLlamaIndexConfig` for structured configuration:

```python
from memori.integrations.llamaindex import MemoriChatEngine, MemoriLlamaIndexConfig

config = MemoriLlamaIndexConfig(
    entity_id="user_123",
    process_id="support-bot",
    retrieval_k=5,
    project_id="my-cloud-project",   # Memori Cloud only
)

engine = MemoriChatEngine(chat_engine=base_engine, memori=mem, **vars(config))
```

| Parameter | Default | Description |
|---|---|---|
| `entity_id` | *(required)* | Identifies the user/entity |
| `process_id` | `"llamaindex"` | Identifies the agent/process |
| `retrieval_k` | `5` | Max memory facts recalled per query |
| `auto_retrieve_memories` | `True` | Inject memories into prompts |
| `enable_chat_memory` | `True` | Enable memory for chat engines |
| `enable_agent_memory` | `True` | Enable memory for agents |
| `session_id` | `None` | Explicit session ID (auto-generated if not set) |
| `project_id` | `None` | Memori Cloud project ID for `capture_agent_turn` |

## Optional: Memori Cloud Callback Handler

For Memori Cloud users who want explicit per-turn records:

```python
from llama_index.core.callbacks import CallbackManager
from memori.integrations.llamaindex import MemoriCallbackHandler

handler = MemoriCallbackHandler(mem, project_id="my-project")
callback_manager = CallbackManager([handler])
```

## Supported LlamaIndex LLMs

| LlamaIndex LLM | Package |
|---|---|
| `llama_index.llms.openai.OpenAI` | `llama-index-llms-openai` |
| `llama_index.llms.anthropic.Anthropic` | `llama-index-llms-anthropic` |
| `llama_index.llms.gemini.Gemini` | `llama-index-llms-gemini` |

## License

Apache 2.0
