# Memori Developer Relations

Memori is an open-source, SQL-native memory layer for AI agents. It captures conversations, extracts structured facts, and enables semantic recall across entities, processes, and sessions with minimal code changes.

This repository is the DevRel hub for Memori. It includes programs, workshops, tutorials, and assets that help developers get to their first successful memory-backed interaction quickly.

## Quickstart (Python)

Prereqs
- Python 3.10+
- An OpenAI API key

Install

```
pip install memori
pip install openai
```

Set env var

```
export OPENAI_API_KEY="your-api-key-here"
```

Minimal example

```python
import os
import sqlite3

from memori import Memori
from openai import OpenAI


def get_sqlite_connection():
    return sqlite3.connect("memori.db")


client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
memori = Memori(conn=get_sqlite_connection).llm.register(client)

# Required: attribution tells Memori who and what this memory belongs to.
memori.attribution(entity_id="123456", process_id="test-ai-agent")

# Build the schema once per database.
memori.config.storage.build()

# Write memory through normal LLM usage.
response = client.chat.completions.create(
    model="gpt-4.1-mini",
    messages=[{"role": "user", "content": "My favorite color is blue."}],
)
print(response.choices[0].message.content + "\n")

# For short-lived scripts, wait for background augmentation to finish.
memori.augmentation.wait()

# Recall stored facts.
facts = memori.recall("favorite color", limit=5)
print(facts)
```

Optional: verify the stored facts directly in SQLite:

```
sqlite3 memori.db "select * from memori_entity_fact;"
```

Optional one-time setup (pre-downloads embeddings for faster first run):

```
python -m memori setup
```

Optional: set a Memori API key for higher Advanced Augmentation limits:

```
export MEMORI_API_KEY="your-memori-key-here"
```

## Core Concepts

- Entity: a person, place, or thing (for example, a user).
- Process: your agent, workflow, or program.
- Session: a group of related interactions.
- Augmentation: background memory enrichment (no extra latency).

## Supported Providers and Datastores

Memori is LLM, database, and framework agnostic, with first-class support for major providers and SQL databases. Today that includes OpenAI, Anthropic, Bedrock, Gemini, and Grok; frameworks like Agno and LangChain; and any DB API 2.0 compatible database driver (such as Postgres, MySQL, SQLite, and more). See the docs for the current list of providers, frameworks, and datastore integrations.

## Docs and Examples

- Docs and API reference: https://memorilabs.ai/docs
- Quickstart: https://memorilabs.ai/docs/getting-started/quick-start
- Troubleshooting: https://memorilabs.ai/docs/troubleshooting
- Cookbook: https://github.com/MemoriLabs/memori-cookbook
- SDK repository: https://github.com/MemoriLabs/Memori

## Developer Programs and Materials (in this repo)

- Programs overview: docs/
- Workshops: workshops/
- Tutorials: tutorials/
- Templates: templates/
- Assets: assets/

## Contributing and Support

- Issues: https://github.com/MemoriLabs/Memori/issues
- Discord: https://discord.gg/abD4eGym6v

If you are new to Memori, start with the Quickstart above, then jump into the cookbook for complete, runnable examples.
