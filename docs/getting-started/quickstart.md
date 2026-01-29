[![Memori Labs](https://s3.us-east-1.amazonaws.com/images.memorilabs.ai/banner.png)](https://memorilabs.ai/)

# Quickstart

Get started with Memori in under 3 minutes.

Memori is LLM, database and framework agnostic and works with the tools you already use today. In this quickstart, we'll show Memori working with SQLite and **either OpenAI or Anthropic**.

- [Supported LLM providers](https://github.com/MemoriLabs/Memori/blob/main/docs/features/llm.md)
- [Supported databases](https://github.com/MemoriLabs/Memori/blob/main/docs/features/databases.md)

## Prerequisites

- Python 3.10 or higher
- An API key for **OpenAI or Anthropic** (this quickstart includes both options)

## Step 1: Install Libraries

Install Memori:

```bash
pip install memori
```

For this example, install **one** of the following (depending on your LLM provider):

```bash
pip install openai
# or
pip install anthropic
```

> If you use a virtualenv on macOS/Homebrew Python, install `sqlalchemy` as well to avoid a first-run import error:
>
> ```bash
> pip install sqlalchemy
> ```

## Step 2: Set environment variables

Choose one option:

### Option A: OpenAI

```bash
export OPENAI_API_KEY="your-api-key-here"
```

### Option B: Anthropic

```bash
export ANTHROPIC_API_KEY="your-api-key-here"
```

(Optional) Memori Labs API key for Advanced Augmentation:

```bash
export MEMORI_API_KEY="your-memori-api-key-here"
```

## Step 3: Run Your First Memori Application

Create a new Python file `quickstart.py`.

### Option A: OpenAI quickstart

```python
import os
import sqlite3

from memori import Memori
from openai import OpenAI


def get_sqlite_connection():
    return sqlite3.connect("memori.db")


client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

memori = Memori(conn=get_sqlite_connection).llm.register(client)
memori.attribution(entity_id="123456", process_id="test-ai-agent")
memori.config.storage.build()

client.chat.completions.create(
    model="gpt-4.1-mini",
    messages=[{"role": "user", "content": "My favorite color is blue."}],
)

# Advanced Augmentation runs asynchronously; short-lived scripts should wait.
memori.augmentation.wait()

# Reset everything so there's no prior context.
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
memori = Memori(conn=get_sqlite_connection).llm.register(client)
memori.attribution(entity_id="123456", process_id="test-ai-agent")

client.chat.completions.create(
    model="gpt-4.1-mini",
    messages=[{"role": "user", "content": "What's my favorite color?"}],
)
```

### Option B: Anthropic quickstart

```python
import os
import sqlite3

from anthropic import Anthropic
from memori import Memori


def get_sqlite_connection():
    return sqlite3.connect("memori.db")


client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

memori = Memori(conn=get_sqlite_connection).llm.register(client)
memori.attribution(entity_id="123456", process_id="test-ai-agent")
memori.config.storage.build()

client.messages.create(
    model="claude-3-5-sonnet-latest",
    max_tokens=256,
    messages=[{"role": "user", "content": "My favorite color is blue."}],
)

memori.augmentation.wait()

# Reset everything so there's no prior context.
client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
memori = Memori(conn=get_sqlite_connection).llm.register(client)
memori.attribution(entity_id="123456", process_id="test-ai-agent")

client.messages.create(
    model="claude-3-5-sonnet-latest",
    max_tokens=256,
    messages=[{"role": "user", "content": "What's my favorite color?"}],
)
```

## Step 4: Run the Application

Execute your Python file:

```bash
python quickstart.py
```

You should see the AI respond to both questions, with the second response correctly recalling that your favorite color is blue!

## Step 5: Check the memories created

```bash
/bin/echo "select * from memori_conversation_message" | /usr/bin/sqlite3 memori.db
/bin/echo "select * from memori_entity_fact" | /usr/bin/sqlite3 memori.db
/bin/echo "select * from memori_process_attribute" | /usr/bin/sqlite3 memori.db
/bin/echo "select * from memori_knowledge_graph" | /usr/bin/sqlite3 memori.db
```

## What Just Happened?

1. **Setup**: You initialized Memori with a SQLite database and registered your LLM client (OpenAI or Anthropic)
2. **Attribution**: You identified the user (`entity_id`) and application (`process_id`) for context tracking
3. **Storage**: The database schema was automatically created
4. **Memory in Action**: Memori captured the first interaction so it can be recalled later
