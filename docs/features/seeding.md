[![Memori Labs](https://s3.us-east-1.amazonaws.com/images.memorilabs.ai/banner.png)](https://memorilabs.ai/)

# Memory Seeding

Bulk import historical conversations to bootstrap user memories. Perfect for onboarding users with existing chat history from other platforms.

## Overview

Memory seeding processes historical conversations through Memori's Advanced Augmentation service to extract facts, preferences, skills, and events - just like real-time conversations, but in bulk.

> **Important:**
> - **API Key Required**: Memory seeding requires a valid `MEMORI_API_KEY` to access the Advanced Augmentation service.
> - **Quota Usage**: Each conversation seeded counts against your Memori creation quota, just like real-time memory creation.

### How It Works

```mermaid
graph LR
    A[Historical Conversations] --> B[Seed API]
    B --> C[Batched Processing]
    C --> D[AA Service]
    D --> E[Fact Extraction]
    E --> F[Your Database]
```

1. **Prepare Data**: Format conversations as `SeedData`
2. **Batch Processing**: Conversations processed in parallel batches
3. **AA Extraction**: Each conversation analyzed for facts/preferences
4. **Storage**: Extracted memories stored in your database

## Quick Start

### Async (Recommended)

```python
import asyncio
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from memori import Memori, SeedData

engine = create_engine("sqlite:///memori.db")
Session = sessionmaker(bind=engine)

conversations = [
    {
        "id": "conv-1",
        "messages": [
            {"role": "user", "content": "I just bought a Tesla Model 3!"},
            {"role": "assistant", "content": "Congratulations! That's exciting."},
        ],
    },
]

async def main():
    m = Memori(conn=Session)
    m.attribution(entity_id="user-123")
    m.config.storage.build()

    seed_data = SeedData.for_conversations(
        entity_id="user-123",
        conversations=conversations,
    )

    result = await m.seed(seed_data)
    print(f"Processed: {result.successful}/{result.total}")

asyncio.run(main())
```

### Sync

```python
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from memori import Memori, SeedData

engine = create_engine("sqlite:///memori.db")
Session = sessionmaker(bind=engine)

m = Memori(conn=Session)
m.attribution(entity_id="user-123")
m.config.storage.build()

seed_data = SeedData.for_conversations(
    entity_id="user-123",
    conversations=conversations,
)

result = m.seed_sync(seed_data)
```

## API Reference

### SeedData

Wrapper for conversations to be seeded.

```python
seed_data = SeedData.for_conversations(
    entity_id="user-123",           # Required: who owns these memories
    conversations=conversations,     # Required: list of conversation dicts
    process_id="onboarding",        # Optional: source identifier
)
```

**Conversation Format:**

```python
{
    "id": "unique-conversation-id",  # Required: unique identifier
    "messages": [                    # Required: list of messages
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hi there!"},
    ],
}
```

### SeedConfig

Configuration options for seeding behavior.

```python
from memori import SeedConfig

config = SeedConfig(
    max_messages_per_request=4000,   # Max messages per AA request
    max_chars_per_request=800_000,   # Max characters per AA request
    max_retries=3,                   # Retry attempts on failure
    retry_delay=1.0,                 # Seconds between retries
    chunk_large_conversations=True,  # Auto-split large conversations
)

result = await m.seed(seed_data, seed_config=config)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_messages_per_request` | 4000 | Maximum messages per single AA request |
| `max_chars_per_request` | 800,000 | Maximum characters per single AA request |
| `max_retries` | 3 | Number of retry attempts on failure |
| `retry_delay` | 1.0 | Delay between retries (seconds) |
| `chunk_large_conversations` | True | Automatically split large conversations |

### SeedResult

Result object returned from seeding operations.

```python
result = await m.seed(seed_data)

print(f"Total: {result.total}")
print(f"Successful: {result.successful}")
print(f"Failed: {result.failed}")
print(f"Triples extracted: {result.total_triples}")
print(f"Duration: {result.duration_ms:.0f}ms")
```

| Property | Type | Description |
|----------|------|-------------|
| `total` | int | Total conversations processed |
| `successful` | int | Successfully processed count |
| `failed` | int | Failed processing count |
| `total_triples` | int | Total facts/memories extracted |
| `duration_ms` | float | Total processing time in milliseconds |
| `results` | list | Per-conversation results |

## Batch Processing

Control parallelism with the `batch_size` parameter:

```python
result = await m.seed(
    seed_data,
    batch_size=10,  # Process 10 conversations concurrently
)
```

Higher batch sizes = faster processing, but more concurrent API calls.

## Progress Tracking

Monitor progress with a callback function:

```python
def on_progress(processed, total, result):
    status = "✓" if result.success else "✗"
    print(f"[{processed}/{total}] {status} {result.conversation_id}")

result = await m.seed(
    seed_data,
    on_progress=on_progress,
)
```

## Large Conversation Handling

Conversations exceeding limits are automatically chunked:

1. **First chunk**: Messages up to the limit
2. **Summary generation**: AI summarizes the first chunk
3. **Next chunk**: Summary prepended to next batch of messages
4. **Repeat**: Until all messages processed

This preserves context across chunks while respecting API limits.

### Limits

| Limit | Default | Description |
|-------|---------|-------------|
| Messages per request | 4000 | Maximum messages in a single AA request |
| Characters per request | 800,000 | Maximum total characters |

To disable auto-chunking:

```python
config = SeedConfig(chunk_large_conversations=False)
```

## CLI Usage

Seed from a JSON file:

```bash
python -m memori seed path/to/conversations.json --entity-id user-123
```

**JSON file format:**

```json
[
    {
        "id": "conv-1",
        "messages": [
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi!"}
        ]
    }
]
```

## Example: Seed and Chat

After seeding, memories are immediately available:

```python
import asyncio
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from memori import Memori, SeedData
from openai import OpenAI

engine = create_engine("sqlite:///memori.db")
Session = sessionmaker(bind=engine)

conversations = [
    {
        "id": "conv-1",
        "messages": [
            {"role": "user", "content": "I'm a software engineer at Google."},
            {"role": "assistant", "content": "Nice! What do you work on?"},
            {"role": "user", "content": "I work on the search team."},
        ],
    },
]

async def main():
    # Seed memories
    m = Memori(conn=Session)
    m.attribution(entity_id="user-123")
    m.config.storage.build()

    seed_data = SeedData.for_conversations(
        entity_id="user-123",
        conversations=conversations,
    )
    await m.seed(seed_data)

    # Chat with memories
    client = OpenAI()
    m = Memori(conn=Session).llm.register(client)
    m.attribution(entity_id="user-123")

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": "Where do I work?"}],
    )
    print(response.choices[0].message.content)
    # Output: You work at Google on the search team!

asyncio.run(main())
```

## Best Practices

1. **Batch size**: Start with 10, increase if your rate limits allow
2. **Error handling**: Check `result.failed` and `result.results` for failures
3. **Large imports**: Use progress callback to monitor long-running seeds
4. **Deduplication**: Use unique conversation IDs to avoid duplicate processing
5. **Attribution**: Set `entity_id` to properly scope memories to users
