[![Memori Labs](https://s3.us-east-1.amazonaws.com/images.memorilabs.ai/banner.png)](https://memorilabs.ai/)

# Memory Seeding

Bulk import historical conversations to bootstrap user memories.

## Quick Start

```python
import asyncio
from memori import Memori, SeedData
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

engine = create_engine("sqlite:///memori.db")
Session = sessionmaker(bind=engine)

conversations = [
    {
        "id": "conv-1",
        "messages": [
            {"role": "user", "content": "I just bought a Tesla Model 3!"},
            {"role": "assistant", "content": "Congratulations!"},
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

## Conversation Format

```python
{
    "id": "unique-id",        # Required
    "messages": [             # Required
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hi!"},
    ],
}
```

## Configuration

```python
from memori import SeedConfig

config = SeedConfig(
    max_messages_per_request=4000,
    max_chars_per_request=800_000,
    max_retries=3,
    retry_delay=1.0,
    chunk_large_conversations=True,
)

result = await m.seed(seed_data, seed_config=config)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_messages_per_request` | 4000 | Max messages per AA request |
| `max_chars_per_request` | 800,000 | Max characters per AA request |
| `max_retries` | 3 | Retry attempts on failure |
| `retry_delay` | 1.0 | Seconds between retries |
| `chunk_large_conversations` | True | Auto-split large conversations |

## Result

```python
result = await m.seed(seed_data)

print(result.total)           # Total conversations
print(result.successful)      # Successful count
print(result.failed)          # Failed count
print(result.total_triples)   # Facts extracted
print(result.duration_ms)     # Processing time (ms)
```

## Progress Tracking

```python
def on_progress(processed, total, result):
    status = "✓" if result.success else "✗"
    print(f"[{processed}/{total}] {status} {result.conversation_id}")

result = await m.seed(seed_data, on_progress=on_progress)
```

## CLI

```bash
python -m memori seed conversations.json --entity-id user-123
```

**File format:**

```json
{
  "entity_id": "user-123",
  "conversations": [
    {"id": "conv-1", "messages": [{"role": "user", "content": "Hello!"}]}
  ]
}
```

## Requirements

- `MEMORI_API_KEY` environment variable
- Each conversation counts against your quota
