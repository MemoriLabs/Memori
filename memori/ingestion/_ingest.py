r"""
 __  __                           _
|  \/  | ___ _ __ ___   ___  _ __(_)
| |\/| |/ _ \ '_ ` _ \ / _ \| '__| |
| |  | |  __/ | | | | | (_) | |  | |
|_|  |_|\___|_| |_| |_|\___/|_|  |_|
                  perfectam memoriam
                       memorilabs.ai
"""

import asyncio
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from memori._config import Config
from memori.ingestion._client import IngestionClient
from memori.ingestion._types import (
    DEFAULT_MAX_CHARS_PER_REQUEST,
    DEFAULT_MAX_MESSAGES_PER_REQUEST,
    DEFAULT_MAX_RETRIES,
    DEFAULT_RETRY_DELAY,
    ConversationResult,
    IngestionConfig,
    IngestResult,
    SeedConfig,
    SeedData,
    SeedResult,
    SeedType,
)
from memori.ingestion._validation import (
    estimate_conversation_size,
    validate_conversation,
    validate_message,
)


async def ingest_conversations(
    config: Config,
    driver,
    entity_id: str,
    conversations: list[dict[str, Any]],
    process_id: str | None = None,
    batch_size: int = 10,
    ingestion_config: IngestionConfig | None = None,
    on_progress: Callable[[int, int, ConversationResult], None] | None = None,
) -> IngestResult:
    client = IngestionClient(
        config=config,
        driver=driver,
        entity_id=entity_id,
        process_id=process_id,
        batch_size=batch_size,
        ingestion_config=ingestion_config,
        on_progress=on_progress,
    )

    return await client.ingest(conversations)


def ingest_from_file(
    config: Config,
    driver,
    file_path: str,
    entity_id: str | None = None,
    process_id: str | None = None,
    batch_size: int = 10,
    ingestion_config: IngestionConfig | None = None,
    on_progress: Callable[[int, int, ConversationResult], None] | None = None,
) -> IngestResult:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(path) as f:
        data = json.load(f)

    final_entity_id = entity_id or data.get("entity_id")
    if not final_entity_id:
        raise ValueError("entity_id must be provided either as argument or in file")

    final_process_id = process_id or data.get("process_id")

    conversations = data.get("conversations", [])
    if not conversations:
        raise ValueError("No conversations found in file")

    return asyncio.run(
        ingest_conversations(
            config=config,
            driver=driver,
            entity_id=final_entity_id,
            conversations=conversations,
            process_id=final_process_id,
            batch_size=batch_size,
            ingestion_config=ingestion_config,
            on_progress=on_progress,
        )
    )


seed_conversations = ingest_conversations
seed_from_file = ingest_from_file

__all__ = [
    "DEFAULT_MAX_CHARS_PER_REQUEST",
    "DEFAULT_MAX_MESSAGES_PER_REQUEST",
    "DEFAULT_MAX_RETRIES",
    "DEFAULT_RETRY_DELAY",
    "ConversationResult",
    "IngestResult",
    "IngestionClient",
    "IngestionConfig",
    "SeedConfig",
    "SeedData",
    "SeedResult",
    "SeedType",
    "estimate_conversation_size",
    "ingest_conversations",
    "ingest_from_file",
    "seed_conversations",
    "seed_from_file",
    "validate_conversation",
    "validate_message",
]
