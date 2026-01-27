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
from memori.seed._client import SeedClient
from memori.seed._types import (
    DEFAULT_MAX_CHARS_PER_REQUEST,
    DEFAULT_MAX_MESSAGES_PER_REQUEST,
    DEFAULT_MAX_RETRIES,
    DEFAULT_RETRY_DELAY,
    ConversationResult,
    SeedConfig,
    SeedData,
    SeedResult,
    SeedType,
    estimate_conversation_size,
    validate_conversation,
    validate_message,
)


async def seed_conversations(
    config: Config,
    driver,
    entity_id: str,
    conversations: list[dict[str, Any]],
    process_id: str | None = None,
    batch_size: int = 10,
    seed_config: SeedConfig | None = None,
    on_progress: Callable[[int, int, ConversationResult], None] | None = None,
) -> SeedResult:
    client = SeedClient(
        config=config,
        driver=driver,
        entity_id=entity_id,
        process_id=process_id,
        batch_size=batch_size,
        seed_config=seed_config,
        on_progress=on_progress,
    )

    return await client.seed(conversations)


def seed_from_file(
    config: Config,
    driver,
    file_path: str,
    entity_id: str | None = None,
    process_id: str | None = None,
    batch_size: int = 10,
    seed_config: SeedConfig | None = None,
    on_progress: Callable[[int, int, ConversationResult], None] | None = None,
) -> SeedResult:
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
        seed_conversations(
            config=config,
            driver=driver,
            entity_id=final_entity_id,
            conversations=conversations,
            process_id=final_process_id,
            batch_size=batch_size,
            seed_config=seed_config,
            on_progress=on_progress,
        )
    )


__all__ = [
    "DEFAULT_MAX_CHARS_PER_REQUEST",
    "DEFAULT_MAX_MESSAGES_PER_REQUEST",
    "DEFAULT_MAX_RETRIES",
    "DEFAULT_RETRY_DELAY",
    "ConversationResult",
    "SeedClient",
    "SeedConfig",
    "SeedData",
    "SeedResult",
    "SeedType",
    "estimate_conversation_size",
    "seed_conversations",
    "seed_from_file",
    "validate_conversation",
    "validate_message",
]
