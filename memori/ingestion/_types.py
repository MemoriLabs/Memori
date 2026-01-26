r"""
 __  __                           _
|  \/  | ___ _ __ ___   ___  _ __(_)
| |\/| |/ _ \ '_ ` _ \ / _ \| '__| |
| |  | |  __/ | | | | | (_) | |  | |
|_|  |_|\___|_| |_| |_|\___/|_|  |_|
                  perfectam memoriam
                       memorilabs.ai
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

DEFAULT_MAX_MESSAGES_PER_REQUEST = 4000
DEFAULT_MAX_CHARS_PER_REQUEST = 800_000
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_DELAY = 2.0


class SeedType(Enum):
    CONVERSATION = "conversation"


@dataclass
class SeedData:
    seed_type: SeedType
    entity_id: str
    process_id: str | None
    data: Any

    @classmethod
    def for_conversations(
        cls,
        entity_id: str,
        conversations: list[dict[str, Any]],
        process_id: str | None = None,
    ) -> "SeedData":
        return cls(
            seed_type=SeedType.CONVERSATION,
            entity_id=entity_id,
            process_id=process_id,
            data=conversations,
        )


@dataclass
class SeedResult:
    total: int = 0
    successful: int = 0
    failed: int = 0
    total_triples: int = 0
    duration_ms: float = 0
    conversations: list["ConversationResult"] = field(default_factory=list)
    chunked_conversations: int = 0

    @property
    def success_rate(self) -> float:
        if self.total == 0:
            return 0.0
        return (self.successful / self.total) * 100


@dataclass
class ConversationResult:
    conversation_id: str
    success: bool
    triples_count: int = 0
    summary: str | None = None
    error: str | None = None
    duration_ms: float = 0
    chunks_processed: int = 1
    warnings: list[str] = field(default_factory=list)


IngestResult = SeedResult


@dataclass
class SeedConfig:
    max_messages_per_request: int = DEFAULT_MAX_MESSAGES_PER_REQUEST
    max_chars_per_request: int = DEFAULT_MAX_CHARS_PER_REQUEST
    max_retries: int = DEFAULT_MAX_RETRIES
    retry_delay: float = DEFAULT_RETRY_DELAY
    chunk_large_conversations: bool = True


IngestionConfig = SeedConfig
