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
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from memori._config import Config
from memori._network import Api
from memori.embeddings import embed_texts
from memori.memory._struct import Memories
from memori.memory.augmentation._models import (
    AttributionData,
    AugmentationPayload,
    ConversationData,
    EntityData,
    FrameworkData,
    LlmData,
    MetaData,
    ModelData,
    PlatformData,
    ProcessData,
    SdkData,
    SdkVersionData,
    StorageData,
    hash_id,
)

logger = logging.getLogger(__name__)

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


class ValidationError(Exception):
    pass


def validate_message(message: Any, index: int) -> list[str]:
    errors = []

    if not isinstance(message, dict):
        errors.append(
            f"Message {index}: must be a dictionary, got {type(message).__name__}"
        )
        return errors

    if "role" not in message:
        errors.append(f"Message {index}: missing required field 'role'")
    elif message["role"] not in ("user", "assistant", "system"):
        errors.append(
            f"Message {index}: invalid role '{message['role']}', must be 'user', 'assistant', or 'system'"
        )

    if "content" not in message:
        errors.append(f"Message {index}: missing required field 'content'")
    elif not isinstance(message["content"], str):
        errors.append(
            f"Message {index}: 'content' must be a string, got {type(message['content']).__name__}"
        )
    elif len(message["content"].strip()) == 0:
        errors.append(f"Message {index}: 'content' is empty")

    return errors


def validate_conversation(conversation: Any) -> tuple[bool, list[str]]:
    errors = []

    if not isinstance(conversation, dict):
        return False, [
            f"Conversation must be a dictionary, got {type(conversation).__name__}"
        ]

    messages = conversation.get("messages", [])

    if not messages:
        errors.append("Conversation has no messages")
        return False, errors

    if not isinstance(messages, list):
        errors.append(f"'messages' must be a list, got {type(messages).__name__}")
        return False, errors

    for i, msg in enumerate(messages):
        msg_errors = validate_message(msg, i)
        errors.extend(msg_errors)

    return len(errors) == 0, errors


def estimate_conversation_size(messages: list[dict[str, Any]]) -> tuple[int, int]:
    total_chars = sum(len(m.get("content", "")) for m in messages)
    return len(messages), total_chars


class IngestionClient:
    def __init__(
        self,
        config: Config,
        driver,
        entity_id: str,
        process_id: str | None = None,
        batch_size: int = 10,
        ingestion_config: IngestionConfig | None = None,
        on_progress: Callable[[int, int, ConversationResult], None] | None = None,
    ):
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")

        self.config = config
        self.driver = driver
        self.entity_id = entity_id
        self.process_id = process_id
        self.batch_size = batch_size
        self.ingestion_config = ingestion_config or IngestionConfig()
        self.on_progress = on_progress
        self.api = Api(config)

    def _build_payload(
        self, messages: list[dict[str, Any]], summary: str | None = None
    ) -> dict[str, Any]:
        dialect = self.driver.conversation.conn.get_dialect()

        conversation = ConversationData(
            messages=messages,
            summary=summary,
        )

        meta = MetaData(
            attribution=AttributionData(
                entity=EntityData(id=hash_id(self.entity_id)),
                process=ProcessData(id=hash_id(self.process_id)),
            ),
            framework=FrameworkData(provider=self.config.framework.provider or "seed"),
            llm=LlmData(
                model=ModelData(
                    provider=self.config.llm.provider or "seed",
                    sdk=SdkVersionData(
                        version=self.config.llm.provider_sdk_version or "1.0.0"
                    ),
                    version=self.config.llm.version or "seed",
                )
            ),
            platform=PlatformData(provider=self.config.platform.provider or "seed"),
            sdk=SdkData(lang="python", version=self.config.version),
            storage=StorageData(
                cockroachdb=self.config.storage_config.cockroachdb,
                dialect=dialect,
            ),
        )

        payload = AugmentationPayload(conversation=conversation, meta=meta)
        return payload.to_dict()

    def _should_chunk_conversation(self, messages: list[dict[str, Any]]) -> bool:
        msg_count, total_chars = estimate_conversation_size(messages)

        return (
            msg_count > self.ingestion_config.max_messages_per_request
            or total_chars > self.ingestion_config.max_chars_per_request
        )

    def _chunk_messages(
        self, messages: list[dict[str, Any]]
    ) -> list[list[dict[str, Any]]]:
        max_msgs = self.ingestion_config.max_messages_per_request
        max_chars = self.ingestion_config.max_chars_per_request

        chunks = []
        current_chunk = []
        current_chars = 0

        for msg in messages:
            msg_chars = len(msg.get("content", ""))

            would_exceed_msgs = len(current_chunk) >= max_msgs
            would_exceed_chars = current_chars + msg_chars > max_chars and current_chunk

            if would_exceed_msgs or would_exceed_chars:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = [msg]
                current_chars = msg_chars
            else:
                current_chunk.append(msg)
                current_chars += msg_chars

        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    async def _call_aa_with_retry(self, payload: dict[str, Any]) -> dict[str, Any]:
        last_error: Exception | None = None

        for attempt in range(self.ingestion_config.max_retries):
            try:
                response = await self.api.augmentation_async(payload)
                return response
            except Exception as e:
                last_error = e
                error_str = str(e).lower()

                if "422" in error_str or "validation" in error_str:
                    raise
                if "quota" in error_str or "429" in error_str:
                    raise

                if attempt < self.ingestion_config.max_retries - 1:
                    delay = self.ingestion_config.retry_delay * (2**attempt)
                    logger.warning(
                        f"AA request failed (attempt {attempt + 1}/{self.ingestion_config.max_retries}), "
                        f"retrying in {delay:.1f}s: {e}"
                    )
                    await asyncio.sleep(delay)

        if last_error is not None:
            raise last_error
        raise RuntimeError("No retries attempted")

    async def _process_single_conversation(
        self, conversation: dict[str, Any]
    ) -> ConversationResult:
        conv_id = conversation.get("id")
        messages = conversation.get("messages", [])

        if not conv_id:
            return ConversationResult(
                conversation_id="<missing>",
                success=False,
                error="Conversation missing required 'id' field",
            )

        conv_id = str(conv_id)

        is_valid, validation_errors = validate_conversation(conversation)
        if not is_valid:
            return ConversationResult(
                conversation_id=conv_id,
                success=False,
                error=f"Validation failed: {'; '.join(validation_errors)}",
            )

        start = time.perf_counter()
        warnings = []

        try:
            if self._should_chunk_conversation(messages):
                if not self.ingestion_config.chunk_large_conversations:
                    msg_count, char_count = estimate_conversation_size(messages)
                    return ConversationResult(
                        conversation_id=conv_id,
                        success=False,
                        error=f"Conversation too large ({msg_count} messages, {char_count} chars) and chunking disabled",
                        duration_ms=(time.perf_counter() - start) * 1000,
                    )

                return await self._process_chunked_conversation(
                    conv_id, messages, start
                )

            payload = self._build_payload(messages)
            response = await self._call_aa_with_retry(payload)

            duration = (time.perf_counter() - start) * 1000

            if not response:
                return ConversationResult(
                    conversation_id=conv_id,
                    success=False,
                    error="Empty response from AA",
                    duration_ms=duration,
                )

            memories = await self._process_response(response)

            storage_error = await self._store_memories(conv_id, memories)
            if storage_error:
                return ConversationResult(
                    conversation_id=conv_id,
                    success=False,
                    error=f"Storage failed: {storage_error}",
                    duration_ms=duration,
                )

            triples_count = len(memories.entity.semantic_triples or [])
            summary = memories.conversation.summary

            return ConversationResult(
                conversation_id=conv_id,
                success=True,
                triples_count=triples_count,
                summary=summary,
                duration_ms=duration,
                warnings=warnings,
            )

        except Exception as e:
            duration = (time.perf_counter() - start) * 1000
            logger.error(f"Failed to process conversation {conv_id}: {e}")
            return ConversationResult(
                conversation_id=conv_id,
                success=False,
                error=str(e),
                duration_ms=duration,
            )

    async def _process_chunked_conversation(
        self, conv_id: str, messages: list[dict[str, Any]], start_time: float
    ) -> ConversationResult:
        chunks = self._chunk_messages(messages)
        logger.info(f"Conversation {conv_id} split into {len(chunks)} chunks")

        all_triples = []
        current_summary = None
        warnings = [
            f"Large conversation processed in {len(chunks)} chunks (some entity resolution quality may be lost)"
        ]
        successful_chunks = 0

        for i, chunk in enumerate(chunks):
            try:
                payload = self._build_payload(chunk, summary=current_summary)
                response = await self._call_aa_with_retry(payload)

                if not response:
                    warnings.append(f"Chunk {i + 1} returned empty response")
                    continue

                entity_data = response.get("entity", {})
                triples = entity_data.get("triples", [])
                all_triples.extend(triples)

                conv_data = response.get("conversation", {})
                current_summary = conv_data.get("summary")
                successful_chunks += 1

            except Exception as e:
                logger.warning(f"Chunk {i + 1}/{len(chunks)} failed for {conv_id}: {e}")
                warnings.append(f"Chunk {i + 1} failed: {str(e)[:50]}")

        if successful_chunks == 0:
            duration = (time.perf_counter() - start_time) * 1000
            return ConversationResult(
                conversation_id=conv_id,
                success=False,
                error=f"All {len(chunks)} chunks failed to process",
                duration_ms=duration,
                chunks_processed=0,
                warnings=warnings,
            )

        combined_response = {
            "entity": {"triples": all_triples},
            "conversation": {"summary": current_summary},
            "process": {"attributes": []},
        }

        if all_triples:
            facts = [
                f"{t['subject']['name']} {t['predicate']} {t['object']['name']}"
                for t in all_triples
                if t.get("subject") and t.get("predicate") and t.get("object")
            ]
            if facts:
                embeddings_config = self.config.embeddings
                embeddings = await embed_texts(
                    facts,
                    model=embeddings_config.model,
                    async_=True,
                )
                combined_response["entity"]["facts"] = facts
                combined_response["entity"]["fact_embeddings"] = embeddings

        memories = Memories().configure_from_advanced_augmentation(combined_response)

        storage_error = await self._store_memories(conv_id, memories)

        duration = (time.perf_counter() - start_time) * 1000

        if storage_error:
            return ConversationResult(
                conversation_id=conv_id,
                success=False,
                error=f"Storage failed: {storage_error}",
                duration_ms=duration,
                chunks_processed=len(chunks),
                warnings=warnings,
            )

        return ConversationResult(
            conversation_id=conv_id,
            success=True,
            triples_count=len(all_triples),
            summary=current_summary,
            duration_ms=duration,
            chunks_processed=len(chunks),
            warnings=warnings,
        )

    async def _process_response(self, response: dict[str, Any]) -> Memories:
        entity_data = response.get("entity", {})
        facts = entity_data.get("facts", [])
        triples = entity_data.get("triples", [])

        if not facts and triples:
            facts = [
                f"{t['subject']['name']} {t['predicate']} {t['object']['name']}"
                for t in triples
                if t.get("subject") and t.get("predicate") and t.get("object")
            ]

        if facts:
            embeddings_config = self.config.embeddings
            fact_embeddings = await embed_texts(
                facts,
                model=embeddings_config.model,
                async_=True,
            )
            response["entity"]["fact_embeddings"] = fact_embeddings
            response["entity"]["facts"] = facts

        return Memories().configure_from_advanced_augmentation(response)

    async def _store_memories(self, conv_id: str, memories: Memories) -> str | None:
        entity_id = self.driver.entity.create(self.entity_id)
        if not entity_id:
            error_msg = f"Failed to create entity for {self.entity_id}"
            logger.error(error_msg)
            return error_msg

        facts = memories.entity.facts
        embeddings = memories.entity.fact_embeddings

        if memories.entity.semantic_triples and (not facts or not embeddings):
            facts_from_triples = [
                f"{triple.subject_name} {triple.predicate} {triple.object_name}"
                for triple in memories.entity.semantic_triples
            ]

            if facts_from_triples:
                embeddings_config = self.config.embeddings
                embeddings_from_triples = await embed_texts(
                    facts_from_triples,
                    model=embeddings_config.model,
                    async_=True,
                )
                facts = (facts or []) + facts_from_triples
                embeddings = (embeddings or []) + embeddings_from_triples

        if facts and embeddings:
            self.driver.entity_fact.create(entity_id, facts, embeddings)

        if memories.entity.semantic_triples:
            self.driver.knowledge_graph.create(
                entity_id, memories.entity.semantic_triples
            )

        if self.process_id:
            process_id = self.driver.process.create(self.process_id)
            if process_id and memories.process.attributes:
                self.driver.process_attribute.create(
                    process_id, memories.process.attributes
                )

        if memories.conversation.summary:
            self.driver.conversation.update(conv_id, memories.conversation.summary)

        return None

    async def ingest(self, conversations: list[dict[str, Any]]) -> IngestResult:
        start = time.perf_counter()

        conv_ids = [c.get("id") for c in conversations if c.get("id")]
        seen = set()
        duplicates = []
        for cid in conv_ids:
            if cid in seen:
                duplicates.append(cid)
            seen.add(cid)

        if duplicates:
            raise ValueError(
                f"Duplicate conversation IDs found: {duplicates[:5]}{'...' if len(duplicates) > 5 else ''}"
            )

        result = IngestResult(total=len(conversations))

        for batch_start in range(0, len(conversations), self.batch_size):
            batch = conversations[batch_start : batch_start + self.batch_size]

            tasks = [self._process_single_conversation(conv) for conv in batch]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            for i, batch_result in enumerate(batch_results):
                if isinstance(batch_result, BaseException):
                    conv_result = ConversationResult(
                        conversation_id=batch[i].get("id", "unknown"),
                        success=False,
                        error=str(batch_result),
                    )
                else:
                    conv_result = batch_result

                result.conversations.append(conv_result)

                if conv_result.success:
                    result.successful += 1
                    result.total_triples += conv_result.triples_count
                    if conv_result.chunks_processed > 1:
                        result.chunked_conversations += 1
                else:
                    result.failed += 1

                if self.on_progress:
                    processed = batch_start + i + 1
                    self.on_progress(processed, len(conversations), conv_result)

        result.duration_ms = (time.perf_counter() - start) * 1000
        return result


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
