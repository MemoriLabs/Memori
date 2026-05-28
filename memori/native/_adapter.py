"""Python adapter for the native Rust core engine."""

import base64
import json
import logging
import threading
from dataclasses import dataclass, field
from typing import Any

from memori.memory._struct import SemanticTriple
from memori.native._embeddings import _embed_texts_with_cardinality, embed_texts
from memori.native._errors import RustCoreAdapterError
from memori.native._loader import _normalize_model_name, _try_import_memori_python
from memori.storage._connection import connection_context

logger = logging.getLogger(__name__)


def _embed_entity_facts(
    config: Any, facts_str: list[str], model: str | None
) -> list[list[float]] | None:
    rust_core = getattr(config, "rust_core", None)
    embed_fn = getattr(rust_core, "embed_texts", None)
    if callable(embed_fn):
        try:
            return embed_fn(facts_str, model=model)
        except Exception:  # noqa: BLE001
            logger.exception(
                "Failed to embed AA facts with rust core before write; "
                "falling back without embeddings"
            )
            return None

    try:
        return embed_texts(facts_str, model=model)
    except Exception:  # noqa: BLE001
        logger.exception(
            "Failed to embed AA facts before write; falling back without embeddings"
        )
        return None


@dataclass
class RustCoreAdapter:
    config: Any
    _engine: Any | None = None
    _engine_error: Exception | None = field(default=None, init=False, repr=False)
    _engine_lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False
    )

    @classmethod
    def maybe_create(cls, config: Any) -> "RustCoreAdapter | None":
        if not getattr(config, "byodb", False):
            return None
        if not getattr(config, "use_rust_core", True):
            return None

        storage = getattr(config, "storage", None)
        if storage is None or getattr(storage, "conn_factory", None) is None:
            logger.warning(
                "Rust core enabled but storage connection factory is not ready."
            )
            return None

        return cls(config=config)

    def _create_engine(self) -> Any:
        _try_import_memori_python()
        try:
            from memori_python import EngineHandle  # ty: ignore[unresolved-import]
        except ImportError as exc:
            logger.warning("Rust core unavailable: %s", exc)
            raise RustCoreAdapterError("Rust core is unavailable") from exc
        except Exception as exc:  # noqa: BLE001
            logger.exception("Unexpected error importing memori_python EngineHandle")
            raise RustCoreAdapterError("Rust core import failed") from exc

        engine = EngineHandle(
            _normalize_model_name(
                getattr(getattr(self.config, "embeddings", None), "model", None)
            ),
            self._fetch_embeddings_cb(self.config),
            self._fetch_facts_by_ids_cb(self.config),
            self._write_batch_cb(self.config),
        )
        return engine

    @property
    def _active_engine(self) -> Any:
        if self._engine is not None:
            return self._engine
        if self._engine_error is not None:
            raise self._engine_error

        with self._engine_lock:
            if self._engine is not None:
                return self._engine
            if self._engine_error is not None:
                raise self._engine_error

            try:
                self._engine = self._create_engine()
            except Exception as exc:  # noqa: BLE001
                self._engine_error = exc
                raise
            return self._engine

    def embed_texts(
        self, texts: str | list[str], model: str | None = None
    ) -> list[list[float]]:
        engine = self._engine
        if engine is not None:
            return _embed_texts_with_cardinality(
                texts,
                lambda embeddable: [
                    list(row) for row in engine.embed_texts(embeddable)
                ],
            )
        return embed_texts(texts, model=model)

    def retrieve_facts(
        self,
        *,
        query: str,
        entity_id: str,
        limit: int,
        dense_limit: int,
    ) -> list[dict[str, Any]]:
        payload = {
            "entity_id": entity_id,
            "query_text": query,
            "dense_limit": dense_limit,
            "limit": limit,
        }
        data = self._active_engine.retrieve(json.dumps(payload))
        parsed = _parse_json(data, "retrieve response")
        if not isinstance(parsed, list):
            raise RustCoreAdapterError("retrieve response must be a JSON list")
        return [item for item in parsed if isinstance(item, dict)]

    def recall_text(
        self,
        *,
        query: str,
        entity_id: str,
        limit: int,
        dense_limit: int,
    ) -> str:
        payload = {
            "entity_id": entity_id,
            "query_text": query,
            "dense_limit": dense_limit,
            "limit": limit,
        }
        return self._active_engine.recall(json.dumps(payload))

    def submit_augmentation(
        self,
        *,
        entity_id: str | None,
        process_id: str | None,
        conversation_id: int | str | None,
        conversation_messages: list[dict[str, str]],
        llm_provider: str | None,
        llm_model: str | None,
        llm_provider_sdk_version: str | None,
        framework: str | None,
        platform_provider: str | None,
        storage_dialect: str | None,
        storage_cockroachdb: bool,
        sdk_version: str | None,
    ) -> int:
        resolved_storage_dialect = _resolve_storage_dialect(
            self.config, storage_dialect
        )
        payload = {
            "entity_id": entity_id or "",
            "process_id": process_id,
            "conversation_id": str(conversation_id)
            if conversation_id is not None
            else None,
            "conversation_messages": conversation_messages,
            "llm_provider": llm_provider,
            "llm_model": llm_model,
            "llm_provider_sdk_version": llm_provider_sdk_version,
            "framework": framework,
            "platform_provider": platform_provider,
            "storage_dialect": resolved_storage_dialect,
            "storage_cockroachdb": bool(storage_cockroachdb),
            "sdk_version": sdk_version,
            "session_id": str(getattr(self.config, "session_id", "")),
        }
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "submit_augmentation payload: %s", json.dumps(payload, indent=2)
            )
        result = self._active_engine.submit_augmentation(json.dumps(payload))
        try:
            return int(result)
        except (TypeError, ValueError) as exc:
            raise RustCoreAdapterError(
                f"submit_augmentation returned non-integer job id: {result!r}"
            ) from exc

    def wait_for_augmentation(self, timeout: float | None = None) -> bool:
        if self._engine is None:
            return True
        timeout_ms: int | None = None
        if timeout is not None:
            timeout_ms = max(0, int(timeout * 1000))
        return bool(self._engine.wait_for_augmentation(timeout_ms))

    @staticmethod
    def _fetch_embeddings_cb(config: Any):
        def _callback(request_json: str) -> str:
            request = _parse_json_object(request_json, "fetch_embeddings request")
            raw_entity_id = request.get("entity_id")
            try:
                limit = int(request.get("limit", 1000))
            except (TypeError, ValueError) as exc:
                raise RustCoreAdapterError(
                    "fetch_embeddings.limit must be an integer"
                ) from exc
            with connection_context(config.storage.conn_factory) as (
                _conn,
                _adapter,
                driver,
            ):
                entity_id = _resolve_entity_id(driver, raw_entity_id)
                rows = driver.entity_fact.get_embeddings(entity_id, limit)
                out: list[dict[str, Any]] = []
                for row in rows:
                    fact_id = row.get("id")
                    embedding = row.get("content_embedding")
                    embedding_row = _normalize_embedding_row(fact_id, embedding)
                    if embedding_row is not None:
                        out.append(embedding_row)
                return json.dumps(out)

        return _callback

    @staticmethod
    def _fetch_facts_by_ids_cb(config: Any):
        def _callback(request_json: str) -> str:
            request = _parse_json_object(request_json, "fetch_facts_by_ids request")
            ids = request.get("ids", [])
            if not isinstance(ids, list):
                raise RustCoreAdapterError("fetch_facts_by_ids.ids must be a list")
            with connection_context(config.storage.conn_factory) as (
                _conn,
                _adapter,
                driver,
            ):
                fact_ids = _normalize_fact_ids(ids, driver)
                rows = driver.entity_fact.get_facts_by_ids(fact_ids)
                out = []
                for row in rows:
                    out.append(
                        {
                            "id": _normalize_fact_id(row.get("id")),
                            "content": row.get("content", ""),
                            "date_created": str(row.get("date_created", "")),
                            "summaries": _json_safe(row.get("summaries", [])),
                        }
                    )
                return json.dumps(out)

        return _callback

    @staticmethod
    def _write_batch_cb(config: Any):
        def _callback(batch_json: str) -> str:
            batch = _parse_json_object(batch_json, "write_batch request")
            ops = batch.get("ops", [])
            if not isinstance(ops, list):
                raise RustCoreAdapterError("write_batch.ops must be a list")

            written = 0
            with connection_context(config.storage.conn_factory) as (
                _conn,
                _adapter,
                driver,
            ):
                for op in ops:
                    if not isinstance(op, dict):
                        continue
                    op_type = op.get("op_type")
                    payload = op.get("payload", {})
                    if not isinstance(payload, dict):
                        continue
                    if _apply_write_op(config, driver, op_type, payload):
                        written += 1

            return json.dumps({"written_ops": written})

        return _callback


def _resolve_entity_id(driver: Any, raw_entity_id: Any) -> Any:
    if isinstance(raw_entity_id, int):
        return raw_entity_id
    if isinstance(raw_entity_id, str):
        stripped = raw_entity_id.strip()
        if not stripped:
            raise RustCoreAdapterError("entity_id cannot be empty")
        if stripped.isdigit():
            return int(stripped)
        return _normalize_created_id(driver, driver.entity.create(stripped))
    if raw_entity_id is None:
        raise RustCoreAdapterError("entity_id is required")
    return _normalize_created_id(driver, driver.entity.create(str(raw_entity_id)))


def _normalize_fact_ids(ids: list[Any], driver: Any | None = None) -> list[Any]:
    normalized: list[Any] = []
    for fact_id in ids:
        if isinstance(fact_id, int):
            normalized.append(fact_id)
        elif isinstance(fact_id, str) and fact_id.isdigit():
            normalized.append(int(fact_id))
        else:
            normalized.append(_coerce_driver_id(driver, fact_id))
    return normalized


def _normalize_fact_id(fact_id: Any) -> int | str:
    if isinstance(fact_id, int):
        return fact_id
    if isinstance(fact_id, str):
        return fact_id
    return str(fact_id)


def _json_safe(value: Any) -> Any:
    try:
        json.dumps(value)
    except TypeError:
        pass
    else:
        return value

    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, set):
        return [_json_safe(item) for item in value]
    return str(value)


def _normalize_embedding_row(fact_id: Any, embedding: Any) -> dict[str, Any] | None:
    payload: dict[str, Any] = {"id": _normalize_fact_id(fact_id)}
    if embedding is None:
        return None

    if isinstance(embedding, memoryview):
        raw = embedding.tobytes()
        if raw:
            payload["content_embedding_b64"] = base64.b64encode(raw).decode("utf-8")
            return payload

    if isinstance(embedding, (bytes, bytearray)):
        raw = bytes(embedding)
        if raw:
            payload["content_embedding_b64"] = base64.b64encode(raw).decode("utf-8")
            return payload

    if isinstance(embedding, str):
        try:
            parsed = json.loads(embedding)
        except Exception:  # noqa: BLE001
            return None
        if isinstance(parsed, list):
            payload["content_embedding"] = [float(x) for x in parsed]
            return payload
        return None

    if isinstance(embedding, (list, tuple)):
        payload["content_embedding"] = [float(x) for x in embedding]
        return payload

    if hasattr(embedding, "tobytes"):
        raw = embedding.tobytes()
        if raw:
            payload["content_embedding_b64"] = base64.b64encode(raw).decode("utf-8")
            return payload

    if hasattr(embedding, "__iter__"):
        try:
            payload["content_embedding"] = [float(x) for x in embedding]
            return payload
        except Exception:  # noqa: BLE001
            return None

    return None


def _normalize_fact_embeddings(
    value: Any, expected_count: int
) -> list[list[float]] | None:
    if not isinstance(value, list) or len(value) != expected_count:
        return None

    embeddings: list[list[float]] = []
    for row in value:
        if not isinstance(row, (list, tuple)):
            return None
        if not row:
            embeddings.append([])
            continue
        try:
            embeddings.append([float(item) for item in row])
        except (TypeError, ValueError):
            return None
    return embeddings


def _coerce_driver_id(driver: Any | None, value: Any) -> Any:
    if _is_mongodb_driver(driver):
        object_id = _to_mongodb_object_id(value)
        if object_id is not None:
            return object_id
    return value


def _normalize_created_id(driver: Any | None, value: Any) -> Any:
    if _is_mongodb_driver(driver):
        return value
    return int(value)


def _is_mongodb_driver(driver: Any | None) -> bool:
    if driver is None:
        return False
    module = getattr(driver.__class__, "__module__", "")
    return module == "memori.storage.drivers.mongodb._driver"


def _to_mongodb_object_id(value: Any) -> Any | None:
    try:
        from bson import ObjectId
    except ImportError:
        return None

    if isinstance(value, ObjectId):
        return value
    if isinstance(value, str) and ObjectId.is_valid(value):
        return ObjectId(value)
    return None


def _resolve_storage_dialect(config: Any, explicit_dialect: str | None) -> str | None:
    if isinstance(explicit_dialect, str):
        candidate = explicit_dialect.strip()
        if candidate:
            return candidate

    storage = getattr(config, "storage", None)
    adapter = getattr(storage, "adapter", None)
    get_dialect = getattr(adapter, "get_dialect", None)
    if callable(get_dialect):
        detected = get_dialect()
        if isinstance(detected, str):
            candidate = detected.strip()
            if candidate:
                return candidate

    storage_config = getattr(config, "storage_config", None)
    configured = getattr(storage_config, "dialect", None)
    if isinstance(configured, str):
        candidate = configured.strip()
        if candidate:
            return candidate

    return None


def _apply_write_op(
    config: Any, driver: Any, op_type: str, payload: dict[str, Any]
) -> bool:
    if op_type == "entity_fact.create":
        raw_entity = payload.get("entity_id")
        if not raw_entity:
            return False
        entity_id = driver.entity.create(str(raw_entity))
        facts = payload.get("facts", [])
        if not isinstance(facts, list):
            return False
        facts_str = [str(f) for f in facts if isinstance(f, (str, int, float))]
        if not facts_str:
            return False
        conversation_id = payload.get("conversation_id")
        conversation_id_driver_id = _to_optional_driver_id(driver, conversation_id)
        embeddings = _normalize_fact_embeddings(
            payload.get("fact_embeddings"), len(facts_str)
        )
        if embeddings is None:
            embeddings_model = getattr(
                getattr(config, "embeddings", None), "model", None
            )
            if isinstance(embeddings_model, str) and embeddings_model:
                embeddings = _embed_entity_facts(config, facts_str, embeddings_model)
        driver.entity_fact.create(
            entity_id,
            facts_str,
            fact_embeddings=embeddings,
            conversation_id=conversation_id_driver_id,
        )
        return True

    if op_type == "knowledge_graph.create":
        raw_entity = payload.get("entity_id")
        if not raw_entity:
            return False
        entity_id = driver.entity.create(str(raw_entity))
        triples = payload.get("semantic_triples", [])
        triples_struct = _to_semantic_triples(triples)
        if not triples_struct:
            return False
        driver.knowledge_graph.create(entity_id, triples_struct)
        return True

    if op_type == "process_attribute.create":
        raw_process = payload.get("process_id")
        if not raw_process:
            return False
        process_id = driver.process.create(str(raw_process))
        attributes = payload.get("attributes", [])
        attributes_norm = _normalize_attributes(attributes)
        if not attributes_norm:
            return False
        driver.process_attribute.create(process_id, attributes_norm)
        return True

    if op_type == "conversation.update":
        conversation_id_driver_id = _to_optional_driver_id(
            driver, payload.get("conversation_id")
        )
        summary = payload.get("summary")
        if conversation_id_driver_id is None or summary is None:
            return False
        driver.conversation.update(conversation_id_driver_id, str(summary))
        return True

    if op_type == "upsert_fact":
        raw_entity = payload.get("entity_id")
        content = payload.get("content")
        if not raw_entity or not isinstance(content, str) or not content.strip():
            return False
        entity_id = driver.entity.create(str(raw_entity))
        driver.entity_fact.create(
            entity_id, [content], fact_embeddings=None, conversation_id=None
        )
        return True

    logger.debug("Skipping unsupported write op type: %s", op_type)
    return False


def _to_semantic_triples(raw: Any) -> list[SemanticTriple]:
    if not isinstance(raw, list):
        return []
    out: list[SemanticTriple] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        subject = item.get("subject")
        predicate = item.get("predicate")
        obj = item.get("object")

        if isinstance(subject, dict):
            subject_name = subject.get("name")
            subject_type = subject.get("type")
        else:
            subject_name = subject
            subject_type = "entity"

        if isinstance(obj, dict):
            object_name = obj.get("name")
            object_type = obj.get("type")
        else:
            object_name = obj
            object_type = "entity"

        if not subject_name or not predicate or not object_name:
            continue

        triple = SemanticTriple()
        triple.subject_name = str(subject_name)
        triple.subject_type = str(subject_type or "entity")
        triple.predicate = str(predicate)
        triple.object_name = str(object_name)
        triple.object_type = str(object_type or "entity")
        out.append(triple)
    return out


def _normalize_attributes(raw: Any) -> list[str]:
    if isinstance(raw, list):
        return [str(x) for x in raw if str(x).strip()]
    if isinstance(raw, dict):
        return [f"{k}:{v}" for k, v in raw.items()]
    if raw is None:
        return []
    return [str(raw)]


def _to_optional_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


def _to_optional_driver_id(driver: Any, value: Any) -> Any | None:
    if value is None:
        return None
    if _is_mongodb_driver(driver):
        object_id = _to_mongodb_object_id(value)
        if object_id is not None:
            return object_id
    return _to_optional_int(value)


def _parse_json(raw: str, context: str) -> Any:
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RustCoreAdapterError(f"Invalid JSON in {context}") from exc


def _parse_json_object(raw: str, context: str) -> dict[str, Any]:
    parsed = _parse_json(raw, context)
    if not isinstance(parsed, dict):
        raise RustCoreAdapterError(f"{context} must be a JSON object")
    return parsed
