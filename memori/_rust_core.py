import base64
import hashlib
import importlib.util
import json
import logging
import os
import platform
import shutil
import sys
import tarfile
import tempfile
import threading
import time
import zipfile
from dataclasses import dataclass, field
from importlib.machinery import ExtensionFileLoader
from pathlib import Path
from typing import Any

import requests

from memori.memory._struct import SemanticTriple
from memori.storage._connection import connection_context

logger = logging.getLogger(__name__)
_ORT_VERSION = "1.23.2"
_ORT_LOCK_TIMEOUT_SECONDS = 120.0
_ORT_DOWNLOAD_ATTEMPTS = 3
_ORT_ASSET_BY_PLATFORM: dict[tuple[str, str], tuple[str, str]] = {
    (
        "linux",
        "x86_64",
    ): (
        "onnxruntime-linux-x64-1.23.2.tgz",
        "1fa4dcaef22f6f7d5cd81b28c2800414350c10116f5fdd46a2160082551c5f9b",
    ),
    (
        "linux",
        "amd64",
    ): (
        "onnxruntime-linux-x64-1.23.2.tgz",
        "1fa4dcaef22f6f7d5cd81b28c2800414350c10116f5fdd46a2160082551c5f9b",
    ),
    (
        "linux",
        "aarch64",
    ): (
        "onnxruntime-linux-aarch64-1.23.2.tgz",
        "7c63c73560ed76b1fac6cff8204ffe34fe180e70d6582b5332ec094810241e5c",
    ),
    (
        "linux",
        "arm64",
    ): (
        "onnxruntime-linux-aarch64-1.23.2.tgz",
        "7c63c73560ed76b1fac6cff8204ffe34fe180e70d6582b5332ec094810241e5c",
    ),
    (
        "android",
        "aarch64",
    ): (
        "onnxruntime-android-1.23.2.aar",
        "82048d1f462218adae4ba76477089ab0ba76093d84f733540066db1a8ba6b827",
    ),
    (
        "android",
        "arm64",
    ): (
        "onnxruntime-android-1.23.2.aar",
        "82048d1f462218adae4ba76477089ab0ba76093d84f733540066db1a8ba6b827",
    ),
    (
        "android",
        "x86_64",
    ): (
        "onnxruntime-android-1.23.2.aar",
        "82048d1f462218adae4ba76477089ab0ba76093d84f733540066db1a8ba6b827",
    ),
    (
        "android",
        "amd64",
    ): (
        "onnxruntime-android-1.23.2.aar",
        "82048d1f462218adae4ba76477089ab0ba76093d84f733540066db1a8ba6b827",
    ),
    (
        "darwin",
        "x86_64",
    ): (
        "onnxruntime-osx-x86_64-1.23.2.tgz",
        "d10359e16347b57d9959f7e80a225a5b4a66ed7d7e007274a15cae86836485a6",
    ),
    (
        "darwin",
        "arm64",
    ): (
        "onnxruntime-osx-arm64-1.23.2.tgz",
        "b4d513ab2b26f088c66891dbbc1408166708773d7cc4163de7bdca0e9bbb7856",
    ),
    (
        "windows",
        "x86_64",
    ): (
        "onnxruntime-win-x64-1.23.2.zip",
        "0b38df9af21834e41e73d602d90db5cb06dbd1ca618948b8f1d66d607ac9f3cd",
    ),
    (
        "windows",
        "amd64",
    ): (
        "onnxruntime-win-x64-1.23.2.zip",
        "0b38df9af21834e41e73d602d90db5cb06dbd1ca618948b8f1d66d607ac9f3cd",
    ),
    (
        "windows",
        "arm64",
    ): (
        "onnxruntime-win-arm64-1.23.2.zip",
        "1cfe88b6435df3b5fb0e9f6bd7d6f5df1e887b6174de7f6e2a47bab956f3f168",
    ),
}


def embed_texts(*args: Any, **kwargs: Any) -> Any:
    from memori.embeddings import embed_texts as embed_texts_impl

    return embed_texts_impl(*args, **kwargs)


def _current_platform_system() -> str:
    if sys.platform == "android":
        return "android"
    return platform.system().lower()


def _onnxruntime_asset_for_current_platform() -> tuple[str, str] | None:
    return _ORT_ASSET_BY_PLATFORM.get(
        (_current_platform_system(), platform.machine().lower())
    )


def _onnxruntime_lib_filename() -> str:
    system = _current_platform_system()
    if system == "windows":
        return "onnxruntime.dll"
    if system == "darwin":
        return "libonnxruntime.dylib"
    return "libonnxruntime.so"


def _android_abi_for_machine(machine: str) -> str | None:
    normalized = machine.lower()
    if normalized in {"aarch64", "arm64"}:
        return "arm64-v8a"
    if normalized in {"x86_64", "amd64"}:
        return "x86_64"
    return None


def _resolve_onnxruntime_lib_path(lib_dir: Path) -> Path | None:
    direct_path = lib_dir / _onnxruntime_lib_filename()
    if direct_path.exists():
        return direct_path

    system = _current_platform_system()
    if system == "android":
        abi = _android_abi_for_machine(platform.machine())
        if abi is not None:
            abi_path = lib_dir / "jni" / abi / _onnxruntime_lib_filename()
            if abi_path.exists():
                return abi_path

    if system == "darwin":
        fallback_pattern = "libonnxruntime.*.dylib"
    elif system == "windows":
        fallback_pattern = "onnxruntime*.dll"
    else:
        fallback_pattern = "libonnxruntime.so.*"

    for candidate in sorted(lib_dir.glob(fallback_pattern)):
        if candidate.is_file():
            return candidate
    for candidate in sorted(lib_dir.rglob(fallback_pattern)):
        if candidate.is_file():
            return candidate
    for candidate in sorted(lib_dir.rglob(_onnxruntime_lib_filename())):
        if candidate.is_file():
            return candidate
    return None


def _is_within_directory(directory: Path, candidate: Path) -> bool:
    directory = directory.resolve()
    candidate = candidate.resolve()
    return directory == candidate or directory in candidate.parents


def _extract_onnxruntime_archive(archive_path: Path, destination: Path) -> None:
    if archive_path.suffix in {".zip", ".aar"}:
        with zipfile.ZipFile(archive_path, "r") as archive:
            for member in archive.infolist():
                if not member.filename:
                    continue
                target = destination / member.filename
                if not _is_within_directory(destination, target):
                    raise RuntimeError("Unsafe path in ONNX Runtime zip archive")
                if member.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member, "r") as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output)
        return
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive.getmembers():
            if not member.name:
                continue
            target = destination / member.name
            if not _is_within_directory(destination, target):
                raise RuntimeError("Unsafe path in ONNX Runtime tar archive")
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if not (member.isfile() or member.islnk()):
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            source = archive.extractfile(member)
            if source is None:
                continue
            with source, target.open("wb") as output:
                shutil.copyfileobj(source, output)


def _compute_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _download_urls_for_asset(asset_name: str) -> tuple[str, str]:
    if asset_name.endswith(".aar"):
        maven = (
            "https://repo1.maven.org/maven2/com/microsoft/onnxruntime/"
            f"onnxruntime-android/{_ORT_VERSION}/{asset_name}"
        )
        return (maven, maven)

    github = (
        "https://github.com/microsoft/onnxruntime/releases/download/"
        f"v{_ORT_VERSION}/{asset_name}"
    )
    sourceforge = (
        "https://sourceforge.net/projects/onnx-runtime.mirror/files/"
        f"v{_ORT_VERSION}/{asset_name}/download"
    )
    return github, sourceforge


def _download_asset_with_retries(asset_name: str, destination: Path) -> bool:
    urls = _download_urls_for_asset(asset_name)
    for attempt in range(1, _ORT_DOWNLOAD_ATTEMPTS + 1):
        for url in urls:
            try:
                with requests.get(url, stream=True, timeout=(15, 120)) as response:
                    response.raise_for_status()
                    with destination.open("wb") as file_handle:
                        for chunk in response.iter_content(chunk_size=1024 * 1024):
                            if chunk:
                                file_handle.write(chunk)
                return True
            except Exception:  # noqa: BLE001
                logger.warning(
                    "Failed to download %s (attempt %s/%s) from %s",
                    asset_name,
                    attempt,
                    _ORT_DOWNLOAD_ATTEMPTS,
                    url,
                )
    return False


def _acquire_cache_lock(lock_path: Path) -> bool:
    deadline = time.monotonic() + _ORT_LOCK_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            try:
                os.write(fd, str(os.getpid()).encode("utf-8"))
            finally:
                os.close(fd)
            return True
        except FileExistsError:
            time.sleep(0.2)
    return False


def _release_cache_lock(lock_path: Path) -> None:
    try:
        lock_path.unlink()
    except FileNotFoundError:
        return


def _configure_onnxruntime_env(lib_path: Path) -> None:
    os.environ["ORT_DYLIB_PATH"] = str(lib_path)
    if _current_platform_system() == "windows":
        try:
            os.add_dll_directory(str(lib_path.parent))
        except Exception:  # noqa: BLE001
            logger.debug("Failed to add ONNX Runtime directory to DLL search path")


def _ensure_onnxruntime_dylib() -> None:
    current = os.environ.get("ORT_DYLIB_PATH")
    if current and Path(current).exists():
        _configure_onnxruntime_env(Path(current))
        return
    if os.environ.get("MEMORI_ORT_AUTO_DOWNLOAD", "1").lower() in {"0", "false", "no"}:
        return

    asset_info = _onnxruntime_asset_for_current_platform()
    if asset_info is None:
        return
    asset_name, expected_sha = asset_info

    cache_root = Path.home() / ".cache" / "memori" / "onnxruntime" / _ORT_VERSION
    asset_root = (
        asset_name.removesuffix(".tgz").removesuffix(".zip").removesuffix(".aar")
    )
    install_dir = cache_root / asset_root
    lib_path = _resolve_onnxruntime_lib_path(install_dir)
    if lib_path is not None:
        _configure_onnxruntime_env(lib_path)
        return

    cache_root.mkdir(parents=True, exist_ok=True)
    lock_path = cache_root / ".download.lock"
    if not _acquire_cache_lock(lock_path):
        logger.warning("Timed out waiting for ONNX Runtime cache lock")
        return
    try:
        existing_lib_path = _resolve_onnxruntime_lib_path(install_dir)
        if existing_lib_path is not None:
            _configure_onnxruntime_env(existing_lib_path)
            return

        with tempfile.NamedTemporaryFile(
            suffix=Path(asset_name).suffix, dir=cache_root, delete=False
        ) as tmp_file:
            archive_path = Path(tmp_file.name)
        try:
            if not _download_asset_with_retries(asset_name, archive_path):
                return
            actual_sha = _compute_sha256(archive_path)
            if actual_sha != expected_sha:
                logger.error(
                    "ONNX Runtime checksum mismatch for %s: expected %s got %s",
                    asset_name,
                    expected_sha,
                    actual_sha,
                )
                return

            extract_root = Path(
                tempfile.mkdtemp(prefix="onnxruntime-extract-", dir=cache_root)
            )
            try:
                _extract_onnxruntime_archive(archive_path, extract_root)
                extracted_dir = extract_root / asset_root
                source_dir = extracted_dir if extracted_dir.exists() else extract_root
                final_dir = install_dir
                if not final_dir.exists():
                    if source_dir == extract_root:
                        shutil.copytree(source_dir, final_dir)
                    else:
                        os.replace(source_dir, final_dir)
            finally:
                shutil.rmtree(extract_root, ignore_errors=True)
        except Exception:  # noqa: BLE001
            logger.exception("Failed to prepare ONNX Runtime binaries")
            return
        finally:
            archive_path.unlink(missing_ok=True)

        resolved_lib_path = _resolve_onnxruntime_lib_path(install_dir)
        if resolved_lib_path is not None:
            _configure_onnxruntime_env(resolved_lib_path)
    finally:
        _release_cache_lock(lock_path)


class RustCoreAdapterError(RuntimeError):
    pass


def _try_import_memori_python() -> bool:
    _ensure_onnxruntime_dylib()
    env_path = os.environ.get("MEMORI_PYTHON_LIB")
    candidates: list[Path] = []
    if env_path:
        candidates.append(Path(env_path))

    cargo_target_dir = os.environ.get("CARGO_TARGET_DIR")
    if cargo_target_dir:
        target = Path(cargo_target_dir)
        candidates.extend(
            [
                target / "release" / "libmemori_python.dylib",
                target / "release" / "libmemori_python.so",
                target / "release" / "memori_python.dll",
                target / "debug" / "libmemori_python.dylib",
                target / "debug" / "libmemori_python.so",
                target / "debug" / "memori_python.dll",
            ]
        )

    candidates.extend(
        [
            Path("target/release/libmemori_python.dylib"),
            Path("target/release/libmemori_python.so"),
            Path("target/release/memori_python.dll"),
            Path("target/debug/libmemori_python.dylib"),
            Path("target/debug/libmemori_python.so"),
            Path("target/debug/memori_python.dll"),
            Path("core/target/release/libmemori_python.dylib"),
            Path("core/target/release/libmemori_python.so"),
            Path("core/target/release/memori_python.dll"),
            Path("core/target/debug/libmemori_python.dylib"),
            Path("core/target/debug/libmemori_python.so"),
            Path("core/target/debug/memori_python.dll"),
        ]
    )

    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            loader = ExtensionFileLoader("memori_python", str(candidate))
            spec = importlib.util.spec_from_loader("memori_python", loader)
            if spec is None:
                continue
            module = importlib.util.module_from_spec(spec)
            loader.exec_module(module)
            sys.modules["memori_python"] = module
            logger.debug("Loaded memori_python from %s", candidate)
            return True
        except ImportError:
            continue

    try:
        import memori_python  # noqa: F401  # ty: ignore[unresolved-import]

        logger.debug(
            "Loaded memori_python from %s",
            getattr(memori_python, "__file__", "unknown"),
        )
        return True
    except ImportError:
        return False


def _normalize_model_name(model_name: str | None) -> str | None:
    if model_name is None:
        return None
    normalized = model_name.strip()
    if not normalized:
        return None
    lowered = normalized.lower()
    if lowered in {"all-minilm-l6-v2", "allminilml6v2"}:
        return None
    return normalized


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
            logger.warning(
                "Rust core unavailable, falling back to Python path: %s", exc
            )
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
            fact_ids = _normalize_fact_ids(ids)
            with connection_context(config.storage.conn_factory) as (
                _conn,
                _adapter,
                driver,
            ):
                rows = driver.entity_fact.get_facts_by_ids(fact_ids)
                out = []
                for row in rows:
                    out.append(
                        {
                            "id": _normalize_fact_id(row.get("id")),
                            "content": row.get("content", ""),
                            "date_created": str(row.get("date_created", "")),
                            "summaries": row.get("summaries", []),
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


def _resolve_entity_id(driver: Any, raw_entity_id: Any) -> int:
    if isinstance(raw_entity_id, int):
        return raw_entity_id
    if isinstance(raw_entity_id, str):
        stripped = raw_entity_id.strip()
        if not stripped:
            raise RustCoreAdapterError("entity_id cannot be empty")
        if stripped.isdigit():
            return int(stripped)
        return int(driver.entity.create(stripped))
    if raw_entity_id is None:
        raise RustCoreAdapterError("entity_id is required")
    return int(driver.entity.create(str(raw_entity_id)))


def _normalize_fact_ids(ids: list[Any]) -> list[Any]:
    normalized: list[Any] = []
    for fact_id in ids:
        if isinstance(fact_id, int):
            normalized.append(fact_id)
        elif isinstance(fact_id, str) and fact_id.isdigit():
            normalized.append(int(fact_id))
        else:
            normalized.append(fact_id)
    return normalized


def _normalize_fact_id(fact_id: Any) -> int | str:
    if isinstance(fact_id, int):
        return fact_id
    if isinstance(fact_id, str):
        return fact_id
    return str(fact_id)


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
        conversation_id_int = _to_optional_int(conversation_id)
        embeddings: list[list[float]] | None = None
        try:
            embeddings_model = getattr(
                getattr(config, "embeddings", None), "model", None
            )
            if isinstance(embeddings_model, str) and embeddings_model:
                embeddings = embed_texts(facts_str, model=embeddings_model)
        except Exception:  # noqa: BLE001
            logger.exception(
                "Failed to embed AA facts before write; falling back without embeddings"
            )
        driver.entity_fact.create(
            entity_id,
            facts_str,
            fact_embeddings=embeddings,
            conversation_id=conversation_id_int,
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
        conversation_id_int = _to_optional_int(payload.get("conversation_id"))
        summary = payload.get("summary")
        if conversation_id_int is None or summary is None:
            return False
        driver.conversation.update(conversation_id_int, str(summary))
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
