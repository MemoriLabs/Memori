"""Compatibility exports for the native Rust core bridge.

Implementation lives under `memori.native`; this module preserves the legacy
`memori._rust_core` import path used by the SDK and tests.
"""

# ruff: noqa: F401

import os
import platform
import sys
from pathlib import Path

import requests

from memori.native._adapter import (
    RustCoreAdapter,
    _apply_write_op,
    _coerce_driver_id,
    _embed_entity_facts,
    _is_mongodb_driver,
    _json_safe,
    _normalize_attributes,
    _normalize_created_id,
    _normalize_embedding_row,
    _normalize_fact_embeddings,
    _normalize_fact_id,
    _normalize_fact_ids,
    _parse_json,
    _parse_json_object,
    _resolve_entity_id,
    _resolve_storage_dialect,
    _to_mongodb_object_id,
    _to_optional_driver_id,
    _to_optional_int,
    _to_semantic_triples,
)
from memori.native._embeddings import (
    _NATIVE_EMBEDDER_CACHE,
    _NATIVE_EMBEDDER_LOCK,
    _embed_texts_with_cardinality,
    _embed_with_native_cache,
    embed_texts,
)
from memori.native._errors import RustCoreAdapterError
from memori.native._loader import _normalize_model_name, _try_import_memori_python
from memori.native._onnxruntime import (
    _ORT_ASSET_BY_PLATFORM,
    _ORT_DOWNLOAD_ATTEMPTS,
    _ORT_LOCK_TIMEOUT_SECONDS,
    _ORT_VERSION,
    _acquire_cache_lock,
    _android_abi_for_machine,
    _compute_sha256,
    _configure_onnxruntime_env,
    _current_platform_system,
    _download_asset_with_retries,
    _download_urls_for_asset,
    _ensure_onnxruntime_dylib,
    _extract_onnxruntime_archive,
    _is_within_directory,
    _onnxruntime_asset_for_current_platform,
    _onnxruntime_lib_filename,
    _release_cache_lock,
    _resolve_onnxruntime_lib_path,
)
from memori.storage._connection import connection_context

__all__ = [
    "RustCoreAdapter",
    "RustCoreAdapterError",
    "embed_texts",
]
