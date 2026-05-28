"""Native embedding cache and input cardinality helpers."""

import threading
from collections.abc import Callable
from typing import Any

from memori._embedding_input import is_embeddable_text, normalize_embed_texts_input
from memori.native._errors import RustCoreAdapterError
from memori.native._loader import _normalize_model_name, _try_import_memori_python

_NATIVE_EMBEDDER_CACHE: dict[str | None, Any] = {}
_NATIVE_EMBEDDER_LOCK = threading.Lock()


def _embed_with_native_cache(
    inputs: list[str], model: str | None = None
) -> list[list[float]]:
    model_name = _normalize_model_name(model)
    with _NATIVE_EMBEDDER_LOCK:
        engine = _NATIVE_EMBEDDER_CACHE.get(model_name)
        if engine is None:
            _try_import_memori_python()
            try:
                from memori_python import (  # ty: ignore[unresolved-import]
                    NativeEmbedder,
                )
            except ImportError as exc:
                raise RustCoreAdapterError("Rust embeddings are unavailable") from exc
            engine = NativeEmbedder(model_name)
            _NATIVE_EMBEDDER_CACHE[model_name] = engine

    return [list(row) for row in engine.embed_texts(inputs)]


def _embed_texts_with_cardinality(
    texts: str | list[str],
    embed_fn: Callable[[list[str]], list[list[float]]],
) -> list[list[float]]:
    originals = normalize_embed_texts_input(texts)
    if not originals:
        return []

    embeddable = [text for text in originals if is_embeddable_text(text)]
    if not embeddable:
        return [[] for _ in originals]

    embedded = embed_fn(embeddable)
    if len(embedded) != len(embeddable):
        raise RustCoreAdapterError(
            "Native embedder returned "
            f"{len(embedded)} vectors for {len(embeddable)} embeddable inputs"
        )

    result: list[list[float]] = [[] for _ in originals]
    embed_index = 0
    for index, text in enumerate(originals):
        if not is_embeddable_text(text):
            continue
        result[index] = embedded[embed_index]
        embed_index += 1
    return result


def embed_texts(texts: str | list[str], model: str | None = None) -> list[list[float]]:
    return _embed_texts_with_cardinality(
        texts,
        lambda embeddable: _embed_with_native_cache(embeddable, model),
    )
