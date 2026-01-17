from __future__ import annotations

import logging
import os
import threading
from typing import Any

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
from sentence_transformers import SentenceTransformer

from memori.embeddings._utils import embedding_dimension, zero_vectors

logger = logging.getLogger(__name__)


class SentenceTransformersEmbedder:
    def __init__(self, model_name: str) -> None:
        self._model_name = model_name
        self._model: SentenceTransformer | None = None
        self._model_lock = threading.Lock()
        self._encode_lock = threading.Lock()

    def _get_model(self) -> SentenceTransformer:
        with self._model_lock:
            if self._model is None:
                self._model = SentenceTransformer(self._model_name)
            return self._model

    def _load_encoder(self, *, fallback_dimension: int) -> SentenceTransformer | None:
        try:
            return self._get_model()
        except (OSError, RuntimeError, ValueError):
            logger.debug(
                "Failed to load model %s, returning zero embeddings", self._model_name
            )
            return None

    def _encode_batch(
        self, encoder: SentenceTransformer, inputs: list[str]
    ) -> list[list[float]]:
        with self._encode_lock:
            embeddings = encoder.encode(
                inputs, convert_to_numpy=True, normalize_embeddings=True
            )
        return embeddings.tolist()

    def _encode_one_by_one(
        self, encoder: SentenceTransformer, inputs: list[str]
    ) -> list[list[float]]:
        vectors: list[list[float]] = []
        with self._encode_lock:
            for text in inputs:
                single = encoder.encode(
                    [text], convert_to_numpy=True, normalize_embeddings=True
                )
                vectors.append(single[0].tolist())

        dim_set = {len(v) for v in vectors}
        if len(dim_set) != 1:
            raise ValueError("all input arrays must have the same shape")

        return vectors

    def _chunk_size_tokens(self, encoder: SentenceTransformer) -> int | None:
        def _as_int(value: object) -> int | None:
            try:
                if isinstance(value, bool):
                    return None
                if isinstance(value, int):
                    return value
                if isinstance(value, float):
                    return int(value)
                if isinstance(value, str):
                    return int(value.strip())
            except Exception:
                return None
            return None

        max_len: int | None = None
        try:
            max_len = _as_int(encoder.get_max_seq_length())
        except Exception:
            max_len = None

        if max_len is None:
            try:
                raw = getattr(encoder, "max_seq_length", None)
                max_len = _as_int(raw)
            except Exception:
                max_len = None

        if max_len is None or max_len <= 0:
            return None

        # We chunk without special tokens; reserve a small budget so the model
        # can still add special tokens without truncation.
        return max(1, int(max_len) - 2)

    def _tokenizer(self, encoder: SentenceTransformer) -> Any | None:
        return getattr(encoder, "tokenizer", None)

    def _chunk_text(
        self, *, encoder: SentenceTransformer, text: str, chunk_size_tokens: int
    ) -> list[str]:
        tokenizer = self._tokenizer(encoder)
        if tokenizer is None:
            return [text]

        try:
            encoded = tokenizer(
                text,
                add_special_tokens=False,
                return_attention_mask=False,
                return_token_type_ids=False,
            )
            ids = encoded.get("input_ids") if isinstance(encoded, dict) else None
            if not isinstance(ids, list):
                return [text]
        except Exception:
            return [text]

        if len(ids) <= chunk_size_tokens:
            return [text]

        chunks: list[str] = []
        for i in range(0, len(ids), chunk_size_tokens):
            chunk_ids = ids[i : i + chunk_size_tokens]
            chunk_text = tokenizer.decode(
                chunk_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
            if chunk_text:
                chunks.append(chunk_text)

        return chunks or [text]

    def _mean_pool_and_normalize(self, vectors: np.ndarray) -> np.ndarray:
        mean_vec = vectors.mean(axis=0)
        norm = float(np.linalg.norm(mean_vec))
        if norm > 0.0:
            mean_vec = mean_vec / norm
        return mean_vec

    def _encode_chunks(
        self,
        *,
        encoder: SentenceTransformer,
        chunks: list[str],
    ) -> list[float]:
        if len(chunks) == 1:
            return self._encode_batch(encoder, chunks)[0]

        with self._encode_lock:
            chunk_vectors = encoder.encode(
                chunks, convert_to_numpy=True, normalize_embeddings=True
            )
        pooled = self._mean_pool_and_normalize(
            np.asarray(chunk_vectors, dtype=np.float32)
        )
        return pooled.tolist()

    def _encode_inputs(
        self,
        *,
        encoder: SentenceTransformer,
        inputs: list[str],
        chunk_size_tokens: int | None,
    ) -> list[list[float]]:
        if not inputs:
            return []
        if chunk_size_tokens is None:
            return self._encode_batch(encoder, inputs)

        short_inputs: list[str] = []
        short_positions: list[int] = []
        long_items: list[tuple[int, list[str]]] = []

        for idx, text in enumerate(inputs):
            chunks = self._chunk_text(
                encoder=encoder, text=text, chunk_size_tokens=chunk_size_tokens
            )
            if len(chunks) == 1:
                short_inputs.append(text)
                short_positions.append(idx)
            else:
                long_items.append((idx, chunks))

        out: list[list[float]] = [[] for _ in inputs]

        if short_inputs:
            short_vecs = self._encode_batch(encoder, short_inputs)
            for pos, vec in zip(short_positions, short_vecs, strict=False):
                out[pos] = vec

        for pos, chunks in long_items:
            out[pos] = self._encode_chunks(encoder=encoder, chunks=chunks)

        return out

    def _zero_result(
        self,
        *,
        count: int,
        fallback_dimension: int,
        encoder: SentenceTransformer | None,
    ) -> list[list[float]]:
        dim = (
            embedding_dimension(encoder, default=fallback_dimension)
            if encoder is not None
            else fallback_dimension
        )
        logger.warning(
            "Embedding encode failed for model=%s, returning zero embeddings of dim %d",
            self._model_name,
            dim,
        )
        return zero_vectors(count, dim)

    def embed(self, inputs: list[str], *, fallback_dimension: int) -> list[list[float]]:
        if not inputs:
            return []

        logger.debug(
            "Generating embedding using model: %s for %d text(s)",
            self._model_name,
            len(inputs),
        )

        encoder = self._load_encoder(fallback_dimension=fallback_dimension)
        if encoder is None:
            return zero_vectors(len(inputs), fallback_dimension)

        try:
            result = self._encode_inputs(
                encoder=encoder,
                inputs=inputs,
                chunk_size_tokens=self._chunk_size_tokens(encoder),
            )
            if result:
                logger.debug(
                    "Embedding generated - dimension: %d, count: %d",
                    len(result[0]),
                    len(result),
                )
            return result
        except ValueError as e:
            if "same shape" not in str(e):
                raise

            try:
                vectors = self._encode_one_by_one(encoder, inputs)
                if vectors:
                    logger.debug(
                        "Embedding generated (one-by-one) - dimension: %d, count: %d",
                        len(vectors[0]),
                        len(vectors),
                    )
                return vectors
            except Exception:
                return self._zero_result(
                    count=len(inputs),
                    fallback_dimension=fallback_dimension,
                    encoder=encoder,
                )
        except RuntimeError:
            return self._zero_result(
                count=len(inputs),
                fallback_dimension=fallback_dimension,
                encoder=encoder,
            )


_EMBEDDER_CACHE: dict[str, SentenceTransformersEmbedder] = {}
_EMBEDDER_CACHE_LOCK = threading.Lock()


def get_sentence_transformers_embedder(model_name: str) -> SentenceTransformersEmbedder:
    with _EMBEDDER_CACHE_LOCK:
        embedder = _EMBEDDER_CACHE.get(model_name)
        if embedder is None:
            embedder = SentenceTransformersEmbedder(model_name)
            _EMBEDDER_CACHE[model_name] = embedder
        return embedder
