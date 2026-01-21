from __future__ import annotations

import logging
from time import perf_counter
from typing import Any

import numpy as np

from memori.embeddings._chunking import chunk_text_by_tokens
from memori.embeddings._tei import TEI

logger = logging.getLogger(__name__)


def embed_texts_via_tei(
    *,
    text: str,
    model: str,
    tei: TEI,
    tokenizer: Any | None = None,
    chunk_size: int = 128,
) -> list[float]:
    """
    Embed a single text using a TEI-compatible server.

    If a tokenizer is provided, texts are chunked by token count, then chunk
    embeddings are mean-pooled and L2-normalized back to 1 vector.
    """
    if not text:
        return []

    t0 = perf_counter()
    if tokenizer is None:
        logger.debug("embed_texts_via_tei called with no tokenizer")
        out = tei.embed([text], model=model)[0]
        t1 = perf_counter()
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "embed_texts_via_tei(no_chunk): tei=%.2fms total=%.2fms",
                (t1 - t0) * 1000.0,
                (t1 - t0) * 1000.0,
            )
        return out

    t_chunk0 = perf_counter()
    chunks = chunk_text_by_tokens(text=text, tokenizer=tokenizer, chunk_size=chunk_size)
    t_chunk1 = perf_counter()
    chunk_vecs = tei.embed(chunks, model=model)
    t_tei = perf_counter()
    if len(chunk_vecs) != len(chunks):
        raise ValueError("TEI response count does not match input count")

    if len(chunk_vecs) == 1:
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "embed_texts_via_tei(chunked,1): chunks=%d chunk=%.2fms tei=%.2fms total=%.2fms",
                len(chunks),
                (t_chunk1 - t_chunk0) * 1000.0,
                (t_tei - t_chunk1) * 1000.0,
                (t_tei - t0) * 1000.0,
            )
        return chunk_vecs[0]

    t_pool0 = perf_counter()
    embeddings = np.array(chunk_vecs, dtype=np.float32)
    mean_vec = embeddings.mean(axis=0)
    norm = float(np.linalg.norm(mean_vec))
    if norm > 0.0:
        mean_vec = mean_vec / norm
    t_pool1 = perf_counter()
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "embed_texts_via_tei(chunked): chunks=%d chunk=%.2fms tei=%.2fms pool=%.2fms total=%.2fms",
            len(chunks),
            (t_chunk1 - t_chunk0) * 1000.0,
            (t_tei - t_chunk1) * 1000.0,
            (t_pool1 - t_pool0) * 1000.0,
            (t_pool1 - t0) * 1000.0,
        )
    return mean_vec.tolist()
