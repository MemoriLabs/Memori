from __future__ import annotations

import logging
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

    if tokenizer is None:
        logger.debug("embed_texts_via_tei called with no tokenizer")
        return tei.embed([text], model=model)[0]

    chunks = chunk_text_by_tokens(text=text, tokenizer=tokenizer, chunk_size=chunk_size)
    chunk_vecs = tei.embed(chunks, model=model)
    if len(chunk_vecs) != len(chunks):
        raise ValueError("TEI response count does not match input count")

    if len(chunk_vecs) == 1:
        return chunk_vecs[0]

    embeddings = np.array(chunk_vecs, dtype=np.float32)
    mean_vec = embeddings.mean(axis=0)
    norm = float(np.linalg.norm(mean_vec))
    if norm > 0.0:
        mean_vec = mean_vec / norm
    return mean_vec.tolist()
