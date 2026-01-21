from __future__ import annotations

import logging
from typing import Any

import numpy as np

from memori.embeddings._chunking import chunk_text_by_tokens
from memori.embeddings._tei import TEI

logger = logging.getLogger(__name__)


def embed_texts_via_tei(
    *,
    texts: list[str],
    model: str,
    tei: TEI,
    tokenizer: Any | None = None,
    chunk_size: int = 128,
) -> list[list[float]]:
    """
    Embed texts using a TEI-compatible server.

    If a tokenizer is provided, texts are chunked by token count, then chunk
    embeddings are mean-pooled and L2-normalized back to 1 vector per input.
    """
    if not texts:
        return []

    if tokenizer is None:
        logger.debug("embed_texts_via_tei called with no tokenizer")
        return tei.embed(texts, model=model)

    groups = [
        chunk_text_by_tokens(text=t, tokenizer=tokenizer, chunk_size=chunk_size)
        for t in texts
    ]
    flat = [c for g in groups for c in g]
    flat_embeddings = tei.embed(flat, model=model)
    if len(flat_embeddings) != len(flat):
        raise ValueError("TEI response count does not match input count")

    out: list[list[float]] = []
    idx = 0
    for g in groups:
        chunk_vecs = flat_embeddings[idx : idx + len(g)]
        idx += len(g)

        if len(chunk_vecs) == 1:
            out.append(chunk_vecs[0])
            continue

        embeddings = np.array(chunk_vecs, dtype=np.float32)
        mean_vec = embeddings.mean(axis=0)
        norm = float(np.linalg.norm(mean_vec))
        if norm > 0.0:
            mean_vec = mean_vec / norm
        out.append(mean_vec.tolist())
    return out
