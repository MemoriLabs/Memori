from __future__ import annotations

from typing import Any

import numpy as np

from memori.embeddings._chunking import chunk_text_by_tokens
from memori.embeddings._tei import TEI


def _mean_pool_and_normalize(vectors: list[list[float]]) -> list[float]:
    arr = np.asarray(vectors, dtype=np.float32)
    mean_vec = arr.mean(axis=0)
    norm = float(np.linalg.norm(mean_vec))
    if norm > 0.0:
        mean_vec = mean_vec / norm
    return mean_vec.tolist()


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
        out.append(
            chunk_vecs[0]
            if len(chunk_vecs) == 1
            else _mean_pool_and_normalize(chunk_vecs)
        )
    return out
