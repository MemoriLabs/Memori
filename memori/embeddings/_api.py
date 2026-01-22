from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable
from functools import partial
from typing import Literal, overload

from memori.embeddings._sentence_transformers import get_sentence_transformers_embedder
from memori.embeddings._tei import TEI
from memori.embeddings._tei_embed import embed_texts_via_tei
from memori.embeddings._utils import prepare_text_inputs

logger = logging.getLogger(__name__)
_FALLBACK_DIMENSION = 768


def _embed_texts(
    texts: str | list[str],
    model: str,
    *,
    tei: TEI | None = None,
    tokenizer: object | None = None,
    chunk_size: int = 128,
    fallback_dimension: int | None = None,
) -> list[list[float]]:
    inputs = prepare_text_inputs(texts)
    if not inputs:
        logger.debug("embed_texts called with empty input")
        return []
    if tei is not None:
        return [
            embed_texts_via_tei(
                text=t,
                model=model,
                tei=tei,
                tokenizer=tokenizer,
                chunk_size=chunk_size,
            )
            for t in inputs
        ]
    dim = fallback_dimension if fallback_dimension is not None else _FALLBACK_DIMENSION
    return get_sentence_transformers_embedder(model).embed(
        inputs, fallback_dimension=dim
    )


async def _embed_texts_async(
    texts: str | list[str],
    model: str,
    *,
    tei: TEI | None = None,
    tokenizer: object | None = None,
    chunk_size: int = 128,
    fallback_dimension: int | None = None,
) -> list[list[float]]:
    loop = asyncio.get_event_loop()
    fn = partial(
        _embed_texts,
        texts,
        model,
        tei=tei,
        tokenizer=tokenizer,
        chunk_size=chunk_size,
        fallback_dimension=fallback_dimension,
    )
    return await loop.run_in_executor(None, fn)


@overload
def embed_texts(
    texts: str | list[str],
    model: str,
    *,
    async_: Literal[False] = False,
    tei: TEI | None = None,
    tokenizer: object | None = None,
    chunk_size: int = 128,
) -> list[list[float]]: ...


@overload
def embed_texts(
    texts: str | list[str],
    model: str,
    *,
    async_: Literal[True],
    tei: TEI | None = None,
    tokenizer: object | None = None,
    chunk_size: int = 128,
) -> Awaitable[list[list[float]]]: ...


def embed_texts(
    texts: str | list[str],
    model: str,
    *,
    async_: bool = False,
    tei: TEI | None = None,
    tokenizer: object | None = None,
    chunk_size: int = 128,
) -> list[list[float]] | Awaitable[list[list[float]]]:
    """
    Embed text(s) into vectors.

    When async_=True, returns an awaitable that runs the work in a threadpool.
    """
    if async_:
        return _embed_texts_async(
            texts, model, tei=tei, tokenizer=tokenizer, chunk_size=chunk_size
        )
    return _embed_texts(
        texts, model, tei=tei, tokenizer=tokenizer, chunk_size=chunk_size
    )
