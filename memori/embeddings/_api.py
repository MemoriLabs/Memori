from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable
from typing import Literal, overload

from memori.embeddings._sentence_transformers import get_sentence_transformers_embedder
from memori.embeddings._utils import prepare_text_inputs

logger = logging.getLogger(__name__)


def _embed_texts(
    texts: str | list[str],
    model: str,
    fallback_dimension: int,
) -> list[list[float]]:
    inputs = prepare_text_inputs(texts)
    if not inputs:
        logger.debug("embed_texts called with empty input")
        return []
    return get_sentence_transformers_embedder(model).embed(
        inputs, fallback_dimension=fallback_dimension
    )


async def _embed_texts_async(
    texts: str | list[str],
    model: str,
    fallback_dimension: int,
) -> list[list[float]]:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None, _embed_texts, texts, model, fallback_dimension
    )


@overload
def embed_texts(
    texts: str | list[str],
    model: str,
    fallback_dimension: int,
    *,
    async_: Literal[False] = False,
) -> list[list[float]]: ...


@overload
def embed_texts(
    texts: str | list[str],
    model: str,
    fallback_dimension: int,
    *,
    async_: Literal[True],
) -> Awaitable[list[list[float]]]: ...


def embed_texts(
    texts: str | list[str],
    model: str,
    fallback_dimension: int,
    *,
    async_: bool = False,
) -> list[list[float]] | Awaitable[list[list[float]]]:
    """
    Embed text(s) into vectors.

    When async_=True, returns an awaitable that runs the work in a threadpool.
    """
    if async_:
        return _embed_texts_async(texts, model, fallback_dimension)
    return _embed_texts(texts, model, fallback_dimension)
