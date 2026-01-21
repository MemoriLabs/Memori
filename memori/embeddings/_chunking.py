from __future__ import annotations

from collections.abc import Callable
from typing import Any


def chunk_text_by_tokens(
    *,
    text: str,
    tokenizer: Any,
    chunk_size: int,
) -> list[str]:
    """
    Chunk text by token count using a user-provided tokenizer.

    Tokenizer requirements:
    - callable: tokenizer(text, return_tensors=...) -> dict with "input_ids"
    - decode: tokenizer.decode(ids_slice) -> str
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")

    tokens = tokenizer(text, return_tensors="np")
    input_ids_raw = tokens.get("input_ids") if isinstance(tokens, dict) else None
    if input_ids_raw is None:
        return [text]

    try:
        ids = input_ids_raw[0]
        num_tokens = len(ids)
    except (IndexError, TypeError):
        return [text]

    if num_tokens == 0:
        return [text]

    chunks: list[str] = []
    decode: Callable[[Any], str] = tokenizer.decode
    for i in range(0, num_tokens, chunk_size):
        chunk_text = decode(ids[i : i + chunk_size])
        if chunk_text:
            chunks.append(chunk_text)
    return chunks or [text]
