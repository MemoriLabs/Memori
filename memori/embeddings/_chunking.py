from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


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
    num_tokens = len(tokens["input_ids"][0])

    chunks = []
    for i in range(0, num_tokens, chunk_size):
        chunks.append(tokenizer.decode(tokens["input_ids"][0][i : i + chunk_size]))

    return chunks
