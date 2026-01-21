from __future__ import annotations

from collections.abc import Callable
from typing import Any


def _as_token_id_list(value: Any) -> list[int] | None:
    if value is None:
        return None
    if isinstance(value, list):
        return value
    try:
        return value.tolist()  # type: ignore[attr-defined]
    except Exception:
        return None


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

    ids_2d: list[list[int]] | None = None
    if (
        isinstance(input_ids_raw, list)
        and input_ids_raw
        and isinstance(input_ids_raw[0], list)
    ):
        ids_2d = [input_ids_raw[0]]
    else:
        ids_1d = (
            _as_token_id_list(input_ids_raw[0])
            if hasattr(input_ids_raw, "__getitem__")
            else None
        )
        if ids_1d is not None:
            ids_2d = [ids_1d]

    if not ids_2d or not ids_2d[0]:
        return [text]

    ids = ids_2d[0]
    chunks: list[str] = []
    decode: Callable[[Any], str] = tokenizer.decode
    for i in range(0, len(ids), chunk_size):
        chunk_text = decode(ids[i : i + chunk_size])
        if chunk_text:
            chunks.append(chunk_text)
    return chunks or [text]
