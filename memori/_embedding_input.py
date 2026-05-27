from __future__ import annotations

import unicodedata
from collections.abc import Iterable


def is_embeddable_text(text: str) -> bool:
    """Match Rust `prepare_text_inputs` visibility rules."""
    return any(
        not char.isspace()
        and unicodedata.category(char) not in {"Cc", "Cf"}
        and char != "\u200b"
        for char in text
    )


def normalize_embed_texts_input(texts: str | Iterable[str]) -> list[str]:
    if isinstance(texts, str):
        return [texts]
    return list(texts)
