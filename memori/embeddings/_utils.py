r"""
 __  __                           _
|  \/  | ___ _ __ ___   ___  _ __(_)
| |\/| |/ _ \ '_ ` _ \ / _ \| '__| |
| |  | |  __/ | | | | | (_) | |  | |
|_|  |_|\___|_| |_| |_|\___/|_|  |_|
                  perfectam memoriam
                       memorilabs.ai
"""

from __future__ import annotations

from collections.abc import Iterable

from memori._embedding_input import is_embeddable_text, normalize_embed_texts_input


def prepare_text_inputs(texts: str | Iterable[str]) -> list[str]:
    return [
        text for text in normalize_embed_texts_input(texts) if is_embeddable_text(text)
    ]
