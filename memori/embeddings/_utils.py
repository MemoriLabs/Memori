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


def prepare_text_inputs(texts: str | Iterable[str]) -> list[str]:
    if isinstance(texts, str):
        return [texts]
    return [t for t in texts if t]
