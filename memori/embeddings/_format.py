from __future__ import annotations

import struct
from typing import Any


def format_embedding_for_db(embedding: list[float], dialect: str) -> Any:
    binary_data = struct.pack(f"<{len(embedding)}f", *embedding)

    if dialect == "mongodb":
        try:
            import bson

            return bson.Binary(binary_data)
        except ImportError:
            return binary_data
    return binary_data
