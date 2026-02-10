from __future__ import annotations

import json
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
    if dialect == "oceanbase":
        try:
            from pyobvector.util import Vector

            return Vector._to_db(embedding)
        except Exception:
            return json.dumps(embedding)
    return binary_data
