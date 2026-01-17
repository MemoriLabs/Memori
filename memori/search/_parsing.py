from __future__ import annotations

import json
from typing import Any

import numpy as np


def parse_embedding(raw: Any) -> np.ndarray:
    """Parse embedding from database format to numpy array.

    Handles multiple storage formats:
    - Binary (BYTEA/BLOB/BinData): Most common, used by all databases
    - JSON string: Legacy format
    - Native array: Fallback
    """
    if isinstance(raw, bytes | memoryview):
        return np.frombuffer(raw, dtype="<f4")
    if isinstance(raw, str):
        return np.array(json.loads(raw), dtype=np.float32)

    if hasattr(raw, "__bytes__"):
        return np.frombuffer(bytes(raw), dtype="<f4")
    return np.asarray(raw, dtype=np.float32)
