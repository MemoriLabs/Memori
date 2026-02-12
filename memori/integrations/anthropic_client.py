"""Anthropic integration helper.

Memori already supports the official `anthropic` Python SDK via `Memori(...).llm.register(client)`.

This module provides a small convenience helper for first-time users so they don't
feel like everything is "OpenAI-only".

Usage:

```python
from memori.integrations.anthropic_client import anthropic_client
from memori import Memori

client = anthropic_client()  # uses ANTHROPIC_API_KEY
mem = Memori(conn=...).llm.register(client)
```
"""

from __future__ import annotations

import os
from typing import Any


def anthropic_client(*, api_key: str | None = None, **kwargs: Any):
    """Create an Anthropic client using `ANTHROPIC_API_KEY` by default."""
    try:
        from anthropic import Anthropic
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Anthropic Python SDK is required. Install with: pip install anthropic"
        ) from e

    api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Set it in your environment or pass api_key=."
        )

    return Anthropic(api_key=api_key, **kwargs)
