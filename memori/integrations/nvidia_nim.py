"""NVIDIA NIM integration helpers.

Memori's core LLM integration works by wrapping an LLM client object.

NVIDIA NIM exposes an OpenAI-compatible endpoint, so the standard OpenAI Python
client can be used by setting `base_url`.

Docs:
- https://docs.api.nvidia.com/nim/reference/llm-apis

This helper is intentionally small and optional: it creates a correctly-configured
OpenAI client for NVIDIA NIM. Users can still use LiteLLM or any other OpenAI-
compatible gateway the same way (by providing `base_url`).
"""

from __future__ import annotations

import os
from typing import Any


DEFAULT_NIM_BASE_URL = "https://integrate.api.nvidia.com/v1/"


def openai_client_for_nim(
    *,
    api_key: str | None = None,
    base_url: str = DEFAULT_NIM_BASE_URL,
    **kwargs: Any,
):
    """Create an OpenAI Python client configured for NVIDIA NIM.

    Args:
        api_key: NVIDIA API key. Defaults to env var `NVIDIA_API_KEY`.
        base_url: NIM OpenAI-compatible base URL.
        **kwargs: Passed through to `openai.OpenAI(...)`.

    Returns:
        An `openai.OpenAI` client instance.

    Raises:
        RuntimeError: if the OpenAI Python client is not installed.
    """

    try:
        from openai import OpenAI
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "OpenAI Python client is required for NVIDIA NIM. Install with: pip install openai"
        ) from e

    api_key = api_key or os.getenv("NVIDIA_API_KEY")
    if not api_key:
        raise RuntimeError(
            "NVIDIA_API_KEY is not set. Set it in your environment or pass api_key=."
        )

    return OpenAI(api_key=api_key, base_url=base_url, **kwargs)
