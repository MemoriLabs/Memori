"""Internal helpers for the Memori × LlamaIndex integration.

These utilities inspect a LlamaIndex LLM or agent object, locate the
underlying raw SDK client, and register it with Memori's client-patching
machinery.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from memori import Memori

logger = logging.getLogger(__name__)

# Known LlamaIndex LLM wrapper attribute paths for each SDK.
# Listed in priority order; the first one that exists and satisfies the
# hasattr guard is used.
_LLM_CLIENT_ATTRS: list[tuple[str, str]] = [
    # OpenAI  (sync client)
    ("_client", "chat"),
    # Anthropic
    ("_client", "messages"),
    # Google genai
    ("_client", "models"),
    # Async OpenAI (fallback)
    ("_aclient", "chat"),
]


def _extract_raw_sdk_client(llm: Any) -> Any | None:
    """Return the first raw SDK client found on *llm*, or None."""
    for attr, probe in _LLM_CLIENT_ATTRS:
        inner = getattr(llm, attr, None)
        if inner is not None and hasattr(inner, probe):
            return inner
    return None


def register_llamaindex_llm(memori: Memori, llm: Any) -> bool:
    """Register the underlying SDK client of a LlamaIndex LLM with Memori.

    Tries ``memori.llm.register(llm)`` first (uses auto-detection via the
    Registry).  If that raises ``UnsupportedLLMProviderError`` (e.g. the
    LlamaIndex LLM module path is unrecognised), falls back to extracting the
    raw SDK client manually and registering it directly.

    Returns:
        True if registration succeeded, False otherwise.
    """
    from memori._exceptions import UnsupportedLLMProviderError

    # Primary path — auto-detection via registered matcher.
    try:
        memori.llm.register(llm)
        logger.debug(
            "Registered LlamaIndex LLM via auto-detection: %s",
            type(llm).__name__,
        )
        return True
    except UnsupportedLLMProviderError:
        pass

    # Fallback — extract raw client and register it directly.
    inner = _extract_raw_sdk_client(llm)
    if inner is None:
        logger.warning(
            "Could not extract an underlying SDK client from LlamaIndex LLM %s. "
            "Memory will not be active for this LLM.",
            type(llm).__qualname__,
        )
        return False

    try:
        memori.llm.register(inner)
        logger.debug(
            "Registered raw SDK client %s extracted from LlamaIndex LLM %s.",
            type(inner).__name__,
            type(llm).__name__,
        )
        return True
    except UnsupportedLLMProviderError as exc:
        logger.warning(
            "Memori could not register the SDK client for LlamaIndex LLM %s: %s",
            type(llm).__qualname__,
            exc,
        )
        return False


def get_llm_from_engine_or_agent(obj: Any) -> Any | None:
    """Return the LLM object from a LlamaIndex chat engine or agent.

    Checks common attribute names used across LlamaIndex versions.
    """
    for attr in ("llm", "_llm", "llm_predictor"):
        candidate = getattr(obj, attr, None)
        if candidate is not None:
            # Unwrap LLMPredictor if needed (older LlamaIndex versions)
            if hasattr(candidate, "llm"):
                return candidate.llm
            return candidate
    return None
