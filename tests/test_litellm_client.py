"""Unit tests for the LiteLLM client integration in Memori.

Run with:
    pytest tests/test_litellm_client.py -v
"""

from __future__ import annotations

import os
import sys
import types
from unittest.mock import MagicMock

import pytest

# Build a fake litellm module so the real package is not required at test time.
_fake_litellm = types.ModuleType("litellm")
_fake_litellm.completion = MagicMock()
_fake_litellm.acompletion = MagicMock()
_fake_litellm.__version__ = "0.0.0-test"
sys.modules.setdefault("litellm", _fake_litellm)

from memori.llm._utils import client_is_litellm
from memori.llm.clients import LiteLLM


def test_client_is_litellm_matches_module() -> None:
    import litellm

    assert client_is_litellm(litellm) is True


def test_client_is_litellm_rejects_other_modules() -> None:
    assert client_is_litellm(os) is False
    assert client_is_litellm(sys) is False


def test_client_is_litellm_rejects_arbitrary_objects() -> None:
    assert client_is_litellm(object()) is False
    assert client_is_litellm("litellm") is False
    assert client_is_litellm({"name": "litellm"}) is False


def test_client_is_litellm_accepts_submodule() -> None:
    """Submodules like litellm.completion or litellm.utils should also match."""
    fake_submodule = types.ModuleType("litellm.proxy")
    assert client_is_litellm(fake_submodule) is True


def test_litellm_register_requires_completion_attr() -> None:
    """If user passes something other than the litellm module, register() must fail loudly."""
    from memori._config import Config

    config = Config()
    bogus_module = types.ModuleType("not_litellm")

    client = LiteLLM(config)
    with pytest.raises(RuntimeError, match="not the litellm module"):
        client.register(bogus_module)


def test_litellm_register_wraps_completion_and_acompletion() -> None:
    """After register(), litellm.completion / litellm.acompletion should be replaced
    with Invoke-wrapped callables that retain a backup of the originals."""
    from memori._config import Config

    fake_litellm = types.ModuleType("litellm")
    original_completion = MagicMock(return_value=MagicMock())
    original_acompletion = MagicMock(return_value=MagicMock())
    fake_litellm.completion = original_completion
    fake_litellm.acompletion = original_acompletion

    config = Config()
    client = LiteLLM(config)
    client.register(fake_litellm)

    # Backups stored on the module
    assert fake_litellm._completion is original_completion
    assert fake_litellm._acompletion is original_acompletion
    # `completion` / `acompletion` were replaced (not the same identity)
    assert fake_litellm.completion is not original_completion
    assert fake_litellm.acompletion is not original_acompletion
    # Idempotency marker present
    assert fake_litellm._memori_installed is True


def test_litellm_register_is_idempotent() -> None:
    """Calling register twice should not double-wrap."""
    from memori._config import Config

    fake_litellm = types.ModuleType("litellm")
    fake_litellm.completion = MagicMock()
    fake_litellm.acompletion = MagicMock()

    config = Config()
    LiteLLM(config).register(fake_litellm)
    first_wrapped = fake_litellm.completion

    LiteLLM(config).register(fake_litellm)
    second_wrapped = fake_litellm.completion

    assert first_wrapped is second_wrapped


def test_litellm_register_sets_provider_metadata() -> None:
    """The Memori config should be marked with the LiteLLM provider name."""
    from memori._config import Config
    from memori.llm._constants import LITELLM_LLM_PROVIDER

    fake_litellm = types.ModuleType("litellm")
    fake_litellm.completion = MagicMock()
    fake_litellm.acompletion = MagicMock()
    fake_litellm.__version__ = "1.99.99"

    config = Config()
    LiteLLM(config).register(fake_litellm)

    assert config.llm.provider == LITELLM_LLM_PROVIDER
    assert config.llm.provider_sdk_version == "1.99.99"


def test_litellm_registered_in_registry() -> None:
    """LiteLLM should be discoverable through the Registry by passing a litellm module."""
    from memori._config import Config
    from memori.llm._registry import Registry

    fake_litellm = types.ModuleType("litellm")
    fake_litellm.completion = MagicMock()

    registry = Registry()
    config = Config()
    client = registry.client(fake_litellm, config)
    assert isinstance(client, LiteLLM)
