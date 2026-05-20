"""Unit tests for the LiteLLM client integration in Memori.

Run with:
    pytest tests/test_litellm_client.py -v
"""

from __future__ import annotations

import os
import sys
import types
from unittest.mock import MagicMock

import litellm
import pytest

from memori.llm._utils import client_is_litellm
from memori.llm.clients import LiteLLM


def test_client_is_litellm_matches_module() -> None:
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
    """If user passes something without completion, register() must fail loudly."""
    from memori._config import Config

    config = Config()
    bogus_module = types.ModuleType("not_litellm")

    client = LiteLLM(config)
    with pytest.raises(
        RuntimeError, match="expected the litellm module or a LiteLLM Router"
    ):
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


def test_client_is_litellm_matches_router_object() -> None:
    """A litellm.Router instance should be recognized by client_is_litellm()."""

    class FakeRouter:
        """Simulates litellm.Router which lives in litellm.router module."""

        def completion(self, **kwargs):
            pass

        def acompletion(self, **kwargs):
            pass

    # litellm.Router's __module__ is "litellm.router"
    FakeRouter.__module__ = "litellm.router"
    router = FakeRouter()
    assert client_is_litellm(router) is True


def test_litellm_register_wraps_router_instance_methods() -> None:
    """register() should wrap completion/acompletion on a Router-style object."""
    from memori._config import Config

    class FakeRouter:
        pass

    FakeRouter.__module__ = "litellm.router"
    router = FakeRouter()
    router.completion = MagicMock()
    router.acompletion = MagicMock()
    router.__version__ = "1.99.99"
    original_completion = router.completion

    config = Config()
    client = LiteLLM(config)
    client.register(router)

    # Backups stored on the object
    assert router._completion is original_completion
    # completion was replaced
    assert router.completion is not original_completion
    # Idempotency marker present
    assert router._memori_installed is True


def test_litellm_registered_in_registry_via_module() -> None:
    """LiteLLM module should be discoverable through the Registry."""
    from memori._config import Config
    from memori.llm._registry import Registry

    fake_litellm = types.ModuleType("litellm")
    fake_litellm.completion = MagicMock()

    registry = Registry()
    config = Config()
    client = registry.client(fake_litellm, config)
    assert isinstance(client, LiteLLM)


def test_litellm_registered_in_registry_via_router() -> None:
    """litellm.Router instances should also be discoverable through the Registry.

    This is the recommended registration path for app/server use:
        import litellm
        router = litellm.Router(model_list=[...])
        memori.llm.register(router)
    """
    from memori._config import Config
    from memori.llm._registry import Registry

    class FakeRouter:
        pass

    FakeRouter.__module__ = "litellm.router"
    router = FakeRouter()
    router.completion = MagicMock()
    router.acompletion = MagicMock()

    registry = Registry()
    config = Config()
    client = registry.client(router, config)
    assert isinstance(client, LiteLLM)
