from __future__ import annotations

from collections.abc import Callable
from typing import Any

from memori._exceptions import UnsupportedProvisioningProviderError
from memori.provisioning._models import ProvisionResult

Provider = Callable[..., ProvisionResult]


class Registry:
    _providers: dict[str, Provider] = {}

    @classmethod
    def register_provider(cls, name: str) -> Callable[[Provider], Provider]:
        def decorator(provider: Provider) -> Provider:
            cls._providers[name] = provider
            return provider

        return decorator

    def provider(self, name: str) -> Provider:
        try:
            return self._providers[name]
        except KeyError as e:
            supported = ", ".join(sorted(self._providers)) or "none"
            raise UnsupportedProvisioningProviderError(name, supported) from e


def provision(provider: str, **kwargs: Any) -> ProvisionResult:
    return Registry().provider(provider)(**kwargs)
