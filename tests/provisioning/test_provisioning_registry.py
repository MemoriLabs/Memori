import pytest

from memori._exceptions import UnsupportedProvisioningProviderError
from memori.provisioning._registry import Registry


def test_unknown_provider_raises_dedicated_error():
    with pytest.raises(UnsupportedProvisioningProviderError) as exc_info:
        Registry().provider("unknown-provider")

    assert "Unsupported provisioning provider: unknown-provider" in str(exc_info.value)
    assert "tidb-zero" in str(exc_info.value)
