import pytest

import memori.provisioning as provisioning
from memori._exceptions import UnsupportedProvisionedDatabaseFamilyError
from memori.provisioning import ProvisionResult, get_provision_result, provision_memori
from memori.provisioning._cache import ProvisionCache


def test_get_provision_result_reuses_cache(monkeypatch, tmp_path):
    cache_path = tmp_path / "provisioning.json"
    monkeypatch.setattr(
        provisioning,
        "ProvisionCache",
        lambda: ProvisionCache(cache_path),
    )
    calls = []

    def provider(name, **_kwargs):
        calls.append(name)
        return ProvisionResult(
            provider=name,
            family="mysql",
            dsn="mysql://user:pass@host/db",
            expires_at="2999-01-01T00:00:00Z",
        )

    monkeypatch.setattr(provisioning, "provision", provider)

    first = get_provision_result(provider="tidb-zero", tag="memori")
    second = get_provision_result(provider="tidb-zero", tag="memori")

    assert first == second
    assert calls == ["tidb-zero"]


def test_get_provision_result_cache_disabled_does_not_require_home(monkeypatch):
    monkeypatch.delenv("HOME", raising=False)
    monkeypatch.delenv("MEMORI_HOME", raising=False)

    def provider(name, **_kwargs):
        return ProvisionResult(
            provider=name,
            family="mysql",
            dsn="mysql://user:pass@host/db",
        )

    monkeypatch.setattr(provisioning, "provision", provider)

    result = get_provision_result(provider="tidb-zero", cache=False)

    assert result.provider == "tidb-zero"


def test_get_provision_result_rejects_unsupported_family_before_caching(
    monkeypatch, tmp_path
):
    cache_path = tmp_path / "provisioning.json"
    monkeypatch.setattr(
        provisioning, "ProvisionCache", lambda: ProvisionCache(cache_path)
    )

    def provider(name, **_kwargs):
        return ProvisionResult(
            provider=name,
            family="postgres",
            dsn="postgresql://user:pass@host/db",
        )

    monkeypatch.setattr(provisioning, "provision", provider)

    with pytest.raises(UnsupportedProvisionedDatabaseFamilyError):
        get_provision_result(provider="neon-launchpad")

    assert not cache_path.exists()


def test_provision_memori_rejects_unsupported_family(mocker):
    mocker.patch(
        "memori.provisioning.get_provision_result",
        return_value=ProvisionResult(
            provider="neon-launchpad",
            family="postgres",
            dsn="postgresql://user:pass@host/db",
        ),
    )

    with pytest.raises(UnsupportedProvisionedDatabaseFamilyError):
        provision_memori(provider="neon-launchpad")
