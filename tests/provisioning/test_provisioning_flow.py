import pytest

import memori.provisioning as provisioning
from memori._exceptions import (
    MissingPyMySQLError,
    MissingPsycopgError,
    UnsupportedProvisionedDatabaseFamilyError,
)
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
            family="unsupported-family",
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
            family="unsupported-family",
            dsn="postgresql://user:pass@host/db",
        ),
    )

    with pytest.raises(UnsupportedProvisionedDatabaseFamilyError):
        provision_memori(provider="neon-launchpad")


def test_provision_memori_checks_driver_before_provider_or_cache(mocker):
    require_driver = mocker.patch(
        "memori.provisioning.require_mysql_driver",
        side_effect=MissingPyMySQLError("TiDB Zero"),
    )
    get_result = mocker.patch("memori.provisioning.get_provision_result")

    with pytest.raises(MissingPyMySQLError):
        provision_memori(provider="tidb-zero")

    require_driver.assert_called_once_with("TiDB Zero")
    get_result.assert_not_called()


## PG DRIVE FLOW TEST

def test_provision_memori_postgresql_success_flow(mocker):
    require_pg = mocker.patch("memori.provisioning.require_pg_driver")
    result = ProvisionResult(
        provider="neon-launchpad",
        family="postgresql",
        dsn="postgresql://user:pass@host/db",
    )
    mocker.patch(
        "memori.provisioning.get_provision_result",
        return_value=result,
    )
    conn_factory = mocker.Mock()
    mocker.patch(
        "memori.provisioning.pg_connection_factory",
        return_value=conn_factory,
    )
    mem = mocker.Mock()
    mocker.patch("memori.Memori", return_value=mem)
    build = mocker.patch("memori.storage._manager.Manager.build")

    returned = provision_memori(provider="neon-launchpad", build=True)

    require_pg.assert_called_once_with("Neon Launchpad")
    build.assert_called_once_with()
    assert returned is mem


def test_provision_memori_checks_pg_driver_before_provider_or_cache(mocker):
    require_driver = mocker.patch(
        "memori.provisioning.require_pg_driver",
        side_effect=MissingPsycopgError("Neon Launchpad"),
    )
    get_result = mocker.patch("memori.provisioning.get_provision_result")

    with pytest.raises(MissingPsycopgError):
        provision_memori(provider="neon-launchpad")

    require_driver.assert_called_once_with("Neon Launchpad")
    get_result.assert_not_called()