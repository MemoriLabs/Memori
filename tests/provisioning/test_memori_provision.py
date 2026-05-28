from memori import Memori
from memori.provisioning import ProvisionResult


class FakeCursor:
    description = [("version",)]

    def execute(self, _operation, _binds=()):
        return None

    def fetchone(self):
        return ("8.0.11-TiDB",)

    def close(self):
        return None


class FakeConnection:
    def cursor(self):
        return FakeCursor()

    def commit(self):
        return None

    def rollback(self):
        return None


FakeConnection.__module__ = "pymysql.connections"


def test_memori_provision_returns_byodb_instance(mocker):
    result = ProvisionResult(
        provider="tidb-zero",
        family="mysql",
        dsn="mysql://user:secret@example.com:4000/memori?ssl-mode=REQUIRED",
    )
    get_result = mocker.patch("memori.provisioning.get_provision_result")
    get_result.return_value = result
    mocker.patch(
        "memori.provisioning.mysql_connection_factory",
        return_value=lambda: FakeConnection(),
    )
    build = mocker.patch("memori.storage._manager.Manager.build")
    mocker.patch("memori.native.RustCoreAdapter.maybe_create", return_value=None)

    mem = Memori.provision(provider="tidb-zero", build=True)

    assert mem.config.cloud is False
    assert mem.config.byodb is True
    assert mem.config.provision_result == result
    assert mem.config.storage.adapter.get_dialect() == "tidb"
    build.assert_called_once_with()


def test_memori_provision_can_skip_build(mocker):
    result = ProvisionResult(
        provider="tidb-zero",
        family="mysql",
        dsn="mysql://user:secret@example.com:4000/memori",
    )
    mocker.patch("memori.provisioning.get_provision_result", return_value=result)
    mocker.patch(
        "memori.provisioning.mysql_connection_factory",
        return_value=lambda: FakeConnection(),
    )
    build = mocker.patch("memori.storage._manager.Manager.build")
    mocker.patch("memori.native.RustCoreAdapter.maybe_create", return_value=None)

    Memori.provision(provider="tidb-zero", build=False)

    build.assert_not_called()
