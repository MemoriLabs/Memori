import json
import os

from memori.provisioning import ProvisionResult
from memori.provisioning._cache import ProvisionCache, cache_key, default_cache_path


def test_default_cache_path_uses_memori_home(monkeypatch, tmp_path):
    monkeypatch.setenv("MEMORI_HOME", str(tmp_path))

    assert default_cache_path() == tmp_path / ".memori" / "provisioning.json"


def test_cache_reuses_unexpired_result(tmp_path):
    cache = ProvisionCache(tmp_path / "provisioning.json")
    result = ProvisionResult(
        provider="tidb-zero",
        family="mysql",
        dsn="mysql://user:pass@host/db",
        expires_at="2999-01-01T00:00:00Z",
    )

    cache.set("tidb-zero:memori", result)

    assert cache.get("tidb-zero:memori") == result


def test_cache_ignores_expired_result(tmp_path):
    cache = ProvisionCache(tmp_path / "provisioning.json")
    result = ProvisionResult(
        provider="tidb-zero",
        family="mysql",
        dsn="mysql://user:pass@host/db",
        expires_at="2000-01-01T00:00:00Z",
    )

    cache.set("tidb-zero:memori", result)

    assert cache.get("tidb-zero:memori") is None
    assert "tidb-zero:memori" not in json.loads(cache.path.read_text())


def test_cache_treats_malformed_expiry_as_expired(tmp_path):
    cache = ProvisionCache(tmp_path / "provisioning.json")
    result = ProvisionResult(
        provider="tidb-zero",
        family="mysql",
        dsn="mysql://user:pass@host/db",
        expires_at="not-a-date",
    )

    cache.set("tidb-zero:memori", result)

    assert cache.get("tidb-zero:memori") is None
    assert "tidb-zero:memori" not in json.loads(cache.path.read_text())


def test_cache_ignores_corrupted_file(tmp_path):
    path = tmp_path / "provisioning.json"
    path.write_text("{not json")

    assert ProvisionCache(path).get("tidb-zero:memori") is None


def test_cache_key_prefers_override():
    assert cache_key("tidb-zero", "tag", "custom") == "tidb-zero:custom"
    assert cache_key("tidb-zero", "tag", None) == "tidb-zero:tag"
    assert cache_key("tidb-zero", None, None) == "tidb-zero"


def test_cache_writes_json(tmp_path):
    path = tmp_path / "provisioning.json"
    cache = ProvisionCache(path)

    cache.set(
        "tidb-zero:memori",
        ProvisionResult(
            provider="tidb-zero",
            family="mysql",
            dsn="mysql://user:pass@host/db",
        ),
    )

    assert json.loads(path.read_text())["tidb-zero:memori"]["provider"] == "tidb-zero"
    if os.name != "nt":
        assert path.stat().st_mode & 0o777 == 0o600
