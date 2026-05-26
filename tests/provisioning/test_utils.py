import builtins
import sys
from types import SimpleNamespace

import pytest

from memori._exceptions import MissingPyMySQLError
from memori.provisioning._utils import mysql_connection_factory, redact_dsn


@pytest.mark.parametrize(
    ("dsn", "expected"),
    [
        (
            "mysql://user:secret@example.com:4000/db?ssl-mode=REQUIRED",
            "mysql://user:****@example.com:4000/db?ssl-mode=REQUIRED",
        ),
        ("mysql://user@example.com/db", "mysql://user@example.com/db"),
        ("not a dsn", "not a dsn"),
    ],
)
def test_redact_dsn(dsn, expected):
    assert redact_dsn(dsn) == expected


def test_mysql_connection_factory_parses_tidb_dsn(monkeypatch):
    calls = []

    def connect(**kwargs):
        calls.append(kwargs)
        return object()

    monkeypatch.setitem(sys.modules, "pymysql", SimpleNamespace(connect=connect))

    factory = mysql_connection_factory(
        "mysql://user:secret@example.com:4000/memori?ssl-mode=REQUIRED&charset=utf8mb4"
    )

    factory()

    assert calls == [
        {
            "host": "example.com",
            "port": 4000,
            "user": "user",
            "password": "secret",
            "database": "memori",
            "ssl": {},
            "charset": "utf8mb4",
        }
    ]


def test_mysql_connection_factory_missing_pymysql(monkeypatch):
    monkeypatch.delitem(sys.modules, "pymysql", raising=False)
    real_import = builtins.__import__

    def import_without_pymysql(name, *args, **kwargs):
        if name == "pymysql":
            raise ImportError("No module named pymysql")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_pymysql)

    with pytest.raises(MissingPyMySQLError):
        mysql_connection_factory("mysql://user:secret@example.com/db")
