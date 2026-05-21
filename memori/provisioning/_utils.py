from __future__ import annotations

from collections.abc import Callable
from typing import Any
from urllib.parse import parse_qs, quote, unquote, urlparse, urlunparse

from memori._exceptions import MissingPyMySQLError


def redact_dsn(dsn: str) -> str:
    parsed = urlparse(dsn)
    if not parsed.scheme or not parsed.netloc:
        return dsn

    username = parsed.username
    password = parsed.password
    if username is None and password is None:
        return dsn

    host = parsed.hostname or ""
    userinfo = quote(username or "", safe="")
    if password is not None:
        userinfo += ":****"

    if ":" in host and not host.startswith("["):
        host = f"[{host}]"

    netloc = f"{userinfo}@{host}" if userinfo else host
    if parsed.port is not None:
        netloc = f"{netloc}:{parsed.port}"

    return urlunparse(parsed._replace(netloc=netloc))


def mysql_connection_factory(
    dsn: str,
    connect_args: dict[str, Any] | None = None,
) -> Callable[[], Any]:
    try:
        import pymysql
    except ImportError as e:
        raise MissingPyMySQLError("TiDB Zero") from e

    kwargs = _mysql_kwargs_from_dsn(dsn)
    kwargs.update(connect_args or {})

    return lambda: pymysql.connect(**kwargs)


def _mysql_kwargs_from_dsn(dsn: str) -> dict[str, Any]:
    parsed = urlparse(dsn)
    if parsed.scheme not in {"mysql", "mysql+pymysql"}:
        raise ValueError(f"Unsupported TiDB Zero DSN scheme: {parsed.scheme}")

    if parsed.hostname is None:
        raise ValueError("TiDB Zero DSN must include a hostname")

    kwargs: dict[str, Any] = {
        "host": parsed.hostname,
        "port": parsed.port or 4000,
        "user": unquote(parsed.username or ""),
        "password": unquote(parsed.password or ""),
        "database": unquote(parsed.path.lstrip("/")),
    }

    query = parse_qs(parsed.query, keep_blank_values=True)
    ssl_mode = _first(query, "ssl-mode") or _first(query, "sslmode")
    if ssl_mode is not None and ssl_mode.lower() not in {"disable", "disabled"}:
        kwargs["ssl"] = {}

    charset = _first(query, "charset")
    if charset:
        kwargs["charset"] = charset

    return kwargs


def _first(query: dict[str, list[str]], key: str) -> str | None:
    values = query.get(key)
    if not values:
        return None
    return values[0]
