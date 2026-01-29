r"""
 __  __                           _
|  \/  | ___ _ __ ___   ___  _ __(_)
| |\/| |/ _ \ '_ ` _ \ / _ \| '__| |
| |  | |  __/ | | | | | (_) | |  | |
|_|  |_|\___|_| |_| |_|\___/|_|  |_|
                  perfectam memoriam
                       memorilabs.ai
"""

from collections.abc import Callable
from typing import Any

from memori.storage._base import BaseStorageAdapter


class Registry:
    _adapters: dict[Callable[[Any], bool], type[BaseStorageAdapter]] = {}
    _drivers: dict[str, type] = {}

    @classmethod
    def register_adapter(cls, matcher: Callable[[Any], bool]):
        def decorator(adapter_class: type[BaseStorageAdapter]):
            cls._adapters[matcher] = adapter_class
            return adapter_class

        return decorator

    @classmethod
    def register_driver(cls, dialect: str):
        def decorator(driver_class: type):
            cls._drivers[dialect] = driver_class
            return driver_class

        return decorator

    def adapter(self, conn: Any) -> BaseStorageAdapter:
        conn_to_check = conn() if callable(conn) else conn

        # Support factories that return (conn, release) tuples.
        conn_for_match = (
            conn_to_check[0] if isinstance(conn_to_check, tuple) else conn_to_check
        )

        # Support factories that return a context manager (e.g. psycopg_pool).
        if BaseStorageAdapter._is_managed_resource(conn_for_match):
            cm = conn_for_match
            real_conn = cm.__enter__()

            def _release(cm=cm):
                return cm.__exit__(None, None, None)

            conn_to_check = (real_conn, _release)
            conn_for_match = real_conn

        for matcher, adapter_class in self._adapters.items():
            if matcher(conn_for_match):
                return adapter_class(lambda: conn_to_check)

        raise RuntimeError(
            f"Unsupported database: {type(conn_for_match).__module__}.{type(conn_for_match).__name__}"
        )

    def driver(self, conn: BaseStorageAdapter):
        dialect = conn.get_dialect()
        if dialect not in self._drivers:
            raise RuntimeError(f"Unsupported database dialect: {dialect}")
        return self._drivers[dialect](conn)
