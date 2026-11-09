r"""
 __  __                           _
|  \/  | ___ _ __ ___   ___  _ __(_)
| |\/| |/ _ \ '_ ` _ \ / _ \| '__| |
| |  | |  __/ | | | | | (_) | |  | |
|_|  |_|\___|_| |_| |_|\___/|_|  |_|
                  perfectam memoriam
                       memorilabs.ai
"""

from collections.abc import Callable, Generator
from contextlib import contextmanager
from typing import Any

from memori.storage._base import BaseStorageAdapter
from memori.storage._registry import Registry


class ConnectionFactory:
    """A callable connection source that records who owns its connections.

    A factory supplied by the user creates a new connection per call, so
    connection_context should close what it created. A connection passed
    directly to Memori belongs to the caller, so it stays open until
    Memori.close().
    """

    def __init__(
        self, factory: Callable[[], Any], *, should_close: bool = True
    ) -> None:
        self._factory = factory
        self.should_close = should_close

    def __call__(self) -> Any:
        return self._factory()


@contextmanager
def connection_context(
    conn_factory: Callable[[], Any] | None,
) -> Generator[
    tuple[Any, BaseStorageAdapter, Any] | tuple[None, None, None], None, None
]:
    if conn_factory is None:
        yield None, None, None
        return

    conn = conn_factory()
    adapter = Registry().adapter(lambda: conn)
    driver = Registry().driver(adapter)
    close_connection = getattr(conn_factory, "should_close", True)

    try:
        yield conn, adapter, driver
        adapter.commit()
    except Exception:
        try:
            adapter.rollback()
        except Exception:  # nosec B110
            pass
        raise
    finally:
        if close_connection:
            try:
                adapter.close()
            except Exception:  # nosec B110
                pass
