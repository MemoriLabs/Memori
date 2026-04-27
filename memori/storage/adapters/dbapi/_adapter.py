r"""
 __  __                           _
|  \/  | ___ _ __ ___   ___  _ __(_)
| |\/| |/ _ \ '_ ` _ \ / _ \| '__| |
| |  | |  __/ | | | | | (_) | |  | |
|_|  |_|\___|_| |_| |_|\___/|_|  |_|
                  perfectam memoriam
                       memorilabs.ai
"""

from memori.storage._base import BaseStorageAdapter
from memori.storage._registry import Registry


class CursorWrapper:
    def __init__(self, cursor):
        self._cursor = cursor

    def mappings(self):
        return MappingResult(self._cursor)

    def __getattr__(self, name):
        return getattr(self._cursor, name)


class MappingResult:
    def __init__(self, cursor):
        self._cursor = cursor

    def fetchone(self):
        row = self._cursor.fetchone()
        if row is None:
            return None
        columns = [col[0] for col in self._cursor.description]
        return dict(zip(columns, row, strict=True))

    def fetchall(self):
        rows = self._cursor.fetchall()
        columns = [col[0] for col in self._cursor.description]
        return [dict(zip(columns, row, strict=True)) for row in rows]


def is_dbapi_connection(conn):
    if not (
        hasattr(conn, "cursor")
        and hasattr(conn, "commit")
        and hasattr(conn, "rollback")
        and callable(getattr(conn, "cursor", None))
        and callable(getattr(conn, "commit", None))
        and callable(getattr(conn, "rollback", None))
    ):
        return False

    if hasattr(conn, "__class__"):
        module_name = conn.__class__.__module__
        if module_name.startswith("django.db"):
            return False
        class_name = conn.__class__.__name__
        if class_name in ("Session", "scoped_session", "AsyncSession"):
            return False
        if hasattr(conn, "get_bind"):
            return False

    return True


@Registry.register_adapter(is_dbapi_connection)
class Adapter(BaseStorageAdapter):
    def __init__(self, conn):
        super().__init__(conn)
        self._detected_dialect = None

    def commit(self):
        self.conn.commit()
        return self

    def execute(self, operation, binds=()):
        cursor = self.conn.cursor()
        try:
            cursor.execute(operation, binds)
            return CursorWrapper(cursor)
        except Exception:
            cursor.close()
            raise

    def flush(self):
        return self

    def get_dialect(self):
        if self._detected_dialect is not None:
            return self._detected_dialect

        module_name = type(self.conn).__module__
        dialect_mapping = {
            "postgresql": ["psycopg"],
            "mysql": ["mysql", "MySQLdb", "pymysql"],
            "oceanbase": ["pyobvector"],
            "sqlite": ["sqlite"],
            "oracle": ["cx_Oracle", "oracledb"],
        }
        for dialect, identifiers in dialect_mapping.items():
            if any(identifier in module_name for identifier in identifiers):
                detected = dialect
                if detected in {"mysql", "mariadb"} and self._is_tidb_server():
                    detected = "tidb"
                self._detected_dialect = detected
                return detected
        raise ValueError(
            f"Unable to determine dialect from connection module: {module_name}"
        )

    def _is_tidb_server(self) -> bool:
        cursor = self.conn.cursor()
        try:
            cursor.execute("SELECT VERSION()")
            row = cursor.fetchone()
        except Exception:
            return False
        finally:
            cursor.close()

        version = row[0] if row else None
        return isinstance(version, str) and "tidb" in version.lower()

    def rollback(self):
        self.conn.rollback()
        return self
