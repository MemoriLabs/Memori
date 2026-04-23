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


@Registry.register_adapter(
    lambda conn: type(conn).__module__ == "sqlalchemy.orm.session"
)
class Adapter(BaseStorageAdapter):
    def __init__(self, conn):
        super().__init__(conn)
        self._detected_dialect = None

    def commit(self):
        self.conn.commit()
        return self

    def execute(self, operation, binds=()):
        return self.conn.connection().exec_driver_sql(operation, binds)

    def flush(self):
        self.conn.flush()
        return self

    def get_dialect(self):
        if self._detected_dialect is not None:
            return self._detected_dialect

        dialect = self.conn.get_bind().dialect
        module_name = dialect.__class__.__module__
        if module_name.startswith("pyobvector."):
            self._detected_dialect = "oceanbase"
            return self._detected_dialect

        detected = dialect.name
        if detected in {"mysql", "mariadb"} and self._is_tidb_server():
            detected = "tidb"

        self._detected_dialect = detected
        return detected

    def _is_tidb_server(self) -> bool:
        try:
            version = (
                self.conn.connection().exec_driver_sql("SELECT VERSION()").scalar()
            )
        except Exception:
            return False

        return isinstance(version, str) and "tidb" in version.lower()

    def rollback(self):
        self.conn.rollback()
        return self
