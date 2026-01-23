r"""
 __  __                           _
|  \/  | ___ _ __ ___   ___  _ __(_)
| |\/| |/ _ \ '_ ` _ \ / _ \| '__| |
| |  | |  __/ | | | | | (_) | |  | |
|_|  |_|\___|_| |_| |_|\___/|_|  |_|
                  perfectam memoriam
                       memorilabs.ai
"""


class BaseStorageAdapter:
    def __init__(self, conn):
        if not callable(conn):
            raise TypeError("conn must be a callable")
        self._release = None
        self._cm = None

        resource = conn()
        if isinstance(resource, tuple) and len(resource) == 2 and callable(resource[1]):
            self.conn = resource[0]
            self._release = resource[1]
            return

        # Support factories that return a context manager, e.g.
        # psycopg_pool.ConnectionPool.connection() which must be exited to
        # return the connection to the pool.
        if self._is_managed_resource(resource):
            self._cm = resource
            self.conn = resource.__enter__()

            def _release():
                try:
                    self._cm.__exit__(None, None, None)
                finally:
                    self._cm = None

            self._release = _release
            return

        self.conn = resource

    def close(self):
        if self.conn is not None:
            if self._release is not None:
                try:
                    self._release()
                finally:
                    self._release = None
                    self.conn = None
                return

            if hasattr(self.conn, "close"):
                self.conn.close()
            self.conn = None

    @staticmethod
    def _is_managed_resource(obj) -> bool:
        # Only treat as a managed resource if it looks like a context manager
        # and does NOT look like a DB connection/session itself.
        if not (hasattr(obj, "__enter__") and hasattr(obj, "__exit__")):
            return False

        # DB-API connections commonly have cursor/commit/rollback.
        if (
            hasattr(obj, "cursor")
            and hasattr(obj, "commit")
            and hasattr(obj, "rollback")
        ):
            return False

        # SQLAlchemy Session has get_bind.
        if hasattr(obj, "get_bind"):
            return False

        # Django connection has vendor.
        if hasattr(obj, "vendor"):
            return False

        # MongoDB clients/dbs have list_collection_names.
        if hasattr(obj, "list_collection_names"):
            return False

        return True

    def commit(self):
        raise NotImplementedError

    def execute(self, *args, **kwargs):
        raise NotImplementedError

    def flush(self):
        raise NotImplementedError

    def get_dialect(self):
        raise NotImplementedError

    def rollback(self):
        raise NotImplementedError


class BaseConversation:
    def __init__(self, conn: BaseStorageAdapter):
        self.conn = conn

    def create(self, session_id: int, timeout_minutes: int):
        raise NotImplementedError

    def update(self, id: int, summary: str):
        raise NotImplementedError

    def read(self, id: int) -> dict | None:
        raise NotImplementedError


class BaseConversationMessage:
    def __init__(self, conn: BaseStorageAdapter):
        self.conn = conn

    def create(self, conversation_id: int, role: str, type: str, content: str):
        raise NotImplementedError


class BaseConversationMessages:
    def __init__(self, conn: BaseStorageAdapter):
        self.conn = conn

    def read(self, conversation_id: int):
        raise NotImplementedError


class BaseKnowledgeGraph:
    def __init__(self, conn: BaseStorageAdapter):
        self.conn = conn

    def create(self, entity_id: int, semantic_triples: list):
        raise NotImplementedError


class BaseEntity:
    def __init__(self, conn: BaseStorageAdapter):
        self.conn = conn

    def create(self, external_id: str):
        raise NotImplementedError


class BaseEntityFact:
    def __init__(self, conn: BaseStorageAdapter):
        self.conn = conn

    def create(self, entity_id: int, facts: list, fact_embeddings: list | None = None):
        raise NotImplementedError

    def get_embeddings(self, entity_id: int, limit: int = 1000):
        raise NotImplementedError

    def get_facts_by_ids(self, fact_ids: list[int]):
        raise NotImplementedError


class BaseProcess:
    def __init__(self, conn: BaseStorageAdapter):
        self.conn = conn

    def create(self, external_id: str):
        raise NotImplementedError


class BaseProcessAttribute:
    def __init__(self, conn: BaseStorageAdapter):
        self.conn = conn

    def create(self, process_id: int, attributes: list):
        raise NotImplementedError


class BaseSession:
    def __init__(self, conn: BaseStorageAdapter):
        self.conn = conn

    def create(self, uuid: str, entity_id: int, process_id: int):
        raise NotImplementedError


class BaseSchema:
    def __init__(self, conn: BaseStorageAdapter):
        self.conn = conn


class BaseSchemaVersion:
    def __init__(self, conn: BaseStorageAdapter):
        self.conn = conn

    def create(self, num: int):
        raise NotImplementedError

    def delete(self):
        raise NotImplementedError

    def read(self):
        raise NotImplementedError
