r"""
 __  __                           _
|  \/  | ___ _ __ ___   ___  _ __(_)
| |\/| |/ _ \ '_ ` _ \ / _ \| '__| |
| |  | |  __/ | | | | | (_) | |  | |
|_|  |_|\___|_| |_| |_|\___/|_|  |_|
                  perfectam memoriam
                       memorilabs.ai
"""

from uuid import uuid4

from memori._utils import generate_uniq
from memori.storage._registry import Registry
from memori.storage.drivers.mysql._driver import Driver as MysqlDriver
from memori.storage.drivers.mysql._driver import EntityFact as MysqlEntityFact
from memori.storage.migrations._oceanbase import migrations


class EntityFact(MysqlEntityFact):
    def create(
        self,
        entity_id: int,
        facts: list,
        fact_embeddings: list | None = None,
        conversation_id: int | None = None,
    ):
        if facts is None or len(facts) == 0:
            return self

        from memori.embeddings import format_embedding_for_db

        dialect = self.conn.get_dialect()

        for i, fact in enumerate(facts):
            embedding = (
                fact_embeddings[i]
                if fact_embeddings and i < len(fact_embeddings)
                else []
            )
            embedding_formatted = format_embedding_for_db(embedding, dialect)
            uniq = generate_uniq(fact)

            self.conn.execute(
                """
                INSERT INTO memori_entity_fact(
                    uuid,
                    entity_id,
                    content,
                    content_embedding,
                    num_times,
                    date_last_time,
                    uniq
                ) VALUES (
                    %s,
                    %s,
                    %s,
                    %s,
                    %s,
                    current_timestamp(),
                    %s
                )
                ON DUPLICATE KEY UPDATE
                    num_times = num_times + 1,
                    date_last_time = current_timestamp()
                """,
                (
                    uuid4(),
                    entity_id,
                    fact,
                    embedding_formatted,
                    1,
                    uniq,
                ),
            )

            if conversation_id is not None:
                fact_row = (
                    self.conn.execute(
                        """
                        SELECT id
                          FROM memori_entity_fact
                         WHERE entity_id = %s
                           AND uniq = %s
                        """,
                        (entity_id, uniq),
                    )
                    .mappings()
                    .fetchone()
                )
                fact_id = fact_row.get("id") if fact_row else None
                if fact_id is not None:
                    self.conn.execute(
                        """
                        INSERT IGNORE INTO memori_entity_fact_mention(
                            uuid,
                            entity_id,
                            fact_id,
                            conversation_id
                        ) VALUES (
                            %s,
                            %s,
                            %s,
                            %s
                        )
                        """,
                        (uuid4(), entity_id, fact_id, conversation_id),
                    )

        self.conn.commit()

        return self


@Registry.register_driver("oceanbase")
class Driver(MysqlDriver):
    """OceanBase storage driver (MySQL-compatible)."""

    migrations = migrations
    requires_rollback_on_error = True

    def __init__(self, conn):
        super().__init__(conn)
        self.entity_fact = EntityFact(conn)
