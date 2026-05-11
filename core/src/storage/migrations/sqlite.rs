use super::Migration;

pub static MIGRATIONS: &[(u32, &[Migration])] = &[
    (
        1,
        &[
            Migration {
                description: "create table memori_schema_version",
                statements: &[
                    "CREATE TABLE IF NOT EXISTS memori_schema_version(num INTEGER NOT NULL PRIMARY KEY)",
                ],
            },
            Migration {
                description: "create table memori_entity",
                statements: &["CREATE TABLE IF NOT EXISTS memori_entity(\
                        id INTEGER PRIMARY KEY AUTOINCREMENT,\
                        uuid VARCHAR(36) NOT NULL UNIQUE,\
                        external_id VARCHAR(100) NOT NULL UNIQUE,\
                        date_created DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,\
                        date_updated DATETIME DEFAULT NULL\
                    )"],
            },
            Migration {
                description: "create table memori_process",
                statements: &["CREATE TABLE IF NOT EXISTS memori_process(\
                        id INTEGER PRIMARY KEY AUTOINCREMENT,\
                        uuid VARCHAR(36) NOT NULL UNIQUE,\
                        external_id VARCHAR(100) NOT NULL UNIQUE,\
                        date_created DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,\
                        date_updated DATETIME DEFAULT NULL\
                    )"],
            },
            Migration {
                description: "create table memori_session",
                statements: &["CREATE TABLE IF NOT EXISTS memori_session(\
                        id INTEGER PRIMARY KEY AUTOINCREMENT,\
                        uuid VARCHAR(36) NOT NULL UNIQUE,\
                        entity_id INTEGER DEFAULT NULL,\
                        process_id INTEGER DEFAULT NULL,\
                        date_created DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,\
                        date_updated DATETIME DEFAULT NULL,\
                        FOREIGN KEY (entity_id) REFERENCES memori_entity (id) ON DELETE CASCADE,\
                        FOREIGN KEY (process_id) REFERENCES memori_process (id) ON DELETE CASCADE,\
                        UNIQUE (entity_id, id),\
                        UNIQUE (process_id, id)\
                    )"],
            },
            Migration {
                description: "create table memori_conversation",
                statements: &["CREATE TABLE IF NOT EXISTS memori_conversation(\
                        id INTEGER PRIMARY KEY AUTOINCREMENT,\
                        uuid VARCHAR(36) NOT NULL UNIQUE,\
                        session_id INTEGER NOT NULL UNIQUE,\
                        summary TEXT DEFAULT NULL,\
                        date_created DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,\
                        date_updated DATETIME DEFAULT NULL,\
                        FOREIGN KEY (session_id) REFERENCES memori_session (id) ON DELETE CASCADE\
                    )"],
            },
            Migration {
                description: "create table memori_conversation_message",
                statements: &["CREATE TABLE IF NOT EXISTS memori_conversation_message(\
                        id INTEGER PRIMARY KEY AUTOINCREMENT,\
                        uuid VARCHAR(36) NOT NULL UNIQUE,\
                        conversation_id INTEGER NOT NULL,\
                        role VARCHAR(255) NOT NULL,\
                        type VARCHAR(255) DEFAULT NULL,\
                        content TEXT NOT NULL,\
                        date_created DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,\
                        date_updated DATETIME DEFAULT NULL,\
                        FOREIGN KEY (conversation_id) REFERENCES memori_conversation (id) ON DELETE CASCADE,\
                        UNIQUE (conversation_id, id)\
                    )"],
            },
            Migration {
                description: "create table memori_entity_fact",
                statements: &[
                    "CREATE TABLE IF NOT EXISTS memori_entity_fact(\
                        id INTEGER PRIMARY KEY AUTOINCREMENT,\
                        uuid VARCHAR(36) NOT NULL UNIQUE,\
                        entity_id INTEGER NOT NULL,\
                        content TEXT NOT NULL,\
                        content_embedding BLOB NOT NULL,\
                        num_times INTEGER NOT NULL,\
                        date_last_time DATETIME NOT NULL,\
                        uniq CHAR(64) NOT NULL,\
                        date_created DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,\
                        date_updated DATETIME DEFAULT NULL,\
                        FOREIGN KEY (entity_id) REFERENCES memori_entity (id) ON DELETE CASCADE,\
                        UNIQUE (entity_id, id),\
                        UNIQUE (entity_id, uniq)\
                    )",
                    "CREATE INDEX IF NOT EXISTS idx_memori_entity_fact_entity_id_freq \
                     ON memori_entity_fact (entity_id, num_times DESC, date_last_time DESC)",
                    "CREATE INDEX IF NOT EXISTS idx_memori_entity_fact_embedding_search \
                     ON memori_entity_fact (entity_id, id)",
                ],
            },
            Migration {
                description: "create table memori_process_attribute",
                statements: &["CREATE TABLE IF NOT EXISTS memori_process_attribute(\
                        id INTEGER PRIMARY KEY AUTOINCREMENT,\
                        uuid VARCHAR(36) NOT NULL UNIQUE,\
                        process_id INTEGER NOT NULL,\
                        content TEXT NOT NULL,\
                        num_times INTEGER NOT NULL,\
                        date_last_time DATETIME NOT NULL,\
                        uniq CHAR(64) NOT NULL,\
                        date_created DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,\
                        date_updated DATETIME DEFAULT NULL,\
                        FOREIGN KEY (process_id) REFERENCES memori_process (id) ON DELETE CASCADE,\
                        UNIQUE (process_id, id),\
                        UNIQUE (process_id, uniq)\
                    )"],
            },
            Migration {
                description: "create table memori_subject",
                statements: &["CREATE TABLE IF NOT EXISTS memori_subject(\
                        id INTEGER PRIMARY KEY AUTOINCREMENT,\
                        uuid VARCHAR(36) NOT NULL UNIQUE,\
                        name VARCHAR(255) NOT NULL,\
                        type VARCHAR(255) NOT NULL,\
                        uniq CHAR(64) NOT NULL UNIQUE,\
                        date_created DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,\
                        date_updated DATETIME DEFAULT NULL\
                    )"],
            },
            Migration {
                description: "create table memori_predicate",
                statements: &["CREATE TABLE IF NOT EXISTS memori_predicate(\
                        id INTEGER PRIMARY KEY AUTOINCREMENT,\
                        uuid VARCHAR(36) NOT NULL UNIQUE,\
                        content TEXT NOT NULL,\
                        uniq CHAR(64) NOT NULL UNIQUE,\
                        date_created DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,\
                        date_updated DATETIME DEFAULT NULL\
                    )"],
            },
            Migration {
                description: "create table memori_object",
                statements: &["CREATE TABLE IF NOT EXISTS memori_object(\
                        id INTEGER PRIMARY KEY AUTOINCREMENT,\
                        uuid VARCHAR(36) NOT NULL UNIQUE,\
                        name VARCHAR(255) NOT NULL,\
                        type VARCHAR(255) NOT NULL,\
                        uniq CHAR(64) NOT NULL UNIQUE,\
                        date_created DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,\
                        date_updated DATETIME DEFAULT NULL\
                    )"],
            },
            Migration {
                description: "create table memori_knowledge_graph",
                statements: &["CREATE TABLE IF NOT EXISTS memori_knowledge_graph(\
                        id INTEGER PRIMARY KEY AUTOINCREMENT,\
                        uuid VARCHAR(36) NOT NULL UNIQUE,\
                        entity_id INTEGER NOT NULL,\
                        subject_id INTEGER NOT NULL,\
                        predicate_id INTEGER NOT NULL,\
                        object_id INTEGER NOT NULL,\
                        num_times INTEGER NOT NULL,\
                        date_last_time DATETIME NOT NULL,\
                        date_created DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,\
                        date_updated DATETIME DEFAULT NULL,\
                        FOREIGN KEY (entity_id) REFERENCES memori_entity (id) ON DELETE CASCADE,\
                        FOREIGN KEY (subject_id) REFERENCES memori_subject (id) ON DELETE CASCADE,\
                        FOREIGN KEY (predicate_id) REFERENCES memori_predicate (id) ON DELETE CASCADE,\
                        FOREIGN KEY (object_id) REFERENCES memori_object (id) ON DELETE CASCADE,\
                        UNIQUE (entity_id, id),\
                        UNIQUE (subject_id, id),\
                        UNIQUE (predicate_id, id),\
                        UNIQUE (object_id, id),\
                        UNIQUE (entity_id, subject_id, predicate_id, object_id)\
                    )"],
            },
        ],
    ),
    (
        2,
        &[Migration {
            description: "create table memori_entity_fact_mention",
            statements: &[
                "CREATE TABLE IF NOT EXISTS memori_entity_fact_mention(\
                    id INTEGER PRIMARY KEY AUTOINCREMENT,\
                    uuid VARCHAR(36) NOT NULL UNIQUE,\
                    entity_id INTEGER NOT NULL,\
                    fact_id INTEGER NOT NULL,\
                    conversation_id INTEGER NOT NULL,\
                    date_created DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,\
                    date_updated DATETIME DEFAULT NULL,\
                    FOREIGN KEY (entity_id, fact_id) REFERENCES memori_entity_fact (entity_id, id) ON DELETE CASCADE,\
                    FOREIGN KEY (conversation_id) REFERENCES memori_conversation (id) ON DELETE CASCADE,\
                    UNIQUE (entity_id, fact_id, conversation_id)\
                )",
                "CREATE INDEX IF NOT EXISTS idx_memori_ent_fact_mention_entity_conversation \
                 ON memori_entity_fact_mention (entity_id, conversation_id)",
            ],
        }],
    ),
];
