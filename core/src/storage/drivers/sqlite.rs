use super::{embedding_to_bytes, generate_uniq, new_uuid};
use crate::search::FactId;
use crate::storage::connection::{SqlBind, StorageConnection, read_id};
use crate::storage::models::{CandidateFactRow, EmbeddingRow, HostStorageError};

pub fn schema_version_read(conn: &dyn StorageConnection) -> Result<Option<i64>, HostStorageError> {
    let rows = conn.execute("SELECT num FROM memori_schema_version", vec![])?;
    Ok(rows.first().and_then(|r| read_id(r, "num")))
}

pub fn schema_version_delete(conn: &dyn StorageConnection) -> Result<(), HostStorageError> {
    conn.execute("DELETE FROM memori_schema_version", vec![])?;
    Ok(())
}

pub fn schema_version_create(
    conn: &dyn StorageConnection,
    num: i64,
) -> Result<(), HostStorageError> {
    conn.execute(
        "INSERT INTO memori_schema_version(num) VALUES (?)",
        vec![SqlBind::Int(num)],
    )?;
    Ok(())
}

pub fn entity_get_id(
    conn: &dyn StorageConnection,
    external_id: &str,
) -> Result<Option<i64>, HostStorageError> {
    let rows = conn.execute(
        "SELECT id FROM memori_entity WHERE external_id = ?",
        vec![SqlBind::Text(external_id.to_string())],
    )?;
    Ok(rows.first().and_then(|r| read_id(r, "id")))
}

pub fn entity_create(
    conn: &dyn StorageConnection,
    external_id: &str,
) -> Result<Option<i64>, HostStorageError> {
    conn.execute(
        "INSERT INTO memori_entity(uuid, external_id) VALUES (?, ?) ON CONFLICT DO NOTHING",
        vec![
            SqlBind::Text(new_uuid()),
            SqlBind::Text(external_id.to_string()),
        ],
    )?;
    let rows = conn.execute(
        "SELECT id FROM memori_entity WHERE external_id = ?",
        vec![SqlBind::Text(external_id.to_string())],
    )?;
    Ok(rows.first().and_then(|r| read_id(r, "id")))
}

pub fn process_create(
    conn: &dyn StorageConnection,
    external_id: &str,
) -> Result<Option<i64>, HostStorageError> {
    conn.execute(
        "INSERT INTO memori_process(uuid, external_id) VALUES (?, ?) ON CONFLICT DO NOTHING",
        vec![
            SqlBind::Text(new_uuid()),
            SqlBind::Text(external_id.to_string()),
        ],
    )?;
    let rows = conn.execute(
        "SELECT id FROM memori_process WHERE external_id = ?",
        vec![SqlBind::Text(external_id.to_string())],
    )?;
    Ok(rows.first().and_then(|r| read_id(r, "id")))
}

pub fn session_get_id(
    conn: &dyn StorageConnection,
    uuid: &str,
) -> Result<Option<i64>, HostStorageError> {
    let rows = conn.execute(
        "SELECT id FROM memori_session WHERE uuid = ?",
        vec![SqlBind::Text(uuid.to_string())],
    )?;
    Ok(rows.first().and_then(|r| read_id(r, "id")))
}

pub fn conversation_get_id_by_session(
    conn: &dyn StorageConnection,
    session_id: i64,
) -> Result<Option<i64>, HostStorageError> {
    let rows = conn.execute(
        "SELECT id FROM memori_conversation WHERE session_id = ? ORDER BY id DESC LIMIT 1",
        vec![SqlBind::Int(session_id)],
    )?;
    Ok(rows.first().and_then(|r| read_id(r, "id")))
}

pub fn session_create(
    conn: &dyn StorageConnection,
    uuid: &str,
    entity_id: Option<i64>,
    process_id: Option<i64>,
) -> Result<Option<i64>, HostStorageError> {
    conn.execute(
        "INSERT INTO memori_session(uuid, entity_id, process_id) VALUES (?, ?, ?) \
         ON CONFLICT(uuid) DO UPDATE SET \
           entity_id = COALESCE(excluded.entity_id, memori_session.entity_id), \
           process_id = COALESCE(excluded.process_id, memori_session.process_id)",
        vec![
            SqlBind::Text(uuid.to_string()),
            SqlBind::id_or_null(entity_id),
            SqlBind::id_or_null(process_id),
        ],
    )?;
    let rows = conn.execute(
        "SELECT id FROM memori_session WHERE uuid = ?",
        vec![SqlBind::Text(uuid.to_string())],
    )?;
    Ok(rows.first().and_then(|r| read_id(r, "id")))
}

/// Returns the existing conversation for this session if still active, otherwise creates one.
pub fn conversation_create(
    conn: &dyn StorageConnection,
    session_id: i64,
    timeout_minutes: i64,
) -> Result<Option<i64>, HostStorageError> {
    let existing = conn.execute(
        "SELECT c.id, COALESCE(MAX(m.date_created), c.date_created) as last_activity \
         FROM memori_conversation c \
         LEFT JOIN memori_conversation_message m ON m.conversation_id = c.id \
         WHERE c.session_id = ? GROUP BY c.id, c.date_created",
        vec![SqlBind::Int(session_id)],
    )?;

    if let Some(row) = existing.first() {
        if let Some(last_activity) = row["last_activity"].as_str() {
            let elapsed = conn.execute(
                "SELECT (strftime('%s', 'now') - strftime('%s', ?)) / 60 as minutes_since_activity",
                vec![SqlBind::Text(last_activity.to_string())],
            )?;
            if let Some(elapsed_row) = elapsed.first() {
                let minutes = elapsed_row["minutes_since_activity"]
                    .as_i64()
                    .or_else(|| {
                        elapsed_row["minutes_since_activity"]
                            .as_f64()
                            .map(|f| f as i64)
                    })
                    .unwrap_or(i64::MAX);
                if minutes <= timeout_minutes {
                    return Ok(read_id(row, "id"));
                }
            }
        }
    }

    conn.execute(
        "INSERT INTO memori_conversation(uuid, session_id) VALUES (?, ?) ON CONFLICT DO NOTHING",
        vec![SqlBind::Text(new_uuid()), SqlBind::Int(session_id)],
    )?;
    let rows = conn.execute(
        "SELECT id FROM memori_conversation WHERE session_id = ?",
        vec![SqlBind::Int(session_id)],
    )?;
    Ok(rows.first().and_then(|r| read_id(r, "id")))
}

pub fn conversation_update(
    conn: &dyn StorageConnection,
    id: i64,
    summary: &str,
) -> Result<(), HostStorageError> {
    if summary.is_empty() {
        return Ok(());
    }
    conn.execute(
        "UPDATE memori_conversation SET summary = ? WHERE id = ?",
        vec![SqlBind::Text(summary.to_string()), SqlBind::Int(id)],
    )?;
    Ok(())
}

pub fn conversation_message_create(
    conn: &dyn StorageConnection,
    conversation_id: i64,
    role: &str,
    content: &str,
) -> Result<(), HostStorageError> {
    conn.execute(
        "INSERT INTO memori_conversation_message(uuid, conversation_id, role, content) VALUES (?, ?, ?, ?)",
        vec![
            SqlBind::Text(new_uuid()),
            SqlBind::Int(conversation_id),
            SqlBind::Text(role.to_string()),
            SqlBind::Text(content.to_string()),
        ],
    )?;
    Ok(())
}

pub fn conversation_messages_read(
    conn: &dyn StorageConnection,
    conversation_id: i64,
) -> Result<Vec<(String, String)>, HostStorageError> {
    let rows = conn.execute(
        "SELECT role, content FROM memori_conversation_message WHERE conversation_id = ? ORDER BY id",
        vec![SqlBind::Int(conversation_id)],
    )?;
    Ok(rows
        .iter()
        .filter_map(|r| {
            let role = r["role"].as_str()?.to_string();
            let content = r["content"].as_str()?.to_string();
            Some((role, content))
        })
        .collect())
}

pub fn entity_fact_create(
    conn: &dyn StorageConnection,
    entity_id: i64,
    facts: &[String],
    embeddings: &[Vec<f32>],
    conversation_id: Option<i64>,
) -> Result<(), HostStorageError> {
    for (i, fact) in facts.iter().enumerate() {
        let embedding = embeddings.get(i).map(|e| e.as_slice()).unwrap_or(&[]);
        let embedding_bytes = embedding_to_bytes(embedding);
        let uniq = generate_uniq(&[fact.as_str()]);

        conn.execute(
            "INSERT INTO memori_entity_fact(uuid, entity_id, content, content_embedding, num_times, date_last_time, uniq) \
             VALUES (?, ?, ?, ?, 1, CURRENT_TIMESTAMP, ?) \
             ON CONFLICT (entity_id, uniq) DO UPDATE SET num_times = memori_entity_fact.num_times + 1, date_last_time = CURRENT_TIMESTAMP",
            vec![
                SqlBind::Text(new_uuid()),
                SqlBind::Int(entity_id),
                SqlBind::Text(fact.clone()),
                SqlBind::bytes(&embedding_bytes),
                SqlBind::Text(uniq.clone()),
            ],
        )?;

        if let Some(conv_id) = conversation_id {
            let fact_rows = conn.execute(
                "SELECT id FROM memori_entity_fact WHERE entity_id = ? AND uniq = ?",
                vec![SqlBind::Int(entity_id), SqlBind::Text(uniq)],
            )?;
            if let Some(fact_id) = fact_rows.first().and_then(|r| read_id(r, "id")) {
                conn.execute(
                    "INSERT INTO memori_entity_fact_mention(uuid, entity_id, fact_id, conversation_id) \
                     VALUES (?, ?, ?, ?) ON CONFLICT (entity_id, fact_id, conversation_id) DO NOTHING",
                    vec![
                        SqlBind::Text(new_uuid()),
                        SqlBind::Int(entity_id),
                        SqlBind::Int(fact_id),
                        SqlBind::Int(conv_id),
                    ],
                )?;
            }
        }
    }
    Ok(())
}

pub fn entity_fact_create_without_embedding(
    conn: &dyn StorageConnection,
    entity_id: i64,
    content: &str,
) -> Result<(), HostStorageError> {
    let uniq = generate_uniq(&[content]);
    conn.execute(
        "INSERT INTO memori_entity_fact(uuid, entity_id, content, content_embedding, num_times, date_last_time, uniq) \
         VALUES (?, ?, ?, ?, 1, CURRENT_TIMESTAMP, ?) \
         ON CONFLICT (entity_id, uniq) DO UPDATE SET num_times = memori_entity_fact.num_times + 1, date_last_time = CURRENT_TIMESTAMP",
        vec![
            SqlBind::Text(new_uuid()),
            SqlBind::Int(entity_id),
            SqlBind::Text(content.to_string()),
            SqlBind::bytes(&[]),
            SqlBind::Text(uniq),
        ],
    )?;
    Ok(())
}

pub fn entity_fact_get_embeddings(
    conn: &dyn StorageConnection,
    entity_id: i64,
    limit: usize,
) -> Result<Vec<EmbeddingRow>, HostStorageError> {
    let rows = conn.execute(
        "SELECT id, content_embedding FROM memori_entity_fact \
         WHERE entity_id = ? ORDER BY date_last_time DESC, num_times DESC, id DESC LIMIT ?",
        vec![SqlBind::Int(entity_id), SqlBind::Int(limit as i64)],
    )?;

    let mut results = Vec::new();
    for row in &rows {
        let id = match read_id(row, "id") {
            Some(n) => FactId::Int(n),
            None => continue,
        };
        // The TS adapter sends embedding bytes back as base64 in the "bytes" field,
        // or the raw Buffer may be JSON-serialised as an array. We accept both forms.
        let embedding_b64 = row["content_embedding"].as_str().map(|s| s.to_string());
        if embedding_b64
            .as_deref()
            .map(|s| s.is_empty())
            .unwrap_or(true)
        {
            continue;
        }
        results.push(EmbeddingRow {
            id,
            content_embedding: vec![],
            content_embedding_b64: embedding_b64,
        });
    }
    Ok(results)
}

pub fn entity_fact_get_by_ids(
    conn: &dyn StorageConnection,
    ids: &[FactId],
) -> Result<Vec<CandidateFactRow>, HostStorageError> {
    if ids.is_empty() {
        return Ok(vec![]);
    }
    let placeholders = ids.iter().map(|_| "?").collect::<Vec<_>>().join(",");
    let binds: Vec<SqlBind> = ids
        .iter()
        .map(|id| match id {
            FactId::Int(n) => SqlBind::Int(*n),
            FactId::String(s) => SqlBind::Text(s.clone()),
        })
        .collect();

    let fact_rows = conn.execute(
        &format!(
            "SELECT id, content, date_created FROM memori_entity_fact WHERE id IN ({})",
            placeholders
        ),
        binds.clone(),
    )?;

    if fact_rows.is_empty() {
        return Ok(vec![]);
    }

    let mut facts: Vec<CandidateFactRow> = fact_rows
        .iter()
        .filter_map(|row| {
            let id = read_id(row, "id").map(FactId::Int)?;
            let content = row["content"].as_str()?.to_string();
            let date_created = row["date_created"].as_str().unwrap_or("").to_string();
            Some(CandidateFactRow {
                id,
                content,
                date_created,
                summaries: vec![],
            })
        })
        .collect();

    let summary_rows = conn.execute(
        &format!(
            "SELECT m.fact_id, c.summary AS content, COALESCE(c.date_updated, c.date_created) AS date_created \
             FROM memori_entity_fact_mention m \
             JOIN memori_conversation c ON c.id = m.conversation_id \
             WHERE m.fact_id IN ({}) AND c.summary IS NOT NULL AND c.summary <> ''",
            placeholders
        ),
        binds,
    )?;

    for summary_row in &summary_rows {
        let fact_id = match read_id(summary_row, "fact_id") {
            Some(n) => n,
            None => continue,
        };
        let content = match summary_row["content"].as_str() {
            Some(s) => s.to_string(),
            None => continue,
        };
        let date_created = summary_row["date_created"]
            .as_str()
            .unwrap_or("")
            .to_string();
        if let Some(fact) = facts
            .iter_mut()
            .find(|f| matches!(&f.id, FactId::Int(n) if *n == fact_id))
        {
            fact.summaries.push(serde_json::json!({
                "content": content,
                "date_created": date_created,
            }));
        }
    }
    Ok(facts)
}

pub fn knowledge_graph_create(
    conn: &dyn StorageConnection,
    entity_id: i64,
    semantic_triples: &[serde_json::Value],
) -> Result<(), HostStorageError> {
    for triple in semantic_triples {
        let (subj_name, subj_type) = read_triple_entity(triple.get("subject"));
        let pred = triple["predicate"].as_str().unwrap_or("").to_string();
        let (obj_name, obj_type) = read_triple_entity(triple.get("object"));

        let subj_uniq = generate_uniq(&[&subj_name, &subj_type]);
        conn.execute(
            "INSERT INTO memori_subject(uuid, name, type, uniq) VALUES (?, ?, ?, ?) ON CONFLICT DO NOTHING",
            vec![SqlBind::Text(new_uuid()), SqlBind::Text(subj_name), SqlBind::Text(subj_type), SqlBind::Text(subj_uniq.clone())],
        )?;
        let subj_rows = conn.execute(
            "SELECT id FROM memori_subject WHERE uniq = ?",
            vec![SqlBind::Text(subj_uniq)],
        )?;
        let subj_id = match subj_rows.first().and_then(|r| read_id(r, "id")) {
            Some(id) => id,
            None => continue,
        };

        let pred_uniq = generate_uniq(&[&pred]);
        conn.execute(
            "INSERT INTO memori_predicate(uuid, content, uniq) VALUES (?, ?, ?) ON CONFLICT DO NOTHING",
            vec![SqlBind::Text(new_uuid()), SqlBind::Text(pred), SqlBind::Text(pred_uniq.clone())],
        )?;
        let pred_rows = conn.execute(
            "SELECT id FROM memori_predicate WHERE uniq = ?",
            vec![SqlBind::Text(pred_uniq)],
        )?;
        let pred_id = match pred_rows.first().and_then(|r| read_id(r, "id")) {
            Some(id) => id,
            None => continue,
        };

        let obj_uniq = generate_uniq(&[&obj_name, &obj_type]);
        conn.execute(
            "INSERT INTO memori_object(uuid, name, type, uniq) VALUES (?, ?, ?, ?) ON CONFLICT DO NOTHING",
            vec![SqlBind::Text(new_uuid()), SqlBind::Text(obj_name), SqlBind::Text(obj_type), SqlBind::Text(obj_uniq.clone())],
        )?;
        let obj_rows = conn.execute(
            "SELECT id FROM memori_object WHERE uniq = ?",
            vec![SqlBind::Text(obj_uniq)],
        )?;
        let obj_id = match obj_rows.first().and_then(|r| read_id(r, "id")) {
            Some(id) => id,
            None => continue,
        };

        conn.execute(
            "INSERT INTO memori_knowledge_graph(uuid, entity_id, subject_id, predicate_id, object_id, num_times, date_last_time) \
             VALUES (?, ?, ?, ?, ?, 1, CURRENT_TIMESTAMP) \
             ON CONFLICT (entity_id, subject_id, predicate_id, object_id) DO UPDATE SET num_times = memori_knowledge_graph.num_times + 1, date_last_time = CURRENT_TIMESTAMP",
            vec![
                SqlBind::Text(new_uuid()),
                SqlBind::Int(entity_id),
                SqlBind::Int(subj_id),
                SqlBind::Int(pred_id),
                SqlBind::Int(obj_id),
            ],
        )?;
    }
    Ok(())
}

pub fn process_attribute_create(
    conn: &dyn StorageConnection,
    process_id: i64,
    attributes: &[String],
) -> Result<(), HostStorageError> {
    for attribute in attributes {
        let uniq = generate_uniq(&[attribute.as_str()]);
        conn.execute(
            "INSERT INTO memori_process_attribute(uuid, process_id, content, num_times, date_last_time, uniq) \
             VALUES (?, ?, ?, 1, CURRENT_TIMESTAMP, ?) \
             ON CONFLICT (process_id, uniq) DO UPDATE SET num_times = memori_process_attribute.num_times + 1, date_last_time = CURRENT_TIMESTAMP",
            vec![
                SqlBind::Text(new_uuid()),
                SqlBind::Int(process_id),
                SqlBind::Text(attribute.clone()),
                SqlBind::Text(uniq),
            ],
        )?;
    }
    Ok(())
}

fn read_triple_entity(v: Option<&serde_json::Value>) -> (String, String) {
    match v {
        Some(serde_json::Value::String(s)) => (s.clone(), "entity".to_string()),
        Some(serde_json::Value::Object(map)) => {
            let name = map
                .get("name")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let typ = map
                .get("type")
                .and_then(|v| v.as_str())
                .unwrap_or("entity")
                .to_string();
            (name, typ)
        }
        _ => (String::new(), "entity".to_string()),
    }
}
