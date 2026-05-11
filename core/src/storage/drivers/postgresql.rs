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
        "INSERT INTO memori_schema_version(num) VALUES ($1)",
        vec![SqlBind::Int(num)],
    )?;
    Ok(())
}

pub fn entity_create(
    conn: &dyn StorageConnection,
    external_id: &str,
) -> Result<Option<i64>, HostStorageError> {
    conn.execute(
        "INSERT INTO memori_entity(uuid, external_id) VALUES ($1, $2) ON CONFLICT DO NOTHING",
        vec![
            SqlBind::Text(new_uuid()),
            SqlBind::Text(external_id.to_string()),
        ],
    )?;
    let rows = conn.execute(
        "SELECT id FROM memori_entity WHERE external_id = $1",
        vec![SqlBind::Text(external_id.to_string())],
    )?;
    Ok(rows.first().and_then(|r| read_id(r, "id")))
}

pub fn process_create(
    conn: &dyn StorageConnection,
    external_id: &str,
) -> Result<Option<i64>, HostStorageError> {
    conn.execute(
        "INSERT INTO memori_process(uuid, external_id) VALUES ($1, $2) ON CONFLICT DO NOTHING",
        vec![
            SqlBind::Text(new_uuid()),
            SqlBind::Text(external_id.to_string()),
        ],
    )?;
    let rows = conn.execute(
        "SELECT id FROM memori_process WHERE external_id = $1",
        vec![SqlBind::Text(external_id.to_string())],
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
        "INSERT INTO memori_session(uuid, entity_id, process_id) VALUES ($1, $2, $3) ON CONFLICT DO NOTHING",
        vec![
            SqlBind::Text(uuid.to_string()),
            SqlBind::id_or_null(entity_id),
            SqlBind::id_or_null(process_id),
        ],
    )?;
    let rows = conn.execute(
        "SELECT id FROM memori_session WHERE uuid = $1",
        vec![SqlBind::Text(uuid.to_string())],
    )?;
    Ok(rows.first().and_then(|r| read_id(r, "id")))
}

pub fn conversation_create(
    conn: &dyn StorageConnection,
    session_id: i64,
    timeout_minutes: i64,
) -> Result<Option<i64>, HostStorageError> {
    let existing = conn.execute(
        "SELECT c.id, COALESCE(MAX(m.date_created), c.date_created) as last_activity \
         FROM memori_conversation c \
         LEFT JOIN memori_conversation_message m ON m.conversation_id = c.id \
         WHERE c.session_id = $1 GROUP BY c.id, c.date_created",
        vec![SqlBind::Int(session_id)],
    )?;

    if let Some(row) = existing.first() {
        if let Some(last_activity) = row["last_activity"].as_str() {
            let elapsed = conn.execute(
                "SELECT EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - $1::timestamp)) / 60 as minutes_since_activity",
                vec![SqlBind::Text(last_activity.to_string())],
            )?;
            if let Some(elapsed_row) = elapsed.first() {
                let minutes = elapsed_row["minutes_since_activity"]
                    .as_f64()
                    .map(|f| f as i64)
                    .unwrap_or(i64::MAX);
                if minutes <= timeout_minutes {
                    return Ok(read_id(row, "id"));
                }
            }
        }
    }

    conn.execute(
        "INSERT INTO memori_conversation(uuid, session_id) VALUES ($1, $2) ON CONFLICT DO NOTHING",
        vec![SqlBind::Text(new_uuid()), SqlBind::Int(session_id)],
    )?;
    let rows = conn.execute(
        "SELECT id FROM memori_conversation WHERE session_id = $1",
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
        "UPDATE memori_conversation SET summary = $1 WHERE id = $2",
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
        "INSERT INTO memori_conversation_message(uuid, conversation_id, role, type, content) VALUES ($1, $2, $3, $4, $5)",
        vec![
            SqlBind::Text(new_uuid()),
            SqlBind::Int(conversation_id),
            SqlBind::Text(role.to_string()),
            SqlBind::Text("text".to_string()),
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
        "SELECT role, content FROM memori_conversation_message WHERE conversation_id = $1",
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

/// Bulk-inserts facts using PostgreSQL numbered placeholders.
/// Chunks at CHUNK_SIZE to stay within the pg parameter limit.
pub fn entity_fact_create(
    conn: &dyn StorageConnection,
    entity_id: i64,
    facts: &[String],
    embeddings: &[Vec<f32>],
    conversation_id: Option<i64>,
) -> Result<(), HostStorageError> {
    const CHUNK_SIZE: usize = 2000;

    let valid_facts: Vec<(usize, &String, Vec<u8>)> = facts
        .iter()
        .enumerate()
        .filter_map(|(i, fact)| {
            let embedding = embeddings.get(i)?;
            if embedding.is_empty() {
                return None;
            }
            Some((i, fact, embedding_to_bytes(embedding)))
        })
        .collect();

    for chunk in valid_facts.chunks(CHUNK_SIZE) {
        let mut placeholders = Vec::new();
        let mut binds: Vec<SqlBind> = Vec::new();
        let mut param = 1usize;

        for (_, fact, embedding_bytes) in chunk {
            let uniq = generate_uniq(&[fact.as_str()]);
            placeholders.push(format!(
                "(${}, ${}, ${}, ${}, 1, CURRENT_TIMESTAMP, ${})",
                param,
                param + 1,
                param + 2,
                param + 3,
                param + 4
            ));
            binds.push(SqlBind::Text(new_uuid()));
            binds.push(SqlBind::Int(entity_id));
            binds.push(SqlBind::Text(fact.to_string()));
            binds.push(SqlBind::bytes(embedding_bytes));
            binds.push(SqlBind::Text(uniq));
            param += 5;
        }

        if !binds.is_empty() {
            conn.execute(
                &format!(
                    "INSERT INTO memori_entity_fact(uuid, entity_id, content, content_embedding, num_times, date_last_time, uniq) \
                     VALUES {} \
                     ON CONFLICT (entity_id, uniq) DO UPDATE SET num_times = memori_entity_fact.num_times + 1, date_last_time = CURRENT_TIMESTAMP",
                    placeholders.join(", ")
                ),
                binds,
            )?;
        }

        // Link conversation mentions for the facts in this chunk, selected by their uniq values
        // rather than by recency so concurrent writes cannot attach mentions to unrelated facts.
        if let Some(conv_id) = conversation_id {
            let uniq_vals: Vec<String> = chunk
                .iter()
                .map(|(_, fact, _)| generate_uniq(&[fact.as_str()]))
                .collect();
            if !uniq_vals.is_empty() {
                let uniq_placeholders: String = (2..=uniq_vals.len() + 1)
                    .map(|i| format!("${}", i))
                    .collect::<Vec<_>>()
                    .join(", ");
                let mut lookup_binds = vec![SqlBind::Int(entity_id)];
                lookup_binds.extend(uniq_vals.into_iter().map(SqlBind::Text));
                let inserted = conn.execute(
                    &format!(
                        "SELECT id FROM memori_entity_fact WHERE entity_id = $1 AND uniq IN ({})",
                        uniq_placeholders
                    ),
                    lookup_binds,
                )?;

                if !inserted.is_empty() {
                    let mut link_placeholders = Vec::new();
                    let mut link_binds: Vec<SqlBind> = Vec::new();
                    let mut lp = 1usize;

                    for row in &inserted {
                        if let Some(fact_id) = read_id(row, "id") {
                            link_placeholders.push(format!(
                                "(${}, ${}, ${}, ${})",
                                lp,
                                lp + 1,
                                lp + 2,
                                lp + 3
                            ));
                            link_binds.push(SqlBind::Text(new_uuid()));
                            link_binds.push(SqlBind::Int(entity_id));
                            link_binds.push(SqlBind::Int(fact_id));
                            link_binds.push(SqlBind::Int(conv_id));
                            lp += 4;
                        }
                    }

                    if !link_binds.is_empty() {
                        conn.execute(
                            &format!(
                                "INSERT INTO memori_entity_fact_mention(uuid, entity_id, fact_id, conversation_id) \
                                 VALUES {} ON CONFLICT (entity_id, fact_id, conversation_id) DO NOTHING",
                                link_placeholders.join(", ")
                            ),
                            link_binds,
                        )?;
                    }
                }
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
         VALUES ($1, $2, $3, $4, 1, CURRENT_TIMESTAMP, $5) \
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
        &format!(
            "SELECT id, content_embedding FROM memori_entity_fact \
             WHERE entity_id = $1 ORDER BY date_last_time DESC, num_times DESC, id DESC LIMIT {}",
            limit
        ),
        vec![SqlBind::Int(entity_id)],
    )?;

    let mut results = Vec::new();
    for row in &rows {
        let id = match read_id(row, "id") {
            Some(n) => FactId::Int(n),
            None => continue,
        };
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
    let placeholders = (1..=ids.len())
        .map(|i| format!("${i}"))
        .collect::<Vec<_>>()
        .join(",");
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
            fact.summaries
                .push(serde_json::json!({ "content": content, "date_created": date_created }));
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
            "INSERT INTO memori_subject(uuid, name, type, uniq) VALUES ($1, $2, $3, $4) ON CONFLICT DO NOTHING",
            vec![SqlBind::Text(new_uuid()), SqlBind::Text(subj_name), SqlBind::Text(subj_type), SqlBind::Text(subj_uniq.clone())],
        )?;
        let subj_rows = conn.execute(
            "SELECT id FROM memori_subject WHERE uniq = $1",
            vec![SqlBind::Text(subj_uniq)],
        )?;
        let subj_id = match subj_rows.first().and_then(|r| read_id(r, "id")) {
            Some(id) => id,
            None => continue,
        };

        let pred_uniq = generate_uniq(&[&pred]);
        conn.execute(
            "INSERT INTO memori_predicate(uuid, content, uniq) VALUES ($1, $2, $3) ON CONFLICT DO NOTHING",
            vec![SqlBind::Text(new_uuid()), SqlBind::Text(pred), SqlBind::Text(pred_uniq.clone())],
        )?;
        let pred_rows = conn.execute(
            "SELECT id FROM memori_predicate WHERE uniq = $1",
            vec![SqlBind::Text(pred_uniq)],
        )?;
        let pred_id = match pred_rows.first().and_then(|r| read_id(r, "id")) {
            Some(id) => id,
            None => continue,
        };

        let obj_uniq = generate_uniq(&[&obj_name, &obj_type]);
        conn.execute(
            "INSERT INTO memori_object(uuid, name, type, uniq) VALUES ($1, $2, $3, $4) ON CONFLICT DO NOTHING",
            vec![SqlBind::Text(new_uuid()), SqlBind::Text(obj_name), SqlBind::Text(obj_type), SqlBind::Text(obj_uniq.clone())],
        )?;
        let obj_rows = conn.execute(
            "SELECT id FROM memori_object WHERE uniq = $1",
            vec![SqlBind::Text(obj_uniq)],
        )?;
        let obj_id = match obj_rows.first().and_then(|r| read_id(r, "id")) {
            Some(id) => id,
            None => continue,
        };

        conn.execute(
            "INSERT INTO memori_knowledge_graph(uuid, entity_id, subject_id, predicate_id, object_id, num_times, date_last_time) \
             VALUES ($1, $2, $3, $4, $5, 1, CURRENT_TIMESTAMP) \
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
             VALUES ($1, $2, $3, 1, CURRENT_TIMESTAMP, $4) \
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
