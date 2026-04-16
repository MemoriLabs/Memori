use std::collections::HashMap;

use crate::OrchestratorError;
use crate::embeddings::parse_embedding_from_db;
use crate::retrieval::models::RetrievalRequest;
use crate::search::{FactCandidate, FactId, find_similar_embeddings, search_facts};
use crate::storage::{EmbeddingRow, HostStorageError, RankedFact, StorageBridge};
use base64::Engine;

pub fn run_retrieval(
    bridge: &dyn StorageBridge,
    request: &RetrievalRequest,
    query_embedding: &[f32],
) -> Result<Vec<RankedFact>, OrchestratorError> {
    if request.query_text.trim().is_empty() {
        return Err(OrchestratorError::InvalidInput(
            "query_text cannot be empty".to_string(),
        ));
    }
    if request.entity_id.trim().is_empty() {
        return Err(OrchestratorError::InvalidInput(
            "entity_id cannot be empty".to_string(),
        ));
    }
    if request.dense_limit == 0 || request.limit == 0 {
        return Ok(Vec::new());
    }
    if query_embedding.is_empty() {
        return Ok(Vec::new());
    }

    let embedding_rows = bridge
        .fetch_embeddings(&request.entity_id, request.dense_limit)
        .map_err(OrchestratorError::StorageBridge)?;

    let candidate_embeddings: Vec<(FactId, Vec<f32>)> = embedding_rows
        .into_iter()
        .map(row_to_embedding)
        .collect::<Result<Vec<_>, _>>()
        .map_err(OrchestratorError::StorageBridge)?;

    validate_embedding_dimensions(query_embedding, &candidate_embeddings)
        .map_err(OrchestratorError::StorageBridge)?;

    let rows_loaded = candidate_embeddings.len();
    let candidate_limit = dynamic_candidate_limit(request.limit, rows_loaded);
    if candidate_limit == 0 {
        return Ok(Vec::new());
    }

    let dense = find_similar_embeddings(&candidate_embeddings, query_embedding, candidate_limit);
    if dense.is_empty() {
        return Ok(Vec::new());
    }

    let ids: Vec<FactId> = dense.iter().map(|(id, _)| id.clone()).collect();
    let fact_rows = bridge
        .fetch_facts_by_ids(&ids)
        .map_err(OrchestratorError::StorageBridge)?;

    let fact_map: HashMap<FactId, _> = fact_rows
        .into_iter()
        .map(|row| (row.id.clone(), row))
        .collect();

    let candidates: Vec<FactCandidate> = dense
        .into_iter()
        .filter_map(|(id, score)| {
            let row = fact_map.get(&id)?;
            Some(FactCandidate {
                id: id.clone(),
                content: row.content.clone(),
                score,
                date_created: row.date_created.clone(),
                summaries: row.summaries.clone(),
            })
        })
        .collect();

    let ranked = search_facts(candidates, request.limit, Some(&request.query_text));
    Ok(ranked
        .into_iter()
        .map(|item| RankedFact {
            id: item.id,
            content: item.content,
            similarity: item.similarity,
            rank_score: item.rank_score,
            date_created: item.date_created,
            summaries: item.summaries,
        })
        .collect())
}

pub fn format_recall_output(ranked: &[RankedFact]) -> String {
    let mut out = String::new();
    for (idx, fact) in ranked.iter().enumerate() {
        if idx > 0 {
            out.push_str("\n\n");
        }
        out.push_str(&format!(
            "[id={} similarity={:.4} rank_score={:.4}] {}",
            fact.id, fact.similarity, fact.rank_score, fact.content
        ));
    }
    out
}

fn validate_embedding_dimensions(
    query_embedding: &[f32],
    candidates: &[(FactId, Vec<f32>)],
) -> Result<(), HostStorageError> {
    let query_dim = query_embedding.len();
    for (id, embedding) in candidates {
        if embedding.len() != query_dim {
            return Err(HostStorageError::new(
                "invalid_embedding_dimension",
                format!(
                    "embedding dimension mismatch for id {id}: expected {query_dim}, got {}",
                    embedding.len()
                ),
            ));
        }
    }
    Ok(())
}

fn row_to_embedding(row: EmbeddingRow) -> Result<(FactId, Vec<f32>), HostStorageError> {
    if !row.content_embedding.is_empty() {
        return Ok((row.id, row.content_embedding));
    }

    let Some(b64) = row.content_embedding_b64 else {
        return Err(HostStorageError::new(
            "missing_embedding_data",
            "embedding row must include content_embedding or content_embedding_b64",
        ));
    };

    let bytes = base64::engine::general_purpose::STANDARD
        .decode(b64.as_bytes())
        .map_err(|e| HostStorageError::new("invalid_embedding_base64", e.to_string()))?;
    if bytes.len() % 4 != 0 {
        return Err(HostStorageError::new(
            "invalid_embedding_bytes",
            "decoded embedding bytes length must be divisible by 4",
        ));
    }
    let embedding = parse_embedding_from_db(&bytes);
    if embedding.is_empty() {
        return Err(HostStorageError::new(
            "empty_embedding",
            "decoded embedding cannot be empty",
        ));
    }
    Ok((row.id, embedding))
}

fn dynamic_candidate_limit(limit: usize, rows_loaded: usize) -> usize {
    let scaled = limit.saturating_mul(10);
    let floor = scaled.max(50);
    let bounded_by_rows = rows_loaded.min(floor);
    limit.max(bounded_by_rows)
}
