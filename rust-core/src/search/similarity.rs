//! Dense vector similarity: cosine scoring and top-K retrieval from an embedding pool.

use crate::search::models::FactId;

/// Computes the cosine similarity between two equal-length embedding vectors.
///
/// Returns `0.0` if the vectors have different lengths or either is a zero vector.
#[inline(always)]
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }

    let mut dot_product = 0.0;
    let mut norm_a = 0.0;
    let mut norm_b = 0.0;

    for (x, y) in a.iter().zip(b.iter()) {
        dot_product += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    dot_product / (norm_a.sqrt() * norm_b.sqrt())
}

/// Scores every candidate embedding against `query_embedding` and returns the top-`limit` results.
///
/// Uses a partial sort (O(N)) to avoid a full sort of the candidate pool when only a small
/// number of results is needed.
pub fn find_similar_embeddings(
    candidate_embeddings: &[(FactId, Vec<f32>)],
    query_embedding: &[f32],
    limit: usize,
) -> Vec<(FactId, f32)> {
    if candidate_embeddings.is_empty() || query_embedding.is_empty() || limit == 0 {
        return Vec::new();
    }

    let mut results: Vec<(FactId, f32)> = candidate_embeddings
        .iter()
        .map(|(id, emb)| {
            let score = cosine_similarity(emb, query_embedding);
            (id.clone(), score)
        })
        .collect();

    if results.len() > limit {
        results.select_nth_unstable_by(limit, |a, b| b.1.total_cmp(&a.1));
        results.truncate(limit);
    }

    results.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
    results
}
