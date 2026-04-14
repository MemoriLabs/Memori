//! Public search entry point: fuses cosine scores with BM25 re-ranking into a ranked result list.

use crate::search::lexical::{dense_lexical_weights, lexical_scores, tokenize};
use crate::search::models::{FactCandidate, FactSearchResult};

/// Fuses pre-computed cosine scores with BM25 lexical re-ranking and returns the top results.
///
/// When `query_text` is `None`, candidates are ranked by their cosine score alone.
/// Uses a partial sort (O(N)) to avoid a full sort of the candidate pool when only a small
/// number of results is needed.
pub fn search_facts(
    candidates: Vec<FactCandidate>,
    limit: usize,
    query_text: Option<&str>,
) -> Vec<FactSearchResult> {
    if candidates.is_empty() || limit == 0 {
        return Vec::new();
    }

    let mut results: Vec<FactSearchResult> = if let Some(text) = query_text {
        let q_tokens = tokenize(text);
        let lex_scores = lexical_scores(&q_tokens, &candidates);
        let (w_cos, w_lex) = dense_lexical_weights(q_tokens.len());

        candidates
            .into_iter()
            .map(|c| {
                let cos_score = c.score;
                let lex_score = lex_scores.get(&c.id).copied().unwrap_or(0.0);

                FactSearchResult {
                    id: c.id,
                    content: c.content,
                    similarity: cos_score,
                    rank_score: (w_cos * cos_score) + (w_lex * lex_score),
                    date_created: c.date_created,
                    summaries: c.summaries,
                }
            })
            .collect()
    } else {
        candidates
            .into_iter()
            .map(|c| FactSearchResult {
                id: c.id,
                content: c.content,
                similarity: c.score,
                rank_score: c.score,
                date_created: c.date_created,
                summaries: c.summaries,
            })
            .collect()
    };

    if results.len() > limit {
        results.select_nth_unstable_by(limit, |a, b| b.rank_score.total_cmp(&a.rank_score));
        results.truncate(limit);
    }

    results.sort_unstable_by(|a, b| b.rank_score.total_cmp(&a.rank_score));

    results
}
