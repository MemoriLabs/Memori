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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::search::models::FactId;

    fn candidate(id: i64, content: &str, score: f32) -> FactCandidate {
        FactCandidate {
            id: FactId::Int(id),
            content: content.to_string(),
            score,
            date_created: "2026-01-01".to_string(),
            summaries: Vec::new(),
        }
    }

    #[test]
    fn search_facts_returns_empty_when_limit_is_zero() {
        let results = search_facts(vec![candidate(1, "a", 1.0)], 0, None);
        assert!(results.is_empty());
    }

    #[test]
    fn search_facts_returns_empty_when_candidates_empty() {
        let results = search_facts(Vec::new(), 5, Some("rust"));
        assert!(results.is_empty());
    }

    #[test]
    fn search_facts_without_query_preserves_cosine_as_rank_score() {
        let candidates = vec![
            candidate(1, "rust language", 0.8),
            candidate(2, "python language", 0.9),
        ];
        let results = search_facts(candidates, 2, None);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].id, FactId::Int(2));
        assert!((results[0].rank_score - 0.9).abs() < 1e-6);
        assert!((results[0].similarity - 0.9).abs() < 1e-6);
    }

    #[test]
    fn search_facts_blends_cosine_and_lexical_with_query() {
        let candidates = vec![
            candidate(1, "rust memory safety language", 0.5),
            candidate(2, "completely unrelated payload", 0.6),
        ];
        let results = search_facts(candidates, 2, Some("rust language"));
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].id, FactId::Int(1));
        assert!(results[0].rank_score > results[1].rank_score);
    }

    #[test]
    fn search_facts_truncates_to_limit() {
        let candidates: Vec<FactCandidate> = (0..10)
            .map(|i| candidate(i, "content", i as f32 / 10.0))
            .collect();
        let results = search_facts(candidates, 3, None);
        assert_eq!(results.len(), 3);
        for window in results.windows(2) {
            assert!(window[0].rank_score >= window[1].rank_score);
        }
    }
}
