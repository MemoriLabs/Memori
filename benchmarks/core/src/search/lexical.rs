//! BM25 lexical scoring and hybrid weight selection for the re-ranking stage.

use crate::search::models::{FactCandidate, FactId};
use std::collections::{HashMap, HashSet};
use std::env;
use std::sync::OnceLock;

// Must remain sorted — `tokenize` uses `binary_search` for O(log N) stopword filtering.
const STOPWORDS: &[&str] = &[
    "a", "an", "and", "are", "as", "at", "be", "by", "did", "do", "does", "for", "from", "had",
    "has", "have", "how", "i", "in", "is", "it", "of", "on", "or", "that", "the", "their", "then",
    "there", "to", "was", "were", "what", "when", "where", "which", "who", "why", "with", "you",
    "your",
];

static LEX_WEIGHTS: OnceLock<(f32, f32)> = OnceLock::new();

/// Reads lexical blend weights from environment variables, clamped to `[0.05, 0.40]`.
///
/// Cached after the first read — environment variables are not re-read at runtime.
fn get_lex_weights() -> (f32, f32) {
    *LEX_WEIGHTS.get_or_init(|| {
        let base = env::var("MEMORI_RECALL_LEX_WEIGHT")
            .unwrap_or_else(|_| "0.15".to_string())
            .parse::<f32>()
            .unwrap_or(0.15)
            .clamp(0.05, 0.40);

        let short = env::var("MEMORI_RECALL_LEX_WEIGHT_SHORT")
            .unwrap_or_else(|_| "0.30".to_string())
            .parse::<f32>()
            .unwrap_or(0.30)
            .clamp(0.05, 0.40);

        (base, short)
    })
}

/// Lowercases `text`, splits on non-alphanumeric characters, and removes stopwords.
pub fn tokenize(text: &str) -> Vec<String> {
    debug_assert!(STOPWORDS.windows(2).all(|w| w[0] <= w[1]));
    text.to_lowercase()
        .split(|c: char| !c.is_ascii_alphanumeric())
        .filter(|s| !s.is_empty())
        .filter(|s| STOPWORDS.binary_search(s).is_err())
        .map(|s| s.to_string())
        .collect()
}

/// Computes normalized BM25 scores for each candidate against the pre-tokenized query.
///
/// Accepts pre-tokenized query terms to avoid redundant tokenization across multiple calls.
/// Scores are normalized to `[0.0, 1.0]` relative to the highest-scoring document.
/// Returns all-zero scores when no candidate contains any query term.
pub fn lexical_scores(q_tokens: &[String], candidates: &[FactCandidate]) -> HashMap<FactId, f32> {
    if q_tokens.is_empty() {
        return candidates.iter().map(|c| (c.id.clone(), 0.0)).collect();
    }

    let mut docs_tf: HashMap<FactId, HashMap<String, usize>> = HashMap::new();
    let mut doc_len: HashMap<FactId, usize> = HashMap::new();
    let mut total_len = 0;

    for candidate in candidates {
        let toks = tokenize(&candidate.content);
        let len = toks.len();
        total_len += len;
        doc_len.insert(candidate.id.clone(), len);

        let mut tf_map = HashMap::new();
        for t in toks {
            *tf_map.entry(t).or_insert(0) += 1;
        }
        docs_tf.insert(candidate.id.clone(), tf_map);
    }

    let n_docs = candidates.len() as f32;
    let avgdl = if n_docs > 0.0 {
        total_len as f32 / n_docs
    } else {
        0.0
    };

    let q_terms: HashSet<&String> = q_tokens.iter().collect();
    let mut df: HashMap<&String, usize> = HashMap::new();
    for &t in &q_terms {
        let count = docs_tf.values().filter(|tf| tf.contains_key(t)).count();
        df.insert(t, count);
    }

    // Standard BM25 tuning parameters.
    let k1 = 1.2;
    let b = 0.75;

    let idf = |t: &String| -> f32 {
        let dft = *df.get(t).unwrap_or(&0) as f32;
        (1.0 + ((n_docs - dft + 0.5) / (dft + 0.5))).ln()
    };

    let mut raw_scores: HashMap<FactId, f32> = HashMap::new();
    let mut max_score = 0.0_f32;

    for candidate in candidates {
        let tf = docs_tf.get(&candidate.id).unwrap();
        let dl = *doc_len.get(&candidate.id).unwrap() as f32;

        let denom_norm = if avgdl > 0.0 {
            (1.0 - b) + (b * (dl / avgdl))
        } else {
            1.0
        };

        let mut score = 0.0;
        for &t in &q_terms {
            let f = *tf.get(t).unwrap_or(&0) as f32;
            if f == 0.0 {
                continue;
            }
            score += idf(t) * ((f * (k1 + 1.0)) / (f + (k1 * denom_norm)));
        }

        if score > max_score {
            max_score = score;
        }
        raw_scores.insert(candidate.id.clone(), score);
    }

    // No candidate matched any query term — return the already-zeroed map directly.
    if max_score <= 0.0 {
        return raw_scores;
    }

    raw_scores
        .into_iter()
        .map(|(id, score)| (id, score / max_score))
        .collect()
}

/// Returns `(w_cos, w_lex)` blend weights based on query length.
///
/// Short queries (≤ 2 tokens) use a higher lexical weight because they tend to be
/// keyword lookups rather than broad semantic queries.
pub fn dense_lexical_weights(q_token_count: usize) -> (f32, f32) {
    let (base_lex, short_lex) = get_lex_weights();
    let w_lex = if q_token_count <= 2 {
        short_lex
    } else {
        base_lex
    };
    (1.0 - w_lex, w_lex)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn candidate(id: i64, content: &str) -> FactCandidate {
        FactCandidate {
            id: FactId::Int(id),
            content: content.to_string(),
            score: 0.0,
            date_created: "2026-01-01".to_string(),
            summaries: Vec::new(),
        }
    }

    #[test]
    fn tokenize_lowercases_splits_and_removes_stopwords() {
        let tokens = tokenize("The Quick brown-fox, jumps over the LAZY dog.");
        assert_eq!(
            tokens,
            vec!["quick", "brown", "fox", "jumps", "over", "lazy", "dog"]
        );
    }

    #[test]
    fn tokenize_handles_empty_and_whitespace() {
        assert!(tokenize("").is_empty());
        assert!(tokenize("   \n\t").is_empty());
        assert!(tokenize("the a of").is_empty());
    }

    #[test]
    fn tokenize_keeps_alphanumerics_only() {
        let tokens = tokenize("user42 loves Rust2024!");
        assert_eq!(tokens, vec!["user42", "loves", "rust2024"]);
    }

    #[test]
    fn lexical_scores_returns_zero_when_query_is_empty() {
        let candidates = vec![
            candidate(1, "rust language"),
            candidate(2, "python language"),
        ];
        let scores = lexical_scores(&[], &candidates);
        assert_eq!(scores.len(), 2);
        assert!(scores.values().all(|&s| s == 0.0));
    }

    #[test]
    fn lexical_scores_ranks_matching_document_highest() {
        let candidates = vec![
            candidate(1, "rust memory safety language"),
            candidate(2, "javascript frontend framework"),
            candidate(3, "python data science"),
        ];
        let scores = lexical_scores(&["rust".to_string()], &candidates);
        let rust = *scores.get(&FactId::Int(1)).expect("id 1");
        let js = *scores.get(&FactId::Int(2)).expect("id 2");
        let py = *scores.get(&FactId::Int(3)).expect("id 3");
        assert!(rust > js);
        assert!(rust > py);
        assert!(
            (rust - 1.0).abs() < 1e-6,
            "max score should normalize to 1.0"
        );
    }

    #[test]
    fn lexical_scores_all_zero_when_no_candidate_matches() {
        let candidates = vec![candidate(1, "python"), candidate(2, "java")];
        let scores = lexical_scores(&["rust".to_string()], &candidates);
        assert!(scores.values().all(|&s| s == 0.0));
    }

    #[test]
    fn dense_lexical_weights_favor_lexical_on_short_queries() {
        let (w_cos_short, w_lex_short) = dense_lexical_weights(1);
        let (w_cos_long, w_lex_long) = dense_lexical_weights(8);
        assert!(w_lex_short >= w_lex_long);
        assert!((w_cos_short + w_lex_short - 1.0).abs() < 1e-6);
        assert!((w_cos_long + w_lex_long - 1.0).abs() < 1e-6);
    }
}
