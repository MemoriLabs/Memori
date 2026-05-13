//! Shared data types for the search pipeline.

use std::fmt;

use serde::{Deserialize, Serialize};

/// A stable identifier for a fact, supporting both integer and string primary keys.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(untagged)]
pub enum FactId {
    Int(i64),
    String(String),
}

impl fmt::Display for FactId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FactId::Int(value) => write!(f, "{value}"),
            FactId::String(value) => write!(f, "{value}"),
        }
    }
}

/// A fact retrieved from the database and prepared for re-ranking.
///
/// `score` holds the raw cosine similarity assigned during the dense retrieval stage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FactCandidate {
    pub id: FactId,
    pub content: String,
    pub score: f32,
    pub date_created: String,
    #[serde(default)]
    pub summaries: Vec<serde_json::Value>,
}

/// A fully ranked fact returned to the caller after the hybrid re-ranking stage.
///
/// `similarity` preserves the original cosine score for diagnostics, while `rank_score`
/// reflects the final blended score used to order results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FactSearchResult {
    pub id: FactId,
    pub content: String,
    pub similarity: f32,
    pub rank_score: f32,
    pub date_created: String,
    #[serde(default)]
    pub summaries: Vec<serde_json::Value>,
}
