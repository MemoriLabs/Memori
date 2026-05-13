//! Two-stage hybrid search: dense cosine retrieval followed by BM25 lexical re-ranking.

pub mod api;
pub mod lexical;
pub mod models;
pub mod similarity;

pub use api::search_facts;
pub use models::{FactCandidate, FactId, FactSearchResult};
pub use similarity::{cosine_similarity, find_similar_embeddings};
