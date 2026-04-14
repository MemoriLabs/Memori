//! Text embedding backed by sentence-transformers models via `fastembed`.
mod api;
mod chunking;
mod models;
mod utils;

pub use api::embed_texts;
pub use models::SentenceTransformersEmbedder;
pub use utils::{
    format_embedding_for_db, parse_embedding_batch_from_db, parse_embedding_from_db,
    prepare_text_inputs, zero_vectors,
};
