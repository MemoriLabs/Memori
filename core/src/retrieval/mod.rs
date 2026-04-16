pub mod models;
pub mod pipeline;

pub use models::RetrievalRequest;
pub use pipeline::{format_recall_output, run_retrieval};
