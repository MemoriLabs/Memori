use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetrievalRequest {
    pub entity_id: String,
    pub query_text: String,
    pub dense_limit: usize,
    #[serde(alias = "final_limit")]
    pub limit: usize,
}
