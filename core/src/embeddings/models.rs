//! ONNX-backed sentence-transformers embedder via `fastembed`.

use anyhow::{Result, anyhow};
use fastembed::{EmbeddingModel, ModelTrait, TextEmbedding, TextInitOptions, get_cache_dir};
use hf_hub::{Cache, api::sync::ApiBuilder, api::sync::ApiRepo};
use parking_lot::Mutex;
use std::path::PathBuf;
use tokenizers::Tokenizer;

/// Sentence-transformers embedder with metadata required by the chunking pipeline.
pub struct SentenceTransformersEmbedder {
    // `TextEmbedding::embed` requires `&mut self` for internal ONNX runtime state,
    // so we wrap it in a `Mutex` to share the loaded model across threads safely.
    model: Mutex<TextEmbedding>,
    tokenizer: Tokenizer,
    dim: usize,
    chunk_size: usize,
}

impl SentenceTransformersEmbedder {
    /// Reads the model's maximum sequence length from HuggingFace config files, if available.
    fn fetch_max_seq_length(repo: &ApiRepo) -> Option<usize> {
        if let Ok(sbert_path) = repo.get("sentence_bert_config.json") {
            let content = std::fs::read_to_string(sbert_path).ok()?;
            let json = serde_json::from_str::<serde_json::Value>(&content).ok()?;
            if let Some(val) = json.get("max_seq_length").and_then(|v| v.as_u64()) {
                return Some(val as usize);
            }
        }

        if let Ok(config_path) = repo.get("config.json") {
            let content = std::fs::read_to_string(config_path).ok()?;
            let json = serde_json::from_str::<serde_json::Value>(&content).ok()?;
            if let Some(val) = json.get("max_position_embeddings").and_then(|v| v.as_u64()) {
                return Some(val as usize);
            }
        }
        None
    }

    /// Initializes the embedder. Downloads weights and configuration to the local OS cache if not present.
    ///
    /// # Errors
    /// Fails if the HuggingFace hub is unreachable or the model architecture is unsupported.
    pub fn new(model_name: Option<&str>) -> Result<Self> {
        let embedding_model: EmbeddingModel = match model_name {
            Some(name) => name.parse().map_err(|e: String| anyhow!("{e}"))?,
            None => EmbeddingModel::AllMiniLML6V2,
        };

        let model_info = EmbeddingModel::get_model_info(&embedding_model)
            .ok_or_else(|| anyhow!("No model info found for {embedding_model}"))?;

        let cache_dir: PathBuf = get_cache_dir().into();
        let mut model = TextEmbedding::try_new(
            TextInitOptions::new(embedding_model.clone())
                .with_show_download_progress(true)
                .with_cache_dir(cache_dir.clone()),
        )?;

        let dummy_pass = model.embed(vec!["test"], None)?;
        let dim = dummy_pass[0].len();

        let api = ApiBuilder::from_cache(Cache::new(cache_dir))
            .build()
            .map_err(|e| anyhow!("Failed to initialize HF API: {}", e))?;
        let repo = api.model(model_info.model_code.clone());

        let max_seq_length = Self::fetch_max_seq_length(&repo).unwrap_or(256);
        let chunk_size = std::cmp::max(1, max_seq_length.saturating_sub(2));

        let tokenizer_path = repo.get("tokenizer.json").map_err(|e| {
            anyhow!(
                "Could not find tokenizer.json for {}: {}",
                model_info.model_code,
                e
            )
        })?;
        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| anyhow!("Failed to load tokenizer file: {}", e))?;

        Ok(Self {
            model: Mutex::new(model),
            tokenizer,
            dim,
            chunk_size,
        })
    }

    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn chunk_size(&self) -> usize {
        self.chunk_size
    }

    pub fn embed_single(&self, text: &str) -> Result<Vec<f32>> {
        let mut model = self.model.lock();
        let embeddings = model.embed(vec![text], None)?;
        Ok(embeddings[0].clone())
    }

    pub fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let mut model = self.model.lock();
        let embeddings = model.embed(texts, None)?;
        Ok(embeddings)
    }

    pub fn embed_one_by_one(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let mut results = Vec::with_capacity(texts.len());
        let mut model = self.model.lock();

        for text in texts {
            let embeddings = model.embed(vec![text.as_str()], None)?;
            results.push(embeddings[0].clone());
        }

        let dim_set: std::collections::HashSet<usize> = results.iter().map(|v| v.len()).collect();

        if dim_set.len() != 1 {
            return Err(anyhow!("Inconsistent embedding dimensions: {:?}", dim_set));
        }

        Ok(results)
    }
}
