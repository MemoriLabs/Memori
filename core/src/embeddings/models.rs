//! ONNX-backed sentence-transformers embedder via `fastembed`.

use anyhow::{Result, anyhow};
use fastembed::{EmbeddingModel, ModelTrait, TextEmbedding, TextInitOptions, get_cache_dir};
use hf_hub::{Cache, api::sync::ApiBuilder, api::sync::ApiRepo};
use parking_lot::Mutex;
use std::path::PathBuf;
use tokenizers::Tokenizer;

/// Sentence-transformers embedder with metadata required by the chunking pipeline.
pub struct SentenceTransformersEmbedder {
    // `TextEmbedding::embed` requires `&mut self` for internal ONNX runtime state.
    // We initialize lazily so engine startup does not require loading ONNX runtime.
    model: Mutex<Option<TextEmbedding>>,
    embedding_model: EmbeddingModel,
    cache_dir: PathBuf,
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
        let dim = model_info.dim;

        let api = ApiBuilder::from_cache(Cache::new(cache_dir.clone()))
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
            model: Mutex::new(None),
            embedding_model,
            cache_dir,
            tokenizer,
            dim,
            chunk_size,
        })
    }

    fn with_model<T>(&self, f: impl FnOnce(&mut TextEmbedding) -> Result<T>) -> Result<T> {
        let mut model_guard = self.model.lock();
        if model_guard.is_none() {
            let model = TextEmbedding::try_new(
                TextInitOptions::new(self.embedding_model.clone())
                    .with_show_download_progress(true)
                    .with_cache_dir(self.cache_dir.clone()),
            )?;
            *model_guard = Some(model);
        }

        match model_guard.as_mut() {
            Some(model) => f(model),
            None => Err(anyhow!("Failed to initialize embedding model")),
        }
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
        self.with_model(|model| {
            let embeddings = model.embed(vec![text], None)?;
            Ok(embeddings[0].clone())
        })
    }

    pub fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        self.with_model(|model| model.embed(texts, None))
    }

    pub fn embed_one_by_one(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let mut results = Vec::with_capacity(texts.len());
        self.with_model(|model| {
            for text in texts {
                let embeddings = model.embed(vec![text.as_str()], None)?;
                results.push(embeddings[0].clone());
            }
            Ok(())
        })?;

        let dim_set: std::collections::HashSet<usize> = results.iter().map(|v| v.len()).collect();

        if dim_set.len() != 1 {
            return Err(anyhow!("Inconsistent embedding dimensions: {:?}", dim_set));
        }

        Ok(results)
    }
}
