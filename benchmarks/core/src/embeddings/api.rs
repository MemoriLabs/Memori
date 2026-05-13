//! Embedding pipeline: chunks inputs, runs the model in one batch, pools
//! multi-chunk inputs, and flattens the result for FFI transport.

use crate::embeddings::chunking::chunk_text_by_tokens;
use crate::embeddings::models::SentenceTransformersEmbedder;
use crate::embeddings::utils::{prepare_text_inputs, zero_vectors};

/// Represents an execution plan for a single input text.
enum TextPlan {
    /// Reuses the original owned string. Zero allocations.
    Single(String),
    /// Holds newly allocated sub-chunks because the string exceeded the token window.
    Multi(Vec<String>),
}

/// Maps the logical input document to its physical row locations inside the flattened compute batch.
enum Meta {
    Single { idx: usize },
    Multi { start: usize, len: usize },
}

/// Analyzes text length and determines if it requires chunking.
fn plan_text(text: String, embedder: &SentenceTransformersEmbedder) -> TextPlan {
    match chunk_text_by_tokens(&text, embedder.tokenizer(), embedder.chunk_size()) {
        None => TextPlan::Single(text),
        Some(chunks) => TextPlan::Multi(chunks),
    }
}

/// Averages multiple chunk vectors back into a single representative vector, then L2-normalizes.
fn mean_pool_l2(chunk_embeddings: &[Vec<f32>]) -> Vec<f32> {
    debug_assert!(!chunk_embeddings.is_empty());
    let dim = chunk_embeddings[0].len();
    let mut pooled = vec![0.0; dim];

    for vec in chunk_embeddings {
        for (p, &v) in pooled.iter_mut().zip(vec) {
            *p += v;
        }
    }

    let n = chunk_embeddings.len() as f32;
    for v in pooled.iter_mut() {
        *v /= n;
    }

    let norm: f32 = pooled.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for v in pooled.iter_mut() {
            *v /= norm;
        }
    }
    pooled
}

fn build_flat_and_metas(plans: Vec<TextPlan>) -> (Vec<String>, Vec<Meta>) {
    let mut flat = Vec::new();
    let mut metas = Vec::with_capacity(plans.len());
    for plan in plans {
        match plan {
            TextPlan::Single(full) => {
                let idx = flat.len();
                flat.push(full);
                metas.push(Meta::Single { idx });
            }
            TextPlan::Multi(chunks) => {
                let start = flat.len();
                let len = chunks.len();
                flat.extend(chunks);
                metas.push(Meta::Multi { start, len });
            }
        }
    }
    (flat, metas)
}

/// Assembles per-document embeddings from a raw flat-chunk batch result.
///
/// `raw` has one entry per element of `flat` (i.e. one per chunk).
/// Returns one pooled embedding per logical document (one per entry in `metas`).
fn pool_batch_results(raw: Vec<Vec<f32>>, metas: &[Meta]) -> Vec<Vec<f32>> {
    let mut out = Vec::with_capacity(metas.len());
    for meta in metas {
        match meta {
            Meta::Single { idx } => out.push(raw[*idx].clone()),
            Meta::Multi { start, len } => out.push(mean_pool_l2(&raw[*start..*start + *len])),
        }
    }
    out
}

/// Runs the primary fused batch embed and returns one pooled vector per document.
fn run_batch(
    embedder: &SentenceTransformersEmbedder,
    flat: &[String],
    metas: &[Meta],
) -> Result<Vec<Vec<f32>>, String> {
    let raw = embedder
        .embed_batch(flat)
        .map_err(|e| format!("Fused embed failed: {}", e))?;
    Ok(pool_batch_results(raw, metas))
}

/// Sequential fallback: embeds each document's chunks together, then pools Multi docs.
/// Returns one pooled vector per document.
fn fallback_sequential(
    embedder: &SentenceTransformersEmbedder,
    flat: &[String],
    metas: &[Meta],
) -> Result<Vec<Vec<f32>>, String> {
    let mut out = Vec::with_capacity(metas.len());
    for meta in metas {
        match meta {
            Meta::Single { idx } => {
                let emb = embedder
                    .embed_single(&flat[*idx])
                    .map_err(|e| format!("Single embed failed: {}", e))?;
                out.push(emb);
            }
            Meta::Multi { start, len } => {
                let chunk_embeddings = embedder
                    .embed_batch(&flat[*start..*start + *len])
                    .map_err(|e| format!("Chunk batch embed failed: {}", e))?;
                out.push(mean_pool_l2(&chunk_embeddings));
            }
        }
    }
    Ok(out)
}

/// Last-resort fallback: embeds every chunk one at a time, then pools Multi docs.
/// Returns one pooled vector per document.
fn fallback_one_by_one(
    embedder: &SentenceTransformersEmbedder,
    flat: &[String],
    metas: &[Meta],
) -> Result<Vec<Vec<f32>>, String> {
    let raw = embedder
        .embed_one_by_one(flat)
        .map_err(|e| format!("One-by-one embed failed: {}", e))?;

    let mut out = Vec::with_capacity(metas.len());
    for meta in metas {
        match meta {
            Meta::Single { idx } => out.push(raw[*idx].clone()),
            Meta::Multi { start, len } => {
                out.push(mean_pool_l2(&raw[*start..*start + *len]));
            }
        }
    }
    Ok(out)
}

/// Converts `texts` into a flattened embedding buffer `(Vec<f32>, [rows, cols])`.
///
/// On model failure the pipeline degrades through `run_batch` →
/// `fallback_sequential` → `fallback_one_by_one`, and as a last resort
/// returns zero-vectors so callers always receive a well-formed buffer.
pub fn embed_texts(
    embedder: &SentenceTransformersEmbedder,
    texts: Vec<String>,
) -> (Vec<f32>, [usize; 2]) {
    let inputs = prepare_text_inputs(texts);
    let dim = embedder.dim();

    if inputs.is_empty() {
        return (Vec::new(), [0, dim]);
    }

    let num_inputs = inputs.len();
    log::debug!("Generating embeddings for {} text(s)", num_inputs);

    let mut plans = Vec::with_capacity(num_inputs);
    for text in inputs {
        plans.push(plan_text(text, embedder));
    }

    let (flat, metas) = build_flat_and_metas(plans);
    if flat.is_empty() {
        return (Vec::new(), [0, dim]);
    }

    let doc_embs = match run_batch(embedder, &flat, &metas) {
        Ok(e) => e,
        Err(e) => {
            log::warn!("{}, falling back to sequential", e);
            match fallback_sequential(embedder, &flat, &metas) {
                Ok(seq_emb) => seq_emb,
                Err(e2) => {
                    log::warn!("Sequential fallback failed ({}), trying one-by-one", e2);
                    match fallback_one_by_one(embedder, &flat, &metas) {
                        Ok(one_emb) => one_emb,
                        Err(e3) => {
                            log::error!(
                                "All embedding fallbacks failed ({}). Returning zero vectors to maintain pipeline continuity.",
                                e3
                            );
                            zero_vectors(num_inputs, dim)
                        }
                    }
                }
            }
        }
    };

    let mut flat_out = Vec::with_capacity(num_inputs * dim);
    for emb in doc_embs {
        flat_out.extend_from_slice(&emb);
    }

    (flat_out, [num_inputs, dim])
}
