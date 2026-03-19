//! Local ONNX-based embedding client using bge-small-en-v1.5.

use std::path::Path;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use ndarray::{Array2, Axis};
use ort::session::Session;
use ort::value::TensorRef;

use super::EmbeddingClient;
use crate::error::{Error, Result};

/// A local embedding client that runs bge-small-en-v1.5 via ONNX Runtime.
///
/// Produces 384-dimensional embeddings. The ONNX model and tokenizer.json must
/// be available at the configured model directory path.
pub struct LocalEmbeddings {
    session: Arc<Mutex<Session>>,
    tokenizer: Arc<tokenizers::Tokenizer>,
}

impl LocalEmbeddings {
    /// Load the ONNX model and tokenizer from `model_dir`.
    ///
    /// Expects `model_dir/model.onnx` and `model_dir/tokenizer.json`.
    pub fn new(model_dir: &Path, max_seq_len: usize) -> Result<Self> {
        let model_path = model_dir.join("model.onnx");
        let tokenizer_path = model_dir.join("tokenizer.json");

        let session = Session::builder()
            .map_err(|e| Error::Embedding(format!("session builder error: {e}")))?
            .with_intra_threads(1)
            .map_err(|e| Error::Embedding(format!("thread config error: {e}")))?
            .commit_from_file(&model_path)
            .map_err(|e| Error::Embedding(format!("model load error: {e}")))?;

        let mut tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| Error::Embedding(format!("tokenizer load error: {e}")))?;
        tokenizer
            .with_truncation(Some(tokenizers::TruncationParams {
                max_length: max_seq_len,
                ..Default::default()
            }))
            .map_err(|e| Error::Embedding(format!("truncation config error: {e}")))?;

        Ok(Self {
            session: Arc::new(Mutex::new(session)),
            tokenizer: Arc::new(tokenizer),
        })
    }
}

#[async_trait]
impl EmbeddingClient for LocalEmbeddings {
    async fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let session = self.session.clone();
        let tokenizer = self.tokenizer.clone();
        let texts: Vec<String> = texts.iter().map(|s| s.to_string()).collect();

        tokio::task::spawn_blocking(move || embed_sync(&session, &tokenizer, &texts))
            .await
            .map_err(|e| Error::Embedding(format!("spawn_blocking error: {e}")))?
    }

    fn dimensions(&self) -> usize {
        384
    }

    fn model_name(&self) -> &str {
        "bge-small-en-v1.5"
    }
}

fn embed_sync(
    session: &Mutex<Session>,
    tokenizer: &tokenizers::Tokenizer,
    texts: &[String],
) -> Result<Vec<Vec<f32>>> {
    let encodings = tokenizer
        .encode_batch(texts.to_vec(), true)
        .map_err(|e| Error::Embedding(format!("tokenization error: {e}")))?;

    let batch_size = encodings.len();
    let seq_len = encodings
        .iter()
        .map(|e| e.get_ids().len())
        .max()
        .unwrap_or(0);

    // Build input tensors: input_ids, attention_mask, token_type_ids
    let mut input_ids = Array2::<i64>::zeros((batch_size, seq_len));
    let mut attention_mask = Array2::<i64>::zeros((batch_size, seq_len));
    let mut token_type_ids = Array2::<i64>::zeros((batch_size, seq_len));

    for (i, encoding) in encodings.iter().enumerate() {
        for (j, &id) in encoding.get_ids().iter().enumerate() {
            input_ids[[i, j]] = id as i64;
        }
        for (j, &mask) in encoding.get_attention_mask().iter().enumerate() {
            attention_mask[[i, j]] = mask as i64;
        }
        for (j, &type_id) in encoding.get_type_ids().iter().enumerate() {
            token_type_ids[[i, j]] = type_id as i64;
        }
    }

    let input_ids_tensor = TensorRef::from_array_view(input_ids.view())
        .map_err(|e| Error::Embedding(format!("input_ids tensor error: {e}")))?;
    let attention_mask_tensor = TensorRef::from_array_view(attention_mask.view())
        .map_err(|e| Error::Embedding(format!("attention_mask tensor error: {e}")))?;
    let token_type_ids_tensor = TensorRef::from_array_view(token_type_ids.view())
        .map_err(|e| Error::Embedding(format!("token_type_ids tensor error: {e}")))?;

    let mut session = session
        .lock()
        .map_err(|e| Error::Embedding(format!("session lock error: {e}")))?;

    let outputs = session
        .run(ort::inputs![
            "input_ids" => input_ids_tensor,
            "attention_mask" => attention_mask_tensor,
            "token_type_ids" => token_type_ids_tensor,
        ])
        .map_err(|e| Error::Embedding(format!("inference error: {e}")))?;

    // Extract last_hidden_state: shape [batch, seq_len, 384]
    let hidden = outputs[0]
        .try_extract_array::<f32>()
        .map_err(|e| Error::Embedding(format!("output extraction error: {e}")))?
        .into_dimensionality::<ndarray::Ix3>()
        .map_err(|e| Error::Embedding(format!("shape error: {e}")))?;

    // Mean pooling with attention mask
    let mut results = Vec::with_capacity(batch_size);
    for (i, encoding) in encodings.iter().enumerate().take(batch_size) {
        let token_count = encoding
            .get_attention_mask()
            .iter()
            .filter(|&&m| m == 1)
            .count();
        let sentence = hidden.index_axis(Axis(0), i);
        let mut pooled = vec![0.0f32; 384];
        for j in 0..token_count {
            for k in 0..384 {
                pooled[k] += sentence[[j, k]];
            }
        }
        // Mean
        let tc = token_count as f32;
        for x in &mut pooled {
            *x /= tc;
        }
        // L2 normalize
        let norm: f32 = pooled.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in &mut pooled {
                *x /= norm;
            }
        }
        results.push(pooled);
    }

    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn model_dir() -> PathBuf {
        let _ = dotenvy::dotenv();
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models/bge-small-en-v1.5")
    }

    #[tokio::test]
    #[ignore]
    async fn embed_returns_384_dims() {
        let client = LocalEmbeddings::new(&model_dir(), 512).unwrap();
        let result = client.embed(&["Hello, world!"]).await.unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].len(), 384);
    }

    #[tokio::test]
    #[ignore]
    async fn semantic_similarity() {
        let client = LocalEmbeddings::new(&model_dir(), 512).unwrap();
        let vecs = client.embed(&["cat", "dog", "database"]).await.unwrap();
        let cat_dog = cosine_sim(&vecs[0], &vecs[1]);
        let cat_db = cosine_sim(&vecs[0], &vecs[2]);
        assert!(
            cat_dog > cat_db,
            "cat-dog similarity ({cat_dog}) should be > cat-database similarity ({cat_db})"
        );
    }

    fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
        let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        dot / (na * nb)
    }
}
