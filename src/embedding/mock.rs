//! Mock embedding client for testing.

use async_trait::async_trait;

use super::EmbeddingClient;
use crate::error::Result;

/// A deterministic mock embedding client.
///
/// Hashes input text to seed a simple PRNG, producing L2-normalized vectors.
/// Same text always produces the same embedding.
#[derive(Clone)]
pub struct MockEmbeddings {
    dims: usize,
}

impl MockEmbeddings {
    /// Create a new mock embedding client with the given dimensionality.
    pub fn new(dimensions: usize) -> Self {
        Self { dims: dimensions }
    }
}

#[async_trait]
impl EmbeddingClient for MockEmbeddings {
    async fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        Ok(texts
            .iter()
            .map(|t| deterministic_vector(t, self.dims))
            .collect())
    }

    fn dimensions(&self) -> usize {
        self.dims
    }

    fn model_name(&self) -> &str {
        "mock"
    }
}

/// Generate a deterministic, L2-normalized vector from text.
fn deterministic_vector(text: &str, dims: usize) -> Vec<f32> {
    // Simple hash: FNV-1a
    let mut hash: u64 = 0xcbf29ce484222325;
    for b in text.bytes() {
        hash ^= b as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }

    // Use hash as PRNG seed (xorshift64)
    let mut state = hash;
    let mut vec = Vec::with_capacity(dims);
    for _ in 0..dims {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        // Map to [-1, 1]
        let val = (state as f32) / (u64::MAX as f32) * 2.0 - 1.0;
        vec.push(val);
    }

    // L2 normalize
    let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in &mut vec {
            *x /= norm;
        }
    }

    vec
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn embed_returns_correct_dimensions() {
        let client = MockEmbeddings::new(384);
        let result = client.embed(&["hello", "world"]).await.unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].len(), 384);
        assert_eq!(result[1].len(), 384);
    }

    #[tokio::test]
    async fn embed_is_deterministic() {
        let client = MockEmbeddings::new(384);
        let a = client.embed(&["hello"]).await.unwrap();
        let b = client.embed(&["hello"]).await.unwrap();
        assert_eq!(a, b);
    }

    #[tokio::test]
    async fn different_texts_different_embeddings() {
        let client = MockEmbeddings::new(384);
        let result = client.embed(&["hello", "world"]).await.unwrap();
        assert_ne!(result[0], result[1]);
    }

    #[tokio::test]
    async fn embeddings_are_normalized() {
        let client = MockEmbeddings::new(384);
        let result = client.embed(&["test normalization"]).await.unwrap();
        let norm: f32 = result[0].iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5, "norm was {norm}");
    }
}
