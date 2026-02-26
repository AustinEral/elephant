//! Embedding client abstraction and provider implementations.

pub mod local;
pub mod mock;
pub mod openai;

use async_trait::async_trait;

use crate::error::Result;

/// Trait abstraction over embedding providers.
///
/// Every component that needs vector similarity takes `dyn EmbeddingClient`,
/// making the system testable with [`mock::MockEmbeddings`] and swappable
/// between providers.
#[async_trait]
pub trait EmbeddingClient: Send + Sync {
    /// Embed a batch of texts into vectors.
    async fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>>;

    /// The dimensionality of the embedding vectors.
    fn dimensions(&self) -> usize;

    /// The model name used for embeddings (e.g. "bge-small-en-v1.5").
    fn model_name(&self) -> &str;
}

/// Embedding provider selection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EmbeddingProvider {
    /// Local ONNX model (bge-small-en-v1.5, 384 dims).
    Local,
    /// OpenAI-compatible API (text-embedding-3-small, 1536 dims).
    OpenAi,
}

/// Configuration for the embedding provider.
#[derive(Debug, Clone)]
pub struct EmbeddingConfig {
    /// Which provider to use.
    pub provider: EmbeddingProvider,
    /// Path to local ONNX model directory (for [`EmbeddingProvider::Local`]).
    pub model_path: Option<String>,
    /// API key (for [`EmbeddingProvider::OpenAi`]).
    pub api_key: Option<String>,
    /// Model name override.
    pub model: Option<String>,
    /// Base URL for OpenAI-compatible APIs.
    pub base_url: Option<String>,
    /// Override the default dimensionality.
    pub dimensions: Option<usize>,
}

/// Build an embedding client from a configuration.
pub fn build_client(config: &EmbeddingConfig) -> Result<Box<dyn EmbeddingClient>> {
    match config.provider {
        EmbeddingProvider::Local => {
            let model_path = config
                .model_path
                .as_deref()
                .unwrap_or("models/bge-small-en-v1.5");
            let client = local::LocalEmbeddings::new(std::path::Path::new(model_path))?;
            Ok(Box::new(client))
        }
        EmbeddingProvider::OpenAi => {
            let api_key = config
                .api_key
                .clone()
                .ok_or_else(|| crate::error::Error::Embedding("api_key required for OpenAI".into()))?;
            let client = openai::OpenAiEmbeddings::new(
                api_key,
                config.model.clone(),
                config.base_url.clone(),
                config.dimensions,
            );
            Ok(Box::new(client))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn build_mock_and_embed() {
        let client = mock::MockEmbeddings::new(384);
        let result = client.embed(&["test"]).await.unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].len(), 384);
        assert_eq!(client.dimensions(), 384);
    }
}
