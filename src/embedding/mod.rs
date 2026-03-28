//! Embedding client abstraction and provider implementations.

pub mod local;
pub mod mock;
pub mod openai;

use async_trait::async_trait;

use crate::config::ConfigError;
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
    /// OpenAI API.
    OpenAi,
}

/// Configuration for the embedding provider.
#[derive(Debug, Clone)]
pub struct EmbeddingConfig {
    /// Which provider to use.
    pub provider: EmbeddingProvider,
    /// Path to local ONNX model directory (for [`EmbeddingProvider::Local`]).
    pub model_path: Option<String>,
    /// Max sequence length for local tokenizer truncation.
    pub max_seq_len: usize,
    /// API key (for [`EmbeddingProvider::OpenAi`]).
    pub api_key: Option<String>,
    /// Model name (for [`EmbeddingProvider::OpenAi`]).
    pub model: Option<String>,
    /// Embedding dimensions (for [`EmbeddingProvider::OpenAi`]).
    pub dimensions: Option<usize>,
}

impl EmbeddingConfig {
    /// Read embedding configuration from environment.
    pub fn from_env() -> std::result::Result<Self, ConfigError> {
        let provider = match std::env::var("EMBEDDING_PROVIDER")
            .map_err(|e| ConfigError::internal(format!("EMBEDDING_PROVIDER must be set: {e}")))?
            .as_str()
        {
            "openai" => EmbeddingProvider::OpenAi,
            "local" => EmbeddingProvider::Local,
            other => {
                return Err(ConfigError::internal(format!(
                    "unknown EMBEDDING_PROVIDER: {other}"
                )));
            }
        };

        Ok(Self {
            provider,
            model_path: std::env::var("EMBEDDING_MODEL_PATH").ok(),
            max_seq_len: std::env::var("EMBEDDING_MAX_SEQ_LEN")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(512),
            api_key: std::env::var("EMBEDDING_API_KEY").ok(),
            model: std::env::var("EMBEDDING_API_MODEL").ok(),
            dimensions: std::env::var("EMBEDDING_API_DIMS")
                .ok()
                .and_then(|s| s.parse().ok()),
        })
    }
}

/// Read embedding configuration from environment.
pub fn config_from_env() -> Result<EmbeddingConfig> {
    EmbeddingConfig::from_env().map_err(Into::into)
}

/// Build an embedding client from a configuration.
pub fn build_client(config: &EmbeddingConfig) -> Result<Box<dyn EmbeddingClient>> {
    match config.provider {
        EmbeddingProvider::Local => {
            let model_path = config.model_path.as_deref().ok_or_else(|| {
                crate::error::Error::Embedding(
                    "EMBEDDING_MODEL_PATH must be set for local embeddings".into(),
                )
            })?;
            let client =
                local::LocalEmbeddings::new(std::path::Path::new(model_path), config.max_seq_len)?;
            Ok(Box::new(client))
        }
        EmbeddingProvider::OpenAi => {
            let api_key = config.api_key.clone().ok_or_else(|| {
                crate::error::Error::Embedding("EMBEDDING_API_KEY must be set".into())
            })?;
            let model = config.model.clone().ok_or_else(|| {
                crate::error::Error::Embedding("EMBEDDING_API_MODEL must be set".into())
            })?;
            let dimensions = config.dimensions.ok_or_else(|| {
                crate::error::Error::Embedding("EMBEDDING_API_DIMS must be set".into())
            })?;
            let client = openai::OpenAiEmbeddings::new(api_key, model, dimensions);
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
