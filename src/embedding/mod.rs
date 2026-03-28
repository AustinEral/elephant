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
    /// OpenAI API.
    OpenAi,
}

/// Configuration for the embedding provider.
#[derive(Debug, Clone)]
pub struct EmbeddingConfig {
    provider: EmbeddingProvider,
    model_path: Option<String>,
    max_seq_len: usize,
    api_key: Option<String>,
    model: Option<String>,
    dimensions: Option<usize>,
}

impl EmbeddingConfig {
    pub(crate) fn from_parts(
        provider: EmbeddingProvider,
        model_path: Option<String>,
        max_seq_len: usize,
        api_key: Option<String>,
        model: Option<String>,
        dimensions: Option<usize>,
    ) -> Self {
        Self {
            provider,
            model_path,
            max_seq_len,
            api_key,
            model,
            dimensions,
        }
    }

    /// Create a local embedding configuration from a model directory.
    pub fn local(model_path: impl Into<String>) -> Self {
        Self::from_parts(
            EmbeddingProvider::Local,
            Some(model_path.into()),
            512,
            None,
            None,
            None,
        )
    }

    /// Create an OpenAI embedding configuration from explicit credentials.
    pub fn openai(api_key: impl Into<String>, model: impl Into<String>, dimensions: usize) -> Self {
        Self::from_parts(
            EmbeddingProvider::OpenAi,
            None,
            512,
            Some(api_key.into()),
            Some(model.into()),
            Some(dimensions),
        )
    }

    /// Override the maximum tokenized sequence length.
    pub fn with_max_seq_len(mut self, max_seq_len: usize) -> Self {
        self.max_seq_len = max_seq_len;
        self
    }

    /// Return the selected provider.
    pub fn provider(&self) -> EmbeddingProvider {
        self.provider.clone()
    }

    /// Return the local model directory, if configured.
    pub fn model_path(&self) -> Option<&str> {
        self.model_path.as_deref()
    }

    /// Return the tokenizer sequence cap.
    pub fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }

    /// Return the OpenAI API key, if configured.
    pub fn api_key(&self) -> Option<&str> {
        self.api_key.as_deref()
    }

    /// Return the provider model label, if configured.
    pub fn model(&self) -> Option<&str> {
        self.model.as_deref()
    }

    /// Return the embedding dimensionality, if configured.
    pub fn dimensions(&self) -> Option<usize> {
        self.dimensions
    }
}

/// Build an embedding client from a configuration.
pub fn build_client(config: &EmbeddingConfig) -> Result<Box<dyn EmbeddingClient>> {
    match config.provider() {
        EmbeddingProvider::Local => {
            let model_path = config.model_path().ok_or_else(|| {
                crate::error::Error::Embedding(
                    "EMBEDDING_MODEL_PATH must be set for local embeddings".into(),
                )
            })?;
            if model_path.trim().is_empty() {
                return Err(crate::error::Error::Embedding(
                    "EMBEDDING_MODEL_PATH must not be blank for local embeddings".into(),
                ));
            }
            let client = local::LocalEmbeddings::new(
                std::path::Path::new(model_path),
                config.max_seq_len(),
            )?;
            Ok(Box::new(client))
        }
        EmbeddingProvider::OpenAi => {
            let api_key = config.api_key().map(str::to_string).ok_or_else(|| {
                crate::error::Error::Embedding("EMBEDDING_API_KEY must be set".into())
            })?;
            if api_key.trim().is_empty() {
                return Err(crate::error::Error::Embedding(
                    "EMBEDDING_API_KEY must not be blank".into(),
                ));
            }
            let model = config.model().map(str::to_string).ok_or_else(|| {
                crate::error::Error::Embedding("EMBEDDING_API_MODEL must be set".into())
            })?;
            if model.trim().is_empty() {
                return Err(crate::error::Error::Embedding(
                    "EMBEDDING_API_MODEL must not be blank".into(),
                ));
            }
            let dimensions = config.dimensions().ok_or_else(|| {
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
