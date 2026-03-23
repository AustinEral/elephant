//! Vertex AI wrapper over the Gemini `generateContent` transport.

use async_trait::async_trait;

use crate::error::Result;
use crate::llm::{
    CompletionRequest, CompletionResponse, GeminiConfig, LlmClient, VertexConfig, gemini,
};

const API_URL: &str = "https://aiplatform.googleapis.com/v1";

/// Vertex AI client for Gemini models.
pub struct VertexClient {
    inner: gemini::GeminiClient,
}

impl VertexClient {
    /// Create a new Vertex AI client using the Gemini transport implementation.
    pub fn new(config: VertexConfig) -> Result<Self> {
        let base_url = config.base_url().map(str::to_owned).unwrap_or_else(|| {
            format!(
                "{}/projects/{}/locations/{}/publishers/google",
                API_URL,
                config.project(),
                config.location()
            )
        });

        let gemini_config = GeminiConfig::new(config.api_key(), config.model())?
            .with_timeout_secs(config.timeout_secs())?
            .with_function_call_ids(false)
            .with_base_url(base_url)?;

        Ok(Self {
            inner: gemini::GeminiClient::new(gemini_config)?,
        })
    }
}

#[async_trait]
impl LlmClient for VertexClient {
    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse> {
        self.inner.complete(request).await
    }
}
