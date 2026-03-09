//! OpenAI embedding client.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use super::EmbeddingClient;
use crate::error::{Error, Result};

const API_URL: &str = "https://api.openai.com/v1";

/// An embedding client that calls the OpenAI embeddings API.
pub struct OpenAiEmbeddings {
    client: reqwest::Client,
    api_key: String,
    model: String,
    dims: usize,
}

impl OpenAiEmbeddings {
    /// Create a new OpenAI embedding client.
    pub fn new(api_key: String, model: String, dimensions: usize) -> Self {
        Self {
            client: reqwest::Client::new(),
            api_key,
            model,
            dims: dimensions,
        }
    }
}

#[derive(Serialize)]
struct EmbeddingRequest<'a> {
    model: &'a str,
    input: &'a [&'a str],
}

#[derive(Deserialize)]
struct EmbeddingResponse {
    data: Vec<EmbeddingData>,
}

#[derive(Deserialize)]
struct EmbeddingData {
    embedding: Vec<f32>,
}

#[async_trait]
impl EmbeddingClient for OpenAiEmbeddings {
    async fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let request_body = EmbeddingRequest {
            model: &self.model,
            input: texts,
        };

        let response = self
            .client
            .post(format!("{API_URL}/embeddings"))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(&request_body)
            .send()
            .await
            .map_err(|e| Error::Embedding(format!("request failed: {e}")))?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(Error::Embedding(format!("API error {status}: {body}")));
        }

        let result: EmbeddingResponse = response
            .json()
            .await
            .map_err(|e| Error::Embedding(format!("response parse error: {e}")))?;

        Ok(result.data.into_iter().map(|d| d.embedding).collect())
    }

    fn dimensions(&self) -> usize {
        self.dims
    }

    fn model_name(&self) -> &str {
        &self.model
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    #[ignore = "requires OPENAI_API_KEY"]
    async fn embed_single_text() {
        let _ = dotenvy::dotenv();
        let api_key = std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY must be set");
        let client = OpenAiEmbeddings::new(
            api_key,
            std::env::var("EMBEDDING_API_MODEL").expect("EMBEDDING_API_MODEL must be set"),
            std::env::var("EMBEDDING_API_DIMS")
                .expect("EMBEDDING_API_DIMS must be set")
                .parse()
                .expect("EMBEDDING_API_DIMS must be a number"),
        );
        let result = client.embed(&["Hello, world!"]).await.unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].len(), 1536);
    }

    #[tokio::test]
    #[ignore = "requires OPENAI_API_KEY"]
    async fn embed_batch() {
        let _ = dotenvy::dotenv();
        let api_key = std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY must be set");
        let client = OpenAiEmbeddings::new(
            api_key,
            std::env::var("EMBEDDING_API_MODEL").expect("EMBEDDING_API_MODEL must be set"),
            std::env::var("EMBEDDING_API_DIMS")
                .expect("EMBEDDING_API_DIMS must be set")
                .parse()
                .expect("EMBEDDING_API_DIMS must be a number"),
        );
        let result = client.embed(&["cat", "dog", "database"]).await.unwrap();
        assert_eq!(result.len(), 3);
        for v in &result {
            assert_eq!(v.len(), 1536);
        }
    }
}
