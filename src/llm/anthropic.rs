//! Anthropic Claude Messages API client.

use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};
use crate::llm::LlmClient;
use crate::types::llm::{CompletionRequest, CompletionResponse};

const ANTHROPIC_API_URL: &str = "https://api.anthropic.com/v1/messages";
const ANTHROPIC_VERSION: &str = "2023-06-01";

/// Client for the Anthropic Claude Messages API.
pub struct AnthropicClient {
    client: Client,
    api_key: String,
    default_model: String,
}

impl AnthropicClient {
    /// Create a new Anthropic client.
    pub fn new(api_key: String, model: String) -> Self {
        Self {
            client: Client::new(),
            api_key,
            default_model: model,
        }
    }
}

// --- Anthropic API request/response types ---

#[derive(Serialize)]
struct AnthropicRequest {
    model: String,
    messages: Vec<AnthropicMessage>,
    max_tokens: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<String>,
}

#[derive(Serialize)]
struct AnthropicMessage {
    role: String,
    content: String,
}

#[derive(Deserialize)]
struct AnthropicResponse {
    content: Vec<ContentBlock>,
    usage: AnthropicUsage,
}

#[derive(Deserialize)]
struct ContentBlock {
    text: String,
}

#[derive(Deserialize)]
struct AnthropicUsage {
    input_tokens: usize,
    output_tokens: usize,
}

#[derive(Deserialize)]
struct AnthropicError {
    error: AnthropicErrorDetail,
}

#[derive(Deserialize)]
struct AnthropicErrorDetail {
    message: String,
}

#[async_trait]
impl LlmClient for AnthropicClient {
    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse> {
        let model = if request.model.is_empty() {
            self.default_model.clone()
        } else {
            request.model
        };

        let messages: Vec<AnthropicMessage> = request
            .messages
            .into_iter()
            .map(|m| AnthropicMessage {
                role: m.role,
                content: m.content,
            })
            .collect();

        let body = AnthropicRequest {
            model,
            messages,
            max_tokens: request.max_tokens.unwrap_or(4096),
            temperature: request.temperature,
            system: request.system,
        };

        let resp = self
            .client
            .post(ANTHROPIC_API_URL)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", ANTHROPIC_VERSION)
            .header("content-type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| Error::Llm(format!("Anthropic request failed: {e}")))?;

        let status = resp.status();
        let resp_text = resp
            .text()
            .await
            .map_err(|e| Error::Llm(format!("failed to read Anthropic response: {e}")))?;

        if status.is_success() {
            let parsed: AnthropicResponse = serde_json::from_str(&resp_text)
                .map_err(|e| Error::Llm(format!("failed to parse Anthropic response: {e}")))?;

            let content = parsed
                .content
                .into_iter()
                .next()
                .map(|b| b.text)
                .unwrap_or_default();

            return Ok(CompletionResponse {
                content,
                input_tokens: parsed.usage.input_tokens,
                output_tokens: parsed.usage.output_tokens,
            });
        }

        let msg = serde_json::from_str::<AnthropicError>(&resp_text)
            .map(|e| e.error.message)
            .unwrap_or_else(|_| resp_text);

        if status.as_u16() == 429 {
            return Err(Error::RateLimit(format!("Anthropic API ({status}): {msg}")));
        }
        if status.is_server_error() {
            return Err(Error::ServerError(format!("Anthropic API ({status}): {msg}")));
        }

        Err(Error::Llm(format!("Anthropic API error ({status}): {msg}")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::llm::Message;

    #[tokio::test]
    #[ignore = "requires ANTHROPIC_API_KEY"]
    async fn integration_simple_prompt() {
        let api_key =
            std::env::var("ANTHROPIC_API_KEY").expect("ANTHROPIC_API_KEY must be set");
        let client = AnthropicClient::new(api_key, std::env::var("LLM_MODEL").expect("LLM_MODEL must be set"));

        let request = CompletionRequest {
            model: String::new(),
            messages: vec![Message {
                role: "user".into(),
                content: "Say hello in exactly 3 words.".into(),
            }],
            max_tokens: Some(64),
            temperature: Some(0.0),
            system: None,
        };

        let resp = client.complete(request).await.unwrap();
        assert!(!resp.content.is_empty());
        assert!(resp.input_tokens > 0);
        assert!(resp.output_tokens > 0);
    }

    #[tokio::test]
    #[ignore = "requires ANTHROPIC_API_KEY"]
    async fn integration_structured() {
        use crate::llm::complete_structured;
        use serde::Deserialize;

        let api_key =
            std::env::var("ANTHROPIC_API_KEY").expect("ANTHROPIC_API_KEY must be set");
        let client = AnthropicClient::new(api_key, std::env::var("LLM_MODEL").expect("LLM_MODEL must be set"));

        #[derive(Deserialize, Debug)]
        struct Color {
            name: String,
            hex: String,
        }

        let request = CompletionRequest {
            model: String::new(),
            messages: vec![Message {
                role: "user".into(),
                content: "Return a JSON object with fields \"name\" and \"hex\" for the color red. Only output JSON, nothing else.".into(),
            }],
            max_tokens: Some(64),
            temperature: Some(0.0),
            system: None,
        };

        let color: Color = complete_structured(&client, request).await.unwrap();
        assert!(!color.name.is_empty());
        assert!(color.hex.starts_with('#'));
    }
}
