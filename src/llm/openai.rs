//! OpenAI Chat Completions API client.

use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};
use crate::llm::LlmClient;
use crate::types::llm::{CompletionRequest, CompletionResponse};

const API_URL: &str = "https://api.openai.com/v1";

/// Client for the OpenAI Chat Completions API.
pub struct OpenAiClient {
    client: Client,
    api_key: String,
    default_model: String,
    base_url: String,
}

impl OpenAiClient {
    /// Create a new OpenAI client with optional base URL for compatible providers.
    pub fn new(api_key: String, model: String, base_url: Option<String>) -> Self {
        Self {
            client: Client::new(),
            api_key,
            default_model: model,
            base_url: base_url.unwrap_or_else(|| API_URL.to_string()),
        }
    }
}

// --- OpenAI API request/response types ---

#[derive(Serialize)]
struct OpenAiRequest {
    model: String,
    messages: Vec<OpenAiMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
}

#[derive(Serialize)]
struct OpenAiMessage {
    role: String,
    content: String,
}

#[derive(Deserialize)]
struct OpenAiResponse {
    choices: Vec<OpenAiChoice>,
    usage: Option<OpenAiUsage>,
}

#[derive(Deserialize)]
struct OpenAiChoice {
    message: OpenAiMessageResp,
}

#[derive(Deserialize)]
struct OpenAiMessageResp {
    content: Option<String>,
}

#[derive(Deserialize)]
struct OpenAiUsage {
    prompt_tokens: usize,
    completion_tokens: usize,
}

#[derive(Deserialize)]
struct OpenAiError {
    error: OpenAiErrorDetail,
}

#[derive(Deserialize)]
struct OpenAiErrorDetail {
    message: String,
}

#[async_trait]
impl LlmClient for OpenAiClient {
    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse> {
        let model = if request.model.is_empty() {
            self.default_model.clone()
        } else {
            request.model
        };

        // Build messages: system prompt goes as a system message in the array
        let mut messages: Vec<OpenAiMessage> = Vec::new();
        if let Some(system) = request.system {
            messages.push(OpenAiMessage {
                role: "system".into(),
                content: system,
            });
        }
        for m in request.messages {
            messages.push(OpenAiMessage {
                role: m.role,
                content: m.content,
            });
        }

        let body = OpenAiRequest {
            model,
            messages,
            max_tokens: request.max_tokens,
            temperature: request.temperature,
        };

        let url = format!("{}/chat/completions", self.base_url);

        let resp = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| Error::Llm(format!("OpenAI request failed: {e}")))?;

        let status = resp.status();
        let resp_text = resp
            .text()
            .await
            .map_err(|e| Error::Llm(format!("failed to read OpenAI response: {e}")))?;

        if status.is_success() {
            let parsed: OpenAiResponse = serde_json::from_str(&resp_text)
                .map_err(|e| Error::Llm(format!("failed to parse OpenAI response: {e}")))?;

            let content = parsed
                .choices
                .into_iter()
                .next()
                .and_then(|c| c.message.content)
                .unwrap_or_default();

            let (input_tokens, output_tokens) = parsed
                .usage
                .map(|u| (u.prompt_tokens, u.completion_tokens))
                .unwrap_or((0, 0));

            return Ok(CompletionResponse {
                content,
                input_tokens,
                output_tokens,
            });
        }

        let msg = serde_json::from_str::<OpenAiError>(&resp_text)
            .map(|e| e.error.message)
            .unwrap_or_else(|_| resp_text);

        if status.as_u16() == 429 {
            return Err(Error::RateLimit(format!("OpenAI API ({status}): {msg}")));
        }
        if status.is_server_error() {
            return Err(Error::ServerError(format!("OpenAI API ({status}): {msg}")));
        }

        Err(Error::Llm(format!("OpenAI API error ({status}): {msg}")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::llm::Message;

    #[tokio::test]
    #[ignore = "requires OPENAI_API_KEY"]
    async fn integration_simple_prompt() {
        let _ = dotenvy::dotenv();
        let api_key = std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY must be set");
        let client = OpenAiClient::new(api_key, std::env::var("LLM_MODEL").expect("LLM_MODEL must be set"), None);

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
    }

    #[tokio::test]
    #[ignore = "requires OPENAI_API_KEY"]
    async fn integration_structured() {
        let _ = dotenvy::dotenv();
        use crate::llm::complete_structured;
        use serde::Deserialize;

        let api_key = std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY must be set");
        let client = OpenAiClient::new(api_key, std::env::var("LLM_MODEL").expect("LLM_MODEL must be set"), None);

        #[derive(Deserialize, Debug)]
        struct Color {
            name: String,
            hex: String,
        }

        let request = CompletionRequest {
            model: String::new(),
            messages: vec![Message {
                role: "user".into(),
                content: "Return a JSON object with fields \"name\" and \"hex\" for the color blue. Only output JSON, nothing else.".into(),
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
