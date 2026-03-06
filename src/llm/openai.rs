//! OpenAI Chat Completions API client.

use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::llm::LlmClient;
use crate::types::llm::{CompletionRequest, CompletionResponse, ToolCall, ToolChoice};

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
    pub fn new(api_key: String, model: String, base_url: Option<String>) -> Result<Self> {
        let client = Client::builder()
            .timeout(std::time::Duration::from_secs(
                std::env::var("LLM_TIMEOUT_SECS")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(super::DEFAULT_TIMEOUT_SECS),
            ))
            .build()
            .map_err(|e| crate::error::Error::Internal(e.to_string()))?;
        Ok(Self {
            client,
            api_key,
            default_model: model,
            base_url: base_url.unwrap_or_else(|| API_URL.to_string()),
        })
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
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<OpenAiTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<OpenAiToolChoice>,
}

#[derive(Serialize, Default)]
struct OpenAiMessage {
    role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<OpenAiReqToolCall>>,
}

#[derive(Serialize)]
struct OpenAiReqToolCall {
    id: String,
    #[serde(rename = "type")]
    tool_type: String,
    function: OpenAiReqFunctionCall,
}

#[derive(Serialize)]
struct OpenAiReqFunctionCall {
    name: String,
    arguments: String,
}

#[derive(Serialize)]
struct OpenAiTool {
    #[serde(rename = "type")]
    tool_type: String,
    function: OpenAiFunction,
}

#[derive(Serialize)]
struct OpenAiFunction {
    name: String,
    description: String,
    parameters: serde_json::Value,
}

#[derive(Serialize)]
#[serde(untagged)]
enum OpenAiToolChoice {
    String(String),
    Specific(OpenAiToolChoiceSpecific),
}

#[derive(Serialize)]
struct OpenAiToolChoiceSpecific {
    #[serde(rename = "type")]
    tool_type: String,
    function: OpenAiToolChoiceFunction,
}

#[derive(Serialize)]
struct OpenAiToolChoiceFunction {
    name: String,
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
    #[serde(default)]
    tool_calls: Vec<OpenAiRespToolCall>,
}

#[derive(Deserialize)]
struct OpenAiRespToolCall {
    id: String,
    function: OpenAiRespFunctionCall,
}

#[derive(Deserialize)]
struct OpenAiRespFunctionCall {
    name: String,
    arguments: String,
}

#[derive(Deserialize)]
struct OpenAiUsage {
    prompt_tokens: usize,
    completion_tokens: usize,
}

#[async_trait]
impl LlmClient for OpenAiClient {
    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse> {
        let model = super::resolve_model(request.model, &self.default_model);

        // Build messages: system prompt goes as a system message in the array
        let mut messages: Vec<OpenAiMessage> = Vec::new();
        if let Some(system) = request.system {
            messages.push(OpenAiMessage { role: "system".into(), content: Some(system), ..Default::default() });
        }
        for m in &request.messages {
            let tool_calls = if m.tool_calls.is_empty() { None } else {
                Some(m.tool_calls.iter().map(|tc| OpenAiReqToolCall {
                    id: tc.id.clone(),
                    tool_type: "function".into(),
                    function: OpenAiReqFunctionCall {
                        name: tc.name.clone(),
                        arguments: tc.arguments.to_string(),
                    },
                }).collect())
            };
            messages.push(OpenAiMessage {
                role: m.role.clone(),
                content: if m.content.is_empty() { None } else { Some(m.content.clone()) },
                tool_calls,
                ..Default::default()
            });
            // OpenAI requires tool_results as separate role="tool" messages
            for tr in &m.tool_results {
                messages.push(OpenAiMessage {
                    role: "tool".into(),
                    content: Some(tr.content.clone()),
                    tool_call_id: Some(tr.tool_call_id.clone()),
                    ..Default::default()
                });
            }
        }
        // Append legacy tool_results from CompletionRequest
        for tr in &request.tool_results {
            messages.push(OpenAiMessage {
                role: "tool".into(),
                content: Some(tr.content.clone()),
                tool_call_id: Some(tr.tool_call_id.clone()),
                ..Default::default()
            });
        }

        // Map tools
        let tools = request.tools.map(|defs| {
            defs.into_iter()
                .map(|t| OpenAiTool {
                    tool_type: "function".into(),
                    function: OpenAiFunction {
                        name: t.name,
                        description: t.description,
                        parameters: t.input_schema,
                    },
                })
                .collect()
        });

        let tool_choice = request.tool_choice.map(|tc| match tc {
            ToolChoice::Auto => OpenAiToolChoice::String("auto".into()),
            ToolChoice::Required => OpenAiToolChoice::String("required".into()),
            ToolChoice::None => OpenAiToolChoice::String("none".into()),
            ToolChoice::Specific(name) => OpenAiToolChoice::Specific(OpenAiToolChoiceSpecific {
                tool_type: "function".into(),
                function: OpenAiToolChoiceFunction { name },
            }),
        });

        let body = OpenAiRequest {
            model,
            messages,
            max_tokens: request.max_tokens,
            temperature: request.temperature,
            tools,
            tool_choice,
        };

        let url = format!("{}/chat/completions", self.base_url);

        let resp_text = super::send_and_check(
            "OpenAI",
            self.client
                .post(&url)
                .header("Authorization", format!("Bearer {}", self.api_key))
                .header("Content-Type", "application/json")
                .json(&body),
        )
        .await?;

        let parsed: OpenAiResponse = serde_json::from_str(&resp_text)
            .map_err(|e| crate::error::Error::Llm(format!("failed to parse OpenAI response: {e}")))?;

        let choice = parsed.choices.into_iter().next();
        let content = choice
            .as_ref()
            .and_then(|c| c.message.content.clone())
            .unwrap_or_default();

        let tool_calls: Vec<ToolCall> = choice
            .map(|c| {
                c.message
                    .tool_calls
                    .into_iter()
                    .filter_map(|tc| {
                        let arguments: serde_json::Value =
                            serde_json::from_str(&tc.function.arguments).ok()?;
                        Some(ToolCall {
                            id: tc.id,
                            name: tc.function.name,
                            arguments,
                        })
                    })
                    .collect()
            })
            .unwrap_or_default();

        let (input_tokens, output_tokens) = parsed
            .usage
            .map(|u| (u.prompt_tokens, u.completion_tokens))
            .unwrap_or((0, 0));

        Ok(CompletionResponse {
            content,
            input_tokens,
            output_tokens,
            tool_calls,
        })
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
        let client = OpenAiClient::new(api_key, std::env::var("LLM_MODEL").expect("LLM_MODEL must be set"), None).unwrap();

        let request = CompletionRequest {
            messages: vec![Message::text("user", "Say hello in exactly 3 words.")],
            max_tokens: Some(64),
            temperature: Some(0.0),
            ..Default::default()
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
        let client = OpenAiClient::new(api_key, std::env::var("LLM_MODEL").expect("LLM_MODEL must be set"), None).unwrap();

        #[derive(Deserialize, Debug)]
        struct Color {
            name: String,
            hex: String,
        }

        let request = CompletionRequest {
            messages: vec![Message::text("user", "Return a JSON object with fields \"name\" and \"hex\" for the color blue. Only output JSON, nothing else.")],
            max_tokens: Some(64),
            temperature: Some(0.0),
            ..Default::default()
        };

        let color: Color = complete_structured(&client, request).await.unwrap();
        assert!(!color.name.is_empty());
        assert!(color.hex.starts_with('#'));
    }
}
