//! Anthropic Claude Messages API client.

use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};
use crate::llm::{
    AnthropicConfig, AnthropicPromptCacheConfig, AnthropicPromptCacheTtl, CompletionRequest,
    CompletionResponse, LlmClient, PromptCacheUsage, ToolCall, ToolChoice,
};

const ANTHROPIC_API_URL: &str = "https://api.anthropic.com/v1/messages";
const ANTHROPIC_VERSION: &str = "2023-06-01";
const OAUTH_BETA: &str = "oauth-2025-04-20";

/// Authentication mode for the Anthropic API.
#[derive(Debug, Clone)]
pub enum AnthropicAuth {
    /// Standard API key (x-api-key header).
    ApiKey(String),
    /// Claude Code OAuth token (Bearer auth + oauth beta header).
    /// Auto-detected from `sk-ant-oat` prefix.
    OAuth(String),
}

impl AnthropicAuth {
    /// Create from a token string, auto-detecting OAuth from prefix.
    pub fn from_token(token: String) -> Self {
        if token.starts_with("sk-ant-oat") {
            Self::OAuth(token)
        } else {
            Self::ApiKey(token)
        }
    }
}

/// Client for the Anthropic Claude Messages API.
pub struct AnthropicClient {
    client: Client,
    auth: AnthropicAuth,
    default_model: String,
    prompt_cache: Option<AnthropicPromptCacheConfig>,
}

impl AnthropicClient {
    /// Create a new Anthropic client. Auto-detects OAuth from token prefix.
    pub fn new(config: AnthropicConfig) -> Result<Self> {
        let auth = AnthropicAuth::from_token(config.api_key().to_string());
        let client = Client::builder()
            .timeout(std::time::Duration::from_secs(config.timeout_secs()))
            .build()
            .map_err(|e| Error::Internal(e.to_string()))?;
        Ok(Self {
            client,
            auth,
            default_model: config.model().to_string(),
            prompt_cache: config.prompt_cache().cloned(),
        })
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
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<AnthropicTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<AnthropicToolChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    cache_control: Option<AnthropicCacheControl>,
}

#[derive(Serialize)]
struct AnthropicCacheControl {
    #[serde(rename = "type")]
    cache_type: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    ttl: Option<AnthropicPromptCacheTtl>,
}

#[derive(Serialize)]
struct AnthropicTool {
    name: String,
    description: String,
    input_schema: serde_json::Value,
}

#[derive(Serialize)]
#[serde(tag = "type")]
enum AnthropicToolChoice {
    #[serde(rename = "auto")]
    Auto,
    #[serde(rename = "any")]
    Any,
    #[serde(rename = "tool")]
    Tool { name: String },
}

#[derive(Serialize, Deserialize)]
struct AnthropicMessage {
    role: String,
    content: AnthropicContent,
}

#[derive(Serialize, Deserialize)]
#[serde(untagged)]
enum AnthropicContent {
    Text(String),
    Blocks(Vec<ContentBlock>),
}

#[derive(Serialize, Deserialize)]
#[serde(tag = "type")]
enum ContentBlock {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
    #[serde(rename = "tool_result")]
    ToolResult {
        tool_use_id: String,
        content: String,
    },
}

#[derive(Deserialize)]
struct AnthropicResponse {
    content: Vec<ContentBlock>,
    usage: AnthropicUsage,
    stop_reason: Option<String>,
}

#[derive(Deserialize)]
struct AnthropicUsage {
    input_tokens: usize,
    output_tokens: usize,
    #[serde(default)]
    cache_read_input_tokens: Option<usize>,
    #[serde(default)]
    cache_creation_input_tokens: Option<usize>,
}

impl From<&AnthropicUsage> for PromptCacheUsage {
    fn from(usage: &AnthropicUsage) -> Self {
        Self {
            cached_tokens: None,
            cache_read_input_tokens: usage.cache_read_input_tokens,
            cache_creation_input_tokens: usage.cache_creation_input_tokens,
        }
    }
}

impl AnthropicUsage {
    fn prompt_cache_usage(&self) -> Option<PromptCacheUsage> {
        let usage = PromptCacheUsage::from(self);
        if usage.cache_read_input_tokens.is_some() || usage.cache_creation_input_tokens.is_some() {
            Some(usage)
        } else {
            None
        }
    }
}

impl AnthropicResponse {
    fn into_completion_response(self) -> CompletionResponse {
        let mut content = String::new();
        let mut tool_calls = Vec::new();

        for block in self.content {
            match block {
                ContentBlock::Text { text } => {
                    if !content.is_empty() {
                        content.push('\n');
                    }
                    content.push_str(&text);
                }
                ContentBlock::ToolUse { id, name, input } => {
                    tool_calls.push(ToolCall {
                        id,
                        name,
                        arguments: input,
                    });
                }
                ContentBlock::ToolResult { .. } => {}
            }
        }

        CompletionResponse {
            content,
            input_tokens: self.usage.input_tokens,
            output_tokens: self.usage.output_tokens,
            stop_reason: self.stop_reason,
            tool_calls,
            prompt_cache: self.usage.prompt_cache_usage(),
        }
    }
}

#[async_trait]
impl LlmClient for AnthropicClient {
    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse> {
        let model = super::resolve_model(request.model(), &self.default_model);

        let mut messages: Vec<AnthropicMessage> = Vec::new();

        for m in request.messages() {
            let mut blocks = Vec::new();
            if !m.content().is_empty() {
                blocks.push(ContentBlock::Text {
                    text: m.content().to_string(),
                });
            }
            for tc in m.tool_calls() {
                blocks.push(ContentBlock::ToolUse {
                    id: tc.id.clone(),
                    name: tc.name.clone(),
                    input: tc.arguments.clone(),
                });
            }
            for tr in m.tool_results_ref() {
                blocks.push(ContentBlock::ToolResult {
                    tool_use_id: tr.tool_call_id.clone(),
                    content: tr.content.clone(),
                });
            }
            messages.push(AnthropicMessage {
                role: m.role().as_str().to_string(),
                content: AnthropicContent::Blocks(blocks),
            });
        }

        let tools = request.tools().map(|defs| {
            defs.iter()
                .map(|t| AnthropicTool {
                    name: t.name().to_string(),
                    description: t.description().to_string(),
                    input_schema: t.input_schema().clone(),
                })
                .collect()
        });

        let tool_choice = match request.tool_choice().cloned() {
            Some(ToolChoice::Auto) => Some(AnthropicToolChoice::Auto),
            Some(ToolChoice::Required) => Some(AnthropicToolChoice::Any),
            Some(ToolChoice::Specific(name)) => Some(AnthropicToolChoice::Tool { name }),
            Some(ToolChoice::None) => {
                return Err(Error::Configuration(
                    "Anthropic does not support ToolChoice::None".into(),
                ));
            }
            None => None,
        };

        let body = AnthropicRequest {
            model,
            messages,
            max_tokens: request.max_tokens().unwrap_or(4096),
            temperature: request.temperature(),
            system: request.system().map(str::to_string),
            tools,
            tool_choice,
            cache_control: self.prompt_cache.as_ref().map(AnthropicCacheControl::from),
        };

        let mut req = self.client.post(ANTHROPIC_API_URL);

        req = match &self.auth {
            AnthropicAuth::ApiKey(key) => req.header("x-api-key", key),
            AnthropicAuth::OAuth(token) => req
                .header("Authorization", format!("Bearer {token}"))
                .header("anthropic-beta", OAUTH_BETA),
        };

        let resp_text = super::send_and_check(
            "Anthropic",
            req.header("anthropic-version", ANTHROPIC_VERSION)
                .header("content-type", "application/json")
                .json(&body),
        )
        .await?;

        let parsed: AnthropicResponse = serde_json::from_str(&resp_text).map_err(|e| {
            crate::error::Error::Llm(format!("failed to parse Anthropic response: {e}"))
        })?;

        Ok(parsed.into_completion_response())
    }
}

impl From<&AnthropicPromptCacheConfig> for AnthropicCacheControl {
    fn from(config: &AnthropicPromptCacheConfig) -> Self {
        Self {
            cache_type: "ephemeral",
            ttl: config.ttl(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::Message;
    use serde_json::json;

    #[test]
    fn serializes_cache_control_on_request() {
        let body = AnthropicRequest {
            model: "claude-sonnet".into(),
            messages: vec![AnthropicMessage {
                role: "user".into(),
                content: AnthropicContent::Blocks(vec![ContentBlock::Text {
                    text: "hello".into(),
                }]),
            }],
            max_tokens: 128,
            temperature: Some(0.0),
            system: None,
            tools: None,
            tool_choice: None,
            cache_control: Some(AnthropicCacheControl::from(
                &AnthropicPromptCacheConfig::new().with_ttl(AnthropicPromptCacheTtl::Hours1),
            )),
        };

        let value = serde_json::to_value(&body).unwrap();
        assert_eq!(value["cache_control"]["type"], "ephemeral");
        assert_eq!(value["cache_control"]["ttl"], "1h");
    }

    #[test]
    fn parses_response_with_cache_usage_and_tool_calls() {
        let parsed: AnthropicResponse = serde_json::from_value(json!({
            "content": [
                { "type": "text", "text": "done" },
                {
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "lookup",
                    "input": { "q": "hi" }
                }
            ],
            "usage": {
                "input_tokens": 1500,
                "output_tokens": 55,
                "cache_read_input_tokens": 1200,
                "cache_creation_input_tokens": 300
            },
            "stop_reason": "tool_use"
        }))
        .unwrap();

        let response = parsed.into_completion_response();
        assert_eq!(response.content, "done");
        assert_eq!(response.input_tokens, 1500);
        assert_eq!(response.output_tokens, 55);
        assert_eq!(response.stop_reason.as_deref(), Some("tool_use"));
        assert_eq!(response.tool_calls.len(), 1);
        assert_eq!(response.tool_calls[0].id, "toolu_1");
        assert_eq!(
            response.prompt_cache,
            Some(PromptCacheUsage {
                cached_tokens: None,
                cache_read_input_tokens: Some(1200),
                cache_creation_input_tokens: Some(300),
            })
        );
    }

    #[test]
    fn omits_prompt_cache_when_usage_has_no_cache_fields() {
        let usage = AnthropicUsage {
            input_tokens: 10,
            output_tokens: 5,
            cache_read_input_tokens: None,
            cache_creation_input_tokens: None,
        };

        assert_eq!(usage.prompt_cache_usage(), None);
    }

    #[tokio::test]
    #[ignore = "requires LLM_API_KEY"]
    async fn integration_simple_prompt() {
        let _ = dotenvy::dotenv();
        let api_key = std::env::var("LLM_API_KEY").expect("LLM_API_KEY must be set");
        let client = AnthropicClient::new(
            AnthropicConfig::new(
                api_key,
                std::env::var("LLM_MODEL")
                    .or_else(|_| std::env::var("RETAIN_LLM_MODEL"))
                    .expect("LLM_MODEL or RETAIN_LLM_MODEL must be set"),
            )
            .unwrap()
            .with_timeout_secs(crate::llm::DEFAULT_TIMEOUT_SECS)
            .unwrap(),
        )
        .unwrap();

        let request = CompletionRequest::builder()
            .message(Message::user("Say hello in exactly 3 words."))
            .max_tokens(64)
            .temperature(0.0)
            .build();

        let resp = client.complete(request).await.unwrap();
        assert!(!resp.content.is_empty());
        assert!(resp.input_tokens > 0);
        assert!(resp.output_tokens > 0);
    }

    #[tokio::test]
    #[ignore = "requires LLM_API_KEY"]
    async fn integration_structured() {
        let _ = dotenvy::dotenv();
        use crate::llm::complete_structured;
        use serde::Deserialize;

        let api_key = std::env::var("LLM_API_KEY").expect("LLM_API_KEY must be set");
        let client = AnthropicClient::new(
            AnthropicConfig::new(
                api_key,
                std::env::var("LLM_MODEL")
                    .or_else(|_| std::env::var("RETAIN_LLM_MODEL"))
                    .expect("LLM_MODEL or RETAIN_LLM_MODEL must be set"),
            )
            .unwrap()
            .with_timeout_secs(crate::llm::DEFAULT_TIMEOUT_SECS)
            .unwrap(),
        )
        .unwrap();

        #[derive(Deserialize, Debug)]
        struct Color {
            name: String,
            hex: String,
        }

        let request = CompletionRequest::builder()
            .message(Message::user(
                "Return a JSON object with fields \"name\" and \"hex\" for the color red. Only output JSON, nothing else.",
            ))
            .max_tokens(64)
            .temperature(0.0)
            .build();

        let color: Color = complete_structured(&client, request).await.unwrap();
        assert!(!color.name.is_empty());
        assert!(color.hex.starts_with('#'));
    }

    #[tokio::test]
    #[ignore = "requires LLM_API_KEY"]
    async fn integration_oauth() {
        let _ = dotenvy::dotenv();
        let api_key = std::env::var("LLM_API_KEY").expect("LLM_API_KEY must be set");
        assert!(
            api_key.starts_with("sk-ant-oat"),
            "this test requires an OAuth token (sk-ant-oat...)"
        );
        let client = AnthropicClient::new(
            AnthropicConfig::new(api_key, "claude-sonnet-4-20250514")
                .unwrap()
                .with_timeout_secs(crate::llm::DEFAULT_TIMEOUT_SECS)
                .unwrap(),
        )
        .unwrap();

        let request = CompletionRequest::builder()
            .message(Message::user("Say hello in exactly 3 words."))
            .max_tokens(64)
            .temperature(0.0)
            .build();

        let resp = client.complete(request).await.unwrap();
        println!("OAuth response: {}", resp.content);
        assert!(!resp.content.is_empty());
        assert!(resp.input_tokens > 0);
    }
}
