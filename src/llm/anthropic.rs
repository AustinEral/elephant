//! Anthropic Claude Messages API client.

use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::llm::{LlmClient, PromptCachingConfig};
use crate::types::llm::{
    CacheStatus, CompletionRequest, CompletionResponse, CompletionUsage, ToolCall, ToolChoice,
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
    prompt_caching: PromptCachingConfig,
}

impl AnthropicClient {
    /// Create a new Anthropic client. Auto-detects OAuth from token prefix.
    pub fn new(api_key: String, model: String) -> Result<Self> {
        Self::new_with_prompt_caching(api_key, model, PromptCachingConfig::default())
    }

    /// Create a new Anthropic client with explicit prompt-caching settings.
    pub fn new_with_prompt_caching(
        api_key: String,
        model: String,
        prompt_caching: PromptCachingConfig,
    ) -> Result<Self> {
        let auth = AnthropicAuth::from_token(api_key);
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
            auth,
            default_model: model,
            prompt_caching,
        })
    }
}

// --- Anthropic API request/response types ---

#[derive(Serialize)]
struct AnthropicCacheControl {
    #[serde(rename = "type")]
    kind: &'static str,
}

#[derive(Serialize)]
struct AnthropicRequest {
    model: String,
    messages: Vec<AnthropicMessage>,
    max_tokens: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    cache_control: Option<AnthropicCacheControl>,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<AnthropicTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<AnthropicToolChoice>,
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

fn normalize_anthropic_usage(usage: &AnthropicUsage) -> CompletionUsage {
    let cache_hit_prompt_tokens = usage.cache_read_input_tokens.unwrap_or(0);
    let cache_write_prompt_tokens = usage.cache_creation_input_tokens.unwrap_or(0);
    let uncached_prompt_tokens = usage.input_tokens;
    let prompt_tokens =
        uncached_prompt_tokens + cache_hit_prompt_tokens + cache_write_prompt_tokens;

    let cache_status = if usage.cache_read_input_tokens.is_none()
        && usage.cache_creation_input_tokens.is_none()
    {
        CacheStatus::Unsupported
    } else if cache_hit_prompt_tokens == 0 && cache_write_prompt_tokens == 0 {
        CacheStatus::NoActivity
    } else if cache_hit_prompt_tokens == 0 {
        CacheStatus::WriteOnly
    } else if cache_write_prompt_tokens == 0 {
        CacheStatus::Hit
    } else {
        CacheStatus::HitAndWrite
    };

    CompletionUsage {
        prompt_tokens,
        uncached_prompt_tokens,
        cache_hit_prompt_tokens,
        cache_write_prompt_tokens,
        completion_tokens: usage.output_tokens,
        cache_status,
    }
}

fn anthropic_cache_control(prompt_caching: &PromptCachingConfig) -> Option<AnthropicCacheControl> {
    prompt_caching.enabled.then_some(AnthropicCacheControl { kind: "ephemeral" })
}

fn build_anthropic_request(
    request: CompletionRequest,
    default_model: &str,
    prompt_caching: &PromptCachingConfig,
) -> AnthropicRequest {
    let model = super::resolve_model(request.model, default_model);

    // Build messages, including tool results if present.
    let mut messages: Vec<AnthropicMessage> = Vec::new();

    for m in &request.messages {
        let mut blocks = Vec::new();
        if !m.content.is_empty() {
            blocks.push(ContentBlock::Text {
                text: m.content.clone(),
            });
        }
        for tc in &m.tool_calls {
            blocks.push(ContentBlock::ToolUse {
                id: tc.id.clone(),
                name: tc.name.clone(),
                input: tc.arguments.clone(),
            });
        }
        for tr in &m.tool_results {
            blocks.push(ContentBlock::ToolResult {
                tool_use_id: tr.tool_call_id.clone(),
                content: tr.content.clone(),
            });
        }
        messages.push(AnthropicMessage {
            role: m.role.clone(),
            content: AnthropicContent::Blocks(blocks),
        });
    }

    if !request.tool_results.is_empty() {
        let blocks: Vec<ContentBlock> = request
            .tool_results
            .iter()
            .map(|tr| ContentBlock::ToolResult {
                tool_use_id: tr.tool_call_id.clone(),
                content: tr.content.clone(),
            })
            .collect();
        messages.push(AnthropicMessage {
            role: "user".into(),
            content: AnthropicContent::Blocks(blocks),
        });
    }

    let tools = request.tools.map(|defs| {
        defs.into_iter()
            .map(|t| AnthropicTool {
                name: t.name,
                description: t.description,
                input_schema: t.input_schema,
            })
            .collect()
    });

    let tool_choice = request.tool_choice.map(|tc| match tc {
        ToolChoice::Auto => AnthropicToolChoice::Auto,
        ToolChoice::Required => AnthropicToolChoice::Any,
        ToolChoice::None => AnthropicToolChoice::Auto,
        ToolChoice::Specific(name) => AnthropicToolChoice::Tool { name },
    });

    AnthropicRequest {
        model,
        messages,
        max_tokens: request.max_tokens.unwrap_or(4096),
        temperature: request.temperature,
        cache_control: anthropic_cache_control(prompt_caching),
        system: request.system,
        tools,
        tool_choice,
    }
}

#[doc(hidden)]
pub fn anthropic_request_json(
    request: &CompletionRequest,
    default_model: &str,
    prompt_caching: &PromptCachingConfig,
) -> serde_json::Result<String> {
    serde_json::to_string(&build_anthropic_request(
        request.clone(),
        default_model,
        prompt_caching,
    ))
}

#[async_trait]
impl LlmClient for AnthropicClient {
    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse> {
        let body = build_anthropic_request(request, &self.default_model, &self.prompt_caching);

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

        let mut content = String::new();
        let mut tool_calls = Vec::new();

        for block in parsed.content {
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

        let usage = normalize_anthropic_usage(&parsed.usage);

        Ok(CompletionResponse {
            content,
            input_tokens: usage.prompt_tokens,
            output_tokens: usage.completion_tokens,
            usage,
            stop_reason: parsed.stop_reason,
            tool_calls,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::llm::{CacheStatus, Message};

    fn anthropic_request_json(prompt_caching: PromptCachingConfig) -> String {
        let request = AnthropicRequest {
            model: "claude-sonnet".into(),
            messages: vec![AnthropicMessage {
                role: "user".into(),
                content: AnthropicContent::Blocks(vec![ContentBlock::Text {
                    text: "Cache this prompt".into(),
                }]),
            }],
            max_tokens: 256,
            temperature: Some(0.0),
            cache_control: anthropic_cache_control(&prompt_caching),
            system: Some("You are helpful.".into()),
            tools: None,
            tool_choice: None,
        };

        let json = serde_json::to_string(&request).unwrap();
        if prompt_caching.enabled {
            assert!(
                json.contains("\"cache_control\":{\"type\":\"ephemeral\"}"),
                "enabled prompt caching should serialize cache_control: {json}"
            );
        } else {
            assert!(
                !json.contains("cache_control"),
                "disabled prompt caching should omit cache_control: {json}"
            );
        }
        json
    }

    #[test]
    fn normalize_usage_anthropic_totals() {
        let usage = AnthropicUsage {
            input_tokens: 7,
            output_tokens: 5,
            cache_read_input_tokens: Some(11),
            cache_creation_input_tokens: Some(13),
        };

        let normalized = normalize_anthropic_usage(&usage);
        assert_eq!(normalized.prompt_tokens, 31);
        assert_eq!(normalized.uncached_prompt_tokens, 7);
        assert_eq!(normalized.cache_hit_prompt_tokens, 11);
        assert_eq!(normalized.cache_write_prompt_tokens, 13);
        assert_eq!(normalized.completion_tokens, 5);
        assert_eq!(normalized.cache_status, CacheStatus::HitAndWrite);
    }

    #[test]
    fn normalize_usage_anthropic_unsupported_defaults() {
        let usage = AnthropicUsage {
            input_tokens: 7,
            output_tokens: 5,
            cache_read_input_tokens: None,
            cache_creation_input_tokens: None,
        };

        let normalized = normalize_anthropic_usage(&usage);
        assert_eq!(normalized.prompt_tokens, 7);
        assert_eq!(normalized.uncached_prompt_tokens, 7);
        assert_eq!(normalized.cache_hit_prompt_tokens, 0);
        assert_eq!(normalized.cache_write_prompt_tokens, 0);
        assert_eq!(normalized.completion_tokens, 5);
        assert_eq!(normalized.cache_status, CacheStatus::Unsupported);
    }

    #[test]
    fn anthropic_prompt_caching_request_enabled() {
        let json = anthropic_request_json(PromptCachingConfig { enabled: true });
        assert!(json.contains("\"cache_control\":{\"type\":\"ephemeral\"}"));
    }

    #[test]
    fn anthropic_prompt_caching_request_disabled() {
        let json = anthropic_request_json(PromptCachingConfig::default());
        assert!(!json.contains("cache_control"));
    }

    #[tokio::test]
    #[ignore = "requires LLM_API_KEY"]
    async fn integration_simple_prompt() {
        let _ = dotenvy::dotenv();
        let api_key = std::env::var("LLM_API_KEY").expect("LLM_API_KEY must be set");
        let client = AnthropicClient::new(
            api_key,
            std::env::var("LLM_MODEL")
                .or_else(|_| std::env::var("RETAIN_LLM_MODEL"))
                .expect("LLM_MODEL or RETAIN_LLM_MODEL must be set"),
        )
        .unwrap();

        let request = CompletionRequest {
            model: String::new(),
            messages: vec![Message::text("user", "Say hello in exactly 3 words.")],
            max_tokens: Some(64),
            temperature: Some(0.0),
            system: None,
            ..Default::default()
        };

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
            api_key,
            std::env::var("LLM_MODEL")
                .or_else(|_| std::env::var("RETAIN_LLM_MODEL"))
                .expect("LLM_MODEL or RETAIN_LLM_MODEL must be set"),
        )
        .unwrap();

        #[derive(Deserialize, Debug)]
        struct Color {
            name: String,
            hex: String,
        }

        let request = CompletionRequest {
            model: String::new(),
            messages: vec![Message::text(
                "user",
                "Return a JSON object with fields \"name\" and \"hex\" for the color red. Only output JSON, nothing else.",
            )],
            max_tokens: Some(64),
            temperature: Some(0.0),
            system: None,
            ..Default::default()
        };

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
        let client = AnthropicClient::new(api_key, "claude-sonnet-4-20250514".into()).unwrap();

        let request = CompletionRequest {
            model: String::new(),
            messages: vec![Message::text("user", "Say hello in exactly 3 words.")],
            max_tokens: Some(64),
            temperature: Some(0.0),
            system: None,
            ..Default::default()
        };

        let resp = client.complete(request).await.unwrap();
        println!("OAuth response: {}", resp.content);
        assert!(!resp.content.is_empty());
        assert!(resp.input_tokens > 0);
    }
}
