//! OpenAI Chat Completions API client.

use async_trait::async_trait;
use reqwest::Client;
use serde::ser::SerializeMap;
use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::llm::LlmClient;
use crate::types::llm::{
    CompletionRequest, CompletionResponse, OpenAiPromptCacheConfig,
    OpenAiPromptCacheRetention, PromptCacheConfig, PromptCacheUsage, ToolCall, ToolChoice,
};

const API_URL: &str = "https://api.openai.com/v1";

/// Client for the OpenAI Chat Completions API.
pub struct OpenAiClient {
    client: Client,
    api_key: String,
    default_model: String,
    base_url: String,
    prompt_cache: Option<OpenAiPromptCacheConfig>,
}

impl OpenAiClient {
    /// Create a new OpenAI client with optional base URL for compatible providers.
    pub fn new(
        api_key: String,
        model: String,
        base_url: Option<String>,
        timeout_secs: u64,
        prompt_cache: PromptCacheConfig,
    ) -> Result<Self> {
        let client = Client::builder()
            .timeout(std::time::Duration::from_secs(timeout_secs))
            .build()
            .map_err(|e| crate::error::Error::Internal(e.to_string()))?;
        Ok(Self {
            client,
            api_key,
            default_model: model,
            base_url: base_url.unwrap_or_else(|| API_URL.to_string()),
            prompt_cache: match prompt_cache {
                PromptCacheConfig::OpenAi(config) => Some(config),
                _ => None,
            },
        })
    }
}

// --- OpenAI API request/response types ---

struct OpenAiRequest {
    model: String,
    messages: Vec<OpenAiMessage>,
    max_tokens: Option<usize>,
    temperature: Option<f32>,
    tools: Option<Vec<OpenAiTool>>,
    tool_choice: Option<OpenAiToolChoice>,
    prompt_cache_key: Option<String>,
    prompt_cache_retention: Option<OpenAiPromptCacheRetention>,
}

impl OpenAiRequest {
    fn uses_max_completion_tokens(model: &str) -> bool {
        model.starts_with("gpt-5")
            || model.starts_with("o1")
            || model.starts_with("o3")
            || model.starts_with("o4")
    }

    fn temperature_for_model(model: &str, temperature: Option<f32>) -> Option<f32> {
        if model.starts_with("gpt-5")
            && !model.starts_with("gpt-5.1")
            && !model.starts_with("gpt-5.2")
        {
            None
        } else {
            temperature
        }
    }
}

impl Serialize for OpenAiRequest {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let uses_max_completion_tokens = Self::uses_max_completion_tokens(&self.model);
        let mut map = serializer.serialize_map(None)?;
        map.serialize_entry("model", &self.model)?;
        map.serialize_entry("messages", &self.messages)?;
        if uses_max_completion_tokens {
            if let Some(max_tokens) = self.max_tokens {
                map.serialize_entry("max_completion_tokens", &max_tokens)?;
            }
        } else if let Some(max_tokens) = self.max_tokens {
            map.serialize_entry("max_tokens", &max_tokens)?;
        }
        if let Some(temperature) = Self::temperature_for_model(&self.model, self.temperature) {
            map.serialize_entry("temperature", &temperature)?;
        }
        if let Some(tools) = &self.tools {
            map.serialize_entry("tools", tools)?;
        }
        if let Some(tool_choice) = &self.tool_choice {
            map.serialize_entry("tool_choice", tool_choice)?;
        }
        if let Some(prompt_cache_key) = &self.prompt_cache_key {
            map.serialize_entry("prompt_cache_key", prompt_cache_key)?;
        }
        if let Some(prompt_cache_retention) = self.prompt_cache_retention {
            map.serialize_entry("prompt_cache_retention", &prompt_cache_retention)?;
        }
        map.end()
    }
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
    finish_reason: Option<String>,
}

#[derive(Deserialize)]
struct OpenAiMessageResp {
    content: Option<OpenAiMessageContent>,
    #[serde(default)]
    refusal: Option<String>,
    #[serde(default)]
    tool_calls: Vec<OpenAiRespToolCall>,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum OpenAiMessageContent {
    Text(String),
    Parts(Vec<OpenAiContentPart>),
}

#[derive(Deserialize)]
struct OpenAiContentPart {
    #[serde(rename = "type")]
    part_type: String,
    #[serde(default)]
    text: Option<String>,
    #[serde(default)]
    refusal: Option<String>,
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
    #[serde(default)]
    prompt_tokens_details: Option<OpenAiPromptTokensDetails>,
}

#[derive(Deserialize)]
struct OpenAiPromptTokensDetails {
    #[serde(default)]
    cached_tokens: usize,
}

impl From<OpenAiPromptTokensDetails> for PromptCacheUsage {
    fn from(details: OpenAiPromptTokensDetails) -> Self {
        Self {
            cached_tokens: Some(details.cached_tokens),
            cache_read_input_tokens: None,
            cache_creation_input_tokens: None,
        }
    }
}

impl OpenAiResponse {
    fn into_completion_response(self) -> CompletionResponse {
        let choice = self.choices.into_iter().next();
        let content = choice
            .as_ref()
            .map(|c| c.message.content_text())
            .filter(|content| !content.is_empty())
            .or_else(|| choice.as_ref().and_then(|c| c.message.refusal_text()))
            .unwrap_or_default();
        let stop_reason = choice.as_ref().and_then(|c| {
            if c.message.refusal_text().is_some() {
                Some("refusal".into())
            } else {
                c.finish_reason.clone()
            }
        });

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

        let (input_tokens, output_tokens, prompt_cache) = self
            .usage
            .map(|u| {
                let prompt_cache = u.prompt_tokens_details.map(PromptCacheUsage::from);
                (u.prompt_tokens, u.completion_tokens, prompt_cache)
            })
            .unwrap_or((0, 0, None));

        CompletionResponse {
            content,
            input_tokens,
            output_tokens,
            stop_reason,
            tool_calls,
            prompt_cache,
        }
    }
}

impl OpenAiMessageResp {
    fn content_text(&self) -> String {
        match &self.content {
            Some(OpenAiMessageContent::Text(text)) => text.clone(),
            Some(OpenAiMessageContent::Parts(parts)) => parts
                .iter()
                .filter_map(|part| (part.part_type == "text").then_some(part.text.as_deref()))
                .flatten()
                .collect::<Vec<_>>()
                .join(""),
            None => String::new(),
        }
    }

    fn refusal_text(&self) -> Option<String> {
        self.refusal.clone().or_else(|| match &self.content {
            Some(OpenAiMessageContent::Parts(parts)) => {
                let refusal = parts
                    .iter()
                    .filter_map(|part| {
                        (part.part_type == "refusal").then_some(part.refusal.as_deref())
                    })
                    .flatten()
                    .collect::<Vec<_>>()
                    .join("");
                (!refusal.is_empty()).then_some(refusal)
            }
            _ => None,
        })
    }
}

#[async_trait]
impl LlmClient for OpenAiClient {
    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse> {
        let model = super::resolve_model(request.model, &self.default_model);

        // Build messages: system prompt goes as a system message in the array
        let mut messages: Vec<OpenAiMessage> = Vec::new();
        if let Some(system) = request.system {
            messages.push(OpenAiMessage {
                role: "system".into(),
                content: Some(system),
                ..Default::default()
            });
        }
        for m in &request.messages {
            let tool_calls = if m.tool_calls.is_empty() {
                None
            } else {
                Some(
                    m.tool_calls
                        .iter()
                        .map(|tc| OpenAiReqToolCall {
                            id: tc.id.clone(),
                            tool_type: "function".into(),
                            function: OpenAiReqFunctionCall {
                                name: tc.name.clone(),
                                arguments: tc.arguments.to_string(),
                            },
                        })
                        .collect(),
                )
            };
            let should_emit_message =
                !m.content.is_empty() || tool_calls.is_some() || m.tool_results.is_empty();
            if should_emit_message {
                messages.push(OpenAiMessage {
                    role: m.role.clone(),
                    content: if m.content.is_empty() {
                        None
                    } else {
                        Some(m.content.clone())
                    },
                    tool_calls,
                    ..Default::default()
                });
            }
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
            model: model.clone(),
            messages,
            max_tokens: request.max_tokens,
            temperature: request.temperature,
            tools,
            tool_choice,
            prompt_cache_key: self
                .prompt_cache
                .as_ref()
                .and_then(|config| config.key.clone()),
            prompt_cache_retention: self
                .prompt_cache
                .as_ref()
                .and_then(|config| config.retention),
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

        let parsed: OpenAiResponse = serde_json::from_str(&resp_text).map_err(|e| {
            crate::error::Error::Llm(format!("failed to parse OpenAI response: {e}"))
        })?;

        Ok(parsed.into_completion_response())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::llm::Message;
    use serde_json::json;

    #[test]
    fn serializes_prompt_cache_fields_on_request() {
        let body = OpenAiRequest {
            model: "gpt-5".into(),
            messages: vec![OpenAiMessage {
                role: "user".into(),
                content: Some("hello".into()),
                ..Default::default()
            }],
            max_tokens: Some(128),
            temperature: Some(0.0),
            tools: None,
            tool_choice: None,
            prompt_cache_key: Some("elephant:reflect".into()),
            prompt_cache_retention: Some(OpenAiPromptCacheRetention::Hours24),
        };

        let value = serde_json::to_value(&body).unwrap();
        assert_eq!(value["prompt_cache_key"], "elephant:reflect");
        assert_eq!(value["prompt_cache_retention"], "24h");
    }

    #[test]
    fn uses_max_completion_tokens_for_gpt5_models() {
        let body = OpenAiRequest {
            model: "gpt-5".into(),
            messages: vec![OpenAiMessage {
                role: "user".into(),
                content: Some("hello".into()),
                ..Default::default()
            }],
            max_tokens: Some(128),
            temperature: Some(0.0),
            tools: None,
            tool_choice: None,
            prompt_cache_key: None,
            prompt_cache_retention: None,
        };

        let value = serde_json::to_value(&body).unwrap();
        assert!(OpenAiRequest::uses_max_completion_tokens("gpt-5"));
        assert!(OpenAiRequest::uses_max_completion_tokens("o3"));
        assert!(!OpenAiRequest::uses_max_completion_tokens("gpt-4o"));
        assert_eq!(value["max_completion_tokens"], 128);
        assert!(value.get("max_tokens").is_none());
    }

    #[test]
    fn uses_max_tokens_for_legacy_chat_models() {
        let body = OpenAiRequest {
            model: "gpt-4o".into(),
            messages: vec![OpenAiMessage {
                role: "user".into(),
                content: Some("hello".into()),
                ..Default::default()
            }],
            max_tokens: Some(128),
            temperature: Some(0.0),
            tools: None,
            tool_choice: None,
            prompt_cache_key: None,
            prompt_cache_retention: None,
        };

        let value = serde_json::to_value(&body).unwrap();
        assert!(!OpenAiRequest::uses_max_completion_tokens("gpt-4o"));
        assert_eq!(value["max_tokens"], 128);
        assert!(value.get("max_completion_tokens").is_none());
    }

    #[test]
    fn routes_token_limits_by_model_family() {
        assert!(!OpenAiRequest::uses_max_completion_tokens("gpt-4o"));
        assert!(OpenAiRequest::uses_max_completion_tokens("gpt-5"));
        assert!(OpenAiRequest::uses_max_completion_tokens("o3"));
    }

    #[test]
    fn omits_temperature_for_older_gpt5_chat_models() {
        assert_eq!(
            OpenAiRequest::temperature_for_model("gpt-5", Some(0.1)),
            None
        );
        assert_eq!(
            OpenAiRequest::temperature_for_model("gpt-5-mini", Some(0.0)),
            None
        );
        assert_eq!(
            OpenAiRequest::temperature_for_model("gpt-5-chat-latest", Some(1.0)),
            None
        );
    }

    #[test]
    fn preserves_temperature_for_supported_chat_models() {
        assert_eq!(
            OpenAiRequest::temperature_for_model("gpt-4o", Some(0.3)),
            Some(0.3)
        );
        assert_eq!(
            OpenAiRequest::temperature_for_model("gpt-5.1", Some(0.3)),
            Some(0.3)
        );
        assert_eq!(
            OpenAiRequest::temperature_for_model("gpt-5.2", Some(0.3)),
            Some(0.3)
        );
    }

    #[test]
    fn parses_response_with_cached_tokens_and_tool_calls() {
        let parsed: OpenAiResponse = serde_json::from_value(json!({
            "choices": [{
                "message": {
                    "content": "done",
                    "tool_calls": [{
                        "id": "call_1",
                        "function": {
                            "name": "lookup",
                            "arguments": "{\"q\":\"hi\"}"
                        }
                    }]
                },
                "finish_reason": "tool_calls"
            }],
            "usage": {
                "prompt_tokens": 1200,
                "completion_tokens": 42,
                "prompt_tokens_details": {
                    "cached_tokens": 1024
                }
            }
        }))
        .unwrap();

        let response = parsed.into_completion_response();
        assert_eq!(response.content, "done");
        assert_eq!(response.input_tokens, 1200);
        assert_eq!(response.output_tokens, 42);
        assert_eq!(response.stop_reason.as_deref(), Some("tool_calls"));
        assert_eq!(response.tool_calls.len(), 1);
        assert_eq!(response.tool_calls[0].name, "lookup");
        assert_eq!(response.tool_calls[0].arguments, json!({"q": "hi"}));
        assert_eq!(
            response.prompt_cache,
            Some(PromptCacheUsage {
                cached_tokens: Some(1024),
                cache_read_input_tokens: None,
                cache_creation_input_tokens: None,
            })
        );
    }

    #[test]
    fn parses_refusal_message_as_refusal_stop_reason() {
        let parsed: OpenAiResponse = serde_json::from_value(json!({
            "choices": [{
                "message": {
                    "content": null,
                    "refusal": "I can't help with that."
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 4
            }
        }))
        .unwrap();

        let response = parsed.into_completion_response();
        assert_eq!(response.stop_reason.as_deref(), Some("refusal"));
        assert_eq!(response.content, "I can't help with that.");
    }

    #[test]
    fn parses_text_content_parts() {
        let parsed: OpenAiResponse = serde_json::from_value(json!({
            "choices": [{
                "message": {
                    "content": [
                        {"type": "text", "text": "hello "},
                        {"type": "text", "text": "world"}
                    ]
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 2
            }
        }))
        .unwrap();

        let response = parsed.into_completion_response();
        assert_eq!(response.stop_reason.as_deref(), Some("stop"));
        assert_eq!(response.content, "hello world");
    }

    #[test]
    fn tool_results_are_emitted_as_tool_messages_without_placeholder_user_message() {
        let mut messages = vec![OpenAiMessage {
            role: "assistant".into(),
            content: None,
            tool_call_id: None,
            tool_calls: Some(vec![OpenAiReqToolCall {
                id: "call_123".into(),
                tool_type: "function".into(),
                function: OpenAiReqFunctionCall {
                    name: "recall".into(),
                    arguments: "{\"query\":\"hi\"}".into(),
                },
            }]),
        }];

        let message = Message {
            role: "user".into(),
            content: String::new(),
            tool_calls: vec![],
            tool_results: vec![crate::types::llm::ToolResult {
                tool_call_id: "call_123".into(),
                content: "result".into(),
            }],
        };
        let tool_calls = if message.tool_calls.is_empty() {
            None
        } else {
            unreachable!()
        };
        let should_emit_message =
            !message.content.is_empty() || tool_calls.is_some() || message.tool_results.is_empty();
        if should_emit_message {
            messages.push(OpenAiMessage {
                role: message.role.clone(),
                content: Some(message.content.clone()),
                tool_calls,
                ..Default::default()
            });
        }
        for tr in &message.tool_results {
            messages.push(OpenAiMessage {
                role: "tool".into(),
                content: Some(tr.content.clone()),
                tool_call_id: Some(tr.tool_call_id.clone()),
                ..Default::default()
            });
        }

        assert_eq!(messages.len(), 2);
        assert_eq!(messages[1].role, "tool");
        assert_eq!(messages[1].tool_call_id.as_deref(), Some("call_123"));
        assert_eq!(messages[1].content.as_deref(), Some("result"));
    }

    #[tokio::test]
    #[ignore = "requires OPENAI_API_KEY"]
    async fn integration_simple_prompt() {
        let _ = dotenvy::dotenv();
        let api_key = std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY must be set");
        let client = OpenAiClient::new(
            api_key,
            std::env::var("LLM_MODEL").expect("LLM_MODEL must be set"),
            None,
            crate::llm::DEFAULT_TIMEOUT_SECS,
            PromptCacheConfig::Disabled,
        )
        .unwrap();

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
        let client = OpenAiClient::new(
            api_key,
            std::env::var("LLM_MODEL").expect("LLM_MODEL must be set"),
            None,
            crate::llm::DEFAULT_TIMEOUT_SECS,
            PromptCacheConfig::Disabled,
        )
        .unwrap();

        #[derive(Deserialize, Debug)]
        struct Color {
            name: String,
            hex: String,
        }

        let request = CompletionRequest {
            messages: vec![Message::text(
                "user",
                "Return a JSON object with fields \"name\" and \"hex\" for the color blue. Only output JSON, nothing else.",
            )],
            max_tokens: Some(64),
            temperature: Some(0.0),
            ..Default::default()
        };

        let color: Color = complete_structured(&client, request).await.unwrap();
        assert!(!color.name.is_empty());
        assert!(color.hex.starts_with('#'));
    }
}
