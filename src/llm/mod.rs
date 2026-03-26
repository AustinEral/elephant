//! LLM client abstraction and provider implementations.

pub mod anthropic;
mod config;
pub mod gemini;
pub mod mock;
pub mod openai;
pub mod retry;
mod types;
pub mod vertex;

use async_trait::async_trait;
use serde::de::DeserializeOwned;

pub use config::{
    AnthropicConfig, AnthropicPromptCacheConfig, AnthropicPromptCacheTtl, ClientConfig,
    DEFAULT_TIMEOUT_SECS, GeminiConfig, LlmConfig, OpenAiConfig, OpenAiPromptCacheConfig,
    OpenAiPromptCacheRetention, Provider, VertexConfig, judge_client_config_from_env,
    runtime_config_from_env,
};
pub use types::{
    CompletionRequest, CompletionRequestBuilder, CompletionResponse, Message, MessageRole,
    PromptCacheUsage, ReasoningEffort, ReasoningEffortConfig, ToolCall, ToolChoice, ToolDefinition,
    ToolResult,
};

use crate::error::{Error, Result};

/// Trait abstraction over LLM providers.
///
/// Every component that calls an LLM takes `dyn LlmClient`, making the system
/// testable with [`mock::MockLlmClient`] and swappable between providers.
#[async_trait]
pub trait LlmClient: Send + Sync {
    /// Send a completion request and get a text response.
    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum StructuredResponseErrorKind {
    Refusal,
    Empty,
    NoJson,
    JsonParse,
    JsonStructure,
}

#[derive(Debug)]
pub(crate) struct StructuredResponseError {
    pub error: Error,
    pub kind: StructuredResponseErrorKind,
    pub raw_response: String,
    pub extracted_json: Option<String>,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct StructuredOutputRetryOptions {
    pub max_attempts: usize,
    pub context: &'static str,
}

const STRUCTURED_OUTPUT_LOG_PREVIEW_CHARS: usize = 1_000;

fn preview_for_log(text: &str, max_chars: usize) -> String {
    let total_chars = text.chars().count();
    if total_chars <= max_chars {
        return text.to_string();
    }

    let head_chars = max_chars / 2;
    let tail_chars = max_chars - head_chars;
    let head = text.chars().take(head_chars).collect::<String>();
    let tail = text
        .chars()
        .skip(total_chars.saturating_sub(tail_chars))
        .collect::<String>();
    let omitted = total_chars.saturating_sub(head_chars + tail_chars);

    format!("{head}...<{omitted} chars omitted>...{tail}")
}

fn parse_structured_response<T: DeserializeOwned>(
    response: &CompletionResponse,
) -> std::result::Result<T, StructuredResponseError> {
    if response.stop_reason.as_deref() == Some("refusal") {
        return Err(StructuredResponseError {
            error: Error::LlmRefusal,
            kind: StructuredResponseErrorKind::Refusal,
            raw_response: response.content.clone(),
            extracted_json: None,
        });
    }
    if response.content.is_empty() {
        return Err(StructuredResponseError {
            error: Error::Llm("model returned empty response".into()),
            kind: StructuredResponseErrorKind::Empty,
            raw_response: response.content.clone(),
            extracted_json: None,
        });
    }

    let (value, extracted_json) = match serde_json::from_str::<serde_json::Value>(&response.content)
    {
        Ok(value) => (value, None),
        Err(_) => {
            let json_str =
                extract_json(&response.content).map_err(|error| StructuredResponseError {
                    kind: match error {
                        Error::LlmNoJson => StructuredResponseErrorKind::NoJson,
                        Error::Llm(_) => StructuredResponseErrorKind::JsonParse,
                        _ => StructuredResponseErrorKind::JsonParse,
                    },
                    error,
                    raw_response: response.content.clone(),
                    extracted_json: None,
                })?;
            let value = serde_json::from_str::<serde_json::Value>(&json_str).map_err(|error| {
                StructuredResponseError {
                    error: Error::Llm(format!("JSON parse error: {error}")),
                    kind: StructuredResponseErrorKind::JsonParse,
                    raw_response: response.content.clone(),
                    extracted_json: Some(json_str.clone()),
                }
            })?;
            (value, Some(json_str))
        }
    };

    serde_json::from_value::<T>(value).map_err(|error| StructuredResponseError {
        error: Error::Llm(format!("JSON structure error: {error}")),
        kind: StructuredResponseErrorKind::JsonStructure,
        raw_response: response.content.clone(),
        extracted_json,
    })
}

/// Send a completion request and parse the response as structured JSON.
///
/// Calls [`LlmClient::complete`], extracts JSON from the response text
/// (handling markdown fences and surrounding prose), then deserializes with serde.
///
/// Parses through `serde_json::Value` first so that duplicate keys (which LLMs
/// occasionally produce) are silently resolved via last-wins semantics instead of
/// causing a hard parse error.
pub async fn complete_structured<T: DeserializeOwned>(
    client: &dyn LlmClient,
    request: CompletionRequest,
) -> Result<T> {
    let response = client.complete(request).await?;
    parse_structured_response(&response).map_err(|error| error.error)
}

pub(crate) async fn complete_structured_with_retries<T, F>(
    client: &dyn LlmClient,
    request: CompletionRequest,
    options: StructuredOutputRetryOptions,
    should_retry: F,
) -> Result<T>
where
    T: DeserializeOwned,
    F: Fn(StructuredResponseErrorKind) -> bool,
{
    assert!(
        options.max_attempts > 0,
        "max_attempts must be greater than zero"
    );

    for attempt in 0..options.max_attempts {
        let response = client.complete(request.clone()).await?;
        match parse_structured_response(&response) {
            Ok(parsed) => return Ok(parsed),
            Err(error) => {
                let will_retry = attempt + 1 < options.max_attempts && should_retry(error.kind);
                let extracted_json = error
                    .extracted_json
                    .as_deref()
                    .map(|json| preview_for_log(json, STRUCTURED_OUTPUT_LOG_PREVIEW_CHARS))
                    .unwrap_or_else(|| "<none>".into());
                tracing::warn!(
                    context = options.context,
                    attempt = attempt + 1,
                    max_attempts = options.max_attempts,
                    will_retry,
                    error = %error.error,
                    raw_response = %preview_for_log(
                        &error.raw_response,
                        STRUCTURED_OUTPUT_LOG_PREVIEW_CHARS
                    ),
                    extracted_json = %extracted_json,
                    "structured output invalid"
                );
                if will_retry {
                    continue;
                }
                return Err(error.error);
            }
        }
    }

    unreachable!("structured output retry loop must return")
}

/// Extract a JSON value from text that may contain markdown fences or surrounding prose.
///
/// Strategy:
/// 1. Strip markdown fences (```json ... ```)
/// 2. Find first `{` or `[` and match to last `}` or `]`
/// 3. Return that substring
pub fn extract_json(text: &str) -> Result<String> {
    let stripped = if let Some(start) = text.find("```") {
        let after_fence = &text[start + 3..];
        let content_start = after_fence.find('\n').map(|i| i + 1).unwrap_or(0);
        let content = &after_fence[content_start..];
        if let Some(end) = content.find("```") {
            content[..end].trim()
        } else {
            content.trim()
        }
    } else {
        text.trim()
    };

    if serde_json::from_str::<serde_json::Value>(stripped).is_ok() {
        return Ok(stripped.to_string());
    }

    let obj_start = stripped.find('{');
    let arr_start = stripped.find('[');

    let start = match (obj_start, arr_start) {
        (Some(o), Some(a)) => o.min(a),
        (Some(o), None) => o,
        (None, Some(a)) => a,
        (None, None) => return Err(Error::LlmNoJson),
    };

    let (open, close) = if stripped.as_bytes()[start] == b'{' {
        (b'{', b'}')
    } else {
        (b'[', b']')
    };

    let mut depth = 0i32;
    let mut in_string = false;
    let mut escape_next = false;
    let mut end_pos = None;

    for (i, &b) in stripped.as_bytes()[start..].iter().enumerate() {
        if escape_next {
            escape_next = false;
            continue;
        }
        if b == b'\\' && in_string {
            escape_next = true;
            continue;
        }
        if b == b'"' {
            in_string = !in_string;
            continue;
        }
        if in_string {
            continue;
        }
        if b == open {
            depth += 1;
        } else if b == close {
            depth -= 1;
            if depth == 0 {
                end_pos = Some(start + i + 1);
                break;
            }
        }
    }

    let end = end_pos.ok_or_else(|| Error::Llm("unbalanced JSON in response".into()))?;
    let candidate = &stripped[start..end];

    serde_json::from_str::<serde_json::Value>(candidate)
        .map_err(|e| Error::Llm(format!("extracted JSON is invalid: {e}")))?;

    Ok(candidate.to_string())
}

/// Map an HTTP error response to the appropriate error variant.
pub(crate) fn classify_api_error(
    provider: &str,
    status: reqwest::StatusCode,
    message: String,
) -> Error {
    if status.as_u16() == 429 {
        Error::RateLimit(format!("{provider} API ({status}): {message}"))
    } else if status.is_server_error() {
        Error::ServerError(format!("{provider} API ({status}): {message}"))
    } else {
        Error::Llm(format!("{provider} API error ({status}): {message}"))
    }
}

/// Shared API error response shape: `{"error": {"message": "..."}}`.
#[derive(serde::Deserialize)]
pub(crate) struct ApiErrorBody {
    pub error: ApiErrorDetail,
}

#[derive(serde::Deserialize)]
pub(crate) struct ApiErrorDetail {
    pub message: String,
}

/// Resolve the model: use the request's model if set, otherwise the client default.
pub(crate) fn resolve_model(request_model: &str, default_model: &str) -> String {
    if request_model.is_empty() {
        default_model.to_string()
    } else {
        request_model.to_string()
    }
}

/// Send a request and handle the response/error boilerplate shared by all providers.
pub(crate) async fn send_and_check(
    provider: &str,
    request_builder: reqwest::RequestBuilder,
) -> Result<String> {
    let resp = request_builder.send().await.map_err(|e| {
        if e.is_timeout() {
            Error::ServerError(format!("{provider} request timed out"))
        } else {
            Error::Llm(format!("{provider} request failed: {e}"))
        }
    })?;

    let status = resp.status();
    let resp_text = resp
        .text()
        .await
        .map_err(|e| Error::Llm(format!("failed to read {provider} response: {e}")))?;

    if status.is_success() {
        return Ok(resp_text);
    }

    let msg = serde_json::from_str::<ApiErrorBody>(&resp_text)
        .map(|e| e.error.message)
        .unwrap_or(resp_text);

    Err(classify_api_error(provider, status, msg))
}

/// Build an LLM client from a validated configuration.
pub fn build_client(config: &ClientConfig) -> Result<Box<dyn LlmClient>> {
    match config {
        ClientConfig::Anthropic(config) => {
            Ok(Box::new(anthropic::AnthropicClient::new(config.clone())?))
        }
        ClientConfig::OpenAi(config) => Ok(Box::new(openai::OpenAiClient::new(config.clone())?)),
        ClientConfig::Gemini(config) => Ok(Box::new(gemini::GeminiClient::new(config.clone())?)),
        ClientConfig::Vertex(config) => Ok(Box::new(vertex::VertexClient::new(config.clone())?)),
    }
}

#[cfg(test)]
mod tests {
    use super::{extract_json, preview_for_log};

    #[test]
    fn extract_json_clean() {
        let input = r#"{"name": "alice", "age": 30}"#;
        let result = extract_json(input).unwrap();
        assert_eq!(result, input);
    }

    #[test]
    fn extract_json_markdown_fences() {
        let input = "Here is the result:\n```json\n{\"name\": \"bob\"}\n```\nDone.";
        let result = extract_json(input).unwrap();
        assert_eq!(result, r#"{"name": "bob"}"#);
    }

    #[test]
    fn extract_json_surrounding_prose() {
        let input = "Sure! Here is the JSON:\n{\"key\": \"value\"}\nHope that helps!";
        let result = extract_json(input).unwrap();
        assert_eq!(result, r#"{"key": "value"}"#);
    }

    #[test]
    fn extract_json_array() {
        let input = "Results: [{\"a\": 1}, {\"a\": 2}] end";
        let result = extract_json(input).unwrap();
        assert_eq!(result, r#"[{"a": 1}, {"a": 2}]"#);
    }

    #[test]
    fn extract_json_nested() {
        let input = r#"{"outer": {"inner": [1, 2, 3]}}"#;
        let result = extract_json(input).unwrap();
        assert_eq!(result, input);
    }

    #[test]
    fn extract_json_no_json() {
        let input = "This is just plain text with no JSON.";
        let result = extract_json(input);
        assert!(result.is_err());
    }

    #[test]
    fn extract_json_strings_with_braces() {
        let input = r#"{"msg": "hello {world}"}"#;
        let result = extract_json(input).unwrap();
        assert_eq!(result, input);
    }

    #[test]
    fn preview_for_log_returns_full_text_when_short() {
        let input = "short text";
        let result = preview_for_log(input, 20);
        assert_eq!(result, input);
    }

    #[test]
    fn preview_for_log_keeps_head_and_tail_when_long() {
        let input = "abcdefghij";
        let result = preview_for_log(input, 6);
        assert_eq!(result, "abc...<4 chars omitted>...hij");
    }
}
