//! LLM client abstraction and provider implementations.

pub mod anthropic;
pub mod mock;
pub mod openai;
pub mod retry;

use async_trait::async_trait;
use serde::de::DeserializeOwned;

/// Default HTTP timeout for LLM API requests (seconds).
pub const DEFAULT_TIMEOUT_SECS: u64 = 600;

use crate::error::{Error, Result};
use crate::types::llm::{CompletionRequest, CompletionResponse};

/// Trait abstraction over LLM providers.
///
/// Every component that calls an LLM takes `dyn LlmClient`, making the system
/// testable with [`mock::MockLlmClient`] and swappable between providers.
#[async_trait]
pub trait LlmClient: Send + Sync {
    /// Send a completion request and get a text response.
    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse>;
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
    // Handle refusals — model declined to respond
    if response.stop_reason.as_deref() == Some("refusal") {
        return Err(Error::LlmRefusal);
    }
    // Handle empty responses
    if response.content.is_empty() {
        return Err(Error::Llm("model returned empty response".into()));
    }
    // Fast path: parse via Value (tolerates duplicate keys)
    let value: serde_json::Value = match serde_json::from_str(&response.content) {
        Ok(v) => v,
        Err(_) => {
            // Slow path: extract JSON from surrounding text
            let json_str = extract_json(&response.content).map_err(|e| {
                tracing::warn!(
                    content_len = response.content.len(),
                    output_tokens = response.output_tokens,
                    stop_reason = ?response.stop_reason,
                    content_preview = &response.content[..response.content.len().min(200)],
                    "failed to extract JSON from LLM response"
                );
                e
            })?;
            serde_json::from_str(&json_str)
                .map_err(|e| Error::Llm(format!("JSON parse error: {e}")))?
        }
    };
    serde_json::from_value::<T>(value).map_err(|e| Error::Llm(format!("JSON structure error: {e}")))
}

/// Extract a JSON value from text that may contain markdown fences or surrounding prose.
///
/// Strategy:
/// 1. Strip markdown fences (```json ... ```)
/// 2. Find first `{` or `[` and match to last `}` or `]`
/// 3. Return that substring
pub fn extract_json(text: &str) -> Result<String> {
    // Strip markdown fences
    let stripped = if let Some(start) = text.find("```") {
        let after_fence = &text[start + 3..];
        // Skip optional language tag on the same line
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

    // Try parsing stripped content directly
    if serde_json::from_str::<serde_json::Value>(stripped).is_ok() {
        return Ok(stripped.to_string());
    }

    // Find first { or [ and match to last } or ]
    let obj_start = stripped.find('{');
    let arr_start = stripped.find('[');

    let start = match (obj_start, arr_start) {
        (Some(o), Some(a)) => o.min(a),
        (Some(o), None) => o,
        (None, Some(a)) => a,
        (None, None) => {
            return Err(Error::LlmNoJson);
        }
    };

    let (open, close) = if stripped.as_bytes()[start] == b'{' {
        (b'{', b'}')
    } else {
        (b'[', b']')
    };

    // Walk forward counting braces to find the matching close
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

    // Validate it actually parses
    serde_json::from_str::<serde_json::Value>(candidate)
        .map_err(|e| Error::Llm(format!("extracted JSON is invalid: {e}")))?;

    Ok(candidate.to_string())
}

/// Map an HTTP error response to the appropriate error variant.
///
/// Shared by both provider implementations — the error classification logic
/// (429 → RateLimit, 5xx → ServerError, else → Llm) is the same regardless
/// of wire format.
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
///
/// Both OpenAI and Anthropic use this identical structure for error responses.
#[derive(serde::Deserialize)]
pub(crate) struct ApiErrorBody {
    pub error: ApiErrorDetail,
}

#[derive(serde::Deserialize)]
pub(crate) struct ApiErrorDetail {
    pub message: String,
}

/// Resolve the model: use the request's model if set, otherwise the client default.
pub(crate) fn resolve_model(request_model: String, default_model: &str) -> String {
    if request_model.is_empty() {
        default_model.to_string()
    } else {
        request_model
    }
}

/// Send a request and handle the response/error boilerplate shared by all providers.
///
/// On success, returns the raw response text for provider-specific parsing.
/// On failure, parses the `{"error": {"message": ...}}` body and classifies the error.
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

/// LLM provider selection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Provider {
    /// Anthropic Claude API.
    Anthropic,
    /// OpenAI API.
    OpenAi,
}

/// Retention intent for request-level prompt caching controls.
///
/// This is intentionally semantic rather than provider-specific:
/// - `Standard` maps to the shorter provider-native retention window
/// - `Extended` maps to the longer provider-native retention window
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum PromptCacheRetention {
    /// Provider-native standard retention.
    #[default]
    Standard,
    /// Provider-native extended retention.
    Extended,
}

/// Explicit request-level prompt caching controls.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ExplicitPromptCachingConfig {
    /// Requested provider-native retention window.
    pub retention: PromptCacheRetention,
    /// Optional stable cache scope key for providers that support it.
    pub scope_key: Option<String>,
}

/// Shared prompt-caching policy across providers.
///
/// `ProviderDefault` preserves the provider's natural behavior:
/// - OpenAI Chat Completions: automatic caching on supported requests/models
/// - Anthropic Messages: no prompt caching unless explicit cache controls are sent
///
/// `Explicit` requests provider-specific prompt caching controls where they exist.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum PromptCachingConfig {
    /// Preserve provider-native defaults.
    #[default]
    ProviderDefault,
    /// Apply explicit request-level prompt caching controls.
    Explicit(ExplicitPromptCachingConfig),
}

impl PromptCachingConfig {
    /// Preserve provider-native prompt caching behavior.
    pub fn provider_default() -> Self {
        Self::ProviderDefault
    }

    /// Apply explicit request-level prompt caching controls using standard retention.
    pub fn explicit() -> Self {
        Self::Explicit(ExplicitPromptCachingConfig::default())
    }

    /// Apply explicit request-level prompt caching controls with a chosen retention intent.
    pub fn explicit_with_retention(retention: PromptCacheRetention) -> Self {
        Self::Explicit(ExplicitPromptCachingConfig {
            retention,
            scope_key: None,
        })
    }

    /// Attach a stable cache scope key to an explicit prompt-caching policy.
    pub fn with_scope_key(self, scope_key: impl Into<String>) -> Self {
        let scope_key = scope_key.into();
        let mut explicit = match self {
            Self::ProviderDefault => ExplicitPromptCachingConfig::default(),
            Self::Explicit(explicit) => explicit,
        };
        explicit.scope_key = Some(scope_key);
        Self::Explicit(explicit)
    }

    /// Return the explicit request-level prompt caching controls, if any.
    pub fn explicit_config(&self) -> Option<&ExplicitPromptCachingConfig> {
        match self {
            Self::ProviderDefault => None,
            Self::Explicit(explicit) => Some(explicit),
        }
    }
}

fn parse_prompt_cache_retention(
    var_name: &str,
    value: Option<String>,
) -> Result<Option<PromptCacheRetention>> {
    value
        .map(|value| {
            let normalized = value.trim().to_ascii_lowercase();
            match normalized.as_str() {
                "standard" | "short" | "5m" | "in_memory" | "in-memory" => {
                    Ok(PromptCacheRetention::Standard)
                }
                "extended" | "long" | "1h" | "24h" => Ok(PromptCacheRetention::Extended),
                other => Err(Error::Internal(format!(
                    "{var_name} must be one of standard|short|5m|in_memory|extended|1h|24h, got: {other}"
                ))),
            }
        })
        .transpose()
}

/// Parse prompt-caching configuration from already-resolved string values.
///
/// This keeps provider-default semantics explicit while allowing richer request-level
/// controls when a provider supports them.
pub fn parse_prompt_caching_config(
    mode_var_name: &str,
    mode: Option<String>,
    retention_var_name: &str,
    retention: Option<String>,
    scope_key_var_name: &str,
    scope_key: Option<String>,
) -> Result<PromptCachingConfig> {
    let parsed_retention = parse_prompt_cache_retention(retention_var_name, retention.clone())?;
    let scope_key = scope_key.and_then(|value| {
        let trimmed = value.trim();
        (!trimmed.is_empty()).then(|| trimmed.to_string())
    });

    let explicit_from_options = parsed_retention.is_some() || scope_key.is_some();
    let parsed_mode = match mode {
        Some(value) => {
            let normalized = value.trim().to_ascii_lowercase();
            match normalized.as_str() {
                "default" | "provider_default" | "provider-default" | "auto" => {
                    PromptCachingConfig::ProviderDefault
                }
                "explicit" | "enabled" | "enable" | "on" | "true" | "yes" | "1" => {
                    PromptCachingConfig::explicit()
                }
                // Backward-compatible shorthand: the old boolean "off" means
                // "do not request explicit controls", i.e. preserve provider defaults.
                "disabled" | "disable" | "off" | "false" | "no" | "0" => {
                    PromptCachingConfig::ProviderDefault
                }
                other => {
                    return Err(Error::Internal(format!(
                        "{mode_var_name} must be one of default|auto|explicit|on|off, got: {other}"
                    )));
                }
            }
        }
        None if explicit_from_options => PromptCachingConfig::explicit(),
        None => PromptCachingConfig::ProviderDefault,
    };

    match parsed_mode {
        PromptCachingConfig::ProviderDefault => {
            if explicit_from_options {
                return Err(Error::Internal(format!(
                    "{mode_var_name}=provider_default cannot be combined with {retention_var_name} or {scope_key_var_name}"
                )));
            }
            Ok(PromptCachingConfig::ProviderDefault)
        }
        PromptCachingConfig::Explicit(mut explicit) => {
            if let Some(retention) = parsed_retention {
                explicit.retention = retention;
            }
            explicit.scope_key = scope_key;
            Ok(PromptCachingConfig::Explicit(explicit))
        }
    }
}

/// Configuration for a single LLM provider.
#[derive(Debug, Clone)]
pub struct ProviderConfig {
    /// Which provider to use.
    pub provider: Provider,
    /// API key for authentication.
    pub api_key: String,
    /// Model name/ID to use.
    pub model: String,
    /// Optional base URL override for OpenAI-compatible providers.
    pub base_url: Option<String>,
    /// Prompt-caching behavior for the provider client.
    pub prompt_caching: PromptCachingConfig,
}

/// Configuration for LLM usage across the system.
///
/// Supports per-operation model selection: a strong model for retain/extraction,
/// a cheaper model for reflect, etc.
#[derive(Debug, Clone)]
pub struct LlmConfig {
    /// Default provider config used for all operations.
    pub default: ProviderConfig,
    /// Optional override for retain/extraction operations.
    pub retain_override: Option<ProviderConfig>,
    /// Optional override for reflect operations.
    pub reflect_override: Option<ProviderConfig>,
}

/// Build an LLM client from a provider configuration.
pub fn build_client(config: &ProviderConfig) -> crate::error::Result<Box<dyn LlmClient>> {
    match config.provider {
        Provider::Anthropic => Ok(Box::new(
            anthropic::AnthropicClient::new(config.api_key.clone(), config.model.clone())?
                .prompt_caching(config.prompt_caching.clone()),
        )),
        Provider::OpenAi => Ok(Box::new(
            openai::OpenAiClient::new(
                config.api_key.clone(),
                config.model.clone(),
                config.base_url.clone(),
            )?
            .prompt_caching(config.prompt_caching.clone()),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn prompt_caching_parse_defaults_to_provider_default() {
        let parsed = parse_prompt_caching_config(
            "LLM_PROMPT_CACHING",
            None,
            "LLM_PROMPT_CACHE_RETENTION",
            None,
            "LLM_PROMPT_CACHE_KEY",
            None,
        )
        .unwrap();

        assert_eq!(parsed, PromptCachingConfig::ProviderDefault);
    }

    #[test]
    fn prompt_caching_parse_truthy_mode_becomes_explicit_standard() {
        let parsed = parse_prompt_caching_config(
            "LLM_PROMPT_CACHING",
            Some("true".into()),
            "LLM_PROMPT_CACHE_RETENTION",
            None,
            "LLM_PROMPT_CACHE_KEY",
            None,
        )
        .unwrap();

        assert_eq!(
            parsed,
            PromptCachingConfig::Explicit(ExplicitPromptCachingConfig {
                retention: PromptCacheRetention::Standard,
                scope_key: None,
            })
        );
    }

    #[test]
    fn prompt_caching_parse_options_imply_explicit_when_mode_unspecified() {
        let parsed = parse_prompt_caching_config(
            "LLM_PROMPT_CACHING",
            None,
            "LLM_PROMPT_CACHE_RETENTION",
            Some("24h".into()),
            "LLM_PROMPT_CACHE_KEY",
            Some("benchmark-seed".into()),
        )
        .unwrap();

        assert_eq!(
            parsed,
            PromptCachingConfig::Explicit(ExplicitPromptCachingConfig {
                retention: PromptCacheRetention::Extended,
                scope_key: Some("benchmark-seed".into()),
            })
        );
    }

    #[test]
    fn prompt_caching_parse_rejects_default_mode_with_explicit_options() {
        let err = parse_prompt_caching_config(
            "LLM_PROMPT_CACHING",
            Some("default".into()),
            "LLM_PROMPT_CACHE_RETENTION",
            Some("extended".into()),
            "LLM_PROMPT_CACHE_KEY",
            None,
        )
        .unwrap_err();

        assert!(matches!(err, Error::Internal(_)));
    }
}
