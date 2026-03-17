//! Prompt-agnostic LLM judge infrastructure shared by all benchmark harnesses.

use std::env;
use std::sync::Arc;

use serde::Deserialize;

use elephant::llm::retry::{RetryPolicy, RetryingLlmClient};
use elephant::llm::{self, LlmClient, Provider, ProviderConfig};
use elephant::metrics::{LlmStage, MeteredLlmClient, MetricsCollector};
use elephant::types::{CompletionRequest, Message};

/// Parsed judge LLM response.
#[derive(Debug, Deserialize)]
pub struct JudgeResponse {
    pub reasoning: String,
    pub label: String,
}

pub const JUDGE_TEMPERATURE: f32 = 0.0;
pub const JUDGE_MAX_TOKENS: usize = 200;
pub const JUDGE_MAX_ATTEMPTS: usize = 3;

fn judge_prompt_caching_config() -> llm::PromptCachingConfig {
    fn parse_prompt_caching(var_name: &str, value: &str) -> llm::PromptCachingConfig {
        let enabled = match value.trim().to_ascii_lowercase().as_str() {
            "1" | "true" | "yes" | "on" => true,
            "0" | "false" | "no" | "off" => false,
            other => panic!("{var_name} must be a boolean, got: {other}"),
        };

        llm::PromptCachingConfig { enabled }
    }

    env::var("JUDGE_PROMPT_CACHING")
        .ok()
        .map(|value| parse_prompt_caching("JUDGE_PROMPT_CACHING", &value))
        .or_else(|| {
            env::var("LLM_PROMPT_CACHING")
                .ok()
                .map(|value| parse_prompt_caching("LLM_PROMPT_CACHING", &value))
        })
        .unwrap_or_default()
}

fn judge_provider_config(override_model: Option<String>) -> ProviderConfig {
    let judge_provider_str = env::var("JUDGE_PROVIDER")
        .or_else(|_| env::var("LLM_PROVIDER"))
        .expect("JUDGE_PROVIDER or LLM_PROVIDER must be set");
    let judge_api_key = env::var("JUDGE_API_KEY")
        .or_else(|_| env::var("LLM_API_KEY"))
        .expect("JUDGE_API_KEY or LLM_API_KEY must be set");
    let judge_model = override_model.unwrap_or_else(|| {
        env::var("JUDGE_MODEL")
            .or_else(|_| env::var("LLM_MODEL"))
            .expect("JUDGE_MODEL or LLM_MODEL must be set")
    });
    let provider = match judge_provider_str.as_str() {
        "openai" => Provider::OpenAi,
        _ => Provider::Anthropic,
    };

    ProviderConfig {
        provider,
        api_key: judge_api_key,
        model: judge_model,
        base_url: env::var("JUDGE_BASE_URL")
            .ok()
            .or_else(|| env::var("LLM_BASE_URL").ok()),
        prompt_caching: judge_prompt_caching_config(),
    }
}

/// Send a pre-rendered prompt to the judge LLM and parse the CORRECT/WRONG verdict.
///
/// The caller is responsible for rendering the prompt template with the appropriate
/// placeholders (question, answer, response, etc.) before calling this function.
///
/// Returns `Ok((correct, reasoning))` on success, `Err(description)` on failure.
pub async fn llm_judge(
    judge: &dyn LlmClient,
    rendered_prompt: &str,
) -> Result<(bool, String), String> {
    let request = CompletionRequest {
        model: String::new(),
        messages: vec![Message::text("user", rendered_prompt)],
        max_tokens: Some(JUDGE_MAX_TOKENS),
        temperature: Some(JUDGE_TEMPERATURE),
        system: None,
        ..Default::default()
    };

    for attempt in 0..JUDGE_MAX_ATTEMPTS {
        let result = judge.complete(request.clone()).await;
        match result {
            Ok(resp) => {
                if let Ok(parsed) = serde_json::from_str::<JudgeResponse>(&resp.content) {
                    let correct = parsed.label.eq_ignore_ascii_case("CORRECT");
                    return Ok((correct, parsed.reasoning));
                }
                if let Ok(json_str) = llm::extract_json(&resp.content)
                    && let Ok(parsed) = serde_json::from_str::<JudgeResponse>(&json_str)
                {
                    let correct = parsed.label.eq_ignore_ascii_case("CORRECT");
                    return Ok((correct, parsed.reasoning));
                }
                if attempt + 1 == JUDGE_MAX_ATTEMPTS {
                    return Err(format!(
                        "could not parse judge response: {}",
                        &resp.content[..resp.content.len().min(120)]
                    ));
                }
            }
            Err(e) => {
                return Err(format!("judge error: {e}"));
            }
        }
    }
    Err("judge failed after retries".into())
}

/// Build an LLM client configured for judge duty.
///
/// Uses env var fallback chain: JUDGE_PROVIDER -> LLM_PROVIDER, JUDGE_API_KEY -> LLM_API_KEY,
/// JUDGE_MODEL -> LLM_MODEL. The `override_model` replaces the env var chain for model selection.
pub fn build_judge_client(
    metrics: Arc<MetricsCollector>,
    override_model: Option<String>,
) -> Arc<dyn LlmClient> {
    let judge_config = judge_provider_config(override_model);
    let inner: Arc<dyn LlmClient> = Arc::from(llm::build_client(&judge_config).unwrap());
    let metered: Arc<dyn LlmClient> =
        Arc::new(MeteredLlmClient::new(inner, metrics, LlmStage::Judge));
    Arc::new(RetryingLlmClient::new(metered, RetryPolicy::default()))
}

/// Returns "provider/model" string identifying the judge configuration.
pub fn judge_label(override_model: &Option<String>) -> String {
    let provider = env::var("JUDGE_PROVIDER")
        .or_else(|_| env::var("LLM_PROVIDER"))
        .expect("JUDGE_PROVIDER or LLM_PROVIDER must be set");
    let model = override_model.clone().unwrap_or_else(|| {
        env::var("JUDGE_MODEL")
            .or_else(|_| env::var("LLM_MODEL"))
            .expect("JUDGE_MODEL or LLM_MODEL must be set")
    });
    format!("{provider}/{model}")
}

#[cfg(test)]
mod tests {
    use super::judge_provider_config;
    use elephant::llm::Provider;
    use std::sync::{Mutex, OnceLock};

    fn env_lock() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
    }

    fn clear_env(var_name: &str) {
        unsafe {
            std::env::remove_var(var_name);
        }
    }

    fn set_env(var_name: &str, value: &str) {
        unsafe {
            std::env::set_var(var_name, value);
        }
    }

    #[test]
    fn judge_anthropic_prompt_caching_follows_runtime_env() {
        let _guard = env_lock().lock().unwrap();
        set_env("LLM_PROVIDER", "anthropic");
        set_env("LLM_API_KEY", "test-key");
        set_env("LLM_MODEL", "claude-sonnet");
        set_env("LLM_PROMPT_CACHING", "true");
        set_env("JUDGE_PROMPT_CACHING", "false");

        let config = judge_provider_config(None);

        assert_eq!(config.provider, Provider::Anthropic);
        assert!(!config.prompt_caching.enabled);
        clear_env("JUDGE_PROMPT_CACHING");
        clear_env("LLM_PROMPT_CACHING");
        clear_env("LLM_MODEL");
        clear_env("LLM_API_KEY");
        clear_env("LLM_PROVIDER");
    }

    #[test]
    fn judge_openai_prompt_caching_follows_runtime_env() {
        let _guard = env_lock().lock().unwrap();
        set_env("JUDGE_PROVIDER", "openai");
        set_env("JUDGE_API_KEY", "test-key");
        set_env("JUDGE_MODEL", "gpt-4o-mini");
        clear_env("JUDGE_PROMPT_CACHING");
        set_env("LLM_PROMPT_CACHING", "on");

        let config = judge_provider_config(None);

        assert_eq!(config.provider, Provider::OpenAi);
        assert!(config.prompt_caching.enabled);
        clear_env("JUDGE_MODEL");
        clear_env("JUDGE_API_KEY");
        clear_env("JUDGE_PROVIDER");
        clear_env("LLM_PROMPT_CACHING");
    }

    #[test]
    fn judge_prompt_caching_invalid_env_panics() {
        let _guard = env_lock().lock().unwrap();
        set_env("LLM_PROVIDER", "anthropic");
        set_env("LLM_API_KEY", "test-key");
        set_env("LLM_MODEL", "claude-sonnet");
        set_env("JUDGE_PROMPT_CACHING", "definitely-not-bool");

        let panic = std::panic::catch_unwind(|| judge_provider_config(None))
            .expect_err("invalid judge prompt caching env should panic");

        let message = panic
            .downcast_ref::<String>()
            .map(String::as_str)
            .or_else(|| panic.downcast_ref::<&str>().copied())
            .unwrap_or("");

        assert!(message.contains("JUDGE_PROMPT_CACHING must be a boolean"));
        clear_env("JUDGE_PROMPT_CACHING");
        clear_env("LLM_MODEL");
        clear_env("LLM_API_KEY");
        clear_env("LLM_PROVIDER");
    }
}
