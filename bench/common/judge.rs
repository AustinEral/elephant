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
    let judge_config = ProviderConfig {
        provider,
        api_key: judge_api_key,
        model: judge_model,
        base_url: env::var("JUDGE_BASE_URL")
            .ok()
            .or_else(|| env::var("LLM_BASE_URL").ok()),
    };
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
