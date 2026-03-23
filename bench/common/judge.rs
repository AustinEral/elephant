//! Prompt-agnostic LLM judge infrastructure shared by all benchmark harnesses.
#![allow(dead_code)]

use std::sync::Arc;

use serde::Deserialize;

use crate::common::failure;
use elephant::llm::retry::{RetryPolicy, RetryingLlmClient};
use elephant::llm::{self, CompletionRequest, LlmClient, Message};
use elephant::metrics::{LlmStage, MeteredLlmClient, MetricsCollector};

/// Parsed judge LLM response.
#[derive(Debug, Deserialize)]
pub struct JudgeResponse {
    pub reasoning: String,
    pub label: String,
}

pub const JUDGE_TEMPERATURE: f32 = 0.0;
pub const JUDGE_MAX_TOKENS: usize = 200;
pub const JUDGE_MAX_ATTEMPTS: usize = 3;

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct JudgeOverrides {
    pub provider: Option<llm::Provider>,
    pub model: Option<String>,
}

#[derive(Debug, thiserror::Error)]
pub enum JudgeError {
    #[error("judge error: {0}")]
    Llm(elephant::error::Error),
    #[error("could not parse judge response: {0}")]
    Parse(String),
    #[error("judge failed after retries")]
    RetriesExhausted,
}

impl JudgeError {
    pub fn is_fatal(&self) -> bool {
        match self {
            Self::Llm(err) => failure::is_fatal_bench_error(err),
            Self::Parse(_) | Self::RetriesExhausted => false,
        }
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
) -> Result<(bool, String), JudgeError> {
    let request = CompletionRequest::builder()
        .message(Message::user(rendered_prompt))
        .max_tokens(JUDGE_MAX_TOKENS)
        .temperature(JUDGE_TEMPERATURE)
        .build();

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
                    return Err(JudgeError::Parse(format!(
                        "could not parse judge response: {}",
                        &resp.content[..resp.content.len().min(120)]
                    )));
                }
            }
            Err(e) => {
                return Err(JudgeError::Llm(e));
            }
        }
    }
    Err(JudgeError::RetriesExhausted)
}

/// Build an LLM client configured for judge duty.
///
/// Uses env var fallback chain: JUDGE_PROVIDER -> LLM_PROVIDER, JUDGE_API_KEY -> LLM_API_KEY,
/// JUDGE_MODEL -> LLM_MODEL, and JUDGE_BASE_URL -> LLM_BASE_URL. `overrides` can replace
/// the resolved provider and/or model.
pub fn build_judge_client(
    metrics: Arc<MetricsCollector>,
    overrides: &JudgeOverrides,
) -> Arc<dyn LlmClient> {
    let judge_config =
        llm::judge_client_config_from_env(overrides.provider, overrides.model.as_deref()).unwrap();
    let inner: Arc<dyn LlmClient> = Arc::from(llm::build_client(&judge_config).unwrap());
    let metered: Arc<dyn LlmClient> =
        Arc::new(MeteredLlmClient::new(inner, metrics, LlmStage::Judge));
    Arc::new(RetryingLlmClient::new(metered, RetryPolicy::default()))
}

/// Returns "provider/model" string identifying the judge configuration.
pub fn judge_label(overrides: &JudgeOverrides) -> String {
    llm::judge_client_config_from_env(overrides.provider, overrides.model.as_deref())
        .unwrap()
        .label()
}
