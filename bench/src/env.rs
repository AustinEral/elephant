//! Typed benchmark-only configuration types.

use std::fmt;
use std::sync::Arc;

use elephant::llm::retry::{RetryPolicy, RetryingLlmClient};
use elephant::llm::{ClientConfig, DeterminismRequirement, LlmClient, build_client};
use elephant::metrics::{LlmStage, MeteredLlmClient, MetricsCollector};

/// Validated benchmark-only startup configuration.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct BenchConfig {
    determinism_requirement: Option<DeterminismRequirement>,
}

impl BenchConfig {
    /// Create benchmark configuration from validated values.
    pub(crate) fn new(determinism_requirement: Option<DeterminismRequirement>) -> Self {
        Self {
            determinism_requirement,
        }
    }

    /// Return the benchmark determinism requirement, if configured.
    pub fn determinism_requirement(&self) -> Option<DeterminismRequirement> {
        self.determinism_requirement
    }
}

/// Validated judge client configuration for benchmark evaluation.
#[derive(Clone)]
pub struct BenchJudgeConfig {
    client: ClientConfig,
    temperature: Option<f32>,
    max_tokens: usize,
    max_attempts: usize,
}

impl fmt::Debug for BenchJudgeConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BenchJudgeConfig")
            .field("client_label", &self.client.label())
            .finish()
    }
}

impl BenchJudgeConfig {
    /// Create a benchmark judge configuration from a validated client config.
    pub(crate) fn new(
        client: ClientConfig,
        temperature: Option<f32>,
        max_tokens: usize,
        max_attempts: usize,
    ) -> Self {
        Self {
            client,
            temperature,
            max_tokens,
            max_attempts,
        }
    }

    /// Return a stable provider/model label for the judge.
    pub fn label(&self) -> String {
        self.client.label()
    }

    /// Return the optional judge temperature override.
    pub fn temperature(&self) -> Option<f32> {
        self.temperature
    }

    /// Return the maximum completion tokens for one judge attempt.
    pub fn max_tokens(&self) -> usize {
        self.max_tokens
    }

    /// Return the maximum number of judge attempts.
    pub fn max_attempts(&self) -> usize {
        self.max_attempts
    }

    /// Build a metered, retrying judge client from the validated configuration.
    pub fn build_client(
        &self,
        metrics: Arc<MetricsCollector>,
    ) -> elephant::Result<Arc<dyn LlmClient>> {
        let inner: Arc<dyn LlmClient> = Arc::from(build_client(&self.client)?);
        let metered: Arc<dyn LlmClient> =
            Arc::new(MeteredLlmClient::new(inner, metrics, LlmStage::Judge));
        Ok(Arc::new(RetryingLlmClient::new(
            metered,
            RetryPolicy::default(),
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use elephant::llm::OpenAiConfig;

    #[test]
    fn bench_config_new_preserves_requirement() {
        let config = BenchConfig::new(Some(DeterminismRequirement::Strong));
        assert_eq!(
            config.determinism_requirement(),
            Some(DeterminismRequirement::Strong)
        );
    }

    #[test]
    fn judge_config_new_uses_validated_client() {
        let client = ClientConfig::OpenAi(OpenAiConfig::new("sk-test", "gpt-4o-mini").unwrap());
        let config = BenchJudgeConfig::new(client, None, 200, 3);
        assert_eq!(config.label(), "openai/gpt-4o-mini");
    }

    #[test]
    fn judge_config_debug_redacts_client_secrets() {
        let client =
            ClientConfig::OpenAi(OpenAiConfig::new("judge-secret", "gpt-4o-mini").unwrap());
        let config = BenchJudgeConfig::new(client, None, 200, 3);
        let debug = format!("{config:?}");
        assert!(debug.contains("openai/gpt-4o-mini"));
        assert!(!debug.contains("judge-secret"));
    }
}
