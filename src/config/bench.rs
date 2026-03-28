//! Typed configuration for benchmark startup policy.

#[cfg(test)]
use std::env;
use std::fmt;

use crate::llm::{ClientConfig, DeterminismRequirement, Provider, VertexConfig};

use super::client_env::{
    SharedClientEnvConfig, build_client_config, optional_string_any, required_string_any,
};
use super::error::{ConfigError, Result};

/// Validated benchmark-only startup configuration loaded from environment.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct BenchConfig {
    determinism_requirement: Option<DeterminismRequirement>,
}

impl BenchConfig {
    /// Create benchmark configuration from validated values.
    pub fn new(determinism_requirement: Option<DeterminismRequirement>) -> Self {
        Self {
            determinism_requirement,
        }
    }

    /// Load benchmark configuration from the process environment.
    pub fn from_env() -> Result<Self> {
        let determinism_requirement = match std::env::var("BENCH_DETERMINISM_REQUIREMENT") {
            Ok(raw) => {
                let value = raw.trim().to_ascii_lowercase();
                let requirement = match value.as_str() {
                    "best_effort" | "best-effort" | "1" | "true" | "yes" | "on" => {
                        DeterminismRequirement::BestEffort
                    }
                    "strong" => DeterminismRequirement::Strong,
                    _ => {
                        return Err(ConfigError::configuration(format!(
                            "BENCH_DETERMINISM_REQUIREMENT must be one of: best_effort, strong; got: {raw}"
                        )));
                    }
                };
                Some(requirement)
            }
            Err(std::env::VarError::NotPresent) => None,
            Err(err) => {
                return Err(ConfigError::configuration(format!(
                    "BENCH_DETERMINISM_REQUIREMENT must be set: {err}"
                )));
            }
        };

        Ok(Self::new(determinism_requirement))
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
    pub fn new(client: ClientConfig) -> Self {
        Self { client }
    }

    /// Load the benchmark judge client configuration from environment with optional overrides.
    pub fn from_env(
        override_provider: Option<Provider>,
        override_model: Option<&str>,
    ) -> Result<Self> {
        let provider = match override_provider {
            Some(provider) => provider,
            None => required_string_any(&["JUDGE_PROVIDER", "LLM_PROVIDER"])?
                .parse::<Provider>()
                .map_err(ConfigError::from)?,
        };
        let api_key = required_string_any(&["JUDGE_API_KEY", "LLM_API_KEY"])?;
        let model = if let Some(override_model) = override_model {
            override_model.to_string()
        } else {
            required_string_any(&["JUDGE_MODEL", "LLM_MODEL"])?
        };
        let vertex_project = if provider == Provider::Vertex {
            Some(required_string_any(&[
                "JUDGE_VERTEX_PROJECT",
                "LLM_VERTEX_PROJECT",
                "GOOGLE_CLOUD_PROJECT",
            ])?)
        } else {
            None
        };
        let vertex_location = if provider == Provider::Vertex {
            Some(
                optional_string_any(&[
                    "JUDGE_VERTEX_LOCATION",
                    "LLM_VERTEX_LOCATION",
                    "GOOGLE_CLOUD_LOCATION",
                ])
                .unwrap_or_else(|| VertexConfig::DEFAULT_LOCATION.into()),
            )
        } else {
            None
        };
        let base_url = optional_string_any(&["JUDGE_BASE_URL", "LLM_BASE_URL"]);
        let shared = SharedClientEnvConfig::new(
            provider,
            api_key,
            vertex_project,
            vertex_location,
            base_url,
        )?;

        Ok(Self::new(build_client_config(
            &shared,
            model,
            "JUDGE_VERTEX_PROJECT, LLM_VERTEX_PROJECT, or GOOGLE_CLOUD_PROJECT must be set for provider=vertex",
        )?))
    }

    /// Return the validated judge client configuration.
    pub fn client(&self) -> &ClientConfig {
        &self.client
    }

    /// Consume the typed config and return the validated judge client configuration.
    pub fn into_client(self) -> ClientConfig {
        self.client
    }

    /// Return a stable provider/model label for the judge.
    pub fn label(&self) -> String {
        self.client.label()
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::OpenAiConfig;
    use std::sync::{Mutex, OnceLock};

    fn env_lock() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
    }

    #[test]
    fn bench_config_new_preserves_requirement() {
        let config = BenchConfig::new(Some(DeterminismRequirement::Strong));
        assert_eq!(
            config.determinism_requirement(),
            Some(DeterminismRequirement::Strong)
        );
    }

    #[test]
    fn parses_best_effort_requirement() {
        let _guard = env_lock().lock().unwrap();
        unsafe {
            env::set_var("BENCH_DETERMINISM_REQUIREMENT", "best_effort");
        }

        let config = BenchConfig::from_env().unwrap();
        assert_eq!(
            config.determinism_requirement(),
            Some(DeterminismRequirement::BestEffort)
        );

        unsafe {
            env::remove_var("BENCH_DETERMINISM_REQUIREMENT");
        }
    }

    #[test]
    fn rejects_invalid_requirement() {
        let _guard = env_lock().lock().unwrap();
        unsafe {
            env::set_var("BENCH_DETERMINISM_REQUIREMENT", "maybe");
        }

        let err = BenchConfig::from_env().unwrap_err();
        assert!(err.to_string().contains("BENCH_DETERMINISM_REQUIREMENT"));

        unsafe {
            env::remove_var("BENCH_DETERMINISM_REQUIREMENT");
        }
    }

    #[test]
    fn judge_config_new_uses_validated_client() {
        let client = ClientConfig::OpenAi(OpenAiConfig::new("sk-test", "gpt-4o-mini").unwrap());
        let config = BenchJudgeConfig::new(client);
        assert_eq!(config.label(), "openai/gpt-4o-mini");
    }

    #[test]
    fn judge_config_uses_fallback_chain() {
        let _guard = env_lock().lock().unwrap();
        unsafe {
            env::set_var("LLM_PROVIDER", "openai");
            env::set_var("LLM_API_KEY", "sk-test");
            env::set_var("LLM_MODEL", "gpt-4o-mini");
        }

        let config = BenchJudgeConfig::from_env(None, None).unwrap();
        assert_eq!(config.label(), "openai/gpt-4o-mini");

        unsafe {
            env::remove_var("LLM_PROVIDER");
            env::remove_var("LLM_API_KEY");
            env::remove_var("LLM_MODEL");
        }
    }

    #[test]
    fn judge_config_accepts_model_override() {
        let _guard = env_lock().lock().unwrap();
        unsafe {
            env::set_var("LLM_PROVIDER", "openai");
            env::set_var("LLM_API_KEY", "sk-test");
        }

        let config = BenchJudgeConfig::from_env(None, Some("gpt-4o")).unwrap();
        assert_eq!(config.label(), "openai/gpt-4o");

        unsafe {
            env::remove_var("LLM_PROVIDER");
            env::remove_var("LLM_API_KEY");
        }
    }

    #[test]
    fn judge_config_debug_redacts_client_secrets() {
        let _guard = env_lock().lock().unwrap();
        unsafe {
            env::set_var("LLM_PROVIDER", "openai");
            env::set_var("LLM_API_KEY", "judge-secret");
            env::set_var("LLM_MODEL", "gpt-4o-mini");
        }

        let config = BenchJudgeConfig::from_env(None, None).unwrap();
        let debug = format!("{config:?}");
        assert!(debug.contains("openai/gpt-4o-mini"));
        assert!(!debug.contains("judge-secret"));

        unsafe {
            env::remove_var("LLM_PROVIDER");
            env::remove_var("LLM_API_KEY");
            env::remove_var("LLM_MODEL");
        }
    }
}
