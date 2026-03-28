//! Typed configuration for benchmark startup policy.

use std::{env, fmt};

use crate::llm::{
    AnthropicConfig, AnthropicPromptCacheConfig, AnthropicPromptCacheTtl, ClientConfig,
    DEFAULT_TIMEOUT_SECS, DeterminismRequirement, GeminiConfig, OpenAiConfig,
    OpenAiPromptCacheConfig, OpenAiPromptCacheRetention, Provider, VertexConfig,
};

use super::error::{ConfigError, Result};

#[derive(Debug, Clone, PartialEq, Eq)]
enum PromptCacheConfig {
    Disabled,
    OpenAi(OpenAiPromptCacheConfig),
    Anthropic(AnthropicPromptCacheConfig),
}

/// Validated benchmark-only startup configuration loaded from environment.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct BenchConfig {
    determinism_requirement: Option<DeterminismRequirement>,
}

impl BenchConfig {
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

        Ok(Self {
            determinism_requirement,
        })
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
        let timeout_secs = parse_timeout_secs()?;
        let prompt_cache = prompt_cache_config(provider)?;

        Ok(Self {
            client: build_client_config(
                provider,
                api_key,
                model,
                vertex_project,
                vertex_location,
                base_url,
                timeout_secs,
                prompt_cache,
            )?,
        })
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

fn required_string_any(names: &[&str]) -> Result<String> {
    for name in names {
        if let Ok(value) = env::var(name) {
            return Ok(value);
        }
    }

    Err(ConfigError::configuration(format!(
        "{} must be set",
        names.join(" or ")
    )))
}

fn optional_string_any(names: &[&str]) -> Option<String> {
    for name in names {
        if let Ok(value) = env::var(name) {
            return Some(value);
        }
    }
    None
}

fn parse_timeout_secs() -> Result<u64> {
    match env::var("LLM_TIMEOUT_SECS") {
        Ok(value) => {
            let timeout_secs = value.parse::<u64>().map_err(|_| {
                ConfigError::configuration(format!(
                    "LLM_TIMEOUT_SECS must be a positive integer, got: {value}"
                ))
            })?;
            if timeout_secs == 0 {
                return Err(ConfigError::configuration(
                    "LLM_TIMEOUT_SECS must be greater than zero",
                ));
            }
            Ok(timeout_secs)
        }
        Err(_) => Ok(DEFAULT_TIMEOUT_SECS),
    }
}

fn prompt_cache_enabled() -> Result<bool> {
    match env::var("LLM_PROMPT_CACHE_ENABLED") {
        Ok(value) => match value.trim().to_ascii_lowercase().as_str() {
            "1" | "true" | "yes" | "on" => Ok(true),
            "0" | "false" | "no" | "off" => Ok(false),
            other => Err(ConfigError::configuration(format!(
                "LLM_PROMPT_CACHE_ENABLED must be a boolean, got: {other}"
            ))),
        },
        Err(_) => Ok(false),
    }
}

fn prompt_cache_config(provider: Provider) -> Result<PromptCacheConfig> {
    if !prompt_cache_enabled()? {
        return Ok(PromptCacheConfig::Disabled);
    }

    match provider {
        Provider::OpenAi => {
            let retention = match env::var("OPENAI_PROMPT_CACHE_RETENTION") {
                Ok(value) => match value.trim().to_ascii_lowercase().as_str() {
                    "in_memory" | "in-memory" => Some(OpenAiPromptCacheRetention::InMemory),
                    "24h" => Some(OpenAiPromptCacheRetention::Hours24),
                    other => {
                        return Err(ConfigError::configuration(format!(
                            "OPENAI_PROMPT_CACHE_RETENTION must be one of: in_memory, in-memory, 24h; got: {other}"
                        )));
                    }
                },
                Err(_) => None,
            };

            let mut config = OpenAiPromptCacheConfig::new();
            if let Ok(key) = env::var("OPENAI_PROMPT_CACHE_KEY") {
                config = config.with_key(key);
            }
            if let Some(retention) = retention {
                config = config.with_retention(retention);
            }
            Ok(PromptCacheConfig::OpenAi(config))
        }
        Provider::Anthropic => {
            let ttl = match env::var("ANTHROPIC_PROMPT_CACHE_TTL") {
                Ok(value) => match value.trim().to_ascii_lowercase().as_str() {
                    "5m" => Some(AnthropicPromptCacheTtl::Minutes5),
                    "1h" => Some(AnthropicPromptCacheTtl::Hours1),
                    other => {
                        return Err(ConfigError::configuration(format!(
                            "ANTHROPIC_PROMPT_CACHE_TTL must be one of: 5m, 1h; got: {other}"
                        )));
                    }
                },
                Err(_) => None,
            };

            let mut config = AnthropicPromptCacheConfig::new();
            if let Some(ttl) = ttl {
                config = config.with_ttl(ttl);
            }
            Ok(PromptCacheConfig::Anthropic(config))
        }
        Provider::Gemini | Provider::Vertex => Ok(PromptCacheConfig::Disabled),
    }
}

#[allow(clippy::too_many_arguments)]
fn build_client_config(
    provider: Provider,
    api_key: String,
    model: String,
    vertex_project: Option<String>,
    vertex_location: Option<String>,
    base_url: Option<String>,
    timeout_secs: u64,
    prompt_cache: PromptCacheConfig,
) -> Result<ClientConfig> {
    Ok(match provider {
        Provider::Anthropic => {
            let mut config = AnthropicConfig::new(api_key, model)
                .map_err(ConfigError::from)?
                .with_timeout_secs(timeout_secs)
                .map_err(ConfigError::from)?;
            if let PromptCacheConfig::Anthropic(prompt_cache) = prompt_cache {
                config = config.with_prompt_cache(prompt_cache);
            }
            ClientConfig::Anthropic(config)
        }
        Provider::OpenAi => {
            let mut config = OpenAiConfig::new(api_key, model)
                .map_err(ConfigError::from)?
                .with_timeout_secs(timeout_secs)
                .map_err(ConfigError::from)?;
            if let Some(base_url) = base_url {
                config = config.with_base_url(base_url).map_err(ConfigError::from)?;
            }
            if let PromptCacheConfig::OpenAi(prompt_cache) = prompt_cache {
                config = config.with_prompt_cache(prompt_cache);
            }
            ClientConfig::OpenAi(config)
        }
        Provider::Gemini => {
            let mut config = GeminiConfig::new(api_key, model)
                .map_err(ConfigError::from)?
                .with_timeout_secs(timeout_secs)
                .map_err(ConfigError::from)?;
            if let Some(base_url) = base_url {
                config = config.with_base_url(base_url).map_err(ConfigError::from)?;
            }
            ClientConfig::Gemini(config)
        }
        Provider::Vertex => {
            let project = vertex_project.ok_or_else(|| {
                ConfigError::configuration(
                    "JUDGE_VERTEX_PROJECT, LLM_VERTEX_PROJECT, or GOOGLE_CLOUD_PROJECT must be set for provider=vertex",
                )
            })?;
            let mut config = VertexConfig::new(api_key, model, project)
                .map_err(ConfigError::from)?
                .with_timeout_secs(timeout_secs)
                .map_err(ConfigError::from)?;
            if let Some(location) = vertex_location {
                config = config.with_location(location).map_err(ConfigError::from)?;
            }
            if let Some(base_url) = base_url {
                config = config.with_base_url(base_url).map_err(ConfigError::from)?;
            }
            ClientConfig::Vertex(config)
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;
    use std::sync::{Mutex, OnceLock};

    fn env_lock() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
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
