//! Shared provider/client environment loading helpers for config modules.

use std::env;

use crate::llm::{
    AnthropicConfig, AnthropicPromptCacheConfig, AnthropicPromptCacheTtl, ClientConfig,
    DEFAULT_TIMEOUT_SECS, GeminiConfig, OpenAiConfig, OpenAiPromptCacheConfig,
    OpenAiPromptCacheRetention, Provider, VertexConfig,
};

use super::env as config_env;
use super::error::{ConfigError, Result};

#[derive(Clone)]
pub(crate) struct SharedClientEnvConfig {
    provider: Provider,
    api_key: String,
    vertex_project: Option<String>,
    vertex_location: Option<String>,
    base_url: Option<String>,
    timeout_secs: u64,
    openai_prompt_cache: Option<OpenAiPromptCacheConfig>,
    anthropic_prompt_cache: Option<AnthropicPromptCacheConfig>,
}

impl SharedClientEnvConfig {
    pub(crate) fn new(
        provider: Provider,
        api_key: String,
        vertex_project: Option<String>,
        vertex_location: Option<String>,
        base_url: Option<String>,
    ) -> Result<Self> {
        Ok(Self {
            provider,
            api_key,
            vertex_project,
            vertex_location,
            base_url,
            timeout_secs: parse_timeout_secs()?,
            openai_prompt_cache: if provider == Provider::OpenAi {
                parse_openai_prompt_cache()?
            } else {
                None
            },
            anthropic_prompt_cache: if provider == Provider::Anthropic {
                parse_anthropic_prompt_cache()?
            } else {
                None
            },
        })
    }
}

pub(crate) fn required_string_any(names: &[&str]) -> Result<String> {
    for name in names {
        if let Some(value) = config_env::optional_string(name) {
            return Ok(value);
        }
    }

    Err(ConfigError::configuration(format!(
        "{} must be set",
        names.join(" or ")
    )))
}

pub(crate) fn optional_string_any(names: &[&str]) -> Option<String> {
    names
        .iter()
        .find_map(|name| config_env::optional_string(name))
}

pub(crate) fn build_client_config(
    shared: &SharedClientEnvConfig,
    model: String,
    vertex_project_error: &'static str,
) -> Result<ClientConfig> {
    Ok(match shared.provider {
        Provider::Anthropic => {
            let mut config = AnthropicConfig::new(shared.api_key.clone(), model)
                .map_err(ConfigError::from)?
                .with_timeout_secs(shared.timeout_secs)
                .map_err(ConfigError::from)?;
            if let Some(prompt_cache) = shared.anthropic_prompt_cache.clone() {
                config = config.with_prompt_cache(prompt_cache);
            }
            ClientConfig::Anthropic(config)
        }
        Provider::OpenAi => {
            let mut config = OpenAiConfig::new(shared.api_key.clone(), model)
                .map_err(ConfigError::from)?
                .with_timeout_secs(shared.timeout_secs)
                .map_err(ConfigError::from)?;
            if let Some(base_url) = shared.base_url.clone() {
                config = config.with_base_url(base_url).map_err(ConfigError::from)?;
            }
            if let Some(prompt_cache) = shared.openai_prompt_cache.clone() {
                config = config.with_prompt_cache(prompt_cache);
            }
            ClientConfig::OpenAi(config)
        }
        Provider::Gemini => {
            let mut config = GeminiConfig::new(shared.api_key.clone(), model)
                .map_err(ConfigError::from)?
                .with_timeout_secs(shared.timeout_secs)
                .map_err(ConfigError::from)?;
            if let Some(base_url) = shared.base_url.clone() {
                config = config.with_base_url(base_url).map_err(ConfigError::from)?;
            }
            ClientConfig::Gemini(config)
        }
        Provider::Vertex => {
            let project = shared
                .vertex_project
                .clone()
                .ok_or_else(|| ConfigError::configuration(vertex_project_error))?;
            let mut config = VertexConfig::new(shared.api_key.clone(), model, project)
                .map_err(ConfigError::from)?
                .with_timeout_secs(shared.timeout_secs)
                .map_err(ConfigError::from)?;
            if let Some(location) = shared.vertex_location.clone() {
                config = config.with_location(location).map_err(ConfigError::from)?;
            }
            if let Some(base_url) = shared.base_url.clone() {
                config = config.with_base_url(base_url).map_err(ConfigError::from)?;
            }
            ClientConfig::Vertex(config)
        }
    })
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
        Err(env::VarError::NotPresent) => Ok(DEFAULT_TIMEOUT_SECS),
        Err(err) => Err(ConfigError::configuration(format!(
            "LLM_TIMEOUT_SECS must be set: {err}"
        ))),
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
        Err(env::VarError::NotPresent) => Ok(false),
        Err(err) => Err(ConfigError::configuration(format!(
            "LLM_PROMPT_CACHE_ENABLED must be set: {err}"
        ))),
    }
}

fn parse_openai_prompt_cache() -> Result<Option<OpenAiPromptCacheConfig>> {
    if !prompt_cache_enabled()? {
        return Ok(None);
    }

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
        Err(env::VarError::NotPresent) => None,
        Err(err) => {
            return Err(ConfigError::configuration(format!(
                "OPENAI_PROMPT_CACHE_RETENTION must be set: {err}"
            )));
        }
    };

    let mut config = OpenAiPromptCacheConfig::new();
    if let Some(key) = config_env::optional_string("OPENAI_PROMPT_CACHE_KEY") {
        config = config.with_key(key);
    }
    if let Some(retention) = retention {
        config = config.with_retention(retention);
    }
    Ok(Some(config))
}

fn parse_anthropic_prompt_cache() -> Result<Option<AnthropicPromptCacheConfig>> {
    if !prompt_cache_enabled()? {
        return Ok(None);
    }

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
        Err(env::VarError::NotPresent) => None,
        Err(err) => {
            return Err(ConfigError::configuration(format!(
                "ANTHROPIC_PROMPT_CACHE_TTL must be set: {err}"
            )));
        }
    };

    let mut config = AnthropicPromptCacheConfig::new();
    if let Some(ttl) = ttl {
        config = config.with_ttl(ttl);
    }
    Ok(Some(config))
}
