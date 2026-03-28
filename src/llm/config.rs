//! Provider selection and client configuration for the `llm` module.

#[cfg(test)]
use std::env;
use std::str::FromStr;

use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};

/// Default HTTP timeout for LLM API requests (seconds).
pub const DEFAULT_TIMEOUT_SECS: u64 = 600;

/// Prompt caching configuration for a concrete provider client.
#[cfg(test)]
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
enum PromptCacheConfig {
    /// Disable prompt caching.
    Disabled,
    /// OpenAI prompt caching configuration.
    OpenAi(OpenAiPromptCacheConfig),
    /// Anthropic prompt caching configuration.
    Anthropic(AnthropicPromptCacheConfig),
}

/// OpenAI prompt caching settings.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct OpenAiPromptCacheConfig {
    key: Option<String>,
    retention: Option<OpenAiPromptCacheRetention>,
}

impl OpenAiPromptCacheConfig {
    /// Create an empty OpenAI prompt-cache configuration.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the optional routing hint used for cache locality.
    pub fn with_key(mut self, key: impl Into<String>) -> Self {
        self.key = Some(key.into());
        self
    }

    /// Set the optional retention policy.
    pub fn with_retention(mut self, retention: OpenAiPromptCacheRetention) -> Self {
        self.retention = Some(retention);
        self
    }

    /// Return the optional routing hint.
    pub fn key(&self) -> Option<&str> {
        self.key.as_deref()
    }

    /// Return the optional retention policy.
    pub fn retention(&self) -> Option<OpenAiPromptCacheRetention> {
        self.retention
    }
}

/// OpenAI prompt cache retention policy.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum OpenAiPromptCacheRetention {
    /// Retain in memory.
    #[serde(rename = "in_memory")]
    InMemory,
    /// Retain for 24 hours.
    #[serde(rename = "24h")]
    Hours24,
}

/// Anthropic prompt caching settings.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct AnthropicPromptCacheConfig {
    ttl: Option<AnthropicPromptCacheTtl>,
}

impl AnthropicPromptCacheConfig {
    /// Create an empty Anthropic prompt-cache configuration.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the optional cache TTL.
    pub fn with_ttl(mut self, ttl: AnthropicPromptCacheTtl) -> Self {
        self.ttl = Some(ttl);
        self
    }

    /// Return the optional cache TTL.
    pub fn ttl(&self) -> Option<AnthropicPromptCacheTtl> {
        self.ttl
    }
}

/// Anthropic prompt cache TTL.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum AnthropicPromptCacheTtl {
    /// Five minute TTL.
    #[serde(rename = "5m")]
    Minutes5,
    /// One hour TTL.
    #[serde(rename = "1h")]
    Hours1,
}

/// LLM provider selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Provider {
    /// Anthropic Claude API.
    Anthropic,
    /// OpenAI Responses API.
    OpenAi,
    /// Google Gemini API.
    Gemini,
    /// Google Gemini on Vertex AI.
    Vertex,
}

impl Provider {
    /// Returns the canonical environment value for this provider.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Anthropic => "anthropic",
            Self::OpenAi => "openai",
            Self::Gemini => "gemini",
            Self::Vertex => "vertex",
        }
    }
}

impl FromStr for Provider {
    type Err = Error;

    fn from_str(value: &str) -> Result<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "anthropic" => Ok(Self::Anthropic),
            "openai" => Ok(Self::OpenAi),
            "gemini" => Ok(Self::Gemini),
            "vertex" => Ok(Self::Vertex),
            other => Err(Error::Configuration(format!(
                "LLM provider must be one of: anthropic, openai, gemini, vertex; got: {other}"
            ))),
        }
    }
}

/// Validated client configuration for one provider implementation.
#[derive(Debug, Clone)]
pub enum ClientConfig {
    /// Anthropic client configuration.
    Anthropic(AnthropicConfig),
    /// OpenAI client configuration.
    OpenAi(OpenAiConfig),
    /// Gemini client configuration.
    Gemini(GeminiConfig),
    /// Vertex AI Gemini client configuration.
    Vertex(VertexConfig),
}

impl ClientConfig {
    /// Return the selected provider.
    pub fn provider(&self) -> Provider {
        match self {
            Self::Anthropic(_) => Provider::Anthropic,
            Self::OpenAi(_) => Provider::OpenAi,
            Self::Gemini(_) => Provider::Gemini,
            Self::Vertex(_) => Provider::Vertex,
        }
    }

    /// Return the configured model label.
    pub fn model(&self) -> &str {
        match self {
            Self::Anthropic(config) => config.model(),
            Self::OpenAi(config) => config.model(),
            Self::Gemini(config) => config.model(),
            Self::Vertex(config) => config.model(),
        }
    }

    /// Return a stable `provider/model` label.
    pub fn label(&self) -> String {
        format!("{}/{}", self.provider().as_str(), self.model())
    }
}

/// Anthropic-specific client configuration.
#[derive(Debug, Clone)]
pub struct AnthropicConfig {
    api_key: String,
    model: String,
    timeout_secs: u64,
    prompt_cache: Option<AnthropicPromptCacheConfig>,
}

impl AnthropicConfig {
    /// Create a new Anthropic client configuration.
    pub fn new(api_key: impl Into<String>, model: impl Into<String>) -> Result<Self> {
        Ok(Self {
            api_key: validate_nonempty("api_key", api_key.into())?,
            model: validate_nonempty("model", model.into())?,
            timeout_secs: DEFAULT_TIMEOUT_SECS,
            prompt_cache: None,
        })
    }

    /// Set the HTTP timeout in seconds.
    pub fn with_timeout_secs(mut self, timeout_secs: u64) -> Result<Self> {
        self.timeout_secs = validate_timeout_secs(timeout_secs)?;
        Ok(self)
    }

    /// Enable Anthropic prompt caching.
    pub fn with_prompt_cache(mut self, prompt_cache: AnthropicPromptCacheConfig) -> Self {
        self.prompt_cache = Some(prompt_cache);
        self
    }

    /// Return the configured API key.
    pub fn api_key(&self) -> &str {
        &self.api_key
    }

    /// Return the configured model.
    pub fn model(&self) -> &str {
        &self.model
    }

    /// Return the configured timeout.
    pub fn timeout_secs(&self) -> u64 {
        self.timeout_secs
    }

    /// Return the prompt-cache configuration, if enabled.
    pub fn prompt_cache(&self) -> Option<&AnthropicPromptCacheConfig> {
        self.prompt_cache.as_ref()
    }
}

/// OpenAI-specific client configuration.
#[derive(Debug, Clone)]
pub struct OpenAiConfig {
    api_key: String,
    model: String,
    base_url: Option<String>,
    timeout_secs: u64,
    prompt_cache: Option<OpenAiPromptCacheConfig>,
}

/// Gemini-specific client configuration.
#[derive(Debug, Clone)]
pub struct GeminiConfig {
    api_key: String,
    model: String,
    base_url: Option<String>,
    timeout_secs: u64,
    supports_function_call_ids: bool,
}

/// Vertex AI-specific client configuration.
#[derive(Debug, Clone)]
pub struct VertexConfig {
    api_key: String,
    model: String,
    project: String,
    location: String,
    base_url: Option<String>,
    timeout_secs: u64,
}

impl GeminiConfig {
    /// Create a new Gemini client configuration.
    pub fn new(api_key: impl Into<String>, model: impl Into<String>) -> Result<Self> {
        Ok(Self {
            api_key: validate_nonempty("api_key", api_key.into())?,
            model: validate_nonempty("model", model.into())?,
            base_url: None,
            timeout_secs: DEFAULT_TIMEOUT_SECS,
            supports_function_call_ids: true,
        })
    }

    /// Override the API base URL.
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Result<Self> {
        self.base_url = Some(validate_nonempty("base_url", base_url.into())?);
        Ok(self)
    }

    /// Set the HTTP timeout in seconds.
    pub fn with_timeout_secs(mut self, timeout_secs: u64) -> Result<Self> {
        self.timeout_secs = validate_timeout_secs(timeout_secs)?;
        Ok(self)
    }

    /// Control whether function-call IDs are sent in tool history parts.
    pub(crate) fn with_function_call_ids(mut self, enabled: bool) -> Self {
        self.supports_function_call_ids = enabled;
        self
    }

    /// Return the configured API key.
    pub fn api_key(&self) -> &str {
        &self.api_key
    }

    /// Return the configured model.
    pub fn model(&self) -> &str {
        &self.model
    }

    /// Return the optional base URL override.
    pub fn base_url(&self) -> Option<&str> {
        self.base_url.as_deref()
    }

    /// Return the configured timeout.
    pub fn timeout_secs(&self) -> u64 {
        self.timeout_secs
    }

    /// Return whether the transport supports function-call IDs in request history.
    pub(crate) fn supports_function_call_ids(&self) -> bool {
        self.supports_function_call_ids
    }
}

impl VertexConfig {
    /// Default Vertex AI location.
    pub const DEFAULT_LOCATION: &'static str = "global";

    /// Create a new Vertex AI client configuration.
    pub fn new(
        api_key: impl Into<String>,
        model: impl Into<String>,
        project: impl Into<String>,
    ) -> Result<Self> {
        Ok(Self {
            api_key: validate_nonempty("api_key", api_key.into())?,
            model: validate_nonempty("model", model.into())?,
            project: validate_nonempty("project", project.into())?,
            location: Self::DEFAULT_LOCATION.into(),
            base_url: None,
            timeout_secs: DEFAULT_TIMEOUT_SECS,
        })
    }

    /// Override the Vertex project location.
    pub fn with_location(mut self, location: impl Into<String>) -> Result<Self> {
        self.location = validate_nonempty("location", location.into())?;
        Ok(self)
    }

    /// Override the API base URL.
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Result<Self> {
        self.base_url = Some(validate_nonempty("base_url", base_url.into())?);
        Ok(self)
    }

    /// Set the HTTP timeout in seconds.
    pub fn with_timeout_secs(mut self, timeout_secs: u64) -> Result<Self> {
        self.timeout_secs = validate_timeout_secs(timeout_secs)?;
        Ok(self)
    }

    /// Return the configured API key.
    pub fn api_key(&self) -> &str {
        &self.api_key
    }

    /// Return the configured model.
    pub fn model(&self) -> &str {
        &self.model
    }

    /// Return the Google Cloud project ID.
    pub fn project(&self) -> &str {
        &self.project
    }

    /// Return the Vertex location.
    pub fn location(&self) -> &str {
        &self.location
    }

    /// Return the optional base URL override.
    pub fn base_url(&self) -> Option<&str> {
        self.base_url.as_deref()
    }

    /// Return the configured timeout.
    pub fn timeout_secs(&self) -> u64 {
        self.timeout_secs
    }
}

impl OpenAiConfig {
    /// Create a new OpenAI client configuration.
    pub fn new(api_key: impl Into<String>, model: impl Into<String>) -> Result<Self> {
        Ok(Self {
            api_key: validate_nonempty("api_key", api_key.into())?,
            model: validate_nonempty("model", model.into())?,
            base_url: None,
            timeout_secs: DEFAULT_TIMEOUT_SECS,
            prompt_cache: None,
        })
    }

    /// Override the API base URL.
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Result<Self> {
        self.base_url = Some(validate_nonempty("base_url", base_url.into())?);
        Ok(self)
    }

    /// Set the HTTP timeout in seconds.
    pub fn with_timeout_secs(mut self, timeout_secs: u64) -> Result<Self> {
        self.timeout_secs = validate_timeout_secs(timeout_secs)?;
        Ok(self)
    }

    /// Enable OpenAI prompt caching.
    pub fn with_prompt_cache(mut self, prompt_cache: OpenAiPromptCacheConfig) -> Self {
        self.prompt_cache = Some(prompt_cache);
        self
    }

    /// Return the configured API key.
    pub fn api_key(&self) -> &str {
        &self.api_key
    }

    /// Return the configured model.
    pub fn model(&self) -> &str {
        &self.model
    }

    /// Return the optional base URL override.
    pub fn base_url(&self) -> Option<&str> {
        self.base_url.as_deref()
    }

    /// Return the configured timeout.
    pub fn timeout_secs(&self) -> u64 {
        self.timeout_secs
    }

    /// Return the prompt-cache configuration, if enabled.
    pub fn prompt_cache(&self) -> Option<&OpenAiPromptCacheConfig> {
        self.prompt_cache.as_ref()
    }
}

/// Resolved LLM configuration for runtime construction.
#[derive(Debug, Clone)]
pub struct LlmConfig {
    retain: ClientConfig,
    reflect: ClientConfig,
}

impl LlmConfig {
    /// Create a new runtime LLM configuration.
    pub fn new(retain: ClientConfig, reflect: ClientConfig) -> Self {
        Self { retain, reflect }
    }

    /// Return the retain-stage client configuration.
    pub fn retain(&self) -> &ClientConfig {
        &self.retain
    }

    /// Return the reflect-stage client configuration.
    pub fn reflect(&self) -> &ClientConfig {
        &self.reflect
    }
}

fn validate_nonempty(name: &str, value: String) -> Result<String> {
    if value.trim().is_empty() {
        Err(Error::Configuration(format!("{name} must not be empty")))
    } else {
        Ok(value)
    }
}

fn validate_timeout_secs(timeout_secs: u64) -> Result<u64> {
    if timeout_secs == 0 {
        Err(Error::Configuration(
            "LLM_TIMEOUT_SECS must be greater than zero".into(),
        ))
    } else {
        Ok(timeout_secs)
    }
}

#[cfg(test)]
fn env_bool(name: &str, default: bool) -> Result<bool> {
    match env::var(name) {
        Ok(value) => match value.trim().to_ascii_lowercase().as_str() {
            "1" | "true" | "yes" | "on" => Ok(true),
            "0" | "false" | "no" | "off" => Ok(false),
            other => Err(Error::Configuration(format!(
                "{name} must be a boolean, got: {other}"
            ))),
        },
        Err(_) => Ok(default),
    }
}

#[cfg(test)]
fn required_env_any(names: &[&str]) -> Result<String> {
    for name in names {
        if let Ok(value) = env::var(name) {
            return Ok(value);
        }
    }
    Err(Error::Configuration(format!(
        "{} must be set",
        names.join(" or ")
    )))
}

#[cfg(test)]
fn optional_env_any(names: &[&str]) -> Option<String> {
    for name in names {
        if let Ok(value) = env::var(name) {
            return Some(value);
        }
    }
    None
}

#[cfg(test)]
fn timeout_secs_from_env() -> Result<u64> {
    match env::var("LLM_TIMEOUT_SECS") {
        Ok(value) => validate_timeout_secs(value.parse().map_err(|_| {
            Error::Configuration(format!(
                "LLM_TIMEOUT_SECS must be a positive integer, got: {value}"
            ))
        })?),
        Err(_) => Ok(DEFAULT_TIMEOUT_SECS),
    }
}

#[cfg(test)]
fn vertex_project_from_env(names: &[&str]) -> Result<String> {
    required_env_any(names)
}

#[cfg(test)]
fn vertex_location_from_env(names: &[&str]) -> Result<String> {
    match optional_env_any(names) {
        Some(value) => validate_nonempty("location", value),
        None => Ok(VertexConfig::DEFAULT_LOCATION.into()),
    }
}

#[cfg(test)]
fn prompt_cache_config_from_env(provider: Provider) -> Result<PromptCacheConfig> {
    if !env_bool("LLM_PROMPT_CACHE_ENABLED", false)? {
        return Ok(PromptCacheConfig::Disabled);
    }

    match provider {
        Provider::OpenAi => {
            let retention = match env::var("OPENAI_PROMPT_CACHE_RETENTION") {
                Ok(value) => match value.trim().to_ascii_lowercase().as_str() {
                    "in_memory" | "in-memory" => Some(OpenAiPromptCacheRetention::InMemory),
                    "24h" => Some(OpenAiPromptCacheRetention::Hours24),
                    other => {
                        return Err(Error::Configuration(format!(
                            "OPENAI_PROMPT_CACHE_RETENTION must be one of: in_memory, in-memory, 24h; got: {other}"
                        )));
                    }
                },
                Err(_) => None,
            };

            let mut config = OpenAiPromptCacheConfig::new();
            if let Some(key) = env::var("OPENAI_PROMPT_CACHE_KEY").ok() {
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
                        return Err(Error::Configuration(format!(
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

#[cfg(test)]
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
    match provider {
        Provider::Anthropic => {
            let mut config =
                AnthropicConfig::new(api_key, model)?.with_timeout_secs(timeout_secs)?;
            if let PromptCacheConfig::Anthropic(prompt_cache) = prompt_cache {
                config = config.with_prompt_cache(prompt_cache);
            }
            Ok(ClientConfig::Anthropic(config))
        }
        Provider::OpenAi => {
            let mut config = OpenAiConfig::new(api_key, model)?.with_timeout_secs(timeout_secs)?;
            if let Some(base_url) = base_url {
                config = config.with_base_url(base_url)?;
            }
            if let PromptCacheConfig::OpenAi(prompt_cache) = prompt_cache {
                config = config.with_prompt_cache(prompt_cache);
            }
            Ok(ClientConfig::OpenAi(config))
        }
        Provider::Gemini => {
            let mut config = GeminiConfig::new(api_key, model)?.with_timeout_secs(timeout_secs)?;
            if let Some(base_url) = base_url {
                config = config.with_base_url(base_url)?;
            }
            Ok(ClientConfig::Gemini(config))
        }
        Provider::Vertex => {
            let project = vertex_project.ok_or_else(|| {
                Error::Configuration(
                    "LLM_VERTEX_PROJECT or GOOGLE_CLOUD_PROJECT must be set for provider=vertex"
                        .into(),
                )
            })?;
            let mut config =
                VertexConfig::new(api_key, model, project)?.with_timeout_secs(timeout_secs)?;
            if let Some(location) = vertex_location {
                config = config.with_location(location)?;
            }
            if let Some(base_url) = base_url {
                config = config.with_base_url(base_url)?;
            }
            Ok(ClientConfig::Vertex(config))
        }
    }
}

/// Load a single validated client configuration from one or more environment-variable names.
#[cfg(test)]
fn client_config_from_env_vars(
    provider_vars: &[&str],
    api_key_vars: &[&str],
    model_vars: &[&str],
    vertex_project_vars: &[&str],
    vertex_location_vars: &[&str],
    base_url_vars: &[&str],
) -> Result<ClientConfig> {
    let provider = required_env_any(provider_vars)?.parse::<Provider>()?;
    let api_key = required_env_any(api_key_vars)?;
    let model = required_env_any(model_vars)?;
    let vertex_project = optional_env_any(vertex_project_vars);
    let vertex_location = optional_env_any(vertex_location_vars);
    let base_url = optional_env_any(base_url_vars);
    let timeout_secs = timeout_secs_from_env()?;
    let prompt_cache = prompt_cache_config_from_env(provider)?;

    build_client_config(
        provider,
        api_key,
        model,
        vertex_project,
        vertex_location,
        base_url,
        timeout_secs,
        prompt_cache,
    )
}

#[cfg(test)]
/// Build the retain and reflect client configs from the standard runtime environment.
fn runtime_config_from_env() -> Result<LlmConfig> {
    let provider = required_env_any(&["LLM_PROVIDER"])?.parse::<Provider>()?;
    let api_key = required_env_any(&["LLM_API_KEY"])?;
    let retain_model = required_env_any(&["RETAIN_LLM_MODEL", "LLM_MODEL"])?;
    let reflect_model = required_env_any(&["REFLECT_LLM_MODEL", "LLM_MODEL"])?;
    let vertex_project = if provider == Provider::Vertex {
        Some(vertex_project_from_env(&[
            "LLM_VERTEX_PROJECT",
            "GOOGLE_CLOUD_PROJECT",
        ])?)
    } else {
        None
    };
    let vertex_location = if provider == Provider::Vertex {
        Some(vertex_location_from_env(&[
            "LLM_VERTEX_LOCATION",
            "GOOGLE_CLOUD_LOCATION",
        ])?)
    } else {
        None
    };
    let base_url = optional_env_any(&["LLM_BASE_URL"]);
    let timeout_secs = timeout_secs_from_env()?;
    let prompt_cache = prompt_cache_config_from_env(provider)?;

    let retain = build_client_config(
        provider,
        api_key.clone(),
        retain_model,
        vertex_project.clone(),
        vertex_location.clone(),
        base_url.clone(),
        timeout_secs,
        prompt_cache.clone(),
    )?;

    let reflect = build_client_config(
        provider,
        api_key,
        reflect_model,
        vertex_project,
        vertex_location,
        base_url,
        timeout_secs,
        prompt_cache,
    )?;

    Ok(LlmConfig::new(retain, reflect))
}

#[cfg(test)]
/// Build the benchmark judge client configuration from the judge fallback chain.
fn judge_client_config_from_env(
    override_provider: Option<Provider>,
    override_model: Option<&str>,
) -> Result<ClientConfig> {
    let provider = match override_provider {
        Some(provider) => provider,
        None => required_env_any(&["JUDGE_PROVIDER", "LLM_PROVIDER"])?.parse::<Provider>()?,
    };
    let api_key = required_env_any(&["JUDGE_API_KEY", "LLM_API_KEY"])?;
    let model = override_model
        .map(str::to_string)
        .map(Ok)
        .unwrap_or_else(|| required_env_any(&["JUDGE_MODEL", "LLM_MODEL"]))?;
    let vertex_project = if provider == Provider::Vertex {
        Some(vertex_project_from_env(&[
            "JUDGE_VERTEX_PROJECT",
            "LLM_VERTEX_PROJECT",
            "GOOGLE_CLOUD_PROJECT",
        ])?)
    } else {
        None
    };
    let vertex_location = if provider == Provider::Vertex {
        Some(vertex_location_from_env(&[
            "JUDGE_VERTEX_LOCATION",
            "LLM_VERTEX_LOCATION",
            "GOOGLE_CLOUD_LOCATION",
        ])?)
    } else {
        None
    };
    let base_url = optional_env_any(&["JUDGE_BASE_URL", "LLM_BASE_URL"]);
    let timeout_secs = timeout_secs_from_env()?;
    let prompt_cache = prompt_cache_config_from_env(provider)?;

    build_client_config(
        provider,
        api_key,
        model,
        vertex_project,
        vertex_location,
        base_url,
        timeout_secs,
        prompt_cache,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;
    use std::sync::{Mutex, OnceLock};

    fn env_lock() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
    }

    fn with_env_vars<F>(overrides: &[(&str, Option<&str>)], test: F)
    where
        F: FnOnce(),
    {
        let _guard = env_lock().lock().unwrap();
        let keys = [
            "LLM_PROVIDER",
            "LLM_API_KEY",
            "LLM_MODEL",
            "RETAIN_LLM_MODEL",
            "REFLECT_LLM_MODEL",
            "LLM_VERTEX_PROJECT",
            "LLM_VERTEX_LOCATION",
            "GOOGLE_CLOUD_PROJECT",
            "GOOGLE_CLOUD_LOCATION",
            "LLM_BASE_URL",
            "JUDGE_PROVIDER",
            "JUDGE_API_KEY",
            "JUDGE_MODEL",
            "JUDGE_VERTEX_PROJECT",
            "JUDGE_VERTEX_LOCATION",
            "JUDGE_BASE_URL",
            "LLM_PROMPT_CACHE_ENABLED",
            "OPENAI_PROMPT_CACHE_KEY",
            "OPENAI_PROMPT_CACHE_RETENTION",
            "ANTHROPIC_PROMPT_CACHE_TTL",
            "LLM_TIMEOUT_SECS",
        ];

        let mut original = BTreeMap::new();
        for key in keys {
            original.insert(key, std::env::var(key).ok());
            unsafe { std::env::remove_var(key) };
        }

        for (key, value) in overrides {
            match value {
                Some(value) => unsafe { std::env::set_var(key, value) },
                None => unsafe { std::env::remove_var(key) },
            }
        }

        test();

        for (key, value) in original {
            match value {
                Some(value) => unsafe { std::env::set_var(key, value) },
                None => unsafe { std::env::remove_var(key) },
            }
        }
    }

    #[test]
    fn provider_rejects_removed_openai_responses_alias() {
        let err = "openai-responses".parse::<Provider>().unwrap_err();
        assert!(matches!(err, Error::Configuration(_)));
    }

    #[test]
    fn runtime_config_loads_openai() {
        with_env_vars(
            &[
                ("LLM_PROVIDER", Some("openai")),
                ("LLM_API_KEY", Some("sk-test")),
                ("RETAIN_LLM_MODEL", Some("gpt-5.1")),
                ("REFLECT_LLM_MODEL", Some("gpt-5.1-mini")),
                ("LLM_BASE_URL", Some("https://example.com/v1")),
                ("LLM_PROMPT_CACHE_ENABLED", Some("1")),
                ("OPENAI_PROMPT_CACHE_KEY", Some("elephant:reflect")),
            ],
            || {
                let config = runtime_config_from_env().unwrap();
                assert_eq!(config.retain().provider(), Provider::OpenAi);
                assert_eq!(config.reflect().provider(), Provider::OpenAi);
            },
        );
    }

    #[test]
    fn runtime_config_loads_gemini() {
        with_env_vars(
            &[
                ("LLM_PROVIDER", Some("gemini")),
                ("LLM_API_KEY", Some("gemini-test-key")),
                ("RETAIN_LLM_MODEL", Some("gemini-2.5-flash")),
                ("REFLECT_LLM_MODEL", Some("gemini-2.5-pro")),
                (
                    "LLM_BASE_URL",
                    Some("https://generativelanguage.googleapis.com/v1beta"),
                ),
            ],
            || {
                let config = runtime_config_from_env().unwrap();
                assert_eq!(config.retain().provider(), Provider::Gemini);
                assert_eq!(config.reflect().provider(), Provider::Gemini);
            },
        );
    }

    #[test]
    fn timeout_rejects_zero() {
        with_env_vars(&[("LLM_TIMEOUT_SECS", Some("0"))], || {
            let err = client_config_from_env_vars(
                &["LLM_PROVIDER"],
                &["LLM_API_KEY"],
                &["LLM_MODEL"],
                &[],
                &[],
                &[],
            )
            .unwrap_err();
            assert!(matches!(err, Error::Configuration(_)));
        });
    }

    #[test]
    fn runtime_config_loads_vertex() {
        with_env_vars(
            &[
                ("LLM_PROVIDER", Some("vertex")),
                ("LLM_API_KEY", Some("vertex-test-key")),
                ("LLM_VERTEX_PROJECT", Some("vertex-project")),
                ("LLM_VERTEX_LOCATION", Some("us-central1")),
                ("RETAIN_LLM_MODEL", Some("gemini-3.1-pro-preview")),
                ("REFLECT_LLM_MODEL", Some("gemini-3.1-pro-preview")),
            ],
            || {
                let config = runtime_config_from_env().unwrap();
                assert_eq!(config.retain().provider(), Provider::Vertex);
                assert_eq!(config.reflect().provider(), Provider::Vertex);
            },
        );
    }

    #[test]
    fn judge_config_can_override_provider_and_base_url() {
        with_env_vars(
            &[
                ("LLM_PROVIDER", Some("anthropic")),
                ("LLM_API_KEY", Some("sk-test")),
                ("LLM_MODEL", Some("claude-sonnet")),
                ("LLM_BASE_URL", Some("https://anthropic.invalid")),
                ("JUDGE_BASE_URL", Some("https://openai.invalid/v1")),
            ],
            || {
                let config =
                    judge_client_config_from_env(Some(Provider::OpenAi), Some("gpt-4o")).unwrap();
                match config {
                    ClientConfig::OpenAi(config) => {
                        assert_eq!(config.model(), "gpt-4o");
                        assert_eq!(config.base_url(), Some("https://openai.invalid/v1"));
                    }
                    other => panic!("expected openai config, got {other:?}"),
                }
            },
        );
    }
}
