use std::env;

use elephant::llm::{
    self, AnthropicConfig, AnthropicPromptCacheConfig, AnthropicPromptCacheTtl, ClientConfig,
    LlmClient, OpenAiConfig, OpenAiPromptCacheConfig, OpenAiPromptCacheRetention, Provider,
};

fn required_env_any(names: &[&str]) -> Result<String, String> {
    for name in names {
        if let Ok(value) = env::var(name) {
            return Ok(value);
        }
    }
    Err(format!("{} must be set", names.join(" or ")))
}

fn optional_env_any(names: &[&str]) -> Option<String> {
    for name in names {
        if let Ok(value) = env::var(name) {
            return Some(value);
        }
    }
    None
}

fn env_bool(name: &str, default: bool) -> Result<bool, String> {
    match env::var(name) {
        Ok(value) => match value.trim().to_ascii_lowercase().as_str() {
            "1" | "true" | "yes" | "on" => Ok(true),
            "0" | "false" | "no" | "off" => Ok(false),
            other => Err(format!("{name} must be a boolean, got: {other}")),
        },
        Err(_) => Ok(default),
    }
}

fn timeout_secs_from_env() -> Result<u64, String> {
    match env::var("LLM_TIMEOUT_SECS") {
        Ok(value) => value
            .parse::<u64>()
            .map_err(|_| format!("LLM_TIMEOUT_SECS must be a positive integer, got: {value}"))
            .and_then(|timeout_secs| {
                if timeout_secs == 0 {
                    Err("LLM_TIMEOUT_SECS must be greater than zero".into())
                } else {
                    Ok(timeout_secs)
                }
            }),
        Err(_) => Ok(llm::DEFAULT_TIMEOUT_SECS),
    }
}

pub fn client_config_from_env(model_vars: &[&str]) -> Result<ClientConfig, String> {
    let provider = required_env_any(&["LLM_PROVIDER"])?
        .parse::<Provider>()
        .map_err(|e| format!("invalid LLM_PROVIDER: {e}"))?;
    let api_key = required_env_any(&["LLM_API_KEY"])?;
    let model = required_env_any(model_vars)?;
    let base_url = optional_env_any(&["LLM_BASE_URL"]);
    let timeout_secs = timeout_secs_from_env()?;
    let prompt_cache_enabled = env_bool("LLM_PROMPT_CACHE_ENABLED", false)?;

    match provider {
        Provider::Anthropic => {
            let mut config = AnthropicConfig::new(api_key, model)
                .map_err(|e| e.to_string())?
                .with_timeout_secs(timeout_secs)
                .map_err(|e| e.to_string())?;
            if prompt_cache_enabled {
                let mut prompt_cache = AnthropicPromptCacheConfig::new();
                if let Ok(ttl) = env::var("ANTHROPIC_PROMPT_CACHE_TTL") {
                    let ttl = match ttl.trim().to_ascii_lowercase().as_str() {
                        "5m" => AnthropicPromptCacheTtl::Minutes5,
                        "1h" => AnthropicPromptCacheTtl::Hours1,
                        other => {
                            return Err(format!(
                                "ANTHROPIC_PROMPT_CACHE_TTL must be one of: 5m, 1h; got: {other}"
                            ));
                        }
                    };
                    prompt_cache = prompt_cache.with_ttl(ttl);
                }
                config = config.with_prompt_cache(prompt_cache);
            }
            Ok(ClientConfig::Anthropic(config))
        }
        Provider::OpenAi => {
            let mut config = OpenAiConfig::new(api_key, model)
                .map_err(|e| e.to_string())?
                .with_timeout_secs(timeout_secs)
                .map_err(|e| e.to_string())?;
            if let Some(base_url) = base_url {
                config = config.with_base_url(base_url).map_err(|e| e.to_string())?;
            }
            if prompt_cache_enabled {
                let mut prompt_cache = OpenAiPromptCacheConfig::new();
                if let Ok(key) = env::var("OPENAI_PROMPT_CACHE_KEY") {
                    prompt_cache = prompt_cache.with_key(key);
                }
                if let Ok(retention) = env::var("OPENAI_PROMPT_CACHE_RETENTION") {
                    let retention = match retention.trim().to_ascii_lowercase().as_str() {
                        "in_memory" | "in-memory" => OpenAiPromptCacheRetention::InMemory,
                        "24h" => OpenAiPromptCacheRetention::Hours24,
                        other => {
                            return Err(format!(
                                "OPENAI_PROMPT_CACHE_RETENTION must be one of: in_memory, in-memory, 24h; got: {other}"
                            ));
                        }
                    };
                    prompt_cache = prompt_cache.with_retention(retention);
                }
                config = config.with_prompt_cache(prompt_cache);
            }
            Ok(ClientConfig::OpenAi(config))
        }
    }
}

pub fn build_client_from_env(model_vars: &[&str]) -> Result<Box<dyn LlmClient>, String> {
    let config = client_config_from_env(model_vars)?;
    llm::build_client(&config).map_err(|e| format!("failed to build LLM client: {e}"))
}
