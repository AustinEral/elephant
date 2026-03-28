//! Typed configuration for server startup and server-only policy.

use std::net::SocketAddr;

use super::env as config_env;
use super::error::{ConfigError, ConfigErrorKind, Result};

/// Supported logging formats for the HTTP/MCP server binary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogFormat {
    /// Human-readable plain text logs.
    Text,
    /// JSON-formatted structured logs.
    Json,
}

impl std::str::FromStr for LogFormat {
    type Err = ConfigError;

    fn from_str(value: &str) -> std::result::Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "text" => Ok(Self::Text),
            "json" => Ok(Self::Json),
            other => Err(ConfigError::configuration(format!(
                "LOG_FORMAT must be one of: text, json; got: {other}"
            ))),
        }
    }
}

/// Validated server startup configuration loaded from environment.
#[derive(Debug, Clone)]
pub struct ServerConfig {
    listen_addr: SocketAddr,
    log_format: LogFormat,
    background_consolidation: BackgroundConsolidationConfig,
}

impl ServerConfig {
    /// Create server startup configuration from validated values.
    pub fn new(
        listen_addr: SocketAddr,
        log_format: LogFormat,
        background_consolidation: BackgroundConsolidationConfig,
    ) -> Self {
        Self {
            listen_addr,
            log_format,
            background_consolidation,
        }
    }

    /// Load server startup configuration from the process environment.
    pub fn from_env() -> Result<Self> {
        let listen_addr =
            match config_env::optional_string("LISTEN_ADDR", ConfigErrorKind::Configuration)? {
                Some(raw) => raw.parse::<SocketAddr>().map_err(|_| {
                    ConfigError::configuration(format!(
                        "LISTEN_ADDR must be a valid socket address, got: {raw}"
                    ))
                })?,
                None => SocketAddr::from(([0, 0, 0, 0], 3001)),
            };
        let log_format =
            match config_env::optional_string("LOG_FORMAT", ConfigErrorKind::Configuration)? {
                Some(raw) => raw.parse::<LogFormat>()?,
                None => LogFormat::Text,
            };

        Ok(Self::new(
            listen_addr,
            log_format,
            BackgroundConsolidationConfig::from_env()?,
        ))
    }

    /// Return the address the server should bind to.
    pub fn listen_addr(&self) -> SocketAddr {
        self.listen_addr
    }

    /// Return the configured log format.
    pub fn log_format(&self) -> LogFormat {
        self.log_format
    }

    /// Return the background consolidation policy.
    pub fn background_consolidation(&self) -> &BackgroundConsolidationConfig {
        &self.background_consolidation
    }
}

/// Validated server-only background consolidation policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BackgroundConsolidationConfig {
    enabled: bool,
    min_unconsolidated_facts: usize,
    cooldown_secs: u64,
    merge_opinions_after: bool,
}

impl BackgroundConsolidationConfig {
    /// Load the background consolidation policy from the process environment.
    pub fn from_env() -> Result<Self> {
        let enabled = config_env::parse_optional_bool(
            "SERVER_AUTO_CONSOLIDATION",
            ConfigErrorKind::Configuration,
        )?
        .unwrap_or(true);
        let min_unconsolidated_facts = config_env::parse_optional_usize(
            "SERVER_AUTO_CONSOLIDATION_MIN_FACTS",
            ConfigErrorKind::Configuration,
        )?
        .unwrap_or(32);
        let cooldown_secs = config_env::parse_optional_u64(
            "SERVER_AUTO_CONSOLIDATION_COOLDOWN_SECS",
            ConfigErrorKind::Configuration,
        )?
        .unwrap_or(30);
        let merge_opinions_after = config_env::parse_optional_bool(
            "SERVER_AUTO_CONSOLIDATION_MERGE_OPINIONS",
            ConfigErrorKind::Configuration,
        )?
        .unwrap_or(false);

        Self::new(
            enabled,
            min_unconsolidated_facts,
            cooldown_secs,
            merge_opinions_after,
        )
    }

    /// Create a background consolidation policy from validated values.
    pub fn new(
        enabled: bool,
        min_unconsolidated_facts: usize,
        cooldown_secs: u64,
        merge_opinions_after: bool,
    ) -> Result<Self> {
        if min_unconsolidated_facts == 0 {
            return Err(ConfigError::configuration(
                "SERVER_AUTO_CONSOLIDATION_MIN_FACTS must be greater than 0",
            ));
        }

        Ok(Self {
            enabled,
            min_unconsolidated_facts,
            cooldown_secs,
            merge_opinions_after,
        })
    }

    /// Return whether background consolidation is enabled.
    pub fn enabled(&self) -> bool {
        self.enabled
    }

    /// Return the minimum unconsolidated fact threshold.
    pub fn min_unconsolidated_facts(&self) -> usize {
        self.min_unconsolidated_facts
    }

    /// Return the scheduler cooldown in seconds.
    pub fn cooldown_secs(&self) -> u64 {
        self.cooldown_secs
    }

    /// Return whether opinion merge runs after consolidation.
    pub fn merge_opinions_after(&self) -> bool {
        self.merge_opinions_after
    }
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
    fn rejects_zero_minimum_facts() {
        let err = BackgroundConsolidationConfig::new(true, 0, 30, false).unwrap_err();
        assert_eq!(err.kind(), ConfigErrorKind::Configuration);
        assert!(
            err.to_string()
                .contains("SERVER_AUTO_CONSOLIDATION_MIN_FACTS")
        );
    }

    #[test]
    fn server_config_rejects_invalid_log_format() {
        let _guard = env_lock().lock().unwrap();
        unsafe {
            env::set_var("LOG_FORMAT", "yaml");
        }

        let err = ServerConfig::from_env().unwrap_err();
        assert_eq!(err.kind(), ConfigErrorKind::Configuration);
        assert!(err.to_string().contains("LOG_FORMAT"));

        unsafe {
            env::remove_var("LOG_FORMAT");
        }
    }

    #[test]
    fn server_config_rejects_invalid_listen_addr() {
        let _guard = env_lock().lock().unwrap();
        unsafe {
            env::set_var("LISTEN_ADDR", "localhost");
        }

        let err = ServerConfig::from_env().unwrap_err();
        assert_eq!(err.kind(), ConfigErrorKind::Configuration);
        assert!(err.to_string().contains("LISTEN_ADDR"));

        unsafe {
            env::remove_var("LISTEN_ADDR");
        }
    }
}
