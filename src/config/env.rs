//! Shared environment parsing helpers for typed config.

use std::env;

use super::error::{ConfigError, ConfigErrorKind, Result};

fn build_error(kind: ConfigErrorKind, message: impl Into<String>) -> ConfigError {
    match kind {
        ConfigErrorKind::Configuration => ConfigError::configuration(message),
        ConfigErrorKind::Internal => ConfigError::internal(message),
    }
}

pub fn required_string(name: &'static str, kind: ConfigErrorKind) -> Result<String> {
    env::var(name).map_err(|err| build_error(kind, format!("{name} must be set: {err}")))
}

pub fn optional_string(name: &str) -> Option<String> {
    env::var(name).ok()
}

pub fn optional_usize_lossy(name: &str) -> Option<usize> {
    env::var(name)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
}

pub fn parse_optional_bool(name: &'static str, kind: ConfigErrorKind) -> Result<Option<bool>> {
    match env::var(name) {
        Ok(value) => match value.trim().to_ascii_lowercase().as_str() {
            "1" | "true" | "yes" | "on" => Ok(Some(true)),
            "0" | "false" | "no" | "off" => Ok(Some(false)),
            other => Err(build_error(
                kind,
                format!("{name} must be a boolean, got: {other}"),
            )),
        },
        Err(env::VarError::NotPresent) => Ok(None),
        Err(err) => Err(build_error(kind, format!("{name} must be set: {err}"))),
    }
}

pub fn parse_optional_usize(name: &'static str, kind: ConfigErrorKind) -> Result<Option<usize>> {
    match env::var(name) {
        Ok(value) => value.parse::<usize>().map(Some).map_err(|_| {
            build_error(
                kind,
                format!("{name} must be an unsigned integer, got: {value}"),
            )
        }),
        Err(env::VarError::NotPresent) => Ok(None),
        Err(err) => Err(build_error(kind, format!("{name} must be set: {err}"))),
    }
}

pub fn parse_optional_u64(name: &'static str, kind: ConfigErrorKind) -> Result<Option<u64>> {
    match env::var(name) {
        Ok(value) => value.parse::<u64>().map(Some).map_err(|_| {
            build_error(
                kind,
                format!("{name} must be an unsigned integer, got: {value}"),
            )
        }),
        Err(env::VarError::NotPresent) => Ok(None),
        Err(err) => Err(build_error(kind, format!("{name} must be set: {err}"))),
    }
}
