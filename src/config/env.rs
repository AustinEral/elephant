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

pub fn optional_string(name: &str, kind: ConfigErrorKind) -> Result<Option<String>> {
    match env::var(name) {
        Ok(value) => Ok(Some(value)),
        Err(env::VarError::NotPresent) => Ok(None),
        Err(err) => Err(build_error(kind, format!("{name} must be set: {err}"))),
    }
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

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use std::ffi::OsString;
    use std::os::unix::ffi::OsStringExt;
    use std::sync::{Mutex, OnceLock};

    fn env_lock() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
    }

    #[test]
    fn optional_string_rejects_invalid_unicode() {
        let _guard = env_lock().lock().unwrap();
        let name = "INVALID_UNICODE_OPTIONAL_STRING";
        let invalid = OsString::from_vec(vec![0xff, 0xfe, 0xfd]);

        unsafe {
            env::set_var(name, &invalid);
        }

        let err = optional_string(name, ConfigErrorKind::Configuration).unwrap_err();
        assert_eq!(err.kind(), ConfigErrorKind::Configuration);
        assert!(err.to_string().contains(name));

        unsafe {
            env::remove_var(name);
        }
    }
}
