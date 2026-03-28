//! Typed configuration for benchmark startup policy.

use crate::llm::DeterminismRequirement;

use super::error::{ConfigError, Result};

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
}
