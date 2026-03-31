use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use serde::Serialize;

use elephant::ConfigError;

type Result<T> = std::result::Result<T, ConfigError>;

const RUNTIME_API_KEY: &str = "ELEPHANT_BENCH_RUNTIME_API_KEY";
const JUDGE_API_KEY: &str = "ELEPHANT_BENCH_JUDGE_API_KEY";
const EMBEDDING_API_KEY: &str = "ELEPHANT_BENCH_EMBEDDING_API_KEY";
const RERANKER_API_KEY: &str = "ELEPHANT_BENCH_RERANKER_API_KEY";

#[derive(Debug, Clone)]
pub(crate) struct BenchSecrets {
    runtime_api_key: Option<String>,
    judge_api_key: Option<String>,
    embedding_api_key: Option<String>,
    reranker_api_key: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct RedactedBenchSecrets {
    runtime_api_key: bool,
    judge_api_key: bool,
    embedding_api_key: bool,
    reranker_api_key: bool,
}

impl BenchSecrets {
    pub(crate) fn load(env_file: Option<&Path>) -> Result<Self> {
        let file_vars = if let Some(path) = env_file {
            parse_env_file(path)?
        } else {
            BTreeMap::new()
        };

        Ok(Self {
            runtime_api_key: read_secret(&file_vars, RUNTIME_API_KEY)?,
            judge_api_key: read_secret(&file_vars, JUDGE_API_KEY)?,
            embedding_api_key: read_secret(&file_vars, EMBEDDING_API_KEY)?,
            reranker_api_key: read_secret(&file_vars, RERANKER_API_KEY)?,
        })
    }

    pub(crate) fn runtime_api_key(&self) -> Option<&str> {
        self.runtime_api_key.as_deref()
    }

    pub(crate) fn judge_api_key(&self) -> Option<&str> {
        self.judge_api_key.as_deref()
    }

    pub(crate) fn embedding_api_key(&self) -> Option<&str> {
        self.embedding_api_key.as_deref()
    }

    pub(crate) fn reranker_api_key(&self) -> Option<&str> {
        self.reranker_api_key.as_deref()
    }

    pub(crate) fn redacted(&self) -> RedactedBenchSecrets {
        RedactedBenchSecrets {
            runtime_api_key: self.runtime_api_key.is_some(),
            judge_api_key: self.judge_api_key.is_some(),
            embedding_api_key: self.embedding_api_key.is_some(),
            reranker_api_key: self.reranker_api_key.is_some(),
        }
    }
}

fn read_secret(file_vars: &BTreeMap<String, String>, name: &str) -> Result<Option<String>> {
    if let Some(value) = std::env::var_os(name) {
        let value = value.into_string().map_err(|_| {
            ConfigError::configuration(format!("{name} must contain valid Unicode"))
        })?;
        if value.trim().is_empty() {
            return Err(ConfigError::configuration(format!(
                "{name} must not be blank"
            )));
        }
        return Ok(Some(value));
    }

    match file_vars.get(name) {
        Some(value) if value.trim().is_empty() => Err(ConfigError::configuration(format!(
            "{name} must not be blank"
        ))),
        Some(value) => Ok(Some(value.clone())),
        None => Ok(None),
    }
}

fn parse_env_file(path: &Path) -> Result<BTreeMap<String, String>> {
    let resolved = resolve_workspace_path(path);
    let raw = fs::read_to_string(&resolved).map_err(|error| {
        ConfigError::configuration(format!("failed to read {}: {error}", path.display()))
    })?;
    let mut vars = BTreeMap::new();
    for (line_no, line) in raw.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let (name, value) = line.split_once('=').ok_or_else(|| {
            ConfigError::configuration(format!(
                "invalid benchmark secrets env at {}:{}: expected NAME=value",
                path.display(),
                line_no + 1
            ))
        })?;
        vars.insert(name.trim().to_string(), unquote(value.trim()));
    }
    Ok(vars)
}

fn unquote(value: &str) -> String {
    let bytes = value.as_bytes();
    if bytes.len() >= 2
        && ((bytes[0] == b'"' && bytes[bytes.len() - 1] == b'"')
            || (bytes[0] == b'\'' && bytes[bytes.len() - 1] == b'\''))
    {
        value[1..value.len() - 1].to_string()
    } else {
        value.to_string()
    }
}

fn resolve_workspace_path(path: &Path) -> std::path::PathBuf {
    if path.is_absolute() || !path.starts_with("bench") {
        return path.to_path_buf();
    }

    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("bench crate must live under the workspace root")
        .join(path)
}
