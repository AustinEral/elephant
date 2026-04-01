use std::fs;
use std::path::Path;
use std::sync::Arc;

use serde::Serialize;

use elephant::consolidation::{ConsolidationConfig, opinion_merger};
use elephant::embedding::EmbeddingConfig;
use elephant::llm::{
    AnthropicConfig, ClientConfig, GeminiConfig, LlmConfig, OpenAiConfig, ReasoningEffortConfig,
    VertexConfig,
};
use elephant::metrics::MetricsCollector;
use elephant::recall::reranker::RerankerConfig;
use elephant::retain::{extractor, graph_builder};
use elephant::{ConfigError, ReflectConfig, RetrievalConfig, RuntimeConfig};

use crate::env::BenchJudgeConfig;
use crate::harness::{BenchHarness, BenchHarnessBuilder};

use super::contract::{
    BenchmarkKind, JudgeContract, LocomoContractFile, LongMemEvalConsolidationMode,
    LongMemEvalContractFile, LongMemEvalIngestFormat, ProviderKind, RerankerProviderKind,
    ResolvedLocomoContract, ResolvedLongMemEvalContract, RuntimeContract,
};
use super::execution::{
    BenchExecution, BenchExecutionOverlayFile, LongMemEvalExecution,
    LongMemEvalExecutionOverlayFile,
};
use super::secrets::{BenchSecrets, RedactedBenchSecrets};

type Result<T> = std::result::Result<T, ConfigError>;

/// Fully resolved benchmark config with redacted-print support and typed builders.
#[derive(Debug, Clone)]
pub struct ResolvedBenchConfig {
    contract: ResolvedLocomoContract,
    execution: BenchExecution,
    secrets: BenchSecrets,
    contract_hash: String,
}

impl ResolvedBenchConfig {
    /// Return the stable hash for the resolved contract only.
    pub fn contract_hash(&self) -> &str {
        &self.contract_hash
    }

    /// Return the protocol version for this resolved contract.
    pub fn protocol_version(&self) -> &str {
        &self.contract.protocol_version
    }

    /// Return the dataset fingerprint for this resolved contract.
    pub fn dataset_fingerprint(&self) -> &str {
        &self.contract.dataset_fingerprint
    }

    /// Return the category filter for this resolved contract.
    pub fn category_filter(&self) -> &[u8] {
        &self.contract.category_filter
    }

    /// Return the selected conversation slice for this resolved contract.
    pub fn conversations(&self) -> &[String] {
        &self.contract.conversations
    }

    /// Return the optional session limit for this resolved contract.
    pub fn session_limit(&self) -> Option<usize> {
        self.contract.session_limit
    }

    /// Return the optional question limit for this resolved contract.
    pub fn question_limit(&self) -> Option<usize> {
        self.contract.question_limit
    }

    /// Return the resolved ingest mode.
    pub fn ingest_mode(&self) -> &'static str {
        match self.contract.ingest {
            super::contract::LocomoIngestMode::Turn => "turn",
            super::contract::LocomoIngestMode::Session => "session",
            super::contract::LocomoIngestMode::RawJson => "raw-json",
        }
    }

    /// Return the resolved consolidation mode.
    pub fn consolidation_mode(&self) -> &'static str {
        match self.contract.consolidation {
            super::contract::LocomoConsolidationMode::End => "end",
            super::contract::LocomoConsolidationMode::PerSession => "per-session",
            super::contract::LocomoConsolidationMode::Off => "off",
        }
    }

    /// Return the image policy derived from the ingest contract.
    pub fn image_policy(&self) -> &str {
        self.contract.image_policy
    }

    /// Return the optional determinism requirement for this resolved contract.
    pub fn determinism_requirement(&self) -> Option<elephant::llm::DeterminismRequirement> {
        self.contract.determinism_requirement
    }

    /// Return the dataset path from the resolved execution config.
    pub fn dataset_path(&self) -> &Path {
        &self.execution.dataset_path
    }

    /// Return the output directory from the resolved execution config.
    pub fn output_dir(&self) -> &Path {
        &self.execution.output_dir
    }

    /// Return the optional execution tag.
    pub fn tag(&self) -> Option<&str> {
        self.execution.tag.as_deref()
    }

    /// Return the resolved conversation concurrency.
    pub fn conversation_jobs(&self) -> usize {
        self.execution.conversation_jobs
    }

    /// Return the resolved question concurrency.
    pub fn question_jobs(&self) -> usize {
        self.execution.question_jobs
    }

    /// Return the reflect budget token setting from the resolved contract.
    pub fn reflect_budget_tokens(&self) -> usize {
        self.contract.runtime.tuning.reflect_budget_tokens
    }

    /// Return a redacted JSON snapshot of the resolved contract.
    pub fn redacted_contract_json(&self) -> serde_json::Value {
        serde_json::to_value(&self.contract).expect("resolved contract serializes")
    }

    /// Return a redacted JSON snapshot of the resolved execution config.
    pub fn redacted_execution_json(&self) -> serde_json::Value {
        serde_json::to_value(RedactedBenchExecution::from(&self.execution))
            .expect("resolved execution serializes")
    }

    /// Return a cloned resolved config with an explicit judge-model override applied.
    pub fn with_judge_model_override(&self, judge_model: Option<&str>) -> Result<Self> {
        let Some(judge_model) = judge_model else {
            return Ok(self.clone());
        };
        validate_nonblank("judge.model", judge_model)?;

        let mut resolved = self.clone();
        resolved.contract.judge.model = judge_model.to_string();
        resolved.contract_hash = contract_hash_for(&resolved.contract)?;
        resolved.build_judge_config()?;
        Ok(resolved)
    }

    /// Render the resolved benchmark config as pretty JSON with secrets redacted.
    pub fn to_pretty_redacted_json(&self) -> String {
        let payload = RedactedResolvedBenchConfig {
            contract: &self.contract,
            execution: RedactedBenchExecution::from(&self.execution),
            secrets: self.secrets.redacted(),
            contract_hash: &self.contract_hash,
        };
        serde_json::to_string_pretty(&payload).expect("resolved config serializes")
    }

    /// Build a typed runtime config from the resolved contract, execution, and secrets.
    pub fn build_runtime_config(&self) -> Result<RuntimeConfig> {
        build_runtime_config_from_parts(
            &self.contract.runtime,
            &self.execution.database_url,
            &self.execution,
            &self.secrets,
        )
    }

    /// Build a typed benchmark judge config from the resolved contract, execution, and secrets.
    pub fn build_judge_config(&self) -> Result<BenchJudgeConfig> {
        build_judge_config_from_parts(&self.contract.judge, &self.execution, &self.secrets)
    }

    /// Build a benchmark harness from the resolved runtime contract.
    pub async fn build_harness(
        &self,
        metrics: Arc<MetricsCollector>,
    ) -> elephant::Result<BenchHarness> {
        let runtime_config = self.build_runtime_config()?;
        let bench_config = crate::env::BenchConfig::new(self.determinism_requirement());
        BenchHarnessBuilder::new(runtime_config, bench_config)
            .metrics(metrics)
            .build()
            .await
    }
}

/// Fully resolved LongMemEval benchmark config with redacted-print support and typed builders.
#[derive(Debug, Clone)]
pub struct ResolvedLongMemEvalBenchConfig {
    contract: ResolvedLongMemEvalContract,
    execution: LongMemEvalExecution,
    secrets: BenchSecrets,
    contract_hash: String,
}

impl ResolvedLongMemEvalBenchConfig {
    /// Return the stable hash for the resolved contract only.
    pub fn contract_hash(&self) -> &str {
        &self.contract_hash
    }

    /// Return the protocol version for this resolved contract.
    pub fn protocol_version(&self) -> &str {
        &self.contract.protocol_version
    }

    /// Return the dataset fingerprint for this resolved contract.
    pub fn dataset_fingerprint(&self) -> &str {
        &self.contract.dataset_fingerprint
    }

    /// Return the selected instance ids for this resolved contract.
    pub fn instances(&self) -> &[String] {
        &self.contract.instances
    }

    /// Return the optional session limit for this resolved contract.
    pub fn session_limit(&self) -> Option<usize> {
        self.contract.session_limit
    }

    /// Return the optional instance limit for this resolved contract.
    pub fn instance_limit(&self) -> Option<usize> {
        self.contract.instance_limit
    }

    /// Return the instance offset for this resolved contract.
    pub fn instance_offset(&self) -> usize {
        self.contract.instance_offset
    }

    /// Return the resolved ingest format.
    pub fn ingest_format(&self) -> &'static str {
        match self.contract.ingest_format {
            LongMemEvalIngestFormat::Text => "text",
            LongMemEvalIngestFormat::Json => "json",
        }
    }

    /// Return the resolved consolidation mode.
    pub fn consolidation_mode(&self) -> &'static str {
        match self.contract.consolidation {
            LongMemEvalConsolidationMode::End => "end",
            LongMemEvalConsolidationMode::PerSession => "per-session",
            LongMemEvalConsolidationMode::Off => "off",
        }
    }

    /// Return the optional determinism requirement for this resolved contract.
    pub fn determinism_requirement(&self) -> Option<elephant::llm::DeterminismRequirement> {
        self.contract.determinism_requirement
    }

    /// Return the dataset path from the resolved execution config.
    pub fn dataset_path(&self) -> &Path {
        &self.execution.dataset_path
    }

    /// Return the output directory from the resolved execution config.
    pub fn output_dir(&self) -> &Path {
        &self.execution.output_dir
    }

    /// Return the optional execution tag.
    pub fn tag(&self) -> Option<&str> {
        self.execution.tag.as_deref()
    }

    /// Return the resolved instance concurrency.
    pub fn instance_jobs(&self) -> usize {
        self.execution.instance_jobs
    }

    /// Return the reflect budget token setting from the resolved contract.
    pub fn reflect_budget_tokens(&self) -> usize {
        self.contract.runtime.tuning.reflect_budget_tokens
    }

    /// Return a redacted JSON snapshot of the resolved contract.
    pub fn redacted_contract_json(&self) -> serde_json::Value {
        serde_json::to_value(&self.contract).expect("resolved contract serializes")
    }

    /// Return a redacted JSON snapshot of the resolved execution config.
    pub fn redacted_execution_json(&self) -> serde_json::Value {
        serde_json::to_value(RedactedLongMemEvalExecution::from(&self.execution))
            .expect("resolved execution serializes")
    }

    /// Return a cloned resolved config with an explicit judge-model override applied.
    pub fn with_judge_model_override(&self, judge_model: Option<&str>) -> Result<Self> {
        let Some(judge_model) = judge_model else {
            return Ok(self.clone());
        };
        validate_nonblank("judge.model", judge_model)?;

        let mut resolved = self.clone();
        resolved.contract.judge.model = judge_model.to_string();
        resolved.contract_hash = contract_hash_for(&resolved.contract)?;
        resolved.build_judge_config()?;
        Ok(resolved)
    }

    /// Render the resolved benchmark config as pretty JSON with secrets redacted.
    pub fn to_pretty_redacted_json(&self) -> String {
        let payload = RedactedLongMemEvalResolvedBenchConfig {
            contract: &self.contract,
            execution: RedactedLongMemEvalExecution::from(&self.execution),
            secrets: self.secrets.redacted(),
            contract_hash: &self.contract_hash,
        };
        serde_json::to_string_pretty(&payload).expect("resolved config serializes")
    }

    /// Build a typed runtime config from the resolved contract, execution, and secrets.
    pub fn build_runtime_config(&self) -> Result<RuntimeConfig> {
        build_runtime_config_from_parts(
            &self.contract.runtime,
            &self.execution.database_url,
            &self.execution,
            &self.secrets,
        )
    }

    /// Build a typed benchmark judge config from the resolved contract, execution, and secrets.
    pub fn build_judge_config(&self) -> Result<BenchJudgeConfig> {
        build_judge_config_from_parts(&self.contract.judge, &self.execution, &self.secrets)
    }

    /// Build a benchmark harness from the resolved runtime contract.
    pub async fn build_harness(
        &self,
        metrics: Arc<MetricsCollector>,
    ) -> elephant::Result<BenchHarness> {
        let runtime_config = self.build_runtime_config()?;
        let bench_config = crate::env::BenchConfig::new(self.determinism_requirement());
        BenchHarnessBuilder::new(runtime_config, bench_config)
            .metrics(metrics)
            .build()
            .await
    }
}

#[derive(Debug, Serialize)]
struct RedactedResolvedBenchConfig<'a> {
    contract: &'a ResolvedLocomoContract,
    execution: RedactedBenchExecution<'a>,
    secrets: RedactedBenchSecrets,
    contract_hash: &'a str,
}

#[derive(Debug, Serialize)]
struct RedactedLongMemEvalResolvedBenchConfig<'a> {
    contract: &'a ResolvedLongMemEvalContract,
    execution: RedactedLongMemEvalExecution<'a>,
    secrets: RedactedBenchSecrets,
    contract_hash: &'a str,
}

#[derive(Debug, Serialize)]
struct RedactedBenchExecution<'a> {
    dataset_path: String,
    output_dir: String,
    tag: Option<&'a str>,
    conversation_jobs: usize,
    question_jobs: usize,
    database_url: &'static str,
    llm_base_url: Option<&'a str>,
    llm_timeout_secs: u64,
    llm_vertex_project: Option<&'a str>,
    llm_vertex_location: Option<&'a str>,
    embedding_model_path: Option<String>,
    embedding_max_seq_len: usize,
    reranker_model_path: Option<String>,
    reranker_api_url: Option<&'a str>,
    reranker_max_seq_len: usize,
    judge_base_url: Option<&'a str>,
    judge_timeout_secs: u64,
    judge_vertex_project: Option<&'a str>,
    judge_vertex_location: Option<&'a str>,
}

impl<'a> From<&'a BenchExecution> for RedactedBenchExecution<'a> {
    fn from(value: &'a BenchExecution) -> Self {
        Self {
            dataset_path: value.dataset_path.display().to_string(),
            output_dir: value.output_dir.display().to_string(),
            tag: value.tag.as_deref(),
            conversation_jobs: value.conversation_jobs,
            question_jobs: value.question_jobs,
            database_url: "<redacted>",
            llm_base_url: value.llm_base_url.as_deref(),
            llm_timeout_secs: value.llm_timeout_secs,
            llm_vertex_project: value.llm_vertex_project.as_deref(),
            llm_vertex_location: value.llm_vertex_location.as_deref(),
            embedding_model_path: value
                .embedding_model_path
                .as_ref()
                .map(|path| path.display().to_string()),
            embedding_max_seq_len: value.embedding_max_seq_len,
            reranker_model_path: value
                .reranker_model_path
                .as_ref()
                .map(|path| path.display().to_string()),
            reranker_api_url: value.reranker_api_url.as_deref(),
            reranker_max_seq_len: value.reranker_max_seq_len,
            judge_base_url: value.judge_base_url.as_deref(),
            judge_timeout_secs: value.judge_timeout_secs,
            judge_vertex_project: value.judge_vertex_project.as_deref(),
            judge_vertex_location: value.judge_vertex_location.as_deref(),
        }
    }
}

#[derive(Debug, Serialize)]
struct RedactedLongMemEvalExecution<'a> {
    dataset_path: String,
    output_dir: String,
    tag: Option<&'a str>,
    instance_jobs: usize,
    database_url: &'static str,
    llm_base_url: Option<&'a str>,
    llm_timeout_secs: u64,
    llm_vertex_project: Option<&'a str>,
    llm_vertex_location: Option<&'a str>,
    embedding_model_path: Option<String>,
    embedding_max_seq_len: usize,
    reranker_model_path: Option<String>,
    reranker_api_url: Option<&'a str>,
    reranker_max_seq_len: usize,
    judge_base_url: Option<&'a str>,
    judge_timeout_secs: u64,
    judge_vertex_project: Option<&'a str>,
    judge_vertex_location: Option<&'a str>,
}

impl<'a> From<&'a LongMemEvalExecution> for RedactedLongMemEvalExecution<'a> {
    fn from(value: &'a LongMemEvalExecution) -> Self {
        Self {
            dataset_path: value.dataset_path.display().to_string(),
            output_dir: value.output_dir.display().to_string(),
            tag: value.tag.as_deref(),
            instance_jobs: value.instance_jobs,
            database_url: "<redacted>",
            llm_base_url: value.llm_base_url.as_deref(),
            llm_timeout_secs: value.llm_timeout_secs,
            llm_vertex_project: value.llm_vertex_project.as_deref(),
            llm_vertex_location: value.llm_vertex_location.as_deref(),
            embedding_model_path: value
                .embedding_model_path
                .as_ref()
                .map(|path| path.display().to_string()),
            embedding_max_seq_len: value.embedding_max_seq_len,
            reranker_model_path: value
                .reranker_model_path
                .as_ref()
                .map(|path| path.display().to_string()),
            reranker_api_url: value.reranker_api_url.as_deref(),
            reranker_max_seq_len: value.reranker_max_seq_len,
            judge_base_url: value.judge_base_url.as_deref(),
            judge_timeout_secs: value.judge_timeout_secs,
            judge_vertex_project: value.judge_vertex_project.as_deref(),
            judge_vertex_location: value.judge_vertex_location.as_deref(),
        }
    }
}

/// Resolve a checked-in LoCoMo profile, optional execution overlay, and benchmark secrets.
pub fn resolve_locomo_bench_config(
    profile_path: &Path,
    execution_overlay_path: Option<&Path>,
    secrets_env_file: Option<&Path>,
) -> Result<ResolvedBenchConfig> {
    let profile = load_toml::<LocomoContractFile>(profile_path)?;
    validate_profile(&profile)?;
    let overlay = execution_overlay_path
        .map(load_toml::<BenchExecutionOverlayFile>)
        .transpose()?;
    let execution = BenchExecution::from_overlay(overlay);
    validate_execution(&execution)?;
    let secrets = BenchSecrets::load(secrets_env_file)?;

    let dataset_path = resolve_workspace_path(&execution.dataset_path);
    let dataset_bytes = fs::read(&dataset_path).map_err(|error| {
        ConfigError::configuration(format!(
            "failed to read dataset {}: {error}",
            execution.dataset_path.display()
        ))
    })?;
    let dataset_fingerprint = fnv1a64_hex(&dataset_bytes);
    if let Some(expected) = &profile.dataset.expected_fingerprint
        && expected != &dataset_fingerprint
    {
        return Err(ConfigError::configuration(format!(
            "dataset fingerprint mismatch for {}: expected {}, got {}",
            profile.dataset.identifier, expected, dataset_fingerprint
        )));
    }

    let resolved_contract = ResolvedLocomoContract {
        benchmark: "locomo",
        schema_version: profile.schema_version,
        protocol_version: profile.protocol_version,
        dataset_identifier: profile.dataset.identifier,
        dataset_fingerprint,
        expected_dataset_fingerprint: profile.dataset.expected_fingerprint,
        category_filter: profile.slice.category_filter,
        conversations: profile.slice.conversations,
        session_limit: profile.slice.session_limit,
        question_limit: profile.slice.question_limit,
        ingest: profile.ingest,
        image_policy: profile.ingest.image_policy(),
        consolidation: profile.consolidation,
        determinism_requirement: profile.determinism_requirement,
        runtime: profile.runtime,
        judge: profile.judge,
        judge_prompt_hash: judge_prompt_hash(),
    };

    let contract_hash = contract_hash_for(&resolved_contract)?;

    let resolved = ResolvedBenchConfig {
        contract: resolved_contract,
        execution,
        secrets,
        contract_hash,
    };

    resolved.build_runtime_config()?;
    resolved.build_judge_config()?;
    Ok(resolved)
}

/// Resolve a checked-in LongMemEval profile, optional execution overlay, and benchmark secrets.
pub fn resolve_longmemeval_bench_config(
    profile_path: &Path,
    execution_overlay_path: Option<&Path>,
    secrets_env_file: Option<&Path>,
) -> Result<ResolvedLongMemEvalBenchConfig> {
    let profile = load_toml::<LongMemEvalContractFile>(profile_path)?;
    validate_longmemeval_profile(&profile)?;
    let overlay = execution_overlay_path
        .map(load_toml::<LongMemEvalExecutionOverlayFile>)
        .transpose()?;
    let execution = LongMemEvalExecution::from_overlay(overlay);
    validate_longmemeval_execution(&execution)?;
    let secrets = BenchSecrets::load(secrets_env_file)?;

    let dataset_path = resolve_workspace_path(&execution.dataset_path);
    let dataset_bytes = fs::read(&dataset_path).map_err(|error| {
        ConfigError::configuration(format!(
            "failed to read dataset {}: {error}",
            execution.dataset_path.display()
        ))
    })?;
    let dataset_fingerprint = fnv1a64_hex(&dataset_bytes);
    if let Some(expected) = &profile.dataset.expected_fingerprint
        && expected != &dataset_fingerprint
    {
        return Err(ConfigError::configuration(format!(
            "dataset fingerprint mismatch for {}: expected {}, got {}",
            profile.dataset.identifier, expected, dataset_fingerprint
        )));
    }

    let resolved_contract = ResolvedLongMemEvalContract {
        benchmark: "longmemeval",
        schema_version: profile.schema_version,
        protocol_version: profile.protocol_version,
        dataset_identifier: profile.dataset.identifier,
        dataset_fingerprint,
        expected_dataset_fingerprint: profile.dataset.expected_fingerprint,
        instances: profile.slice.instances,
        session_limit: profile.slice.session_limit,
        instance_limit: profile.slice.instance_limit,
        instance_offset: profile.slice.instance_offset,
        ingest_format: profile.ingest_format,
        consolidation: profile.consolidation,
        determinism_requirement: profile.determinism_requirement,
        runtime: profile.runtime,
        judge: profile.judge,
        judge_prompt_hash: longmemeval_judge_prompt_hash(),
    };

    let contract_hash = contract_hash_for(&resolved_contract)?;

    let resolved = ResolvedLongMemEvalBenchConfig {
        contract: resolved_contract,
        execution,
        secrets,
        contract_hash,
    };

    resolved.build_runtime_config()?;
    resolved.build_judge_config()?;
    Ok(resolved)
}

fn contract_hash_for<T: Serialize>(contract: &T) -> Result<String> {
    serde_json::to_vec(contract)
        .map(|bytes| fnv1a64_hex(&bytes))
        .map_err(|error| {
            ConfigError::configuration(format!("failed to serialize resolved contract: {error}"))
        })
}

fn load_toml<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T> {
    let resolved = resolve_workspace_path(path);
    let raw = fs::read_to_string(&resolved).map_err(|error| {
        ConfigError::configuration(format!("failed to read {}: {error}", path.display()))
    })?;
    toml::from_str(&raw).map_err(|error| {
        ConfigError::configuration(format!("failed to parse {}: {error}", path.display()))
    })
}

fn resolve_workspace_path(path: &Path) -> std::path::PathBuf {
    if path.is_absolute() {
        return path.to_path_buf();
    }

    let Some(first) = path.components().next() else {
        return path.to_path_buf();
    };
    let anchor_to_workspace = matches!(
        first.as_os_str().to_str(),
        Some("bench" | "data" | "models")
    );
    if !anchor_to_workspace {
        return path.to_path_buf();
    }

    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("bench crate must live under the workspace root")
        .join(path)
}

fn validate_profile(profile: &LocomoContractFile) -> Result<()> {
    if profile.schema_version == 0 {
        return Err(ConfigError::configuration(
            "schema_version must be greater than 0",
        ));
    }
    if profile.benchmark != BenchmarkKind::Locomo {
        return Err(ConfigError::configuration("benchmark must be \"locomo\""));
    }
    if profile.protocol_version.trim().is_empty() {
        return Err(ConfigError::configuration(
            "protocol_version must not be blank",
        ));
    }
    if profile.dataset.identifier.trim().is_empty() {
        return Err(ConfigError::configuration(
            "dataset.identifier must not be blank",
        ));
    }
    if profile.slice.category_filter.is_empty() {
        return Err(ConfigError::configuration(
            "slice.category_filter must not be empty",
        ));
    }
    for category in &profile.slice.category_filter {
        if !matches!(category, 1..=5) {
            return Err(ConfigError::configuration(format!(
                "slice.category_filter must contain only values in 1..=5, got: {category}"
            )));
        }
    }
    if matches!(profile.slice.session_limit, Some(0)) {
        return Err(ConfigError::configuration(
            "slice.session_limit must be greater than 0 if set",
        ));
    }
    if matches!(profile.slice.question_limit, Some(0)) {
        return Err(ConfigError::configuration(
            "slice.question_limit must be greater than 0 if set",
        ));
    }
    validate_runtime_contract(&profile.runtime)?;
    validate_judge_contract(&profile.judge)?;
    Ok(())
}

fn validate_longmemeval_profile(profile: &LongMemEvalContractFile) -> Result<()> {
    if profile.schema_version == 0 {
        return Err(ConfigError::configuration(
            "schema_version must be greater than 0",
        ));
    }
    if profile.benchmark != BenchmarkKind::LongMemEval {
        return Err(ConfigError::configuration(
            "benchmark must be \"longmemeval\"",
        ));
    }
    if profile.protocol_version.trim().is_empty() {
        return Err(ConfigError::configuration(
            "protocol_version must not be blank",
        ));
    }
    if profile.dataset.identifier.trim().is_empty() {
        return Err(ConfigError::configuration(
            "dataset.identifier must not be blank",
        ));
    }
    if matches!(profile.slice.session_limit, Some(0)) {
        return Err(ConfigError::configuration(
            "slice.session_limit must be greater than 0 if set",
        ));
    }
    if matches!(profile.slice.instance_limit, Some(0)) {
        return Err(ConfigError::configuration(
            "slice.instance_limit must be greater than 0 if set",
        ));
    }
    validate_runtime_contract(&profile.runtime)?;
    validate_judge_contract(&profile.judge)?;
    Ok(())
}

fn validate_runtime_contract(runtime: &RuntimeContract) -> Result<()> {
    validate_nonblank("runtime.llm.retain_model", &runtime.llm.retain_model)?;
    validate_nonblank("runtime.llm.reflect_model", &runtime.llm.reflect_model)?;
    validate_nonblank("runtime.embedding.model", &runtime.embedding.model)?;
    match runtime.embedding.provider {
        super::contract::EmbeddingProviderKind::Local => {}
        super::contract::EmbeddingProviderKind::OpenAi => {
            if runtime.embedding.dimensions.is_none() {
                return Err(ConfigError::configuration(
                    "runtime.embedding.dimensions must be set for provider=open_ai",
                ));
            }
        }
    }
    match runtime.reranker.provider {
        RerankerProviderKind::None => {}
        RerankerProviderKind::Local | RerankerProviderKind::Api => {
            let model = runtime.reranker.model.as_deref().ok_or_else(|| {
                ConfigError::configuration(
                    "runtime.reranker.model must be set when reranker provider is not none",
                )
            })?;
            validate_nonblank("runtime.reranker.model", model)?;
        }
    }
    Ok(())
}

fn validate_judge_contract(judge: &JudgeContract) -> Result<()> {
    validate_nonblank("judge.model", &judge.model)?;
    if judge.max_tokens == 0 {
        return Err(ConfigError::configuration(
            "judge.max_tokens must be greater than 0",
        ));
    }
    if judge.max_attempts == 0 {
        return Err(ConfigError::configuration(
            "judge.max_attempts must be greater than 0",
        ));
    }
    Ok(())
}

fn validate_execution(execution: &BenchExecution) -> Result<()> {
    validate_nonblank("execution.database_url", &execution.database_url)?;
    if execution.conversation_jobs == 0 {
        return Err(ConfigError::configuration(
            "execution.conversation_jobs must be greater than 0",
        ));
    }
    if execution.question_jobs == 0 {
        return Err(ConfigError::configuration(
            "execution.question_jobs must be greater than 0",
        ));
    }
    validate_execution_common(execution)
}

fn validate_longmemeval_execution(execution: &LongMemEvalExecution) -> Result<()> {
    validate_nonblank("execution.database_url", &execution.database_url)?;
    if execution.instance_jobs == 0 {
        return Err(ConfigError::configuration(
            "execution.instance_jobs must be greater than 0",
        ));
    }
    validate_execution_common(execution)
}

fn validate_execution_common<E: ExecutionConfigRef>(execution: &E) -> Result<()> {
    if execution.llm_timeout_secs() == 0 {
        return Err(ConfigError::configuration(
            "execution.llm_timeout_secs must be greater than 0",
        ));
    }
    if execution.embedding_max_seq_len() == 0 {
        return Err(ConfigError::configuration(
            "execution.embedding_max_seq_len must be greater than 0",
        ));
    }
    if execution.reranker_max_seq_len() == 0 {
        return Err(ConfigError::configuration(
            "execution.reranker_max_seq_len must be greater than 0",
        ));
    }
    if execution.judge_timeout_secs() == 0 {
        return Err(ConfigError::configuration(
            "execution.judge_timeout_secs must be greater than 0",
        ));
    }
    Ok(())
}

fn validate_nonblank(name: &str, value: &str) -> Result<()> {
    if value.trim().is_empty() {
        Err(ConfigError::configuration(format!(
            "{name} must not be blank"
        )))
    } else {
        Ok(())
    }
}

trait ExecutionConfigRef {
    fn llm_base_url(&self) -> Option<&str>;
    fn llm_timeout_secs(&self) -> u64;
    fn llm_vertex_project(&self) -> Option<&str>;
    fn llm_vertex_location(&self) -> Option<&str>;
    fn embedding_model_path(&self) -> Option<&Path>;
    fn embedding_max_seq_len(&self) -> usize;
    fn reranker_model_path(&self) -> Option<&Path>;
    fn reranker_api_url(&self) -> Option<&str>;
    fn reranker_max_seq_len(&self) -> usize;
    fn judge_base_url(&self) -> Option<&str>;
    fn judge_timeout_secs(&self) -> u64;
    fn judge_vertex_project(&self) -> Option<&str>;
    fn judge_vertex_location(&self) -> Option<&str>;
}

impl ExecutionConfigRef for BenchExecution {
    fn llm_base_url(&self) -> Option<&str> {
        self.llm_base_url.as_deref()
    }

    fn llm_timeout_secs(&self) -> u64 {
        self.llm_timeout_secs
    }

    fn llm_vertex_project(&self) -> Option<&str> {
        self.llm_vertex_project.as_deref()
    }

    fn llm_vertex_location(&self) -> Option<&str> {
        self.llm_vertex_location.as_deref()
    }

    fn embedding_model_path(&self) -> Option<&Path> {
        self.embedding_model_path.as_deref()
    }

    fn embedding_max_seq_len(&self) -> usize {
        self.embedding_max_seq_len
    }

    fn reranker_model_path(&self) -> Option<&Path> {
        self.reranker_model_path.as_deref()
    }

    fn reranker_api_url(&self) -> Option<&str> {
        self.reranker_api_url.as_deref()
    }

    fn reranker_max_seq_len(&self) -> usize {
        self.reranker_max_seq_len
    }

    fn judge_base_url(&self) -> Option<&str> {
        self.judge_base_url.as_deref()
    }

    fn judge_timeout_secs(&self) -> u64 {
        self.judge_timeout_secs
    }

    fn judge_vertex_project(&self) -> Option<&str> {
        self.judge_vertex_project.as_deref()
    }

    fn judge_vertex_location(&self) -> Option<&str> {
        self.judge_vertex_location.as_deref()
    }
}

impl ExecutionConfigRef for LongMemEvalExecution {
    fn llm_base_url(&self) -> Option<&str> {
        self.llm_base_url.as_deref()
    }

    fn llm_timeout_secs(&self) -> u64 {
        self.llm_timeout_secs
    }

    fn llm_vertex_project(&self) -> Option<&str> {
        self.llm_vertex_project.as_deref()
    }

    fn llm_vertex_location(&self) -> Option<&str> {
        self.llm_vertex_location.as_deref()
    }

    fn embedding_model_path(&self) -> Option<&Path> {
        self.embedding_model_path.as_deref()
    }

    fn embedding_max_seq_len(&self) -> usize {
        self.embedding_max_seq_len
    }

    fn reranker_model_path(&self) -> Option<&Path> {
        self.reranker_model_path.as_deref()
    }

    fn reranker_api_url(&self) -> Option<&str> {
        self.reranker_api_url.as_deref()
    }

    fn reranker_max_seq_len(&self) -> usize {
        self.reranker_max_seq_len
    }

    fn judge_base_url(&self) -> Option<&str> {
        self.judge_base_url.as_deref()
    }

    fn judge_timeout_secs(&self) -> u64 {
        self.judge_timeout_secs
    }

    fn judge_vertex_project(&self) -> Option<&str> {
        self.judge_vertex_project.as_deref()
    }

    fn judge_vertex_location(&self) -> Option<&str> {
        self.judge_vertex_location.as_deref()
    }
}

fn build_runtime_config_from_parts<E: ExecutionConfigRef>(
    runtime: &RuntimeContract,
    database_url: &str,
    execution: &E,
    secrets: &BenchSecrets,
) -> Result<RuntimeConfig> {
    let llm = build_runtime_llm_config(runtime, execution, secrets)?;
    let embedding = build_embedding_config(runtime, execution, secrets)?;
    let reranker = build_reranker_config(runtime, execution, secrets)?;
    let tuning = &runtime.tuning;
    let reasoning_effort = ReasoningEffortConfig {
        retain_extract: tuning.reasoning_effort.retain_extract,
        retain_resolve: tuning.reasoning_effort.retain_resolve,
        retain_graph: tuning.reasoning_effort.retain_graph,
        reflect: tuning.reasoning_effort.reflect,
        consolidate: tuning.reasoning_effort.consolidate,
        opinion_merge: tuning.reasoning_effort.opinion_merge,
    };

    RuntimeConfig::new(database_url.to_string(), llm, embedding, reranker)?
        .with_dedup_threshold(tuning.dedup_threshold)?
        .with_reasoning_effort(reasoning_effort)?
        .with_extraction(extractor::ExtractionConfig {
            structured_output_max_attempts: tuning.extraction.structured_output_max_attempts,
            temperature: tuning.extraction.temperature,
            reasoning_effort: tuning.reasoning_effort.retain_extract,
        })?
        .with_resolve_temperature(tuning.resolve_temperature)?
        .with_graph(graph_builder::GraphConfig {
            semantic_threshold: tuning.graph.semantic_threshold,
            temporal_max_days: tuning.graph.temporal_max_days,
            enable_causal: tuning.graph.enable_causal,
            causal_temperature: tuning.graph.causal_temperature,
            causal_reasoning_effort: tuning.reasoning_effort.retain_graph,
        })?
        .with_consolidation(ConsolidationConfig {
            batch_size: tuning.consolidation.batch_size,
            max_tokens: tuning.consolidation.max_tokens,
            recall_budget: tuning.consolidation.recall_budget,
            structured_output_max_attempts: tuning.consolidation.structured_output_max_attempts,
            temperature: tuning.consolidation.temperature,
            reasoning_effort: tuning.reasoning_effort.consolidate,
        })?
        .with_opinion_merge(opinion_merger::OpinionMergeConfig {
            temperature: tuning.opinion_merge.temperature,
            reasoning_effort: tuning.reasoning_effort.opinion_merge,
        })?
        .with_reflect(ReflectConfig::new(
            tuning.reflect.max_iterations,
            tuning.reflect.max_tokens,
            tuning.reflect.source_limit,
            tuning.reflect.source_max_chars,
            tuning.reflect.enable_source_lookup,
        )?)?
        .with_reflect_temperature(tuning.reflect_temperature)?
        .with_retrieval(RetrievalConfig::new(
            tuning.retrieval.retriever_limit,
            tuning.retrieval.max_facts,
        )?)
}

fn build_judge_config_from_parts<E: ExecutionConfigRef>(
    judge: &JudgeContract,
    execution: &E,
    secrets: &BenchSecrets,
) -> Result<BenchJudgeConfig> {
    let client = build_client_config(
        judge.provider,
        secrets.judge_api_key(),
        &judge.model,
        execution.judge_base_url(),
        execution.judge_timeout_secs(),
        execution.judge_vertex_project(),
        execution.judge_vertex_location(),
    )?;
    Ok(BenchJudgeConfig::new(
        client,
        judge.temperature,
        judge.max_tokens,
        judge.max_attempts,
    ))
}

fn build_runtime_llm_config<E: ExecutionConfigRef>(
    runtime: &RuntimeContract,
    execution: &E,
    secrets: &BenchSecrets,
) -> Result<LlmConfig> {
    let retain = build_client_config(
        runtime.llm.provider,
        secrets.runtime_api_key(),
        &runtime.llm.retain_model,
        execution.llm_base_url(),
        execution.llm_timeout_secs(),
        execution.llm_vertex_project(),
        execution.llm_vertex_location(),
    )?;
    let reflect = build_client_config(
        runtime.llm.provider,
        secrets.runtime_api_key(),
        &runtime.llm.reflect_model,
        execution.llm_base_url(),
        execution.llm_timeout_secs(),
        execution.llm_vertex_project(),
        execution.llm_vertex_location(),
    )?;
    Ok(LlmConfig::new(retain, reflect))
}

fn build_embedding_config<E: ExecutionConfigRef>(
    runtime: &RuntimeContract,
    execution: &E,
    secrets: &BenchSecrets,
) -> Result<EmbeddingConfig> {
    match runtime.embedding.provider {
        super::contract::EmbeddingProviderKind::Local => {
            let path = execution.embedding_model_path().ok_or_else(|| {
                ConfigError::configuration(
                    "execution.embedding_model_path must be set for runtime.embedding.provider=local",
                )
            })?;
            let resolved_path = resolve_workspace_path(path);
            Ok(EmbeddingConfig::local(resolved_path.display().to_string())
                .with_max_seq_len(execution.embedding_max_seq_len()))
        }
        super::contract::EmbeddingProviderKind::OpenAi => {
            let api_key = secrets.embedding_api_key().ok_or_else(|| {
                ConfigError::configuration(
                    "ELEPHANT_BENCH_EMBEDDING_API_KEY must be set for runtime.embedding.provider=open_ai",
                )
            })?;
            let dimensions = runtime.embedding.dimensions.ok_or_else(|| {
                ConfigError::configuration(
                    "runtime.embedding.dimensions must be set for runtime.embedding.provider=open_ai",
                )
            })?;
            Ok(
                EmbeddingConfig::openai(api_key, &runtime.embedding.model, dimensions)
                    .with_max_seq_len(execution.embedding_max_seq_len()),
            )
        }
    }
}

fn build_reranker_config<E: ExecutionConfigRef>(
    runtime: &RuntimeContract,
    execution: &E,
    secrets: &BenchSecrets,
) -> Result<RerankerConfig> {
    match runtime.reranker.provider {
        RerankerProviderKind::None => Ok(RerankerConfig::none()),
        RerankerProviderKind::Local => {
            let path = execution.reranker_model_path().ok_or_else(|| {
                ConfigError::configuration(
                    "execution.reranker_model_path must be set for runtime.reranker.provider=local",
                )
            })?;
            let resolved_path = resolve_workspace_path(path);
            Ok(RerankerConfig::local(resolved_path.display().to_string())
                .with_max_seq_len(execution.reranker_max_seq_len()))
        }
        RerankerProviderKind::Api => {
            let api_key = secrets.reranker_api_key().ok_or_else(|| {
                ConfigError::configuration(
                    "ELEPHANT_BENCH_RERANKER_API_KEY must be set for runtime.reranker.provider=api",
                )
            })?;
            let api_url = execution.reranker_api_url().ok_or_else(|| {
                ConfigError::configuration(
                    "execution.reranker_api_url must be set for runtime.reranker.provider=api",
                )
            })?;
            let model = runtime.reranker.model.as_deref().ok_or_else(|| {
                ConfigError::configuration(
                    "runtime.reranker.model must be set for runtime.reranker.provider=api",
                )
            })?;
            Ok(RerankerConfig::api(api_key, api_url, model)
                .with_max_seq_len(execution.reranker_max_seq_len()))
        }
    }
}

fn build_client_config(
    provider: ProviderKind,
    api_key: Option<&str>,
    model: &str,
    base_url: Option<&str>,
    timeout_secs: u64,
    vertex_project: Option<&str>,
    vertex_location: Option<&str>,
) -> Result<ClientConfig> {
    let api_key = api_key.ok_or_else(|| {
        ConfigError::configuration(format!(
            "missing API key for provider {}",
            provider.as_str()
        ))
    })?;

    match provider {
        ProviderKind::Anthropic => {
            let mut config = AnthropicConfig::new(api_key, model).map_err(ConfigError::from)?;
            config = config
                .with_timeout_secs(nonzero_timeout(timeout_secs)?)
                .map_err(ConfigError::from)?;
            Ok(ClientConfig::Anthropic(config))
        }
        ProviderKind::OpenAi => {
            let mut config = OpenAiConfig::new(api_key, model).map_err(ConfigError::from)?;
            config = config
                .with_timeout_secs(nonzero_timeout(timeout_secs)?)
                .map_err(ConfigError::from)?;
            if let Some(base_url) = base_url {
                config = config.with_base_url(base_url).map_err(ConfigError::from)?;
            }
            Ok(ClientConfig::OpenAi(config))
        }
        ProviderKind::Gemini => {
            let mut config = GeminiConfig::new(api_key, model).map_err(ConfigError::from)?;
            config = config
                .with_timeout_secs(nonzero_timeout(timeout_secs)?)
                .map_err(ConfigError::from)?;
            if let Some(base_url) = base_url {
                config = config.with_base_url(base_url).map_err(ConfigError::from)?;
            }
            Ok(ClientConfig::Gemini(config))
        }
        ProviderKind::Vertex => {
            let project = vertex_project.ok_or_else(|| {
                ConfigError::configuration(
                    "execution llm/judge vertex_project must be set for provider=vertex",
                )
            })?;
            let mut config =
                VertexConfig::new(api_key, model, project).map_err(ConfigError::from)?;
            config = config
                .with_timeout_secs(nonzero_timeout(timeout_secs)?)
                .map_err(ConfigError::from)?;
            if let Some(location) = vertex_location {
                config = config.with_location(location).map_err(ConfigError::from)?;
            }
            if let Some(base_url) = base_url {
                config = config.with_base_url(base_url).map_err(ConfigError::from)?;
            }
            Ok(ClientConfig::Vertex(config))
        }
    }
}

fn nonzero_timeout(timeout_secs: u64) -> Result<u64> {
    if timeout_secs == 0 {
        Err(ConfigError::configuration(
            "timeout seconds must be greater than 0",
        ))
    } else {
        Ok(timeout_secs)
    }
}

fn judge_prompt_hash() -> String {
    fnv1a64_hex(include_str!("../../locomo/judge_answer.txt").as_bytes())
}

fn longmemeval_judge_prompt_hash() -> String {
    let mut combined = String::new();
    combined.push_str(include_str!(
        "../../longmemeval/prompts/judge_abstention.txt"
    ));
    combined.push_str(include_str!("../../longmemeval/prompts/judge_factual.txt"));
    combined.push_str(include_str!(
        "../../longmemeval/prompts/judge_knowledge_update.txt"
    ));
    combined.push_str(include_str!(
        "../../longmemeval/prompts/judge_preference.txt"
    ));
    combined.push_str(include_str!("../../longmemeval/prompts/judge_temporal.txt"));
    fnv1a64_hex(combined.as_bytes())
}

fn fnv1a64_hex(bytes: &[u8]) -> String {
    format!("{:016x}", fnv1a64(bytes))
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    const OFFSET: u64 = 0xcbf29ce484222325;
    const PRIME: u64 = 0x100000001b3;
    let mut hash = OFFSET;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(PRIME);
    }
    hash
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::PathBuf;
    use std::sync::{Mutex, OnceLock};
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::*;

    fn env_lock() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
    }

    fn temp_path(name: &str) -> PathBuf {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("elephant-bench-{unique}-{name}"))
    }

    fn write_file(path: &Path, contents: &str) {
        fs::write(path, contents).unwrap();
    }

    fn sample_profile() -> String {
        r#"
schema_version = 1
benchmark = "locomo"
protocol_version = "2026-03-31-locomo-contract-v1"
ingest = "session"
consolidation = "end"

[dataset]
identifier = "locomo10"

[slice]
category_filter = [1, 2, 3, 4]
conversations = ["conv-26"]
session_limit = 1
question_limit = 5

[runtime.llm]
provider = "openai"
retain_model = "gpt-5.4-mini"
reflect_model = "gpt-5.4-mini"

[runtime.embedding]
provider = "local"
model = "bge-small-en-v1.5"

[runtime.reranker]
provider = "local"
model = "ms-marco-MiniLM-L-6-v2"

[judge]
provider = "openai"
model = "gpt-5.4"
"#
        .into()
    }

    fn sample_overlay(dataset_path: &Path) -> String {
        format!(
            r#"
database_url = "postgres://bench:bench@localhost/elephant"
dataset_path = "{}"
conversation_jobs = 1
question_jobs = 4
embedding_model_path = "models/bge-small-en-v1.5"
reranker_model_path = "models/ms-marco-MiniLM-L-6-v2"
"#,
            dataset_path.display()
        )
    }

    #[test]
    fn resolves_locomo_config_and_hash_is_stable() {
        let _guard = env_lock().lock().unwrap();
        let profile = temp_path("profile.toml");
        let overlay = temp_path("overlay.toml");
        let dataset = temp_path("dataset.json");
        let secrets = temp_path("secrets.env");

        write_file(&profile, &sample_profile());
        write_file(&dataset, r#"{"sample":"ok"}"#);
        write_file(&overlay, &sample_overlay(&dataset));
        write_file(
            &secrets,
            "ELEPHANT_BENCH_RUNTIME_API_KEY=runtime-secret\nELEPHANT_BENCH_JUDGE_API_KEY=judge-secret\n",
        );

        let left = resolve_locomo_bench_config(&profile, Some(&overlay), Some(&secrets)).unwrap();
        let right = resolve_locomo_bench_config(&profile, Some(&overlay), Some(&secrets)).unwrap();

        assert_eq!(left.contract_hash(), right.contract_hash());
        let printed = left.to_pretty_redacted_json();
        assert!(printed.contains("\"contract_hash\""));
        assert!(!printed.contains("runtime-secret"));
        assert!(!printed.contains("judge-secret"));
    }

    #[test]
    fn unknown_profile_field_fails_fast() {
        let profile = temp_path("bad-profile.toml");
        write_file(
            &profile,
            r#"
schema_version = 1
benchmark = "locomo"
protocol_version = "2026-03-31-locomo-contract-v1"
unexpected = true

[runtime.llm]
provider = "openai"
retain_model = "gpt-5.4-mini"
reflect_model = "gpt-5.4-mini"

[runtime.embedding]
provider = "local"
model = "bge-small-en-v1.5"

[runtime.reranker]
provider = "none"

[judge]
provider = "openai"
model = "gpt-5.4"
"#,
        );

        let err = resolve_locomo_bench_config(&profile, None, None).unwrap_err();
        assert!(err.to_string().contains("unexpected"));
    }

    #[test]
    fn missing_secrets_fail_with_field_name() {
        let profile = temp_path("profile.toml");
        let overlay = temp_path("overlay.toml");
        let dataset = temp_path("dataset.json");
        write_file(&profile, &sample_profile());
        write_file(&dataset, r#"{"sample":"ok"}"#);
        write_file(&overlay, &sample_overlay(&dataset));

        let err = resolve_locomo_bench_config(&profile, Some(&overlay), None).unwrap_err();
        assert!(err.to_string().contains("missing API key"));
    }

    #[test]
    fn contract_hash_ignores_execution_concurrency() {
        let _guard = env_lock().lock().unwrap();
        let profile = temp_path("profile.toml");
        let overlay_a = temp_path("overlay-a.toml");
        let overlay_b = temp_path("overlay-b.toml");
        let dataset = temp_path("dataset.json");
        let secrets = temp_path("secrets.env");

        write_file(&profile, &sample_profile());
        write_file(&dataset, r#"{"sample":"ok"}"#);
        write_file(&overlay_a, &sample_overlay(&dataset));
        write_file(
            &overlay_b,
            format!(
                r#"
database_url = "postgres://bench:bench@localhost/elephant"
dataset_path = "{}"
conversation_jobs = 8
question_jobs = 2
embedding_model_path = "models/bge-small-en-v1.5"
reranker_model_path = "models/ms-marco-MiniLM-L-6-v2"
"#,
                dataset.display()
            )
            .as_str(),
        );
        write_file(
            &secrets,
            "ELEPHANT_BENCH_RUNTIME_API_KEY=runtime-secret\nELEPHANT_BENCH_JUDGE_API_KEY=judge-secret\n",
        );

        let a = resolve_locomo_bench_config(&profile, Some(&overlay_a), Some(&secrets)).unwrap();
        let b = resolve_locomo_bench_config(&profile, Some(&overlay_b), Some(&secrets)).unwrap();

        assert_eq!(a.contract_hash(), b.contract_hash());
    }
}
