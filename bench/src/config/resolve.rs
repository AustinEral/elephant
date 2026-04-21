use std::fs;
use std::path::Path;
use std::path::PathBuf;
use std::sync::{Arc, OnceLock};

use serde::Serialize;

use elephant::consolidation::{ConsolidationConfig, opinion_merger};
use elephant::embedding::EmbeddingConfig;
use elephant::llm::{
    AnthropicConfig, AnthropicPromptCacheConfig, ClientConfig, DEFAULT_TIMEOUT_SECS, GeminiConfig,
    LlmConfig, OpenAiConfig, OpenAiPromptCacheConfig, ReasoningEffortConfig, VertexConfig,
};
use elephant::metrics::MetricsCollector;
use elephant::recall::reranker::RerankerConfig;
use elephant::retain::{extractor, graph_builder};
use elephant::types::ChunkConfig;
use elephant::{ConfigError, ReflectConfig, RetrievalConfig, RuntimeConfig};

use crate::env::BenchJudgeConfig;
use crate::harness::{BenchHarness, BenchHarnessBuilder};

use super::contract::{
    BenchmarkKind, EmbeddingProviderKind, JudgeContract, LocomoContractFile,
    LongMemEvalConsolidationMode, LongMemEvalContractFile, LongMemEvalIngestFormat, ProviderKind,
    RerankerProviderKind, ResolvedLocomoContract, ResolvedLongMemEvalContract, RuntimeContract,
};
use super::execution::{
    BenchExecution, BenchExecutionOverlayFile, ClientTargetExecution, LocomoShardExecution,
    LongMemEvalExecution, LongMemEvalExecutionOverlayFile, LongMemEvalShardExecution,
    PromptCacheExecution,
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

    /// Return the execution-side conversation shard for this resolved config.
    pub fn shard_conversations(&self) -> &[String] {
        &self.execution.shard.conversations
    }

    /// Return the effective conversation selection for this resolved run.
    pub fn selected_conversations(&self) -> &[String] {
        if self.execution.shard.conversations.is_empty() {
            &self.contract.conversations
        } else {
            &self.execution.shard.conversations
        }
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

    /// Return the resolved retain chunking configuration.
    pub fn retain_chunk_config(&self) -> ChunkConfig {
        ChunkConfig {
            max_tokens: self.contract.runtime.tuning.retain_chunk_max_tokens,
            overlap_tokens: self.contract.runtime.tuning.retain_chunk_overlap_tokens,
            preserve_turns: true,
        }
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
    ///
    /// # Errors
    ///
    /// Returns an error if the override is blank or if the overridden judge config
    /// fails validation.
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
    ///
    /// # Errors
    ///
    /// Returns an error if the resolved runtime provider, model stack, or tuning is
    /// invalid or incomplete.
    pub fn build_runtime_config(&self) -> Result<RuntimeConfig> {
        build_runtime_config_from_parts(
            &self.contract.runtime,
            &self.execution.database_url,
            self.execution.ort_dylib_path.as_deref(),
            &self.execution.runtime_target,
            &self.secrets,
        )
    }

    /// Build a typed benchmark judge config from the resolved contract, execution, and secrets.
    ///
    /// # Errors
    ///
    /// Returns an error if the resolved judge provider, model, or credentials are
    /// invalid or incomplete.
    pub fn build_judge_config(&self) -> Result<BenchJudgeConfig> {
        build_judge_config_from_parts(
            &self.contract.judge,
            &self.execution.judge_target,
            &self.secrets,
        )
    }

    /// Return a cloned resolved config with CLI execution-only overrides applied.
    ///
    /// # Errors
    ///
    /// Returns an error if the updated execution settings are invalid or if a
    /// dataset override does not match the declared dataset fingerprint.
    pub fn with_cli_execution_overrides(
        &self,
        dataset_path: Option<&Path>,
        tag: Option<&str>,
        conversation_jobs: Option<usize>,
        question_jobs: Option<usize>,
        shard_conversations: &[String],
    ) -> Result<Self> {
        let mut resolved = self.clone();
        if let Some(dataset_path) = dataset_path {
            resolved.execution.dataset_path = dataset_path.to_path_buf();
        }
        if let Some(tag) = tag {
            resolved.execution.tag = Some(tag.to_string());
        }
        if let Some(conversation_jobs) = conversation_jobs {
            resolved.execution.conversation_jobs = conversation_jobs;
        }
        if let Some(question_jobs) = question_jobs {
            resolved.execution.question_jobs = question_jobs;
        }
        if !shard_conversations.is_empty() {
            resolved.execution.shard.conversations = shard_conversations.to_vec();
        }
        validate_execution(&resolved.execution)?;
        validate_locomo_execution_shard(&resolved.contract, &resolved.execution.shard)?;
        if dataset_path.is_some() {
            resolved.contract.dataset_fingerprint = load_dataset_fingerprint(
                &resolved.execution.dataset_path,
                &resolved.contract.dataset_identifier,
                resolved.contract.expected_dataset_fingerprint.as_deref(),
            )?;
            resolved.contract_hash = contract_hash_for(&resolved.contract)?;
        }
        Ok(resolved)
    }

    /// Build a benchmark harness from the resolved runtime contract.
    ///
    /// # Errors
    ///
    /// Returns an error if runtime construction fails or if the benchmark harness
    /// cannot be initialized from the resolved config.
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

    /// Return the execution-side instance shard for this resolved config.
    pub fn shard_instances(&self) -> &[String] {
        &self.execution.shard.instances
    }

    /// Return the effective explicit instance-id selection for this resolved run.
    ///
    /// This reflects either the canonical contract instance list or an execution-side
    /// shard instance list. Window-based shard controls remain available through
    /// `shard_instance_limit()` and `shard_instance_offset()`.
    pub fn selected_instances(&self) -> &[String] {
        if self.execution.shard.instances.is_empty() {
            &self.contract.instances
        } else {
            &self.execution.shard.instances
        }
    }

    /// Return the optional session limit for this resolved contract.
    pub fn session_limit(&self) -> Option<usize> {
        self.contract.session_limit
    }

    /// Return the optional execution-side instance limit for this resolved run.
    pub fn shard_instance_limit(&self) -> Option<usize> {
        self.execution.shard.instance_limit
    }

    /// Return the execution-side instance offset for this resolved run.
    pub fn shard_instance_offset(&self) -> usize {
        self.execution.shard.instance_offset
    }

    /// Return the resolved ingest format.
    pub fn ingest_format(&self) -> &'static str {
        match self.contract.ingest_format {
            LongMemEvalIngestFormat::Text => "text",
            LongMemEvalIngestFormat::Json => "json",
            LongMemEvalIngestFormat::Round => "round",
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

    /// Return the resolved retain chunking configuration.
    pub fn retain_chunk_config(&self) -> ChunkConfig {
        ChunkConfig {
            max_tokens: self.contract.runtime.tuning.retain_chunk_max_tokens,
            overlap_tokens: self.contract.runtime.tuning.retain_chunk_overlap_tokens,
            preserve_turns: true,
        }
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
    ///
    /// # Errors
    ///
    /// Returns an error if the override is blank or if the overridden judge config
    /// fails validation.
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
    ///
    /// # Errors
    ///
    /// Returns an error if the resolved runtime provider, model stack, or tuning is
    /// invalid or incomplete.
    pub fn build_runtime_config(&self) -> Result<RuntimeConfig> {
        build_runtime_config_from_parts(
            &self.contract.runtime,
            &self.execution.database_url,
            self.execution.ort_dylib_path.as_deref(),
            &self.execution.runtime_target,
            &self.secrets,
        )
    }

    /// Build a typed benchmark judge config from the resolved contract, execution, and secrets.
    ///
    /// # Errors
    ///
    /// Returns an error if the resolved judge provider, model, or credentials are
    /// invalid or incomplete.
    pub fn build_judge_config(&self) -> Result<BenchJudgeConfig> {
        build_judge_config_from_parts(
            &self.contract.judge,
            &self.execution.judge_target,
            &self.secrets,
        )
    }

    /// Return a cloned resolved config with CLI execution-only overrides applied.
    ///
    /// # Errors
    ///
    /// Returns an error if the updated execution settings are invalid or if a
    /// dataset override does not match the declared dataset fingerprint.
    pub fn with_cli_execution_overrides(
        &self,
        dataset_path: Option<&Path>,
        tag: Option<&str>,
        instance_jobs: Option<usize>,
        shard_instances: &[String],
        shard_instance_limit: Option<usize>,
        shard_instance_offset: Option<usize>,
    ) -> Result<Self> {
        let mut resolved = self.clone();
        if let Some(dataset_path) = dataset_path {
            resolved.execution.dataset_path = dataset_path.to_path_buf();
        }
        if let Some(tag) = tag {
            resolved.execution.tag = Some(tag.to_string());
        }
        if let Some(instance_jobs) = instance_jobs {
            resolved.execution.instance_jobs = instance_jobs;
        }
        if !shard_instances.is_empty() {
            resolved.execution.shard.instances = shard_instances.to_vec();
        }
        if shard_instance_limit.is_some() {
            resolved.execution.shard.instance_limit = shard_instance_limit;
        }
        if let Some(shard_instance_offset) = shard_instance_offset {
            resolved.execution.shard.instance_offset = shard_instance_offset;
        }
        validate_longmemeval_execution(&resolved.execution)?;
        validate_longmemeval_execution_shard(&resolved.contract, &resolved.execution.shard)?;
        if dataset_path.is_some() {
            resolved.contract.dataset_fingerprint = load_dataset_fingerprint(
                &resolved.execution.dataset_path,
                &resolved.contract.dataset_identifier,
                resolved.contract.expected_dataset_fingerprint.as_deref(),
            )?;
            resolved.contract_hash = contract_hash_for(&resolved.contract)?;
        }
        Ok(resolved)
    }

    /// Build a benchmark harness from the resolved runtime contract.
    ///
    /// # Errors
    ///
    /// Returns an error if runtime construction fails or if the benchmark harness
    /// cannot be initialized from the resolved config.
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
    #[serde(skip_serializing_if = "Option::is_none")]
    ort_dylib_path: Option<String>,
    dataset_path: String,
    output_dir: String,
    tag: Option<&'a str>,
    conversation_jobs: usize,
    question_jobs: usize,
    database_url: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    shard: Option<&'a LocomoShardExecution>,
    #[serde(skip_serializing_if = "Option::is_none")]
    runtime_target: Option<&'a ClientTargetExecution>,
    #[serde(skip_serializing_if = "Option::is_none")]
    judge_target: Option<&'a ClientTargetExecution>,
}

impl<'a> From<&'a BenchExecution> for RedactedBenchExecution<'a> {
    fn from(value: &'a BenchExecution) -> Self {
        Self {
            ort_dylib_path: value
                .ort_dylib_path
                .as_ref()
                .map(|path| path.display().to_string()),
            dataset_path: value.dataset_path.display().to_string(),
            output_dir: value.output_dir.display().to_string(),
            tag: value.tag.as_deref(),
            conversation_jobs: value.conversation_jobs,
            question_jobs: value.question_jobs,
            database_url: "<redacted>",
            shard: (!value.shard.is_empty()).then_some(&value.shard),
            runtime_target: (!value.runtime_target.is_empty()).then_some(&value.runtime_target),
            judge_target: (!value.judge_target.is_empty()).then_some(&value.judge_target),
        }
    }
}

#[derive(Debug, Serialize)]
struct RedactedLongMemEvalExecution<'a> {
    #[serde(skip_serializing_if = "Option::is_none")]
    ort_dylib_path: Option<String>,
    dataset_path: String,
    output_dir: String,
    tag: Option<&'a str>,
    instance_jobs: usize,
    database_url: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    shard: Option<&'a LongMemEvalShardExecution>,
    #[serde(skip_serializing_if = "Option::is_none")]
    runtime_target: Option<&'a ClientTargetExecution>,
    #[serde(skip_serializing_if = "Option::is_none")]
    judge_target: Option<&'a ClientTargetExecution>,
}

impl<'a> From<&'a LongMemEvalExecution> for RedactedLongMemEvalExecution<'a> {
    fn from(value: &'a LongMemEvalExecution) -> Self {
        Self {
            ort_dylib_path: value
                .ort_dylib_path
                .as_ref()
                .map(|path| path.display().to_string()),
            dataset_path: value.dataset_path.display().to_string(),
            output_dir: value.output_dir.display().to_string(),
            tag: value.tag.as_deref(),
            instance_jobs: value.instance_jobs,
            database_url: "<redacted>",
            shard: (!value.shard.is_empty()).then_some(&value.shard),
            runtime_target: (!value.runtime_target.is_empty()).then_some(&value.runtime_target),
            judge_target: (!value.judge_target.is_empty()).then_some(&value.judge_target),
        }
    }
}

/// Resolve a checked-in LoCoMo profile, optional execution overlay, and benchmark secrets.
///
/// # Errors
///
/// Returns an error if the profile or overlay cannot be loaded, if the canonical
/// profile fails explicitness linting, if secrets are invalid, or if the resolved
/// runtime and judge configs fail validation.
pub fn resolve_locomo_bench_config(
    profile_path: &Path,
    execution_overlay_path: Option<&Path>,
    secrets_env_file: Option<&Path>,
) -> Result<ResolvedBenchConfig> {
    let profile = load_contract_toml::<LocomoContractFile>(profile_path, BenchmarkKind::Locomo)?;
    validate_profile(&profile)?;
    let overlay = execution_overlay_path
        .map(load_toml::<BenchExecutionOverlayFile>)
        .transpose()?;
    let mut execution = BenchExecution::from_overlay(
        overlay,
        default_locomo_dataset_path(&profile.dataset.identifier)?,
    );
    execution.ort_dylib_path =
        resolve_local_ort_dylib_path(&profile.runtime, execution.ort_dylib_path)?;
    validate_execution(&execution)?;
    let secrets = BenchSecrets::load(secrets_env_file)?;

    let dataset_fingerprint = load_dataset_fingerprint(
        &execution.dataset_path,
        &profile.dataset.identifier,
        profile.dataset.expected_fingerprint.as_deref(),
    )?;

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

    validate_locomo_execution_shard(&resolved_contract, &execution.shard)?;

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
///
/// # Errors
///
/// Returns an error if the profile or overlay cannot be loaded, if the canonical
/// profile fails explicitness linting, if secrets are invalid, or if the resolved
/// runtime and judge configs fail validation.
pub fn resolve_longmemeval_bench_config(
    profile_path: &Path,
    execution_overlay_path: Option<&Path>,
    secrets_env_file: Option<&Path>,
) -> Result<ResolvedLongMemEvalBenchConfig> {
    let profile =
        load_contract_toml::<LongMemEvalContractFile>(profile_path, BenchmarkKind::LongMemEval)?;
    validate_longmemeval_profile(&profile)?;
    let overlay = execution_overlay_path
        .map(load_toml::<LongMemEvalExecutionOverlayFile>)
        .transpose()?;
    let mut execution = LongMemEvalExecution::from_overlay(
        overlay,
        default_longmemeval_dataset_path(&profile.dataset.identifier)?,
    );
    execution.ort_dylib_path =
        resolve_local_ort_dylib_path(&profile.runtime, execution.ort_dylib_path)?;
    validate_longmemeval_execution(&execution)?;
    let secrets = BenchSecrets::load(secrets_env_file)?;

    let dataset_fingerprint = load_dataset_fingerprint(
        &execution.dataset_path,
        &profile.dataset.identifier,
        profile.dataset.expected_fingerprint.as_deref(),
    )?;

    let resolved_contract = ResolvedLongMemEvalContract {
        benchmark: "longmemeval",
        schema_version: profile.schema_version,
        protocol_version: profile.protocol_version,
        dataset_identifier: profile.dataset.identifier,
        dataset_fingerprint,
        expected_dataset_fingerprint: profile.dataset.expected_fingerprint,
        instances: profile.slice.instances,
        session_limit: profile.slice.session_limit,
        ingest_format: profile.ingest_format,
        consolidation: profile.consolidation,
        determinism_requirement: profile.determinism_requirement,
        runtime: profile.runtime,
        judge: profile.judge,
        judge_prompt_hash: longmemeval_judge_prompt_hash(),
    };

    validate_longmemeval_execution_shard(&resolved_contract, &execution.shard)?;

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

fn load_dataset_fingerprint(
    dataset_path: &Path,
    dataset_identifier: &str,
    expected_fingerprint: Option<&str>,
) -> Result<String> {
    let dataset_path = resolve_workspace_path(dataset_path);
    let dataset_bytes = fs::read(&dataset_path).map_err(|error| {
        ConfigError::configuration(format!(
            "failed to read dataset {}: {error}",
            dataset_path.display()
        ))
    })?;
    let dataset_fingerprint = fnv1a64_hex(&dataset_bytes);
    if let Some(expected_fingerprint) = expected_fingerprint
        && expected_fingerprint != dataset_fingerprint
    {
        return Err(ConfigError::configuration(format!(
            "dataset fingerprint mismatch for {dataset_identifier}: expected {expected_fingerprint}, got {dataset_fingerprint}"
        )));
    }
    Ok(dataset_fingerprint)
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

fn load_contract_toml<T: serde::de::DeserializeOwned>(
    path: &Path,
    benchmark_kind: BenchmarkKind,
) -> Result<T> {
    let resolved = resolve_workspace_path(path);
    let raw = fs::read_to_string(&resolved).map_err(|error| {
        ConfigError::configuration(format!("failed to read {}: {error}", path.display()))
    })?;
    lint_canonical_contract_file(path, &raw, benchmark_kind)?;
    toml::from_str(&raw).map_err(|error| {
        ConfigError::configuration(format!("failed to parse {}: {error}", path.display()))
    })
}

fn lint_canonical_contract_file(
    path: &Path,
    raw: &str,
    benchmark_kind: BenchmarkKind,
) -> Result<()> {
    if !is_checked_in_profile_path(path) {
        return Ok(());
    }

    let parsed = toml::from_str::<toml::Value>(raw).map_err(|error| {
        ConfigError::configuration(format!(
            "failed to parse {} while linting canonical profile explicitness: {error}",
            path.display()
        ))
    })?;

    let mut required = vec![
        "schema_version",
        "benchmark",
        "protocol_version",
        "dataset.identifier",
        "dataset.expected_fingerprint",
        "runtime.llm.provider",
        "runtime.llm.retain_model",
        "runtime.llm.reflect_model",
        "runtime.embedding.provider",
        "runtime.embedding.model",
        "runtime.reranker.provider",
        "judge.provider",
        "judge.model",
        "judge.max_tokens",
        "judge.max_attempts",
        "runtime.tuning.reflect_budget_tokens",
        "runtime.tuning.dedup_threshold",
        "runtime.tuning.retain_chunk_max_tokens",
        "runtime.tuning.retain_chunk_overlap_tokens",
        "runtime.tuning.extraction.structured_output_max_attempts",
        "runtime.tuning.graph.semantic_threshold",
        "runtime.tuning.graph.temporal_max_days",
        "runtime.tuning.graph.enable_causal",
        "runtime.tuning.graph.max_causal_checks",
        "runtime.tuning.consolidation.batch_size",
        "runtime.tuning.consolidation.max_tokens",
        "runtime.tuning.consolidation.recall_budget",
        "runtime.tuning.consolidation.structured_output_max_attempts",
        "runtime.tuning.reflect.max_iterations",
        "runtime.tuning.reflect.source_limit",
        "runtime.tuning.reflect.enable_source_lookup",
        "runtime.tuning.retrieval.retriever_limit",
        "runtime.tuning.retrieval.max_facts",
    ];
    match benchmark_kind {
        BenchmarkKind::Locomo => {
            required.extend(["ingest", "consolidation", "slice.category_filter"]);
        }
        BenchmarkKind::LongMemEval => {
            required.extend(["ingest_format", "consolidation"]);
        }
    }

    let mut missing = required
        .into_iter()
        .filter(|field| !toml_path_exists(&parsed, field))
        .map(str::to_string)
        .collect::<Vec<_>>();

    if toml_path_str(&parsed, "runtime.reranker.provider")
        .is_some_and(|provider| provider != "none")
        && !toml_path_exists(&parsed, "runtime.reranker.model")
    {
        missing.push("runtime.reranker.model".to_string());
    }

    if toml_path_str(&parsed, "runtime.reranker.provider") == Some("api")
        && !toml_path_exists(&parsed, "runtime.reranker.api_url")
    {
        missing.push("runtime.reranker.api_url".to_string());
    }

    if toml_path_str(&parsed, "runtime.embedding.provider") == Some("openai")
        && !toml_path_exists(&parsed, "runtime.embedding.dimensions")
    {
        missing.push("runtime.embedding.dimensions".to_string());
    }

    if missing.is_empty() {
        return Ok(());
    }

    missing.sort();
    missing.dedup();
    Err(ConfigError::configuration(format!(
        "canonical benchmark profile {} must explicitly set: {}",
        path.display(),
        missing.join(", ")
    )))
}

fn is_checked_in_profile_path(path: &Path) -> bool {
    let parts = path
        .components()
        .filter_map(|component| component.as_os_str().to_str())
        .collect::<Vec<_>>();
    parts
        .windows(3)
        .any(|window| window[0] == "bench" && window[2] == "profiles")
}

fn toml_path_exists(value: &toml::Value, dotted_path: &str) -> bool {
    toml_lookup(value, dotted_path).is_some()
}

fn toml_path_str<'a>(value: &'a toml::Value, dotted_path: &str) -> Option<&'a str> {
    toml_lookup(value, dotted_path)?.as_str()
}

fn toml_lookup<'a>(value: &'a toml::Value, dotted_path: &str) -> Option<&'a toml::Value> {
    let mut current = value;
    for segment in dotted_path.split('.') {
        current = current.get(segment)?;
    }
    Some(current)
}

fn default_locomo_dataset_path(identifier: &str) -> Result<std::path::PathBuf> {
    match identifier {
        "locomo10" => Ok(std::path::PathBuf::from("data/locomo10.json")),
        other => Err(ConfigError::configuration(format!(
            "unsupported locomo dataset.identifier for implicit dataset path: {other}"
        ))),
    }
}

fn default_longmemeval_dataset_path(identifier: &str) -> Result<std::path::PathBuf> {
    match identifier {
        "longmemeval-s" => Ok(std::path::PathBuf::from("data/longmemeval_s_cleaned.json")),
        "longmemeval-m" => Ok(std::path::PathBuf::from("data/longmemeval_m_cleaned.json")),
        other => Err(ConfigError::configuration(format!(
            "unsupported longmemeval dataset.identifier for implicit dataset path: {other}"
        ))),
    }
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
        Some("bench" | "data" | "models" | "lib")
    );
    if !anchor_to_workspace {
        return path.to_path_buf();
    }

    workspace_root().join(path)
}

fn workspace_root() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("bench crate must live under the workspace root")
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
    validate_runtime_contract(&profile.runtime)?;
    validate_judge_contract(&profile.judge)?;
    Ok(())
}

fn validate_runtime_contract(runtime: &RuntimeContract) -> Result<()> {
    validate_nonblank("runtime.llm.retain_model", &runtime.llm.retain_model)?;
    validate_nonblank("runtime.llm.reflect_model", &runtime.llm.reflect_model)?;
    validate_nonblank("runtime.embedding.model", &runtime.embedding.model)?;
    if runtime.tuning.retain_chunk_max_tokens == 0 {
        return Err(ConfigError::configuration(
            "runtime.tuning.retain_chunk_max_tokens must be greater than 0",
        ));
    }
    if runtime.tuning.retain_chunk_overlap_tokens >= runtime.tuning.retain_chunk_max_tokens {
        return Err(ConfigError::configuration(
            "runtime.tuning.retain_chunk_overlap_tokens must be less than runtime.tuning.retain_chunk_max_tokens",
        ));
    }
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
        RerankerProviderKind::Local => {
            let model = runtime.reranker.model.as_deref().ok_or_else(|| {
                ConfigError::configuration(
                    "runtime.reranker.model must be set when reranker provider is not none",
                )
            })?;
            validate_nonblank("runtime.reranker.model", model)?;
        }
        RerankerProviderKind::Api => {
            let model = runtime.reranker.model.as_deref().ok_or_else(|| {
                ConfigError::configuration(
                    "runtime.reranker.model must be set when reranker provider is api",
                )
            })?;
            validate_nonblank("runtime.reranker.model", model)?;
            let api_url = runtime.reranker.api_url.as_deref().ok_or_else(|| {
                ConfigError::configuration(
                    "runtime.reranker.api_url must be set for runtime.reranker.provider=api",
                )
            })?;
            validate_nonblank("runtime.reranker.api_url", api_url)?;
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
    validate_client_target("execution.runtime_target", &execution.runtime_target)?;
    validate_client_target("execution.judge_target", &execution.judge_target)?;
    validate_locomo_execution_shard_shape(&execution.shard)?;
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
    Ok(())
}

fn validate_longmemeval_execution(execution: &LongMemEvalExecution) -> Result<()> {
    validate_nonblank("execution.database_url", &execution.database_url)?;
    validate_client_target("execution.runtime_target", &execution.runtime_target)?;
    validate_client_target("execution.judge_target", &execution.judge_target)?;
    validate_longmemeval_execution_shard_shape(&execution.shard)?;
    if execution.instance_jobs == 0 {
        return Err(ConfigError::configuration(
            "execution.instance_jobs must be greater than 0",
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

fn validate_client_target(name: &str, target: &ClientTargetExecution) -> Result<()> {
    if let Some(base_url) = target.base_url.as_deref() {
        validate_nonblank(&format!("{name}.base_url"), base_url)?;
    }
    if let Some(vertex_project) = target.vertex_project.as_deref() {
        validate_nonblank(&format!("{name}.vertex_project"), vertex_project)?;
    }
    if let Some(vertex_location) = target.vertex_location.as_deref() {
        validate_nonblank(&format!("{name}.vertex_location"), vertex_location)?;
    }
    if let Some(key) = target.prompt_cache.key.as_deref() {
        validate_nonblank(&format!("{name}.prompt_cache.key"), key)?;
    }
    if !target.prompt_cache.enabled
        && (target.prompt_cache.key.is_some()
            || target.prompt_cache.retention.is_some()
            || target.prompt_cache.ttl.is_some())
    {
        return Err(ConfigError::configuration(format!(
            "{name}.prompt_cache fields require prompt_cache.enabled = true"
        )));
    }
    Ok(())
}

fn validate_locomo_execution_shard_shape(shard: &LocomoShardExecution) -> Result<()> {
    ensure_unique_nonblank_ids("execution.shard.conversations", &shard.conversations)
}

fn validate_locomo_execution_shard(
    contract: &ResolvedLocomoContract,
    shard: &LocomoShardExecution,
) -> Result<()> {
    validate_locomo_execution_shard_shape(shard)?;
    if contract.conversations.is_empty() || shard.conversations.is_empty() {
        return Ok(());
    }

    let allowed = contract
        .conversations
        .iter()
        .map(String::as_str)
        .collect::<std::collections::HashSet<_>>();
    let invalid = shard
        .conversations
        .iter()
        .filter(|id| !allowed.contains(id.as_str()))
        .cloned()
        .collect::<Vec<_>>();
    if invalid.is_empty() {
        Ok(())
    } else {
        Err(ConfigError::configuration(format!(
            "execution.shard.conversations must be a subset of the contract slice; invalid ids: {}",
            invalid.join(", ")
        )))
    }
}

fn validate_longmemeval_execution_shard_shape(shard: &LongMemEvalShardExecution) -> Result<()> {
    ensure_unique_nonblank_ids("execution.shard.instances", &shard.instances)?;
    if matches!(shard.instance_limit, Some(0)) {
        return Err(ConfigError::configuration(
            "execution.shard.instance_limit must be greater than 0 if set",
        ));
    }
    Ok(())
}

fn validate_longmemeval_execution_shard(
    contract: &ResolvedLongMemEvalContract,
    shard: &LongMemEvalShardExecution,
) -> Result<()> {
    validate_longmemeval_execution_shard_shape(shard)?;
    if contract.instances.is_empty() || shard.instances.is_empty() {
        return Ok(());
    }

    let allowed = contract
        .instances
        .iter()
        .map(String::as_str)
        .collect::<std::collections::HashSet<_>>();
    let invalid = shard
        .instances
        .iter()
        .filter(|id| !allowed.contains(id.as_str()))
        .cloned()
        .collect::<Vec<_>>();
    if invalid.is_empty() {
        Ok(())
    } else {
        Err(ConfigError::configuration(format!(
            "execution.shard.instances must be a subset of the contract slice; invalid ids: {}",
            invalid.join(", ")
        )))
    }
}

fn ensure_unique_nonblank_ids(name: &str, ids: &[String]) -> Result<()> {
    let mut seen = std::collections::HashSet::new();
    for id in ids {
        validate_nonblank(name, id)?;
        if !seen.insert(id.as_str()) {
            return Err(ConfigError::configuration(format!(
                "{name} must not contain duplicates: {id}"
            )));
        }
    }
    Ok(())
}

fn build_runtime_config_from_parts(
    runtime: &RuntimeContract,
    database_url: &str,
    ort_dylib_path: Option<&Path>,
    runtime_target: &ClientTargetExecution,
    secrets: &BenchSecrets,
) -> Result<RuntimeConfig> {
    initialize_local_ort(runtime, ort_dylib_path)?;
    let llm = build_runtime_llm_config(runtime, runtime_target, secrets)?;
    let embedding = build_embedding_config(runtime, secrets)?;
    let reranker = build_reranker_config(runtime, secrets)?;
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
        .with_chunk_config(ChunkConfig {
            max_tokens: tuning.retain_chunk_max_tokens,
            overlap_tokens: tuning.retain_chunk_overlap_tokens,
            preserve_turns: true,
        })?
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
            max_causal_checks: tuning.graph.max_causal_checks,
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

fn needs_local_ort(runtime: &RuntimeContract) -> bool {
    runtime.embedding.provider == EmbeddingProviderKind::Local
        || runtime.reranker.provider == RerankerProviderKind::Local
}

fn resolve_local_ort_dylib_path(
    runtime: &RuntimeContract,
    ort_dylib_path: Option<PathBuf>,
) -> Result<Option<PathBuf>> {
    if !needs_local_ort(runtime) {
        return Ok(None);
    }

    if let Some(path) = ort_dylib_path {
        validate_ort_dylib_path(&path)?;
        return Ok(Some(path));
    }

    let discovered = discover_repo_local_ort_dylib_path().ok_or_else(|| {
        ConfigError::configuration(
            "local embeddings or reranking require ONNX Runtime; set execution.ort_dylib_path or install a repo-local ONNX Runtime under lib/onnxruntime-*/lib"
        )
    })?;
    Ok(Some(discovered))
}

fn validate_ort_dylib_path(path: &Path) -> Result<()> {
    if path.as_os_str().is_empty() {
        return Err(ConfigError::configuration(
            "execution.ort_dylib_path must not be blank",
        ));
    }

    let resolved = resolve_workspace_path(path);
    if !resolved.is_file() {
        return Err(ConfigError::configuration(format!(
            "execution.ort_dylib_path does not point to a file: {}",
            resolved.display()
        )));
    }
    Ok(())
}

fn discover_repo_local_ort_dylib_path() -> Option<PathBuf> {
    let lib_root = resolve_workspace_path(Path::new("lib"));
    let entries = fs::read_dir(lib_root).ok()?;
    let mut candidates = entries
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| path.is_dir())
        .filter_map(|dir| {
            let name = dir.file_name()?.to_str()?;
            if !name.starts_with("onnxruntime-") {
                return None;
            }
            let dylib_dir = dir.join("lib");
            let dylib_entries = fs::read_dir(dylib_dir).ok()?;
            let mut dylibs = dylib_entries
                .flatten()
                .map(|entry| entry.path())
                .filter(|path| path.is_file())
                .filter(|path| {
                    path.file_name()
                        .and_then(|name| name.to_str())
                        .is_some_and(is_ort_dylib_filename)
                })
                .collect::<Vec<_>>();
            dylibs.sort();
            dylibs
                .into_iter()
                .next()
                .and_then(|path| path.strip_prefix(workspace_root()).ok().map(PathBuf::from))
        })
        .collect::<Vec<_>>();
    candidates.sort();
    candidates.into_iter().next()
}

fn is_ort_dylib_filename(name: &str) -> bool {
    #[cfg(target_os = "windows")]
    {
        name.eq_ignore_ascii_case("onnxruntime.dll")
    }
    #[cfg(any(target_os = "linux", target_os = "android"))]
    {
        name == "libonnxruntime.so" || name.starts_with("libonnxruntime.so.")
    }
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    {
        name == "libonnxruntime.dylib" || name.starts_with("libonnxruntime.")
    }
}

fn initialize_local_ort(runtime: &RuntimeContract, ort_dylib_path: Option<&Path>) -> Result<()> {
    static ORT_DYLIB_INIT: OnceLock<PathBuf> = OnceLock::new();

    if !needs_local_ort(runtime) {
        return Ok(());
    }

    let ort_dylib_path = ort_dylib_path.ok_or_else(|| {
        ConfigError::configuration(
            "local embeddings or reranking require a resolved execution.ort_dylib_path",
        )
    })?;
    let resolved = resolve_workspace_path(ort_dylib_path);

    if let Some(existing) = ORT_DYLIB_INIT.get() {
        if existing == &resolved {
            return Ok(());
        }
        return Err(ConfigError::configuration(format!(
            "ONNX Runtime was already initialized from {} and cannot be reconfigured to {} in the same benchmark process",
            existing.display(),
            resolved.display()
        )));
    }

    let _ = ort::init_from(&resolved).map_err(|error| {
        ConfigError::configuration(format!(
            "failed to preload ONNX Runtime from {}: {error}",
            resolved.display()
        ))
    })?;
    let _ = ORT_DYLIB_INIT.set(resolved);
    Ok(())
}

fn build_judge_config_from_parts(
    judge: &JudgeContract,
    judge_target: &ClientTargetExecution,
    secrets: &BenchSecrets,
) -> Result<BenchJudgeConfig> {
    let client = build_client_config(
        judge.provider,
        secrets.judge_api_key(),
        &judge.model,
        judge_target.base_url.as_deref(),
        DEFAULT_TIMEOUT_SECS,
        judge_target.vertex_project.as_deref(),
        judge_target.vertex_location.as_deref(),
        &judge_target.prompt_cache,
    )?;
    Ok(BenchJudgeConfig::new(
        client,
        judge.temperature,
        judge.max_tokens,
        judge.max_attempts,
    ))
}

fn build_runtime_llm_config(
    runtime: &RuntimeContract,
    runtime_target: &ClientTargetExecution,
    secrets: &BenchSecrets,
) -> Result<LlmConfig> {
    let retain = build_client_config(
        runtime.llm.provider,
        secrets.runtime_api_key(),
        &runtime.llm.retain_model,
        runtime_target.base_url.as_deref(),
        DEFAULT_TIMEOUT_SECS,
        runtime_target.vertex_project.as_deref(),
        runtime_target.vertex_location.as_deref(),
        &runtime_target.prompt_cache,
    )?;
    let reflect = build_client_config(
        runtime.llm.provider,
        secrets.runtime_api_key(),
        &runtime.llm.reflect_model,
        runtime_target.base_url.as_deref(),
        DEFAULT_TIMEOUT_SECS,
        runtime_target.vertex_project.as_deref(),
        runtime_target.vertex_location.as_deref(),
        &runtime_target.prompt_cache,
    )?;
    Ok(LlmConfig::new(retain, reflect))
}

fn build_embedding_config(
    runtime: &RuntimeContract,
    secrets: &BenchSecrets,
) -> Result<EmbeddingConfig> {
    match runtime.embedding.provider {
        super::contract::EmbeddingProviderKind::Local => {
            let resolved_path =
                resolve_workspace_path(&Path::new("models").join(&runtime.embedding.model));
            Ok(EmbeddingConfig::local(resolved_path.display().to_string()))
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
            Ok(EmbeddingConfig::openai(
                api_key,
                &runtime.embedding.model,
                dimensions,
            ))
        }
    }
}

fn build_reranker_config(
    runtime: &RuntimeContract,
    secrets: &BenchSecrets,
) -> Result<RerankerConfig> {
    match runtime.reranker.provider {
        RerankerProviderKind::None => Ok(RerankerConfig::none()),
        RerankerProviderKind::Local => {
            let model = runtime.reranker.model.as_deref().ok_or_else(|| {
                ConfigError::configuration(
                    "runtime.reranker.model must be set for runtime.reranker.provider=local",
                )
            })?;
            let resolved_path = resolve_workspace_path(&Path::new("models").join(model));
            Ok(RerankerConfig::local(resolved_path.display().to_string()))
        }
        RerankerProviderKind::Api => {
            let api_key = secrets.reranker_api_key().ok_or_else(|| {
                ConfigError::configuration(
                    "ELEPHANT_BENCH_RERANKER_API_KEY must be set for runtime.reranker.provider=api",
                )
            })?;
            let api_url = runtime.reranker.api_url.as_deref().ok_or_else(|| {
                ConfigError::configuration(
                    "runtime.reranker.api_url must be set for runtime.reranker.provider=api",
                )
            })?;
            let model = runtime.reranker.model.as_deref().ok_or_else(|| {
                ConfigError::configuration(
                    "runtime.reranker.model must be set for runtime.reranker.provider=api",
                )
            })?;
            Ok(RerankerConfig::api(api_key, api_url, model))
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
    prompt_cache: &PromptCacheExecution,
) -> Result<ClientConfig> {
    let api_key = api_key.ok_or_else(|| {
        ConfigError::configuration(format!(
            "missing API key for provider {}",
            provider.as_str()
        ))
    })?;

    match provider {
        ProviderKind::Anthropic => {
            if base_url.is_some() {
                return Err(ConfigError::configuration(
                    "base_url is not supported for provider=anthropic",
                ));
            }
            if prompt_cache.key.is_some() || prompt_cache.retention.is_some() {
                return Err(ConfigError::configuration(
                    "OpenAI prompt cache fields are not supported for provider=anthropic",
                ));
            }
            if vertex_project.is_some() || vertex_location.is_some() {
                return Err(ConfigError::configuration(
                    "vertex_project and vertex_location are only supported for provider=vertex",
                ));
            }
            let mut config = AnthropicConfig::new(api_key, model).map_err(ConfigError::from)?;
            config = config
                .with_timeout_secs(nonzero_timeout(timeout_secs)?)
                .map_err(ConfigError::from)?;
            if prompt_cache.enabled {
                let mut cache = AnthropicPromptCacheConfig::new();
                if let Some(ttl) = prompt_cache.ttl {
                    cache = cache.with_ttl(ttl);
                }
                config = config.with_prompt_cache(cache);
            }
            Ok(ClientConfig::Anthropic(config))
        }
        ProviderKind::OpenAi => {
            if vertex_project.is_some() || vertex_location.is_some() {
                return Err(ConfigError::configuration(
                    "vertex_project and vertex_location are only supported for provider=vertex",
                ));
            }
            if prompt_cache.ttl.is_some() {
                return Err(ConfigError::configuration(
                    "Anthropic prompt cache ttl is not supported for provider=openai",
                ));
            }
            let mut config = OpenAiConfig::new(api_key, model).map_err(ConfigError::from)?;
            config = config
                .with_timeout_secs(nonzero_timeout(timeout_secs)?)
                .map_err(ConfigError::from)?;
            if let Some(base_url) = base_url {
                config = config.with_base_url(base_url).map_err(ConfigError::from)?;
            }
            if prompt_cache.enabled {
                if let Some(key) = prompt_cache.key.as_ref() {
                    config = config
                        .with_prompt_cache(OpenAiPromptCacheConfig::new().with_key(key.clone()));
                }
                if let Some(retention) = prompt_cache.retention {
                    let cache = config
                        .prompt_cache()
                        .cloned()
                        .unwrap_or_else(OpenAiPromptCacheConfig::new)
                        .with_retention(retention);
                    config = config.with_prompt_cache(cache);
                }
                if prompt_cache.key.is_none() && prompt_cache.retention.is_none() {
                    config = config.with_prompt_cache(OpenAiPromptCacheConfig::new());
                }
            }
            Ok(ClientConfig::OpenAi(config))
        }
        ProviderKind::Gemini => {
            if prompt_cache.enabled {
                return Err(ConfigError::configuration(
                    "prompt_cache is not supported for provider=gemini in benchmark execution config",
                ));
            }
            if vertex_project.is_some() || vertex_location.is_some() {
                return Err(ConfigError::configuration(
                    "vertex_project and vertex_location are only supported for provider=vertex",
                ));
            }
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
            if prompt_cache.enabled {
                return Err(ConfigError::configuration(
                    "prompt_cache is not supported for provider=vertex in benchmark execution config",
                ));
            }
            let project = vertex_project.ok_or_else(|| {
                ConfigError::configuration("vertex_project must be set for provider=vertex")
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
protocol_version = "2026-04-15-locomo-contract-v3"
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

[runtime.tuning]
reflect_budget_tokens = 4096
dedup_threshold = 0.95
retain_chunk_max_tokens = 512
retain_chunk_overlap_tokens = 64

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
"#,
            dataset_path.display()
        )
    }

    fn sample_overlay_with_prompt_cache(dataset_path: &Path) -> String {
        format!(
            r#"
database_url = "postgres://bench:bench@localhost/elephant"
dataset_path = "{}"
conversation_jobs = 1
question_jobs = 4

[runtime_target]
base_url = "https://openrouter.ai/api/v1"

[runtime_target.prompt_cache]
enabled = true
key = "elephant:bench:runtime"
retention = "in_memory"

[judge_target]
base_url = "https://openrouter.ai/api/v1"

[judge_target.prompt_cache]
enabled = true
key = "elephant:bench:judge"
retention = "in_memory"
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
    fn resolves_locomo_config_and_auto_discovers_repo_local_ort_path() {
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

        let resolved =
            resolve_locomo_bench_config(&profile, Some(&overlay), Some(&secrets)).unwrap();

        let expected = discover_repo_local_ort_dylib_path().unwrap();
        assert_eq!(
            resolved.redacted_execution_json()["ort_dylib_path"],
            serde_json::Value::String(expected.display().to_string())
        );
    }

    #[test]
    fn resolves_locomo_overlay_with_prompt_cache_targets() {
        let _guard = env_lock().lock().unwrap();
        let profile = temp_path("profile.toml");
        let overlay = temp_path("overlay.toml");
        let dataset = temp_path("dataset.json");
        let secrets = temp_path("secrets.env");

        write_file(&profile, &sample_profile());
        write_file(&dataset, r#"{"sample":"ok"}"#);
        write_file(&overlay, &sample_overlay_with_prompt_cache(&dataset));
        write_file(
            &secrets,
            "ELEPHANT_BENCH_RUNTIME_API_KEY=runtime-secret\nELEPHANT_BENCH_JUDGE_API_KEY=judge-secret\n",
        );

        let resolved =
            resolve_locomo_bench_config(&profile, Some(&overlay), Some(&secrets)).unwrap();

        assert_eq!(
            resolved.redacted_execution_json()["runtime_target"]["prompt_cache"]["enabled"],
            serde_json::Value::Bool(true)
        );
        assert_eq!(
            resolved.redacted_execution_json()["judge_target"]["prompt_cache"]["retention"],
            serde_json::Value::String("in_memory".into())
        );
    }

    #[test]
    fn build_client_config_applies_openai_prompt_cache() {
        let prompt_cache = PromptCacheExecution {
            enabled: true,
            key: Some("elephant:bench".into()),
            retention: Some(elephant::llm::OpenAiPromptCacheRetention::InMemory),
            ttl: None,
        };

        let client = build_client_config(
            ProviderKind::OpenAi,
            Some("runtime-secret"),
            "gpt-5.4-mini",
            Some("https://openrouter.ai/api/v1"),
            DEFAULT_TIMEOUT_SECS,
            None,
            None,
            &prompt_cache,
        )
        .unwrap();

        match client {
            ClientConfig::OpenAi(config) => {
                let prompt_cache = config.prompt_cache().expect("prompt cache should be set");
                assert_eq!(prompt_cache.key(), Some("elephant:bench"));
                assert_eq!(
                    prompt_cache.retention(),
                    Some(elephant::llm::OpenAiPromptCacheRetention::InMemory)
                );
            }
            other => panic!("expected openai client, got {other:?}"),
        }
    }

    #[test]
    fn build_client_config_applies_anthropic_prompt_cache() {
        let prompt_cache = PromptCacheExecution {
            enabled: true,
            key: None,
            retention: None,
            ttl: Some(elephant::llm::AnthropicPromptCacheTtl::Hours1),
        };

        let client = build_client_config(
            ProviderKind::Anthropic,
            Some("runtime-secret"),
            "claude-sonnet",
            None,
            DEFAULT_TIMEOUT_SECS,
            None,
            None,
            &prompt_cache,
        )
        .unwrap();

        match client {
            ClientConfig::Anthropic(config) => {
                let prompt_cache = config.prompt_cache().expect("prompt cache should be set");
                assert_eq!(
                    prompt_cache.ttl(),
                    Some(elephant::llm::AnthropicPromptCacheTtl::Hours1)
                );
            }
            other => panic!("expected anthropic client, got {other:?}"),
        }
    }

    #[test]
    fn unknown_profile_field_fails_fast() {
        let profile = temp_path("bad-profile.toml");
        write_file(
            &profile,
            r#"
schema_version = 1
benchmark = "locomo"
protocol_version = "2026-04-15-locomo-contract-v3"
unexpected = true

[runtime.llm]
provider = "openai"
retain_model = "gpt-5.4-mini"
reflect_model = "gpt-5.4-mini"

[runtime.embedding]
provider = "local"
model = "bge-small-en-v1.5"

[runtime.tuning]
reflect_budget_tokens = 4096
dedup_threshold = 0.95
retain_chunk_max_tokens = 512
retain_chunk_overlap_tokens = 64

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
    fn vertex_runtime_provider_uses_local_execution_target() {
        let _guard = env_lock().lock().unwrap();
        let profile = temp_path("vertex-profile.toml");
        let overlay = temp_path("vertex-overlay.toml");
        let dataset = temp_path("dataset.json");
        let secrets = temp_path("vertex-secrets.env");
        write_file(
            &profile,
            r#"
schema_version = 1
benchmark = "locomo"
protocol_version = "2026-04-15-locomo-contract-v3"

[dataset]
identifier = "locomo10"

[slice]
category_filter = [1, 2, 3, 4]

[runtime.llm]
provider = "vertex"
retain_model = "gemini-2.5-flash"
reflect_model = "gemini-2.5-flash"

[runtime.embedding]
provider = "local"
model = "bge-small-en-v1.5"

[runtime.tuning]
reflect_budget_tokens = 4096
dedup_threshold = 0.95
retain_chunk_max_tokens = 512
retain_chunk_overlap_tokens = 64

[runtime.reranker]
provider = "none"

[judge]
provider = "vertex"
model = "gemini-2.5-flash"
"#,
        );
        write_file(&dataset, r#"{"sample":"ok"}"#);
        let overlay_raw = format!(
            r#"
database_url = "postgres://bench:bench@localhost/elephant"
dataset_path = "{}"

[runtime_target]
vertex_project = "bench-runtime-project"
vertex_location = "us-central1"

[judge_target]
vertex_project = "bench-judge-project"
"#,
            dataset.display()
        );
        write_file(&overlay, &overlay_raw);
        write_file(
            &secrets,
            "ELEPHANT_BENCH_RUNTIME_API_KEY=runtime-secret\nELEPHANT_BENCH_JUDGE_API_KEY=judge-secret\n",
        );

        let resolved =
            resolve_locomo_bench_config(&profile, Some(&overlay), Some(&secrets)).unwrap();
        let printed = resolved.to_pretty_redacted_json();
        assert!(printed.contains("\"runtime_target\""));
        assert!(printed.contains("bench-runtime-project"));
        assert!(printed.contains("bench-judge-project"));
    }

    #[test]
    fn vertex_provider_requires_local_project() {
        let _guard = env_lock().lock().unwrap();
        let profile = temp_path("vertex-missing-project.toml");
        let overlay = temp_path("vertex-missing-project-overlay.toml");
        let dataset = temp_path("dataset.json");
        let secrets = temp_path("vertex-missing-project-secrets.env");
        write_file(
            &profile,
            r#"
schema_version = 1
benchmark = "locomo"
protocol_version = "2026-04-15-locomo-contract-v3"

[dataset]
identifier = "locomo10"

[slice]
category_filter = [1, 2, 3, 4]

[runtime.llm]
provider = "vertex"
retain_model = "gemini-2.5-flash"
reflect_model = "gemini-2.5-flash"

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
        write_file(&dataset, r#"{"sample":"ok"}"#);
        let overlay_raw = format!(
            r#"
database_url = "postgres://bench:bench@localhost/elephant"
dataset_path = "{}"
"#,
            dataset.display()
        );
        write_file(&overlay, &overlay_raw);
        write_file(&secrets, "ELEPHANT_BENCH_RUNTIME_API_KEY=runtime-secret\n");

        let err =
            resolve_locomo_bench_config(&profile, Some(&overlay), Some(&secrets)).unwrap_err();
        assert!(err.to_string().contains("vertex_project must be set"));
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

    #[test]
    fn contract_hash_ignores_local_routing_targets_but_execution_records_them() {
        let _guard = env_lock().lock().unwrap();
        let profile = temp_path("profile.toml");
        let overlay_a = temp_path("overlay-a.toml");
        let overlay_b = temp_path("overlay-b.toml");
        let dataset = temp_path("dataset.json");
        let secrets = temp_path("secrets.env");

        write_file(&profile, &sample_profile());
        write_file(&dataset, r#"{"sample":"ok"}"#);
        write_file(&overlay_a, &sample_overlay(&dataset));
        let overlay_b_raw = format!(
            r#"
database_url = "postgres://bench:bench@localhost/elephant"
dataset_path = "{}"
conversation_jobs = 1
question_jobs = 4

[runtime_target]
base_url = "https://vertex-proxy.example.com"

[judge_target]
base_url = "https://judge-proxy.example.com"
"#,
            dataset.display()
        );
        write_file(&overlay_b, &overlay_b_raw);
        write_file(
            &secrets,
            "ELEPHANT_BENCH_RUNTIME_API_KEY=runtime-secret\nELEPHANT_BENCH_JUDGE_API_KEY=judge-secret\n",
        );

        let a = resolve_locomo_bench_config(&profile, Some(&overlay_a), Some(&secrets)).unwrap();
        let b = resolve_locomo_bench_config(&profile, Some(&overlay_b), Some(&secrets)).unwrap();

        assert_eq!(a.contract_hash(), b.contract_hash());
        assert_ne!(a.redacted_execution_json(), b.redacted_execution_json());
        assert_eq!(
            b.redacted_execution_json()["runtime_target"]["base_url"],
            serde_json::Value::String("https://vertex-proxy.example.com".into())
        );
        assert_eq!(
            b.redacted_execution_json()["judge_target"]["base_url"],
            serde_json::Value::String("https://judge-proxy.example.com".into())
        );
    }

    #[test]
    fn cli_dataset_override_updates_execution_and_contract_hash() {
        let _guard = env_lock().lock().unwrap();
        let profile = temp_path("profile.toml");
        let overlay = temp_path("overlay.toml");
        let dataset_a = temp_path("dataset-a.json");
        let dataset_b = temp_path("dataset-b.json");
        let secrets = temp_path("secrets.env");

        write_file(&profile, &sample_profile());
        write_file(&dataset_a, r#"{"sample":"a"}"#);
        write_file(&dataset_b, r#"{"sample":"b"}"#);
        write_file(&overlay, &sample_overlay(&dataset_a));
        write_file(
            &secrets,
            "ELEPHANT_BENCH_RUNTIME_API_KEY=runtime-secret\nELEPHANT_BENCH_JUDGE_API_KEY=judge-secret\n",
        );

        let resolved =
            resolve_locomo_bench_config(&profile, Some(&overlay), Some(&secrets)).unwrap();
        let overridden = resolved
            .with_cli_execution_overrides(Some(&dataset_b), Some("cli-tag"), Some(7), Some(3), &[])
            .unwrap();

        assert_ne!(resolved.contract_hash(), overridden.contract_hash());
        assert_eq!(overridden.dataset_path(), dataset_b.as_path());
        assert_eq!(overridden.tag(), Some("cli-tag"));
        assert_eq!(overridden.conversation_jobs(), 7);
        assert_eq!(overridden.question_jobs(), 3);
        assert_eq!(
            overridden.redacted_execution_json()["dataset_path"],
            serde_json::Value::String(dataset_b.display().to_string())
        );
    }

    #[test]
    fn contract_hash_ignores_locomo_shard_scope_but_execution_records_it() {
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

        let resolved =
            resolve_locomo_bench_config(&profile, Some(&overlay), Some(&secrets)).unwrap();
        let overridden = resolved
            .with_cli_execution_overrides(None, None, None, None, &["conv-26".to_string()])
            .unwrap();

        assert_eq!(resolved.contract_hash(), overridden.contract_hash());
        assert_ne!(
            resolved.redacted_execution_json(),
            overridden.redacted_execution_json()
        );
        assert_eq!(
            overridden.redacted_execution_json()["shard"]["conversations"],
            serde_json::json!(["conv-26"])
        );
    }

    #[test]
    fn locomo_shard_conversations_must_stay_within_contract_slice() {
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

        let err = resolve_locomo_bench_config(&profile, Some(&overlay), Some(&secrets))
            .unwrap()
            .with_cli_execution_overrides(None, None, None, None, &["conv-99".to_string()])
            .unwrap_err();

        assert!(
            err.to_string()
                .contains("execution.shard.conversations must be a subset")
        );
    }

    #[test]
    fn longmemeval_shard_scope_is_execution_only() {
        let _guard = env_lock().lock().unwrap();
        let profile = temp_path("long-profile.toml");
        let overlay = temp_path("long-overlay.toml");
        let dataset = temp_path("long-dataset.json");
        let secrets = temp_path("long-secrets.env");
        write_file(
            &profile,
            r#"
schema_version = 1
benchmark = "longmemeval"
protocol_version = "2026-04-15-longmemeval-contract-v3"
ingest_format = "round"
consolidation = "end"

[dataset]
identifier = "longmemeval-s"

[slice]
instances = ["q1", "q2"]

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
        write_file(&dataset, r#"{"sample":"ok"}"#);
        write_file(
            &overlay,
            format!(
                r#"
database_url = "postgres://bench:bench@localhost/elephant"
dataset_path = "{}"
"#,
                dataset.display()
            )
            .as_str(),
        );
        write_file(
            &secrets,
            "ELEPHANT_BENCH_RUNTIME_API_KEY=runtime-secret\nELEPHANT_BENCH_JUDGE_API_KEY=judge-secret\n",
        );

        let resolved =
            resolve_longmemeval_bench_config(&profile, Some(&overlay), Some(&secrets)).unwrap();
        let overridden = resolved
            .with_cli_execution_overrides(None, None, None, &["q1".to_string()], Some(1), None)
            .unwrap();

        assert_eq!(resolved.contract_hash(), overridden.contract_hash());
        assert_eq!(overridden.selected_instances(), &["q1".to_string()]);
        assert_eq!(overridden.shard_instance_limit(), Some(1));
        assert_eq!(
            overridden.redacted_execution_json()["shard"]["instances"],
            serde_json::json!(["q1"])
        );
    }

    #[test]
    fn longmemeval_round_ingest_keeps_chunk_config_explicit() {
        let _guard = env_lock().lock().unwrap();
        let profile = temp_path("long-round-profile.toml");
        let overlay = temp_path("long-round-overlay.toml");
        let dataset = temp_path("long-round-dataset.json");
        let secrets = temp_path("long-round-secrets.env");
        write_file(
            &profile,
            r#"
schema_version = 1
benchmark = "longmemeval"
protocol_version = "2026-04-15-longmemeval-contract-v3"
ingest_format = "round"
consolidation = "end"

[dataset]
identifier = "longmemeval-s"

[runtime.llm]
provider = "openai"
retain_model = "gpt-5.4-mini"
reflect_model = "gpt-5.4-mini"

[runtime.embedding]
provider = "local"
model = "bge-small-en-v1.5"

[runtime.reranker]
provider = "none"

[runtime.tuning]
reflect_budget_tokens = 4096
dedup_threshold = 0.95
retain_chunk_max_tokens = 2048
retain_chunk_overlap_tokens = 64

[runtime.tuning.extraction]
structured_output_max_attempts = 3

[runtime.tuning.graph]
semantic_threshold = 0.7
temporal_max_days = 30
enable_causal = true
max_causal_checks = 10

[runtime.tuning.consolidation]
batch_size = 8
max_tokens = 4096
recall_budget = 512
structured_output_max_attempts = 3

[runtime.tuning.reflect]
max_iterations = 8
source_limit = 3
enable_source_lookup = true

[runtime.tuning.retrieval]
retriever_limit = 40
max_facts = 50

[judge]
provider = "openai"
model = "gpt-5.4"
max_tokens = 200
max_attempts = 3
"#,
        );
        write_file(&dataset, r#"{"sample":"ok"}"#);
        write_file(
            &overlay,
            format!(
                r#"
database_url = "postgres://bench:bench@localhost/elephant"
dataset_path = "{}"
"#,
                dataset.display()
            )
            .as_str(),
        );
        write_file(
            &secrets,
            "ELEPHANT_BENCH_RUNTIME_API_KEY=runtime-secret\nELEPHANT_BENCH_JUDGE_API_KEY=judge-secret\n",
        );

        let resolved =
            resolve_longmemeval_bench_config(&profile, Some(&overlay), Some(&secrets)).unwrap();
        assert_eq!(resolved.ingest_format(), "round");
        let json = resolved.redacted_contract_json();
        assert_eq!(json["runtime"]["tuning"]["retain_chunk_max_tokens"], 2048);
        assert_eq!(json["runtime"]["tuning"]["retain_chunk_overlap_tokens"], 64);
        assert_eq!(json["runtime"]["tuning"]["graph"]["max_causal_checks"], 10);
    }

    #[test]
    fn longmemeval_round_ingest_allows_zero_chunk_overlap() {
        let _guard = env_lock().lock().unwrap();
        let profile = temp_path("long-round-zero-overlap-profile.toml");
        let overlay = temp_path("long-round-zero-overlap-overlay.toml");
        let dataset = temp_path("long-round-zero-overlap-dataset.json");
        let secrets = temp_path("long-round-zero-overlap-secrets.env");
        write_file(
            &profile,
            r#"
schema_version = 1
benchmark = "longmemeval"
protocol_version = "2026-04-15-longmemeval-contract-v3"
ingest_format = "round"
consolidation = "end"

[dataset]
identifier = "longmemeval-s"

[runtime.llm]
provider = "openai"
retain_model = "gpt-5.4-mini"
reflect_model = "gpt-5.4-mini"

[runtime.embedding]
provider = "local"
model = "bge-small-en-v1.5"

[runtime.reranker]
provider = "none"

[runtime.tuning]
reflect_budget_tokens = 4096
dedup_threshold = 0.95
retain_chunk_max_tokens = 2048
retain_chunk_overlap_tokens = 0

[runtime.tuning.extraction]
structured_output_max_attempts = 3

[runtime.tuning.graph]
semantic_threshold = 0.7
temporal_max_days = 30
enable_causal = true
max_causal_checks = 10

[runtime.tuning.consolidation]
batch_size = 8
max_tokens = 4096
recall_budget = 512
structured_output_max_attempts = 3

[runtime.tuning.reflect]
max_iterations = 8
source_limit = 3
enable_source_lookup = true

[runtime.tuning.retrieval]
retriever_limit = 40
max_facts = 50

[judge]
provider = "openai"
model = "gpt-5.4"
max_tokens = 200
max_attempts = 3
"#,
        );
        write_file(&dataset, r#"{"sample":"ok"}"#);
        write_file(
            &overlay,
            format!(
                r#"
database_url = "postgres://bench:bench@localhost/elephant"
dataset_path = "{}"
"#,
                dataset.display()
            )
            .as_str(),
        );
        write_file(
            &secrets,
            "ELEPHANT_BENCH_RUNTIME_API_KEY=runtime-secret\nELEPHANT_BENCH_JUDGE_API_KEY=judge-secret\n",
        );

        let resolved =
            resolve_longmemeval_bench_config(&profile, Some(&overlay), Some(&secrets)).unwrap();
        let json = resolved.redacted_contract_json();
        assert_eq!(json["runtime"]["tuning"]["retain_chunk_max_tokens"], 2048);
        assert_eq!(json["runtime"]["tuning"]["retain_chunk_overlap_tokens"], 0);
    }

    #[test]
    fn longmemeval_overlay_shard_offset_survives_without_cli_override() {
        let _guard = env_lock().lock().unwrap();
        let profile = temp_path("long-offset-profile.toml");
        let overlay = temp_path("long-offset-overlay.toml");
        let dataset = temp_path("long-offset-dataset.json");
        let secrets = temp_path("long-offset-secrets.env");
        write_file(
            &profile,
            r#"
schema_version = 1
benchmark = "longmemeval"
protocol_version = "2026-04-15-longmemeval-contract-v3"
ingest_format = "round"
consolidation = "end"

[dataset]
identifier = "longmemeval-s"

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
        write_file(&dataset, r#"{"sample":"ok"}"#);
        write_file(
            &overlay,
            format!(
                r#"
database_url = "postgres://bench:bench@localhost/elephant"
dataset_path = "{}"

[shard]
instance_offset = 1
"#,
                dataset.display()
            )
            .as_str(),
        );
        write_file(
            &secrets,
            "ELEPHANT_BENCH_RUNTIME_API_KEY=runtime-secret\nELEPHANT_BENCH_JUDGE_API_KEY=judge-secret\n",
        );

        let resolved =
            resolve_longmemeval_bench_config(&profile, Some(&overlay), Some(&secrets)).unwrap();
        let overridden = resolved
            .with_cli_execution_overrides(None, None, None, &[], None, None)
            .unwrap();

        assert_eq!(resolved.shard_instance_offset(), 1);
        assert_eq!(overridden.shard_instance_offset(), 1);
    }

    #[test]
    fn default_longmemeval_dataset_path_is_profile_aware() {
        assert_eq!(
            default_longmemeval_dataset_path("longmemeval-s").unwrap(),
            PathBuf::from("data/longmemeval_s_cleaned.json")
        );
        assert_eq!(
            default_longmemeval_dataset_path("longmemeval-m").unwrap(),
            PathBuf::from("data/longmemeval_m_cleaned.json")
        );
    }

    #[test]
    fn canonical_profile_lint_rejects_missing_explicit_contract_fields() {
        let raw = r#"
schema_version = 1
benchmark = "locomo"

[dataset]
identifier = "locomo10"
expected_fingerprint = "8dfd1872bfb58ab8"

[slice]
category_filter = [1, 2, 3, 4]

[runtime.llm]
provider = "openai"
retain_model = "gpt-5.4-mini"
reflect_model = "gpt-5.4-mini"

[runtime.embedding]
provider = "local"
model = "bge-small-en-v1.5"

[runtime.reranker]
provider = "none"

[runtime.tuning]
reflect_budget_tokens = 4096
dedup_threshold = 0.95
retain_chunk_max_tokens = 512
retain_chunk_overlap_tokens = 64

[runtime.tuning.extraction]
structured_output_max_attempts = 3

[runtime.tuning.graph]
semantic_threshold = 0.7
temporal_max_days = 30
enable_causal = true
max_causal_checks = 10

[runtime.tuning.consolidation]
batch_size = 8
max_tokens = 4096
recall_budget = 512
structured_output_max_attempts = 3

[runtime.tuning.reflect]
max_iterations = 8
source_limit = 3
enable_source_lookup = true

[runtime.tuning.retrieval]
retriever_limit = 40
max_facts = 50

[judge]
provider = "openai"
model = "gpt-5.4"
max_tokens = 200
max_attempts = 3
"#;

        let err = lint_canonical_contract_file(
            Path::new("bench/locomo/profiles/test.toml"),
            raw,
            BenchmarkKind::Locomo,
        )
        .unwrap_err();

        assert!(err.to_string().contains("must explicitly set"));
        assert!(err.to_string().contains("ingest"));
        assert!(err.to_string().contains("consolidation"));
    }

    #[test]
    fn noncanonical_profile_path_skips_explicitness_lint() {
        let raw = r#"
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
"#;

        lint_canonical_contract_file(
            Path::new("/tmp/local-profile.toml"),
            raw,
            BenchmarkKind::Locomo,
        )
        .unwrap();
    }
}
