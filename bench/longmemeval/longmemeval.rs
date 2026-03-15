//! LongMemEval benchmark harness for Elephant.

use std::collections::{BTreeMap, HashMap};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::str::FromStr;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use tokio::sync::{Mutex, Semaphore};

use chrono::Utc;
use serde::{Deserialize, Serialize};

#[path = "../common/mod.rs"]
mod common;
mod dataset;
mod ingest;

use common::io::{append_jsonl, sidecar_path};
use dataset::load_dataset;
use ingest::{ConsolidationMode, IngestConfig, IngestFormat};

use elephant::metrics::{LlmStage, MetricsCollector, StageUsage, with_scoped_collector};
use elephant::runtime::{
    BuildRuntimeOptions, ElephantRuntime, RuntimePromptHashes as ElephantPromptHashes,
    RuntimeTuning as ElephantRuntimeTuning, build_runtime_from_env,
};
use elephant::types::ReflectQuery;

// --- Judge prompts ---

const JUDGE_FACTUAL: &str = include_str!("prompts/judge_factual.txt");
const JUDGE_TEMPORAL: &str = include_str!("prompts/judge_temporal.txt");
const JUDGE_KNOWLEDGE_UPDATE: &str = include_str!("prompts/judge_knowledge_update.txt");
const JUDGE_PREFERENCE: &str = include_str!("prompts/judge_preference.txt");
const JUDGE_ABSTENTION: &str = include_str!("prompts/judge_abstention.txt");

const REFLECT_BUDGET_TOKENS: usize = 4096;

/// Select the appropriate judge prompt template for a given question instance.
fn select_judge_prompt(instance: &dataset::LongMemEvalInstance) -> &'static str {
    if instance.is_abstention() {
        return JUDGE_ABSTENTION;
    }
    match instance.question_type {
        dataset::QuestionType::SingleSessionUser
        | dataset::QuestionType::SingleSessionAssistant
        | dataset::QuestionType::MultiSession => JUDGE_FACTUAL,
        dataset::QuestionType::TemporalReasoning => JUDGE_TEMPORAL,
        dataset::QuestionType::KnowledgeUpdate => JUDGE_KNOWLEDGE_UPDATE,
        dataset::QuestionType::SingleSessionPreference => JUDGE_PREFERENCE,
    }
}

/// Render a judge prompt template by replacing placeholders.
fn render_judge_prompt(template: &str, question: &str, answer: &str, response: &str) -> String {
    template
        .replace("{question}", question)
        .replace("{answer}", answer)
        .replace("{response}", response)
}

/// Compute a deterministic hash over the full set of judge prompt templates.
fn judge_prompt_hash() -> String {
    let mut combined = String::new();
    // Sorted alphabetically by prompt name
    combined.push_str(JUDGE_ABSTENTION);
    combined.push_str(JUDGE_FACTUAL);
    combined.push_str(JUDGE_KNOWLEDGE_UPDATE);
    combined.push_str(JUDGE_PREFERENCE);
    combined.push_str(JUDGE_TEMPORAL);
    common::fnv1a64_hex(&combined)
}

/// Resolve the judge model from config + env, defaulting to gpt-4o per EVAL-01.
fn resolve_judge_model(config: &RunConfig) -> Option<String> {
    if config.judge_model.is_some() {
        return config.judge_model.clone();
    }
    if env::var("JUDGE_MODEL").is_ok() {
        return None; // let common::judge use the env var
    }
    Some("gpt-4o".into())
}

// --- CLI ---

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BenchCommand {
    Run,
    Ingest,
    Qa,
}

impl BenchCommand {
    fn as_str(self) -> &'static str {
        match self {
            Self::Run => "run",
            Self::Ingest => "ingest",
            Self::Qa => "qa",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum RunProfile {
    Smoke,
    FullS,
    FullM,
}

impl Default for RunProfile {
    fn default() -> Self {
        Self::FullS
    }
}

impl RunProfile {
    fn as_str(self) -> &'static str {
        match self {
            Self::Smoke => "smoke",
            Self::FullS => "full-s",
            Self::FullM => "full-m",
        }
    }

    fn config_path(self) -> PathBuf {
        PathBuf::from(format!("bench/longmemeval/profiles/{}.json", self.as_str()))
    }
}

impl FromStr for RunProfile {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "smoke" => Ok(Self::Smoke),
            "full-s" => Ok(Self::FullS),
            "full-m" => Ok(Self::FullM),
            other => Err(format!(
                "invalid --profile value: {other} (expected one of: smoke, full-s, full-m)"
            )),
        }
    }
}

// --- Config structs ---

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct FileRunConfig {
    #[serde(default)]
    dataset: Option<PathBuf>,
    #[serde(default)]
    output: Option<PathBuf>,
    #[serde(default)]
    tag: Option<String>,
    #[serde(default)]
    instances: Vec<String>,
    #[serde(default)]
    session_limit: Option<usize>,
    #[serde(default)]
    instance_limit: Option<usize>,
    #[serde(default)]
    ingest_format: Option<IngestFormat>,
    #[serde(default)]
    consolidation: Option<ConsolidationMode>,
    #[serde(default)]
    instance_jobs: Option<usize>,
    #[serde(default)]
    judge_model: Option<String>,
}

impl FileRunConfig {
    fn apply(self, config: &mut RunConfig) {
        if let Some(dataset) = self.dataset {
            config.dataset = dataset;
        }
        if let Some(output) = self.output {
            config.output = Some(output);
        }
        if let Some(tag) = self.tag {
            config.tag = Some(tag);
        }
        if !self.instances.is_empty() {
            config.instances = self.instances;
        }
        if let Some(limit) = self.session_limit {
            config.session_limit = Some(limit);
        }
        if let Some(limit) = self.instance_limit {
            config.instance_limit = Some(limit);
        }
        if let Some(format) = self.ingest_format {
            config.ingest_format = format;
        }
        if let Some(consolidation) = self.consolidation {
            config.consolidation = consolidation;
        }
        if let Some(jobs) = self.instance_jobs {
            config.instance_jobs = jobs;
        }
        if let Some(judge_model) = self.judge_model {
            config.judge_model = Some(judge_model);
        }
    }
}

#[derive(Debug, Clone)]
struct RunConfig {
    profile: RunProfile,
    config_path: Option<PathBuf>,
    dataset: PathBuf,
    output: Option<PathBuf>,
    tag: Option<String>,
    instances: Vec<String>,
    session_limit: Option<usize>,
    instance_limit: Option<usize>,
    ingest_format: IngestFormat,
    consolidation: ConsolidationMode,
    instance_jobs: usize,
    judge_model: Option<String>,
    allow_overwrite: bool,
}

impl Default for RunConfig {
    fn default() -> Self {
        Self {
            profile: RunProfile::FullS,
            config_path: None,
            dataset: PathBuf::from("data/longmemeval_s_cleaned.json"),
            output: None,
            tag: None,
            instances: Vec::new(),
            session_limit: None,
            instance_limit: None,
            ingest_format: IngestFormat::Text,
            consolidation: ConsolidationMode::End,
            instance_jobs: 1,
            judge_model: None,
            allow_overwrite: false,
        }
    }
}

// --- CLI structs ---

#[derive(Debug, Default)]
struct CliOverrides {
    help: bool,
    profile: Option<RunProfile>,
    config_path: Option<PathBuf>,
    dataset: Option<PathBuf>,
    output: Option<PathBuf>,
    tag: Option<String>,
    instances: Vec<String>,
    session_limit: Option<usize>,
    instance_limit: Option<usize>,
    ingest_format: Option<IngestFormat>,
    consolidation: Option<ConsolidationMode>,
    instance_jobs: Option<usize>,
    judge_model: Option<String>,
    allow_overwrite: bool,
}

impl CliOverrides {
    fn apply(self, config: &mut RunConfig) {
        if let Some(dataset) = self.dataset {
            config.dataset = dataset;
        }
        if let Some(output) = self.output {
            config.output = Some(output);
        }
        if let Some(tag) = self.tag {
            config.tag = Some(tag);
        }
        if !self.instances.is_empty() {
            config.instances = self.instances;
        }
        if let Some(limit) = self.session_limit {
            config.session_limit = Some(limit);
        }
        if let Some(limit) = self.instance_limit {
            config.instance_limit = Some(limit);
        }
        if let Some(format) = self.ingest_format {
            config.ingest_format = format;
        }
        if let Some(consolidation) = self.consolidation {
            config.consolidation = consolidation;
        }
        if let Some(jobs) = self.instance_jobs {
            config.instance_jobs = jobs;
        }
        if let Some(judge_model) = self.judge_model {
            config.judge_model = Some(judge_model);
        }
        if self.allow_overwrite {
            config.allow_overwrite = true;
        }
        if let Some(path) = self.config_path {
            config.config_path = Some(path);
        }
    }
}

#[derive(Debug)]
struct ParsedCli {
    command: BenchCommand,
    artifact_path: Option<PathBuf>,
    overrides: CliOverrides,
}

#[derive(Debug, Clone)]
struct BenchInvocation {
    command: BenchCommand,
    artifact_path: Option<PathBuf>,
    config: RunConfig,
}

// --- Parsing ---

fn parse_cli_overrides(raw: &[String]) -> Result<ParsedCli, String> {
    let mut overrides = CliOverrides::default();
    let command = match raw.get(1).map(String::as_str) {
        None => return Err("expected subcommand: run, ingest, or qa".into()),
        Some("--help") | Some("-h") => {
            overrides.help = true;
            return Ok(ParsedCli {
                command: BenchCommand::Run,
                artifact_path: None,
                overrides,
            });
        }
        Some("run") => BenchCommand::Run,
        Some("ingest") => BenchCommand::Ingest,
        Some("qa") => BenchCommand::Qa,
        Some(other) => {
            return Err(format!(
                "unknown subcommand: {other} (expected one of: run, ingest, qa)"
            ));
        }
    };

    let mut artifact_path = None;
    let mut i = 2;
    if matches!(command, BenchCommand::Qa) {
        match raw.get(2).map(String::as_str) {
            None => return Err("expected artifact path after `qa`".into()),
            Some("--help") | Some("-h") => {
                overrides.help = true;
                return Ok(ParsedCli {
                    command,
                    artifact_path: None,
                    overrides,
                });
            }
            Some(path) if path.starts_with('-') => {
                return Err("expected artifact path after `qa`".into());
            }
            Some(path) => {
                artifact_path = Some(PathBuf::from(path));
                i = 3;
            }
        }
    }

    while i < raw.len() {
        match raw[i].as_str() {
            "--help" | "-h" => {
                overrides.help = true;
            }
            "--profile" => {
                i += 1;
                overrides.profile = Some(
                    raw.get(i)
                        .ok_or_else(|| "--profile requires a value".to_string())?
                        .parse()?,
                );
            }
            "--config" => {
                i += 1;
                overrides.config_path = Some(PathBuf::from(
                    raw.get(i)
                        .ok_or_else(|| "--config requires a value".to_string())?,
                ));
            }
            "--dataset" => {
                i += 1;
                overrides.dataset = Some(PathBuf::from(
                    raw.get(i)
                        .ok_or_else(|| "--dataset requires a value".to_string())?,
                ));
            }
            "--out" => {
                i += 1;
                overrides.output = Some(PathBuf::from(
                    raw.get(i)
                        .ok_or_else(|| "--out requires a value".to_string())?,
                ));
            }
            "--tag" => {
                i += 1;
                overrides.tag = Some(
                    raw.get(i)
                        .ok_or_else(|| "--tag requires a value".to_string())?
                        .clone(),
                );
            }
            "--instance" => {
                i += 1;
                overrides.instances.push(
                    raw.get(i)
                        .ok_or_else(|| "--instance requires a value".to_string())?
                        .clone(),
                );
            }
            "--session-limit" => {
                i += 1;
                overrides.session_limit = Some(parse_usize_arg(raw.get(i), "--session-limit")?);
            }
            "--instance-limit" => {
                i += 1;
                overrides.instance_limit = Some(parse_usize_arg(raw.get(i), "--instance-limit")?);
            }
            "--ingest-format" => {
                i += 1;
                overrides.ingest_format = Some(
                    raw.get(i)
                        .ok_or_else(|| "--ingest-format requires a value".to_string())?
                        .parse()?,
                );
            }
            "--consolidation" => {
                i += 1;
                overrides.consolidation = Some(
                    raw.get(i)
                        .ok_or_else(|| "--consolidation requires a value".to_string())?
                        .parse()?,
                );
            }
            "--instance-jobs" => {
                i += 1;
                overrides.instance_jobs = Some(parse_usize_arg(raw.get(i), "--instance-jobs")?);
            }
            "--judge-model" => {
                i += 1;
                overrides.judge_model = Some(
                    raw.get(i)
                        .ok_or_else(|| "--judge-model requires a value".to_string())?
                        .clone(),
                );
            }
            "--force" => {
                overrides.allow_overwrite = true;
            }
            other => return Err(format!("Unknown argument: {other}")),
        }
        i += 1;
    }

    Ok(ParsedCli {
        command,
        artifact_path,
        overrides,
    })
}

fn parse_usize_arg(raw: Option<&String>, flag: &str) -> Result<usize, String> {
    raw.ok_or_else(|| format!("{flag} requires a value"))?
        .parse()
        .map_err(|_| format!("invalid numeric value for {flag}"))
}

fn parse_args_from(raw: &[String]) -> Result<Option<BenchInvocation>, String> {
    let ParsedCli {
        command,
        artifact_path,
        overrides,
    } = parse_cli_overrides(raw)?;
    if overrides.help {
        return Ok(None);
    }

    let config = match command {
        BenchCommand::Run | BenchCommand::Ingest => resolve_fresh_config(overrides)?,
        BenchCommand::Qa => resolve_qa_config(
            artifact_path
                .as_deref()
                .ok_or_else(|| "`qa` requires an artifact path".to_string())?,
            overrides,
        )?,
    };
    Ok(Some(BenchInvocation {
        command,
        artifact_path,
        config,
    }))
}

fn parse_args() -> BenchInvocation {
    let raw: Vec<String> = env::args().collect();
    match parse_args_from(&raw) {
        Ok(Some(invocation)) => invocation,
        Ok(None) => {
            print_help();
            std::process::exit(0);
        }
        Err(err) => {
            eprintln!("{err}");
            eprintln!();
            print_help();
            std::process::exit(1);
        }
    }
}

// --- Config resolution ---

fn load_json_config(path: &Path) -> Result<FileRunConfig, String> {
    let raw =
        fs::read_to_string(path).map_err(|e| format!("failed to read {}: {e}", path.display()))?;
    serde_json::from_str(&raw).map_err(|e| format!("failed to parse {}: {e}", path.display()))
}

fn resolve_fresh_config(overrides: CliOverrides) -> Result<RunConfig, String> {
    let profile = overrides.profile.unwrap_or_default();
    let mut config = RunConfig {
        profile,
        ..RunConfig::default()
    };
    load_json_config(&profile.config_path())?.apply(&mut config);

    if let Some(path) = overrides.config_path.clone() {
        load_json_config(&path)?.apply(&mut config);
        config.config_path = Some(path);
    }

    overrides.apply(&mut config);
    Ok(config)
}

fn resolve_qa_config(artifact_path: &Path, overrides: CliOverrides) -> Result<RunConfig, String> {
    validate_qa_overrides(&overrides)?;
    let artifact = load_benchmark_output(artifact_path)?;
    let mut config = run_config_from_artifact(&artifact)?;

    if let Some(output) = overrides.output {
        config.output = Some(output);
    }
    if let Some(tag) = overrides.tag {
        config.tag = Some(tag);
    }
    if !overrides.instances.is_empty() {
        config.instances = overrides.instances;
    }
    if let Some(jobs) = overrides.instance_jobs {
        config.instance_jobs = jobs;
    }
    if let Some(judge_model) = overrides.judge_model {
        config.judge_model = Some(judge_model);
    }
    if overrides.allow_overwrite {
        config.allow_overwrite = true;
    }

    Ok(config)
}

fn validate_qa_overrides(overrides: &CliOverrides) -> Result<(), String> {
    let mut unsupported = Vec::new();
    if overrides.profile.is_some() {
        unsupported.push("--profile");
    }
    if overrides.config_path.is_some() {
        unsupported.push("--config");
    }
    if overrides.dataset.is_some() {
        unsupported.push("--dataset");
    }
    if overrides.session_limit.is_some() {
        unsupported.push("--session-limit");
    }
    if overrides.instance_limit.is_some() {
        unsupported.push("--instance-limit");
    }
    if overrides.ingest_format.is_some() {
        unsupported.push("--ingest-format");
    }
    if overrides.consolidation.is_some() {
        unsupported.push("--consolidation");
    }

    if unsupported.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "`qa` does not accept {}; it runs against the artifact's existing banks",
            unsupported.join(", ")
        ))
    }
}

// --- Output paths ---

fn default_output_path(
    command: BenchCommand,
    config: &RunConfig,
    artifact_path: Option<&Path>,
) -> PathBuf {
    if let Some(ref p) = config.output {
        return p.clone();
    }

    match command {
        BenchCommand::Qa => {
            if let Some(ref tag) = config.tag {
                PathBuf::from(format!("bench/longmemeval/results/local/{tag}.json"))
            } else {
                artifact_path
                    .map(Path::to_path_buf)
                    .unwrap_or_else(|| PathBuf::from("bench/longmemeval/results/local/qa.json"))
            }
        }
        BenchCommand::Run | BenchCommand::Ingest => {
            let stem = if let Some(ref tag) = config.tag {
                tag.clone()
            } else {
                format!("{}-{}", config.profile.as_str(), command.as_str())
            };
            PathBuf::from(format!("bench/longmemeval/results/local/{stem}.json"))
        }
    }
}

// --- Help ---

fn print_help() {
    eprintln!("Usage:");
    eprintln!("  longmemeval-bench run [OPTIONS]");
    eprintln!("  longmemeval-bench ingest [OPTIONS]");
    eprintln!("  longmemeval-bench qa <ARTIFACT.json> [OPTIONS]");
    eprintln!();
    eprintln!("Subcommands:");
    eprintln!("  run                              Fresh ingest, consolidate, then score QA");
    eprintln!("  ingest                           Ingest and consolidate only; do not run QA");
    eprintln!(
        "  qa <ARTIFACT.json>               Score QA against existing banks; skip ingest and consolidation"
    );
    eprintln!();
    eprintln!("Options:");
    eprintln!("  --profile <NAME>                Named profile for `run`/`ingest` [default: full-s]");
    eprintln!("                                  Profiles: smoke, full-s, full-m");
    eprintln!("  --config <PATH>                 JSON config file for `run`/`ingest`");
    eprintln!(
        "  --dataset <PATH>                Dataset path for `run`/`ingest` [default: data/longmemeval_s_cleaned.json]"
    );
    eprintln!("  --tag <NAME>                    Save to results/local/<tag>.json by default");
    eprintln!("  --out <PATH>                    Output results path (overrides --tag)");
    eprintln!("  --instance <ID>                 Run one specific question instance (repeatable)");
    eprintln!("  --ingest-format <MODE>          text | json (`run`/`ingest` only)");
    eprintln!("  --consolidation <MODE>          end | per-session | off (`run`/`ingest` only)");
    eprintln!("  --instance-jobs <N>             Parallel instances");
    eprintln!("  --judge-model <MODEL>           Override judge model");
    eprintln!("  --force                         Allow overwriting existing output files");
    eprintln!();
    eprintln!("Debug slice options:");
    eprintln!(
        "  --session-limit <N>             Limit sessions per instance (`run`/`ingest` only)"
    );
    eprintln!(
        "  --instance-limit <N>            Limit number of instances (`run`/`ingest` only)"
    );
}

// --- Artifact types ---

#[derive(Debug, Clone, Serialize, Deserialize)]
struct BenchmarkOutput {
    benchmark: String,
    timestamp: String,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    commit: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    tag: Option<String>,
    retain_model: String,
    reflect_model: String,
    embedding_model: String,
    reranker_model: String,
    #[serde(default)]
    judge_model: String,
    consolidation_strategy: String,
    total_questions: usize,
    #[serde(default)]
    accuracy: f64,
    #[serde(default)]
    per_category: HashMap<String, CategoryResult>,
    /// Maps question_id -> bank_id for resume/audit.
    #[serde(default)]
    banks: HashMap<String, String>,
    #[serde(default)]
    manifest: BenchmarkManifest,
    #[serde(default)]
    artifacts: BenchmarkArtifacts,
    #[serde(default)]
    stage_metrics: BTreeMap<LlmStage, StageUsage>,
    #[serde(default)]
    total_time_s: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CategoryResult {
    accuracy: f64,
    count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct BenchmarkManifest {
    #[serde(default)]
    protocol_version: String,
    #[serde(default)]
    profile: String,
    #[serde(default)]
    mode: String,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    config_path: Option<String>,
    #[serde(default)]
    dataset_path: String,
    #[serde(default)]
    dataset_fingerprint: String,
    #[serde(default)]
    command: String,
    #[serde(default)]
    selected_instances: Vec<String>,
    #[serde(default)]
    ingest_format: String,
    #[serde(default)]
    instance_concurrency: usize,
    #[serde(default)]
    consolidation_strategy: String,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    session_limit: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    instance_limit: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    dirty_worktree: Option<bool>,
    #[serde(default)]
    prompt_hashes: BenchmarkPromptHashes,
    #[serde(default)]
    runtime_config: BenchmarkRuntimeConfig,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    source_artifact: Option<SourceArtifact>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct BenchmarkPromptHashes {
    #[serde(default)]
    judge: String,
    #[serde(flatten)]
    elephant: ElephantPromptHashes,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct BenchmarkRuntimeConfig {
    #[serde(flatten)]
    elephant: ElephantRuntimeTuning,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct BenchmarkArtifacts {
    #[serde(default)]
    questions_path: String,
    #[serde(default)]
    debug_path: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct SourceArtifact {
    #[serde(default)]
    path: String,
    #[serde(default)]
    fingerprint: String,
    #[serde(default)]
    mode: String,
    #[serde(default)]
    tag: Option<String>,
    #[serde(default)]
    commit: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct QuestionResult {
    question_id: String,
    category: String,
    judge_correct: bool,
    #[serde(default)]
    judge_reasoning: String,
    #[serde(default)]
    hypothesis: String,
    ground_truth: String,
    bank_id: String,
    #[serde(default)]
    elapsed_s: f64,
    #[serde(default)]
    status: String,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    error: Option<String>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    qa_stage_metrics: BTreeMap<LlmStage, StageUsage>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct QuestionDebugRecord {
    question_id: String,
    question: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    reflect_trace: Vec<ReflectTraceEntry>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    final_done: Option<elephant::types::ReflectDoneTrace>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    retrieved_context: Vec<RetrievedFactEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ReflectTraceEntry {
    iteration: usize,
    tool_name: String,
    query: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct RetrievedFactEntry {
    id: String,
    content: String,
    score: f32,
    network: String,
}

// --- Git helpers ---

fn git_commit_sha() -> Option<String> {
    let output = Command::new("git")
        .args(["rev-parse", "--short=12", "HEAD"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let sha = String::from_utf8(output.stdout).ok()?;
    let sha = sha.trim();
    if sha.is_empty() {
        None
    } else {
        Some(sha.to_string())
    }
}

fn is_generated_bench_artifact_path(path: &str) -> bool {
    let path = path.trim().trim_matches('"');
    path.starts_with("bench/longmemeval/results/")
}

fn git_dirty_worktree() -> Option<bool> {
    let output = Command::new("git")
        .args(["status", "--porcelain"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let stdout = String::from_utf8(output.stdout).ok()?;
    Some(stdout.lines().any(|line| {
        !line.trim().is_empty() && !is_generated_bench_artifact_path(&line[3..])
    }))
}

// --- Output safety ---

fn same_path(left: &Path, right: &Path) -> bool {
    if left == right {
        return true;
    }
    match (fs::canonicalize(left), fs::canonicalize(right)) {
        (Ok(lhs), Ok(rhs)) => lhs == rhs,
        _ => false,
    }
}

fn ensure_output_paths_are_safe(
    command: BenchCommand,
    output_path: &Path,
    artifact_path: Option<&Path>,
    allow_overwrite: bool,
) -> Result<(), String> {
    if allow_overwrite {
        return Ok(());
    }

    if matches!(command, BenchCommand::Qa)
        && artifact_path
            .map(|artifact| same_path(output_path, artifact))
            .unwrap_or(false)
    {
        return Err(format!(
            "refusing to overwrite source artifact {} during `qa`; choose --out or pass --force",
            output_path.display()
        ));
    }

    let questions_path = sidecar_path(output_path, "questions");
    let debug_path = sidecar_path(output_path, "debug");
    let existing = [output_path, questions_path.as_path(), debug_path.as_path()]
        .into_iter()
        .filter(|path| path.exists())
        .map(|path| path.display().to_string())
        .collect::<Vec<_>>();

    if existing.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "refusing to overwrite existing benchmark output: {}. Choose a new --tag/--out or pass --force",
            existing.join(", ")
        ))
    }
}

// --- Artifact loading ---

fn load_benchmark_output(path: &Path) -> Result<BenchmarkOutput, String> {
    let raw =
        fs::read_to_string(path).map_err(|e| format!("failed to read {}: {e}", path.display()))?;
    serde_json::from_str(&raw).map_err(|e| format!("failed to parse {}: {e}", path.display()))
}

fn run_config_from_artifact(artifact: &BenchmarkOutput) -> Result<RunConfig, String> {
    let profile = artifact
        .manifest
        .profile
        .parse::<RunProfile>()
        .unwrap_or_default();

    let consolidation = match artifact.manifest.consolidation_strategy.as_str() {
        "end" => ConsolidationMode::End,
        "per-session" => ConsolidationMode::PerSession,
        "off" => ConsolidationMode::Off,
        other => {
            return Err(format!("unknown consolidation strategy in artifact: {other}"))
        }
    };

    let ingest_format = match artifact.manifest.ingest_format.as_str() {
        "text" => IngestFormat::Text,
        "json" => IngestFormat::Json,
        other => return Err(format!("unknown ingest format in artifact: {other}")),
    };

    Ok(RunConfig {
        profile,
        config_path: artifact.manifest.config_path.as_ref().map(PathBuf::from),
        dataset: PathBuf::from(&artifact.manifest.dataset_path),
        output: None,
        tag: artifact.tag.clone(),
        instances: artifact.manifest.selected_instances.clone(),
        session_limit: artifact.manifest.session_limit,
        instance_limit: artifact.manifest.instance_limit,
        ingest_format,
        consolidation,
        instance_jobs: artifact.manifest.instance_concurrency.max(1),
        judge_model: if artifact.judge_model.is_empty() {
            None
        } else {
            Some(artifact.judge_model.clone())
        },
        allow_overwrite: false,
    })
}

fn benchmark_prompt_hashes(runtime: &ElephantRuntime) -> BenchmarkPromptHashes {
    BenchmarkPromptHashes {
        judge: judge_prompt_hash(),
        elephant: runtime.info.prompt_hashes.clone(),
    }
}

fn benchmark_runtime_config(runtime: &ElephantRuntime) -> BenchmarkRuntimeConfig {
    BenchmarkRuntimeConfig {
        elephant: runtime.info.tuning.clone(),
    }
}

fn compute_per_category(results: &[QuestionResult]) -> HashMap<String, CategoryResult> {
    let mut counts: HashMap<String, (usize, usize)> = HashMap::new();
    for r in results {
        let entry = counts.entry(r.category.clone()).or_insert((0, 0));
        entry.0 += 1;
        if r.judge_correct {
            entry.1 += 1;
        }
    }
    counts
        .into_iter()
        .map(|(cat, (count, correct))| {
            let accuracy = if count > 0 {
                correct as f64 / count as f64
            } else {
                0.0
            };
            (cat, CategoryResult { accuracy, count })
        })
        .collect()
}

fn compute_accuracy(results: &[QuestionResult]) -> f64 {
    if results.is_empty() {
        return 0.0;
    }
    let correct = results.iter().filter(|r| r.judge_correct).count();
    correct as f64 / results.len() as f64
}

// --- Shared state for incremental writes ---

struct SharedState {
    results: Vec<QuestionResult>,
    banks: HashMap<String, String>,
    output_path: PathBuf,
    questions_path: PathBuf,
    debug_path: PathBuf,
    judge_label: String,
    tag: Option<String>,
    retain_model: String,
    reflect_model: String,
    embedding_model: String,
    reranker_model: String,
    consolidation_strategy: String,
    manifest: BenchmarkManifest,
    metrics: Arc<MetricsCollector>,
    run_timestamp: String,
    commit: Option<String>,
    bench_start: Instant,
    total_questions_expected: usize,
}

impl SharedState {
    fn push_and_flush(&mut self, result: QuestionResult, debug: QuestionDebugRecord) {
        append_jsonl(&self.questions_path, &result);
        append_jsonl(&self.debug_path, &debug);
        self.results.push(result);
        self.flush();
    }

    fn record_bank(&mut self, question_id: String, bank_id: String) {
        self.banks.insert(question_id, bank_id);
        self.flush();
    }

    fn flush(&self) {
        let output = BenchmarkOutput {
            benchmark: "longmemeval".into(),
            timestamp: self.run_timestamp.clone(),
            commit: self.commit.clone(),
            tag: self.tag.clone(),
            retain_model: self.retain_model.clone(),
            reflect_model: self.reflect_model.clone(),
            embedding_model: self.embedding_model.clone(),
            reranker_model: self.reranker_model.clone(),
            judge_model: self.judge_label.clone(),
            consolidation_strategy: self.consolidation_strategy.clone(),
            total_questions: self.total_questions_expected,
            accuracy: compute_accuracy(&self.results),
            per_category: compute_per_category(&self.results),
            banks: self.banks.clone(),
            manifest: self.manifest.clone(),
            artifacts: BenchmarkArtifacts {
                questions_path: self.questions_path.display().to_string(),
                debug_path: self.debug_path.display().to_string(),
            },
            stage_metrics: self.metrics.snapshot(),
            total_time_s: self.bench_start.elapsed().as_secs_f64(),
        };
        let json = serde_json::to_string_pretty(&output).expect("serialize output");
        fs::write(&self.output_path, json).expect("write summary");
    }
}

// --- Main ---

#[tokio::main]
async fn main() {
    let _ = dotenvy::dotenv();

    let invocation = parse_args();
    let command = invocation.command;
    let artifact_path = invocation.artifact_path.clone();
    let config = invocation.config;
    let output_path = default_output_path(command, &config, artifact_path.as_deref());

    // Safety check
    ensure_output_paths_are_safe(command, &output_path, artifact_path.as_deref(), config.allow_overwrite)
        .unwrap_or_else(|e| {
            eprintln!("{e}");
            std::process::exit(1);
        });

    // Load artifact state for QA mode
    let (existing_banks, source_artifact) = if let Some(ref path) = artifact_path {
        let art = load_benchmark_output(path).unwrap_or_else(|e| {
            eprintln!("{e}");
            std::process::exit(1);
        });
        let sa = SourceArtifact {
            path: path.display().to_string(),
            fingerprint: format!("{:016x}", common::fingerprint::fnv1a64(&fs::read(path).unwrap_or_default())),
            mode: art.manifest.mode.clone(),
            tag: art.tag.clone(),
            commit: art.commit.clone(),
        };
        (art.banks, Some(sa))
    } else {
        (HashMap::new(), None)
    };

    // Validate dataset exists early
    if !config.dataset.exists() {
        eprintln!(
            "Dataset not found: {}\n\nDownload the LongMemEval dataset and place it at the expected path.\nSee: https://github.com/xiaowu0162/LongMemEval",
            config.dataset.display()
        );
        std::process::exit(1);
    }

    // Load dataset
    let (mut instances, dataset_fingerprint) =
        load_dataset(&config.dataset).unwrap_or_else(|e| {
            eprintln!("{e}");
            std::process::exit(1);
        });

    // Filter by --instance
    if !config.instances.is_empty() {
        let selected: std::collections::HashSet<&str> =
            config.instances.iter().map(String::as_str).collect();
        let before = instances.len();
        instances.retain(|inst| selected.contains(inst.question_id.as_str()));
        let missing: Vec<&str> = selected
            .iter()
            .filter(|id| !instances.iter().any(|inst| inst.question_id == **id))
            .copied()
            .collect();
        if !missing.is_empty() {
            eprintln!(
                "WARNING: {} instance(s) not found in dataset: {:?} (loaded {} of {} requested)",
                missing.len(),
                missing,
                instances.len(),
                before
            );
        }
    }

    // Apply instance_limit
    if let Some(limit) = config.instance_limit {
        instances.truncate(limit);
    }

    // QA mode: validate all selected instances have bank_ids
    if matches!(command, BenchCommand::Qa) {
        let missing: Vec<&str> = instances
            .iter()
            .filter(|inst| !existing_banks.contains_key(&inst.question_id))
            .map(|inst| inst.question_id.as_str())
            .collect();
        if !missing.is_empty() {
            eprintln!(
                "QA mode: {} instance(s) have no bank_id in artifact: {:?}",
                missing.len(),
                missing
            );
            std::process::exit(1);
        }
    }

    // Build runtime with MetricsCollector
    let metrics = Arc::new(MetricsCollector::new());
    let runtime = Arc::new(
        build_runtime_from_env(BuildRuntimeOptions {
            metrics: Some(metrics.clone()),
            max_pool_connections: Some(std::cmp::min(config.instance_jobs as u32 * 3, 50)),
        })
        .await
        .unwrap_or_else(|e| {
            eprintln!("failed to build runtime: {e}");
            std::process::exit(1);
        }),
    );

    // Print config summary
    eprintln!("longmemeval-bench {}", command.as_str());
    eprintln!("  profile:        {}", config.profile.as_str());
    eprintln!("  dataset:        {}", config.dataset.display());
    eprintln!("  output:         {}", output_path.display());
    if let Some(ref tag) = config.tag {
        eprintln!("  tag:            {tag}");
    }
    eprintln!("  ingest_format:  {}", config.ingest_format.as_str());
    eprintln!("  consolidation:  {}", config.consolidation.as_str());
    eprintln!("  instance_jobs:  {}", config.instance_jobs);
    if let Some(limit) = config.session_limit {
        eprintln!("  session_limit:  {limit}");
    }
    if let Some(limit) = config.instance_limit {
        eprintln!("  instance_limit: {limit}");
    }
    if !config.instances.is_empty() {
        eprintln!("  instances:      {:?}", config.instances);
    }
    eprintln!("  retain_model:   {}", runtime.info.retain_model);
    eprintln!("  reflect_model:  {}", runtime.info.reflect_model);
    eprintln!("  embedding_model: {}", runtime.info.embedding_model);
    eprintln!("  reranker_model: {}", runtime.info.reranker_model);
    if let Some(ref judge) = config.judge_model {
        eprintln!("  judge_model:    {judge}");
    }
    eprintln!("  instances:      {}", instances.len());
    eprintln!();

    // Create output dirs, truncate sidecars
    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent).ok();
    }
    let questions_path = sidecar_path(&output_path, "questions");
    let debug_path = sidecar_path(&output_path, "debug");
    let _ = fs::write(&questions_path, "");
    let _ = fs::write(&debug_path, "");

    // Capture full CLI string for manifest
    let cli_command: String = env::args().collect::<Vec<_>>().join(" ");

    // Build manifest
    let manifest = BenchmarkManifest {
        protocol_version: "2026-03-15-longmemeval-v1".into(),
        profile: config.profile.as_str().into(),
        mode: command.as_str().into(),
        config_path: config.config_path.as_ref().map(|p| p.display().to_string()),
        dataset_path: config.dataset.display().to_string(),
        dataset_fingerprint: dataset_fingerprint.clone(),
        command: cli_command,
        selected_instances: if config.instances.is_empty() {
            vec![]
        } else {
            config.instances.clone()
        },
        ingest_format: config.ingest_format.as_str().into(),
        instance_concurrency: config.instance_jobs,
        consolidation_strategy: config.consolidation.as_str().into(),
        session_limit: config.session_limit,
        instance_limit: config.instance_limit,
        dirty_worktree: git_dirty_worktree(),
        prompt_hashes: benchmark_prompt_hashes(&runtime),
        runtime_config: benchmark_runtime_config(&runtime),
        source_artifact,
    };

    // Build judge client (only needed for Run/Qa)
    let judge_override = resolve_judge_model(&config);
    let judge: Option<Arc<dyn elephant::llm::LlmClient>> =
        if !matches!(command, BenchCommand::Ingest) {
            Some(common::judge::build_judge_client(
                metrics.clone(),
                judge_override.clone(),
            ))
        } else {
            None
        };
    let jl = if !matches!(command, BenchCommand::Ingest) {
        common::judge::judge_label(&judge_override)
    } else {
        String::new()
    };

    // Concurrent per-instance loop
    let total_instances = instances.len();
    let bench_start = Instant::now();
    let run_timestamp = Utc::now().to_rfc3339();
    let commit = git_commit_sha();

    let shared = Arc::new(Mutex::new(SharedState {
        results: Vec::new(),
        banks: existing_banks,
        output_path: output_path.clone(),
        questions_path: questions_path.clone(),
        debug_path: debug_path.clone(),
        judge_label: jl.clone(),
        tag: config.tag.clone(),
        retain_model: runtime.info.retain_model.clone(),
        reflect_model: runtime.info.reflect_model.clone(),
        embedding_model: runtime.info.embedding_model.clone(),
        reranker_model: runtime.info.reranker_model.clone(),
        consolidation_strategy: config.consolidation.as_str().into(),
        manifest,
        metrics: metrics.clone(),
        run_timestamp,
        commit,
        bench_start,
        total_questions_expected: total_instances,
    }));

    let semaphore = Arc::new(Semaphore::new(config.instance_jobs));
    let completed = Arc::new(AtomicUsize::new(0));
    let ingest_format = config.ingest_format;
    let consolidation = config.consolidation;
    let session_limit = config.session_limit;

    let mut handles = Vec::new();
    for instance in instances {
        let sem = semaphore.clone();
        let runtime = runtime.clone();
        let judge = judge.clone();
        let metrics = metrics.clone();
        let shared = shared.clone();
        let completed = completed.clone();

        handles.push(tokio::spawn(async move {
            let _permit = sem
                .acquire()
                .await
                .map_err(|e| format!("semaphore closed: {e}"))?;

            let instance_start = Instant::now();

            // Ingest (skip if qa mode with existing bank)
            let bank_id_str = if matches!(command, BenchCommand::Qa) {
                shared
                    .lock()
                    .await
                    .banks
                    .get(&instance.question_id)
                    .cloned()
                    .ok_or_else(|| {
                        format!("no bank_id for {} in QA mode", instance.question_id)
                    })?
            } else {
                let ingest_config = IngestConfig {
                    format: ingest_format,
                    consolidation,
                    session_limit,
                };
                let result = with_scoped_collector(
                    metrics.clone(),
                    ingest::ingest_instance(&instance, &runtime, &ingest_config),
                )
                .await;
                match result {
                    Ok(r) => {
                        let bid = r.bank_id.to_string();
                        shared
                            .lock()
                            .await
                            .record_bank(instance.question_id.clone(), bid.clone());
                        bid
                    }
                    Err(e) => {
                        let err_msg = format!("{e}");
                        eprintln!(
                            "ERROR ingesting {}: {err_msg}",
                            instance.question_id
                        );
                        // Record a failed result instead of aborting
                        let qr = QuestionResult {
                            question_id: instance.question_id.clone(),
                            category: instance.reporting_category().to_string(),
                            judge_correct: false,
                            judge_reasoning: String::new(),
                            hypothesis: String::new(),
                            ground_truth: instance.answer_string(),
                            bank_id: String::new(),
                            elapsed_s: instance_start.elapsed().as_secs_f64(),
                            status: "ingest_error".into(),
                            error: Some(err_msg.clone()),
                            qa_stage_metrics: BTreeMap::new(),
                        };
                        let dr = QuestionDebugRecord {
                            question_id: instance.question_id.clone(),
                            question: instance.question.clone(),
                            reflect_trace: vec![],
                            final_done: None,
                            retrieved_context: vec![],
                        };
                        shared.lock().await.push_and_flush(qr, dr);
                        let done = completed.fetch_add(1, Ordering::Relaxed) + 1;
                        eprintln!(
                            "[{done}/{total_instances}] {} err ingest {:.1}s",
                            instance.question_id,
                            instance_start.elapsed().as_secs_f64(),
                        );
                        return Err(err_msg);
                    }
                }
            };

            let ingest_elapsed = instance_start.elapsed().as_secs_f64();

            // QA evaluation
            if matches!(command, BenchCommand::Ingest) {
                // Ingest-only: write stub result, no reflect/judge
                let qr = QuestionResult {
                    question_id: instance.question_id.clone(),
                    category: instance.reporting_category().to_string(),
                    judge_correct: false,
                    judge_reasoning: String::new(),
                    hypothesis: String::new(),
                    ground_truth: instance.answer_string(),
                    bank_id: bank_id_str,
                    elapsed_s: 0.0,
                    status: "ingest-only".into(),
                    error: None,
                    qa_stage_metrics: BTreeMap::new(),
                };
                let dr = QuestionDebugRecord {
                    question_id: instance.question_id.clone(),
                    question: instance.question.clone(),
                    reflect_trace: vec![],
                    final_done: None,
                    retrieved_context: vec![],
                };
                shared.lock().await.push_and_flush(qr, dr);
                let done = completed.fetch_add(1, Ordering::Relaxed) + 1;
                eprintln!(
                    "[{done}/{total_instances}] {} ok ingest {:.1}s",
                    instance.question_id,
                    ingest_elapsed,
                );
                return Ok(());
            }

            let qa_start = Instant::now();

            // 1. Call reflect with scoped metrics
            let reflect_result = with_scoped_collector(
                metrics.clone(),
                runtime.reflect.reflect(&ReflectQuery {
                    bank_id: bank_id_str.parse().unwrap(),
                    question: instance.question.clone(),
                    budget_tokens: REFLECT_BUDGET_TOKENS,
                    temporal_context: Some(instance.question_date.clone()),
                }),
            )
            .await;

            // 2. Process reflect result
            let (hypothesis, retrieved_context, reflect_trace, final_done, status, error) =
                match reflect_result {
                    Ok(resp) => {
                        let rc: Vec<RetrievedFactEntry> = resp
                            .retrieved_context
                            .iter()
                            .map(|f| RetrievedFactEntry {
                                id: f.id.to_string(),
                                content: f.content.clone(),
                                score: f.score,
                                network: format!("{:?}", f.network),
                            })
                            .collect();
                        let rt: Vec<ReflectTraceEntry> = resp
                            .trace
                            .iter()
                            .map(|s| ReflectTraceEntry {
                                iteration: s.iteration,
                                tool_name: s.tool_name.clone(),
                                query: s.query.clone(),
                            })
                            .collect();
                        (
                            resp.response.clone(),
                            rc,
                            rt,
                            resp.final_done.clone(),
                            "ok".to_string(),
                            None,
                        )
                    }
                    Err(e) => {
                        let err_msg = format!("{e}");
                        eprintln!("  reflect error ({}): {err_msg}", instance.question_id);
                        (
                            String::new(),
                            vec![],
                            vec![],
                            None,
                            "reflect_error".to_string(),
                            Some(err_msg),
                        )
                    }
                };

            // 3. Judge (skip if reflect failed / empty hypothesis)
            let (judge_correct, judge_reasoning, status, error) = if hypothesis.is_empty() {
                (false, error.clone().unwrap_or_default(), status, error)
            } else {
                let prompt_template = select_judge_prompt(&instance);
                let rendered = render_judge_prompt(
                    prompt_template,
                    &instance.question,
                    &instance.answer_string(),
                    &hypothesis,
                );
                match common::judge::llm_judge(judge.as_ref().unwrap().as_ref(), &rendered).await {
                    Ok((correct, reasoning)) => (correct, reasoning, "ok".into(), None),
                    Err(e) => {
                        eprintln!("  judge error ({}): {e}", instance.question_id);
                        (false, e.clone(), "judge_error".into(), Some(e))
                    }
                }
            };

            let qa_elapsed = qa_start.elapsed().as_secs_f64();

            let qr = QuestionResult {
                question_id: instance.question_id.clone(),
                category: instance.reporting_category().to_string(),
                judge_correct,
                judge_reasoning,
                hypothesis,
                ground_truth: instance.answer_string(),
                bank_id: bank_id_str,
                elapsed_s: qa_elapsed,
                status,
                error,
                qa_stage_metrics: BTreeMap::new(),
            };
            let dr = QuestionDebugRecord {
                question_id: instance.question_id.clone(),
                question: instance.question.clone(),
                reflect_trace,
                final_done,
                retrieved_context,
            };
            shared.lock().await.push_and_flush(qr, dr);

            let done = completed.fetch_add(1, Ordering::Relaxed) + 1;
            eprintln!(
                "[{done}/{total_instances}] {} {} ingest {:.1}s qa {:.1}s",
                instance.question_id,
                if judge_correct { "ok" } else { "err" },
                ingest_elapsed,
                qa_elapsed,
            );

            Ok::<(), String>(())
        }));
    }

    for handle in handles {
        match handle.await {
            Ok(Ok(())) => {}
            Ok(Err(e)) => eprintln!("instance failed: {e}"),
            Err(e) => eprintln!("instance task panicked: {e}"),
        }
    }

    // Final summary (SharedState.flush() already keeps summary JSON up to date)
    let shared_snapshot = shared.lock().await;
    eprintln!();
    eprintln!("Summary:   {}", output_path.display());
    eprintln!("Questions: {}", questions_path.display());
    eprintln!("Debug:     {}", debug_path.display());
    eprintln!("Time:      {:.1}s", bench_start.elapsed().as_secs_f64());

    if !matches!(command, BenchCommand::Ingest) {
        let acc = compute_accuracy(&shared_snapshot.results);
        eprintln!(
            "Accuracy:  {:.1}% ({}/{})",
            acc * 100.0,
            shared_snapshot
                .results
                .iter()
                .filter(|r| r.judge_correct)
                .count(),
            shared_snapshot.results.len(),
        );
    }
}

// --- Tests ---

#[cfg(test)]
mod tests {
    use super::*;

    fn s(v: &str) -> String {
        v.to_string()
    }

    // --- BenchCommand ---

    #[test]
    fn bench_command_as_str() {
        assert_eq!(BenchCommand::Run.as_str(), "run");
        assert_eq!(BenchCommand::Ingest.as_str(), "ingest");
        assert_eq!(BenchCommand::Qa.as_str(), "qa");
    }

    // --- RunProfile ---

    #[test]
    fn run_profile_as_str() {
        assert_eq!(RunProfile::Smoke.as_str(), "smoke");
        assert_eq!(RunProfile::FullS.as_str(), "full-s");
        assert_eq!(RunProfile::FullM.as_str(), "full-m");
    }

    #[test]
    fn run_profile_from_str() {
        assert_eq!("smoke".parse::<RunProfile>().unwrap(), RunProfile::Smoke);
        assert_eq!("full-s".parse::<RunProfile>().unwrap(), RunProfile::FullS);
        assert_eq!("full-m".parse::<RunProfile>().unwrap(), RunProfile::FullM);
        assert!("bad".parse::<RunProfile>().is_err());
    }

    #[test]
    fn run_profile_default_is_full_s() {
        assert_eq!(RunProfile::default(), RunProfile::FullS);
    }

    #[test]
    fn run_profile_config_path() {
        assert_eq!(
            RunProfile::Smoke.config_path(),
            PathBuf::from("bench/longmemeval/profiles/smoke.json")
        );
        assert_eq!(
            RunProfile::FullS.config_path(),
            PathBuf::from("bench/longmemeval/profiles/full-s.json")
        );
        assert_eq!(
            RunProfile::FullM.config_path(),
            PathBuf::from("bench/longmemeval/profiles/full-m.json")
        );
    }

    // --- parse_args_from: subcommands ---

    #[test]
    fn parse_run_subcommand() {
        let raw = vec![s("bin"), s("run")];
        let inv = parse_args_from(&raw).unwrap().unwrap();
        assert_eq!(inv.command, BenchCommand::Run);
        assert!(inv.artifact_path.is_none());
    }

    #[test]
    fn parse_ingest_subcommand() {
        let raw = vec![s("bin"), s("ingest")];
        let inv = parse_args_from(&raw).unwrap().unwrap();
        assert_eq!(inv.command, BenchCommand::Ingest);
        assert!(inv.artifact_path.is_none());
    }

    #[test]
    fn parse_qa_subcommand_with_path() {
        let dir = env::temp_dir().join(format!("longmemeval-qa-parse-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let artifact = dir.join("artifact.json");

        let output = BenchmarkOutput {
            benchmark: "longmemeval".into(),
            timestamp: "2026-03-15T00:00:00Z".into(),
            commit: None,
            tag: None,
            retain_model: "m1".into(),
            reflect_model: "m2".into(),
            embedding_model: "m3".into(),
            reranker_model: "m4".into(),
            judge_model: String::new(),
            consolidation_strategy: "end".into(),
            total_questions: 0,
            accuracy: 0.0,
            per_category: HashMap::new(),
            banks: HashMap::new(),
            manifest: BenchmarkManifest {
                profile: "smoke".into(),
                dataset_path: "data/test.json".into(),
                ingest_format: "text".into(),
                consolidation_strategy: "end".into(),
                instance_concurrency: 1,
                ..BenchmarkManifest::default()
            },
            artifacts: BenchmarkArtifacts::default(),
            stage_metrics: BTreeMap::new(),
            total_time_s: 0.0,
        };
        fs::write(&artifact, serde_json::to_string(&output).unwrap()).unwrap();

        let raw = vec![s("bin"), s("qa"), s(&artifact.display().to_string())];
        let inv = parse_args_from(&raw).unwrap().unwrap();
        assert_eq!(inv.command, BenchCommand::Qa);
        assert_eq!(inv.artifact_path, Some(artifact.clone()));

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn parse_qa_missing_path_is_error() {
        let raw = vec![s("bin"), s("qa")];
        let err = parse_args_from(&raw).unwrap_err();
        assert!(err.contains("artifact path"));
    }

    #[test]
    fn parse_missing_subcommand_is_error() {
        let raw = vec![s("bin")];
        let err = parse_args_from(&raw).unwrap_err();
        assert!(err.contains("subcommand"));
    }

    #[test]
    fn parse_unknown_subcommand_is_error() {
        let raw = vec![s("bin"), s("resume")];
        let err = parse_args_from(&raw).unwrap_err();
        assert!(err.contains("unknown subcommand"));
    }

    // --- parse_args_from: flags ---

    #[test]
    fn parse_profile_smoke() {
        let raw = vec![s("bin"), s("run"), s("--profile"), s("smoke")];
        let inv = parse_args_from(&raw).unwrap().unwrap();
        assert_eq!(inv.config.profile, RunProfile::Smoke);
    }

    #[test]
    fn parse_profile_full_s() {
        let raw = vec![s("bin"), s("run"), s("--profile"), s("full-s")];
        let inv = parse_args_from(&raw).unwrap().unwrap();
        assert_eq!(inv.config.profile, RunProfile::FullS);
    }

    #[test]
    fn parse_profile_full_m() {
        let raw = vec![s("bin"), s("run"), s("--profile"), s("full-m")];
        let inv = parse_args_from(&raw).unwrap().unwrap();
        assert_eq!(inv.config.profile, RunProfile::FullM);
    }

    #[test]
    fn parse_instance_accumulates() {
        let raw = vec![
            s("bin"),
            s("run"),
            s("--instance"),
            s("q1"),
            s("--instance"),
            s("q2"),
        ];
        let inv = parse_args_from(&raw).unwrap().unwrap();
        assert_eq!(inv.config.instances, vec!["q1", "q2"]);
    }

    #[test]
    fn parse_config_path() {
        let raw = vec![s("bin"), s("run"), s("--config"), s("my.json")];
        // This will fail because my.json doesn't exist, but we test the CLI parsing
        // by using parse_cli_overrides directly
        let parsed = parse_cli_overrides(&raw).unwrap();
        assert_eq!(parsed.overrides.config_path, Some(PathBuf::from("my.json")));
    }

    #[test]
    fn parse_tag() {
        let raw = vec![s("bin"), s("run"), s("--tag"), s("test1")];
        let inv = parse_args_from(&raw).unwrap().unwrap();
        assert_eq!(inv.config.tag, Some("test1".to_string()));
    }

    #[test]
    fn parse_out() {
        let raw = vec![s("bin"), s("run"), s("--out"), s("out.json")];
        let inv = parse_args_from(&raw).unwrap().unwrap();
        assert_eq!(inv.config.output, Some(PathBuf::from("out.json")));
    }

    #[test]
    fn parse_force() {
        let raw = vec![s("bin"), s("run"), s("--force")];
        let inv = parse_args_from(&raw).unwrap().unwrap();
        assert!(inv.config.allow_overwrite);
    }

    #[test]
    fn parse_session_limit() {
        let raw = vec![s("bin"), s("run"), s("--session-limit"), s("5")];
        let inv = parse_args_from(&raw).unwrap().unwrap();
        assert_eq!(inv.config.session_limit, Some(5));
    }

    #[test]
    fn parse_instance_limit() {
        let raw = vec![s("bin"), s("run"), s("--instance-limit"), s("1")];
        let inv = parse_args_from(&raw).unwrap().unwrap();
        assert_eq!(inv.config.instance_limit, Some(1));
    }

    #[test]
    fn parse_ingest_format_json() {
        let raw = vec![s("bin"), s("run"), s("--ingest-format"), s("json")];
        let inv = parse_args_from(&raw).unwrap().unwrap();
        assert_eq!(inv.config.ingest_format, IngestFormat::Json);
    }

    #[test]
    fn parse_consolidation_off() {
        let raw = vec![s("bin"), s("run"), s("--consolidation"), s("off")];
        let inv = parse_args_from(&raw).unwrap().unwrap();
        assert_eq!(inv.config.consolidation, ConsolidationMode::Off);
    }

    #[test]
    fn parse_instance_jobs() {
        let raw = vec![s("bin"), s("run"), s("--instance-jobs"), s("4")];
        let inv = parse_args_from(&raw).unwrap().unwrap();
        assert_eq!(inv.config.instance_jobs, 4);
    }

    #[test]
    fn parse_judge_model() {
        let raw = vec![s("bin"), s("run"), s("--judge-model"), s("gpt-4o")];
        let inv = parse_args_from(&raw).unwrap().unwrap();
        assert_eq!(inv.config.judge_model, Some("gpt-4o".to_string()));
    }

    #[test]
    fn parse_help_returns_none() {
        let raw = vec![s("bin"), s("--help")];
        assert!(parse_args_from(&raw).unwrap().is_none());
    }

    // --- validate_qa_overrides ---

    #[test]
    fn qa_rejects_profile() {
        let overrides = CliOverrides {
            profile: Some(RunProfile::Smoke),
            ..CliOverrides::default()
        };
        let err = validate_qa_overrides(&overrides).unwrap_err();
        assert!(err.contains("--profile"));
    }

    #[test]
    fn qa_rejects_config() {
        let overrides = CliOverrides {
            config_path: Some(PathBuf::from("x.json")),
            ..CliOverrides::default()
        };
        let err = validate_qa_overrides(&overrides).unwrap_err();
        assert!(err.contains("--config"));
    }

    #[test]
    fn qa_rejects_dataset() {
        let overrides = CliOverrides {
            dataset: Some(PathBuf::from("data.json")),
            ..CliOverrides::default()
        };
        let err = validate_qa_overrides(&overrides).unwrap_err();
        assert!(err.contains("--dataset"));
    }

    #[test]
    fn qa_rejects_session_limit() {
        let overrides = CliOverrides {
            session_limit: Some(5),
            ..CliOverrides::default()
        };
        let err = validate_qa_overrides(&overrides).unwrap_err();
        assert!(err.contains("--session-limit"));
    }

    #[test]
    fn qa_rejects_instance_limit() {
        let overrides = CliOverrides {
            instance_limit: Some(1),
            ..CliOverrides::default()
        };
        let err = validate_qa_overrides(&overrides).unwrap_err();
        assert!(err.contains("--instance-limit"));
    }

    #[test]
    fn qa_rejects_ingest_format() {
        let overrides = CliOverrides {
            ingest_format: Some(IngestFormat::Json),
            ..CliOverrides::default()
        };
        let err = validate_qa_overrides(&overrides).unwrap_err();
        assert!(err.contains("--ingest-format"));
    }

    #[test]
    fn qa_rejects_consolidation() {
        let overrides = CliOverrides {
            consolidation: Some(ConsolidationMode::Off),
            ..CliOverrides::default()
        };
        let err = validate_qa_overrides(&overrides).unwrap_err();
        assert!(err.contains("--consolidation"));
    }

    #[test]
    fn qa_rejects_multiple_flags() {
        let overrides = CliOverrides {
            profile: Some(RunProfile::Smoke),
            config_path: Some(PathBuf::from("x.json")),
            dataset: Some(PathBuf::from("d.json")),
            session_limit: Some(5),
            instance_limit: Some(1),
            ingest_format: Some(IngestFormat::Json),
            consolidation: Some(ConsolidationMode::Off),
            ..CliOverrides::default()
        };
        let err = validate_qa_overrides(&overrides).unwrap_err();
        assert!(err.contains("--profile"));
        assert!(err.contains("--config"));
        assert!(err.contains("--dataset"));
        assert!(err.contains("--session-limit"));
        assert!(err.contains("--instance-limit"));
        assert!(err.contains("--ingest-format"));
        assert!(err.contains("--consolidation"));
    }

    #[test]
    fn qa_allows_out() {
        let overrides = CliOverrides {
            output: Some(PathBuf::from("out.json")),
            ..CliOverrides::default()
        };
        assert!(validate_qa_overrides(&overrides).is_ok());
    }

    #[test]
    fn qa_allows_tag() {
        let overrides = CliOverrides {
            tag: Some("test".into()),
            ..CliOverrides::default()
        };
        assert!(validate_qa_overrides(&overrides).is_ok());
    }

    #[test]
    fn qa_allows_instance() {
        let overrides = CliOverrides {
            instances: vec!["q1".into()],
            ..CliOverrides::default()
        };
        assert!(validate_qa_overrides(&overrides).is_ok());
    }

    #[test]
    fn qa_allows_instance_jobs() {
        let overrides = CliOverrides {
            instance_jobs: Some(4),
            ..CliOverrides::default()
        };
        assert!(validate_qa_overrides(&overrides).is_ok());
    }

    #[test]
    fn qa_allows_judge_model() {
        let overrides = CliOverrides {
            judge_model: Some("gpt-4o".into()),
            ..CliOverrides::default()
        };
        assert!(validate_qa_overrides(&overrides).is_ok());
    }

    #[test]
    fn qa_allows_force() {
        let overrides = CliOverrides {
            allow_overwrite: true,
            ..CliOverrides::default()
        };
        assert!(validate_qa_overrides(&overrides).is_ok());
    }

    // --- resolve_fresh_config ---

    #[test]
    fn resolve_fresh_config_smoke_profile() {
        let overrides = CliOverrides {
            profile: Some(RunProfile::Smoke),
            ..CliOverrides::default()
        };
        let config = resolve_fresh_config(overrides).unwrap();
        assert_eq!(config.profile, RunProfile::Smoke);
        assert_eq!(
            config.dataset,
            PathBuf::from("data/longmemeval_s_cleaned.json")
        );
        assert_eq!(config.instance_limit, Some(1));
        assert_eq!(config.ingest_format, IngestFormat::Text);
        assert_eq!(config.consolidation, ConsolidationMode::End);
        assert_eq!(config.instance_jobs, 1);
    }

    #[test]
    fn resolve_fresh_config_with_config_overlay() {
        let dir = env::temp_dir().join(format!("longmemeval-config-test-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let config_path = dir.join("overlay.json");
        fs::write(
            &config_path,
            r#"{"instance_jobs": 4, "tag": "overlay-tag"}"#,
        )
        .unwrap();

        let overrides = CliOverrides {
            profile: Some(RunProfile::Smoke),
            config_path: Some(config_path),
            ..CliOverrides::default()
        };
        let config = resolve_fresh_config(overrides).unwrap();
        assert_eq!(config.instance_jobs, 4);
        assert_eq!(config.tag, Some("overlay-tag".to_string()));
        // smoke profile's instance_limit should still apply (not overridden by overlay)
        assert_eq!(config.instance_limit, Some(1));

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn resolve_fresh_config_cli_overrides_config() {
        let dir = env::temp_dir().join(format!(
            "longmemeval-cli-override-test-{}",
            std::process::id()
        ));
        fs::create_dir_all(&dir).unwrap();
        let config_path = dir.join("overlay.json");
        fs::write(&config_path, r#"{"instance_jobs": 4}"#).unwrap();

        let overrides = CliOverrides {
            profile: Some(RunProfile::Smoke),
            config_path: Some(config_path),
            instance_jobs: Some(8),
            ..CliOverrides::default()
        };
        let config = resolve_fresh_config(overrides).unwrap();
        // CLI override (8) should beat config file (4) and profile (1)
        assert_eq!(config.instance_jobs, 8);

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn resolve_fresh_config_full_s_defaults() {
        let overrides = CliOverrides {
            profile: Some(RunProfile::FullS),
            ..CliOverrides::default()
        };
        let config = resolve_fresh_config(overrides).unwrap();
        assert_eq!(config.profile, RunProfile::FullS);
        assert_eq!(
            config.dataset,
            PathBuf::from("data/longmemeval_s_cleaned.json")
        );
        assert_eq!(config.instance_limit, None);
    }

    #[test]
    fn resolve_fresh_config_full_m_defaults() {
        let overrides = CliOverrides {
            profile: Some(RunProfile::FullM),
            ..CliOverrides::default()
        };
        let config = resolve_fresh_config(overrides).unwrap();
        assert_eq!(config.profile, RunProfile::FullM);
        assert_eq!(
            config.dataset,
            PathBuf::from("data/longmemeval_m_cleaned.json")
        );
    }

    // --- default_output_path ---

    #[test]
    fn default_output_run_smoke_no_tag() {
        let config = RunConfig {
            profile: RunProfile::Smoke,
            ..RunConfig::default()
        };
        assert_eq!(
            default_output_path(BenchCommand::Run, &config, None),
            PathBuf::from("bench/longmemeval/results/local/smoke-run.json")
        );
    }

    #[test]
    fn default_output_ingest_with_tag() {
        let config = RunConfig {
            tag: Some("test".into()),
            ..RunConfig::default()
        };
        assert_eq!(
            default_output_path(BenchCommand::Ingest, &config, None),
            PathBuf::from("bench/longmemeval/results/local/test.json")
        );
    }

    #[test]
    fn default_output_qa_no_tag_returns_artifact() {
        let config = RunConfig::default();
        let artifact = PathBuf::from("results/prev.json");
        assert_eq!(
            default_output_path(BenchCommand::Qa, &config, Some(&artifact)),
            artifact
        );
    }

    #[test]
    fn default_output_qa_with_tag() {
        let config = RunConfig {
            tag: Some("qa-test".into()),
            ..RunConfig::default()
        };
        let artifact = PathBuf::from("results/prev.json");
        assert_eq!(
            default_output_path(BenchCommand::Qa, &config, Some(&artifact)),
            PathBuf::from("bench/longmemeval/results/local/qa-test.json")
        );
    }

    #[test]
    fn default_output_with_explicit_out() {
        let config = RunConfig {
            output: Some(PathBuf::from("custom/out.json")),
            ..RunConfig::default()
        };
        assert_eq!(
            default_output_path(BenchCommand::Run, &config, None),
            PathBuf::from("custom/out.json")
        );
    }

    #[test]
    fn default_output_run_full_s_no_tag() {
        let config = RunConfig {
            profile: RunProfile::FullS,
            ..RunConfig::default()
        };
        assert_eq!(
            default_output_path(BenchCommand::Run, &config, None),
            PathBuf::from("bench/longmemeval/results/local/full-s-run.json")
        );
    }

    // --- FileRunConfig.apply ---

    #[test]
    fn file_run_config_apply_overrides_set_fields() {
        let file_config = FileRunConfig {
            dataset: Some(PathBuf::from("custom.json")),
            instance_jobs: Some(4),
            tag: Some("mytag".into()),
            ..FileRunConfig::default()
        };
        let mut config = RunConfig::default();
        file_config.apply(&mut config);
        assert_eq!(config.dataset, PathBuf::from("custom.json"));
        assert_eq!(config.instance_jobs, 4);
        assert_eq!(config.tag, Some("mytag".to_string()));
    }

    #[test]
    fn file_run_config_apply_preserves_unset_fields() {
        let file_config = FileRunConfig::default();
        let mut config = RunConfig::default();
        let original_dataset = config.dataset.clone();
        file_config.apply(&mut config);
        assert_eq!(config.dataset, original_dataset);
        assert_eq!(config.instance_jobs, 1);
    }

    // --- CliOverrides.apply ---

    #[test]
    fn cli_overrides_apply_overrides_set_fields() {
        let overrides = CliOverrides {
            tag: Some("cli-tag".into()),
            instance_jobs: Some(2),
            allow_overwrite: true,
            ..CliOverrides::default()
        };
        let mut config = RunConfig::default();
        overrides.apply(&mut config);
        assert_eq!(config.tag, Some("cli-tag".to_string()));
        assert_eq!(config.instance_jobs, 2);
        assert!(config.allow_overwrite);
    }

    #[test]
    fn cli_overrides_apply_preserves_unset_fields() {
        let overrides = CliOverrides::default();
        let mut config = RunConfig {
            tag: Some("existing".into()),
            ..RunConfig::default()
        };
        overrides.apply(&mut config);
        assert_eq!(config.tag, Some("existing".to_string()));
    }

    // --- Profile JSON loading ---

    #[test]
    fn smoke_json_loads() {
        let config = load_json_config(&RunProfile::Smoke.config_path()).unwrap();
        assert_eq!(
            config.dataset,
            Some(PathBuf::from("data/longmemeval_s_cleaned.json"))
        );
        assert_eq!(config.instance_limit, Some(1));
        assert_eq!(config.consolidation, Some(ConsolidationMode::End));
    }

    #[test]
    fn full_s_json_loads() {
        let config = load_json_config(&RunProfile::FullS.config_path()).unwrap();
        assert_eq!(
            config.dataset,
            Some(PathBuf::from("data/longmemeval_s_cleaned.json"))
        );
        assert_eq!(config.instance_limit, None);
        assert_eq!(config.consolidation, Some(ConsolidationMode::End));
    }

    #[test]
    fn full_m_json_loads() {
        let config = load_json_config(&RunProfile::FullM.config_path()).unwrap();
        assert_eq!(
            config.dataset,
            Some(PathBuf::from("data/longmemeval_m_cleaned.json"))
        );
        assert_eq!(config.consolidation, Some(ConsolidationMode::End));
    }

    // --- IngestFormat / ConsolidationMode FromStr ---

    #[test]
    fn ingest_format_from_str() {
        assert_eq!("text".parse::<IngestFormat>().unwrap(), IngestFormat::Text);
        assert_eq!("json".parse::<IngestFormat>().unwrap(), IngestFormat::Json);
        assert!("xml".parse::<IngestFormat>().is_err());
    }

    #[test]
    fn consolidation_mode_from_str() {
        assert_eq!(
            "end".parse::<ConsolidationMode>().unwrap(),
            ConsolidationMode::End
        );
        assert_eq!(
            "per-session".parse::<ConsolidationMode>().unwrap(),
            ConsolidationMode::PerSession
        );
        assert_eq!(
            "off".parse::<ConsolidationMode>().unwrap(),
            ConsolidationMode::Off
        );
        assert!("never".parse::<ConsolidationMode>().is_err());
    }

    #[test]
    fn ingest_format_as_str() {
        assert_eq!(IngestFormat::Text.as_str(), "text");
        assert_eq!(IngestFormat::Json.as_str(), "json");
    }

    #[test]
    fn consolidation_mode_as_str() {
        assert_eq!(ConsolidationMode::End.as_str(), "end");
        assert_eq!(ConsolidationMode::PerSession.as_str(), "per-session");
        assert_eq!(ConsolidationMode::Off.as_str(), "off");
    }

    // --- Artifact type serde roundtrips ---

    #[test]
    fn benchmark_output_serde_roundtrip() {
        let mut banks = HashMap::new();
        banks.insert("q1".to_string(), "bank1".to_string());

        let mut per_category = HashMap::new();
        per_category.insert(
            "multi-session".to_string(),
            CategoryResult {
                accuracy: 0.75,
                count: 4,
            },
        );

        let output = BenchmarkOutput {
            benchmark: "longmemeval".into(),
            timestamp: "2026-03-15T00:00:00Z".into(),
            commit: Some("abc123".into()),
            tag: Some("test".into()),
            retain_model: "model-a".into(),
            reflect_model: "model-b".into(),
            embedding_model: "model-c".into(),
            reranker_model: "model-d".into(),
            judge_model: String::new(),
            consolidation_strategy: "end".into(),
            total_questions: 10,
            accuracy: 0.0,
            per_category,
            banks,
            manifest: BenchmarkManifest::default(),
            artifacts: BenchmarkArtifacts::default(),
            stage_metrics: BTreeMap::new(),
            total_time_s: 42.0,
        };

        let json = serde_json::to_string(&output).unwrap();
        let back: BenchmarkOutput = serde_json::from_str(&json).unwrap();
        assert_eq!(back.benchmark, "longmemeval");
        assert_eq!(back.total_questions, 10);
        assert_eq!(back.banks.get("q1").unwrap(), "bank1");
        assert_eq!(back.per_category["multi-session"].count, 4);
    }

    #[test]
    fn category_result_serde_roundtrip() {
        let cr = CategoryResult {
            accuracy: 0.85,
            count: 20,
        };
        let json = serde_json::to_string(&cr).unwrap();
        let back: CategoryResult = serde_json::from_str(&json).unwrap();
        assert!((back.accuracy - 0.85).abs() < f64::EPSILON);
        assert_eq!(back.count, 20);
    }

    #[test]
    fn benchmark_manifest_serde_roundtrip() {
        let m = BenchmarkManifest {
            protocol_version: "2026-03-15-longmemeval-v1".into(),
            profile: "smoke".into(),
            mode: "run".into(),
            config_path: None,
            dataset_path: "data/test.json".into(),
            dataset_fingerprint: "abc123".into(),
            command: "longmemeval-bench run".into(),
            selected_instances: vec!["q1".into()],
            ingest_format: "text".into(),
            instance_concurrency: 1,
            consolidation_strategy: "end".into(),
            session_limit: Some(5),
            instance_limit: Some(1),
            dirty_worktree: Some(false),
            prompt_hashes: BenchmarkPromptHashes::default(),
            runtime_config: BenchmarkRuntimeConfig::default(),
            source_artifact: None,
        };
        let json = serde_json::to_string(&m).unwrap();
        let back: BenchmarkManifest = serde_json::from_str(&json).unwrap();
        assert_eq!(back.protocol_version, "2026-03-15-longmemeval-v1");
        assert_eq!(back.profile, "smoke");
        assert_eq!(back.session_limit, Some(5));
    }

    #[test]
    fn question_result_serde_roundtrip() {
        let qr = QuestionResult {
            question_id: "q1".into(),
            category: "multi-session".into(),
            judge_correct: true,
            judge_reasoning: "correct".into(),
            hypothesis: "answer".into(),
            ground_truth: "truth".into(),
            bank_id: "bank1".into(),
            elapsed_s: 1.5,
            status: "ok".into(),
            error: None,
            qa_stage_metrics: BTreeMap::new(),
        };
        let json = serde_json::to_string(&qr).unwrap();
        let back: QuestionResult = serde_json::from_str(&json).unwrap();
        assert_eq!(back.question_id, "q1");
        assert!(back.judge_correct);
        assert_eq!(back.status, "ok");
    }

    #[test]
    fn question_debug_record_serde_roundtrip() {
        let dr = QuestionDebugRecord {
            question_id: "q1".into(),
            question: "What happened?".into(),
            reflect_trace: vec![ReflectTraceEntry {
                iteration: 0,
                tool_name: "search_observations".into(),
                query: "test query".into(),
            }],
            final_done: None,
            retrieved_context: vec![RetrievedFactEntry {
                id: "f1".into(),
                content: "some fact".into(),
                score: 0.95,
                network: "world".into(),
            }],
        };
        let json = serde_json::to_string(&dr).unwrap();
        let back: QuestionDebugRecord = serde_json::from_str(&json).unwrap();
        assert_eq!(back.question_id, "q1");
        assert_eq!(back.reflect_trace.len(), 1);
        assert_eq!(back.retrieved_context.len(), 1);
    }

    #[test]
    fn benchmark_artifacts_serde_roundtrip() {
        let a = BenchmarkArtifacts {
            questions_path: "results/test.questions.jsonl".into(),
            debug_path: "results/test.debug.jsonl".into(),
        };
        let json = serde_json::to_string(&a).unwrap();
        let back: BenchmarkArtifacts = serde_json::from_str(&json).unwrap();
        assert_eq!(back.questions_path, "results/test.questions.jsonl");
    }

    #[test]
    fn source_artifact_serde_roundtrip() {
        let sa = SourceArtifact {
            path: "results/prev.json".into(),
            fingerprint: "abc123".into(),
            mode: "ingest".into(),
            tag: Some("prev".into()),
            commit: Some("def456".into()),
        };
        let json = serde_json::to_string(&sa).unwrap();
        let back: SourceArtifact = serde_json::from_str(&json).unwrap();
        assert_eq!(back.path, "results/prev.json");
        assert_eq!(back.tag, Some("prev".into()));
    }

    // --- ensure_output_paths_are_safe ---

    #[test]
    fn safety_blocks_existing_output() {
        let dir = env::temp_dir().join(format!("longmemeval-safety-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let output = dir.join("test.json");
        fs::write(&output, "{}").unwrap();

        let err =
            ensure_output_paths_are_safe(BenchCommand::Run, &output, None, false).unwrap_err();
        assert!(err.contains("refusing to overwrite"));

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn safety_allows_with_force() {
        let dir = env::temp_dir().join(format!("longmemeval-safety-force-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let output = dir.join("test.json");
        fs::write(&output, "{}").unwrap();

        assert!(ensure_output_paths_are_safe(BenchCommand::Run, &output, None, true).is_ok());

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn safety_allows_nonexistent_output() {
        let dir = env::temp_dir().join(format!(
            "longmemeval-safety-nofile-{}",
            std::process::id()
        ));
        let output = dir.join("nonexistent.json");
        assert!(ensure_output_paths_are_safe(BenchCommand::Run, &output, None, false).is_ok());
    }

    #[test]
    fn safety_qa_blocks_overwriting_source_artifact() {
        let dir = env::temp_dir().join(format!(
            "longmemeval-safety-qa-{}",
            std::process::id()
        ));
        fs::create_dir_all(&dir).unwrap();
        let artifact = dir.join("artifact.json");
        fs::write(&artifact, "{}").unwrap();

        let err = ensure_output_paths_are_safe(
            BenchCommand::Qa,
            &artifact,
            Some(&artifact),
            false,
        )
        .unwrap_err();
        assert!(err.contains("source artifact"));

        fs::remove_dir_all(&dir).ok();
    }

    // --- git helpers ---

    #[test]
    fn git_commit_sha_returns_some() {
        // Running in a git repo, should return Some
        let sha = git_commit_sha();
        assert!(sha.is_some());
        let sha = sha.unwrap();
        assert!(!sha.is_empty());
        assert!(sha.len() <= 12);
    }

    #[test]
    fn git_dirty_worktree_returns_some() {
        let result = git_dirty_worktree();
        assert!(result.is_some());
    }

    // --- load_benchmark_output ---

    #[test]
    fn load_benchmark_output_parses_valid_json() {
        let dir = env::temp_dir().join(format!(
            "longmemeval-load-artifact-{}",
            std::process::id()
        ));
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test-artifact.json");

        let output = BenchmarkOutput {
            benchmark: "longmemeval".into(),
            timestamp: "2026-03-15T00:00:00Z".into(),
            commit: None,
            tag: None,
            retain_model: "m1".into(),
            reflect_model: "m2".into(),
            embedding_model: "m3".into(),
            reranker_model: "m4".into(),
            judge_model: String::new(),
            consolidation_strategy: "end".into(),
            total_questions: 5,
            accuracy: 0.0,
            per_category: HashMap::new(),
            banks: HashMap::new(),
            manifest: BenchmarkManifest {
                profile: "smoke".into(),
                mode: "ingest".into(),
                dataset_path: "data/test.json".into(),
                ingest_format: "text".into(),
                consolidation_strategy: "end".into(),
                instance_concurrency: 1,
                ..BenchmarkManifest::default()
            },
            artifacts: BenchmarkArtifacts::default(),
            stage_metrics: BTreeMap::new(),
            total_time_s: 10.0,
        };
        let json = serde_json::to_string_pretty(&output).unwrap();
        fs::write(&path, &json).unwrap();

        let loaded = load_benchmark_output(&path).unwrap();
        assert_eq!(loaded.benchmark, "longmemeval");
        assert_eq!(loaded.total_questions, 5);

        fs::remove_dir_all(&dir).ok();
    }

    // --- run_config_from_artifact ---

    #[test]
    fn run_config_from_artifact_extracts_config() {
        let output = BenchmarkOutput {
            benchmark: "longmemeval".into(),
            timestamp: "2026-03-15T00:00:00Z".into(),
            commit: None,
            tag: Some("test-tag".into()),
            retain_model: "m1".into(),
            reflect_model: "m2".into(),
            embedding_model: "m3".into(),
            reranker_model: "m4".into(),
            judge_model: "gpt-4o".into(),
            consolidation_strategy: "end".into(),
            total_questions: 5,
            accuracy: 0.0,
            per_category: HashMap::new(),
            banks: HashMap::new(),
            manifest: BenchmarkManifest {
                profile: "smoke".into(),
                mode: "ingest".into(),
                dataset_path: "data/longmemeval_s_cleaned.json".into(),
                ingest_format: "text".into(),
                consolidation_strategy: "end".into(),
                instance_concurrency: 2,
                session_limit: Some(10),
                selected_instances: vec!["q1".into(), "q2".into()],
                ..BenchmarkManifest::default()
            },
            artifacts: BenchmarkArtifacts::default(),
            stage_metrics: BTreeMap::new(),
            total_time_s: 10.0,
        };

        let config = run_config_from_artifact(&output).unwrap();
        assert_eq!(config.profile, RunProfile::Smoke);
        assert_eq!(
            config.dataset,
            PathBuf::from("data/longmemeval_s_cleaned.json")
        );
        assert_eq!(config.consolidation, ConsolidationMode::End);
        assert_eq!(config.ingest_format, IngestFormat::Text);
        assert_eq!(config.session_limit, Some(10));
        assert_eq!(config.instance_jobs, 2);
        assert_eq!(config.tag, Some("test-tag".into()));
        assert_eq!(config.judge_model, Some("gpt-4o".into()));
        assert_eq!(config.instances, vec!["q1", "q2"]);
    }

    // --- compute_per_category ---

    #[test]
    fn compute_per_category_groups_and_counts() {
        let results = vec![
            QuestionResult {
                question_id: "q1".into(),
                category: "multi-session".into(),
                judge_correct: true,
                judge_reasoning: String::new(),
                hypothesis: String::new(),
                ground_truth: String::new(),
                bank_id: String::new(),
                elapsed_s: 0.0,
                status: "ok".into(),
                error: None,
                qa_stage_metrics: BTreeMap::new(),
            },
            QuestionResult {
                question_id: "q2".into(),
                category: "multi-session".into(),
                judge_correct: false,
                judge_reasoning: String::new(),
                hypothesis: String::new(),
                ground_truth: String::new(),
                bank_id: String::new(),
                elapsed_s: 0.0,
                status: "ok".into(),
                error: None,
                qa_stage_metrics: BTreeMap::new(),
            },
            QuestionResult {
                question_id: "q3".into(),
                category: "temporal-reasoning".into(),
                judge_correct: true,
                judge_reasoning: String::new(),
                hypothesis: String::new(),
                ground_truth: String::new(),
                bank_id: String::new(),
                elapsed_s: 0.0,
                status: "ok".into(),
                error: None,
                qa_stage_metrics: BTreeMap::new(),
            },
        ];

        let cats = compute_per_category(&results);
        assert_eq!(cats["multi-session"].count, 2);
        assert!((cats["multi-session"].accuracy - 0.5).abs() < f64::EPSILON);
        assert_eq!(cats["temporal-reasoning"].count, 1);
        assert!((cats["temporal-reasoning"].accuracy - 1.0).abs() < f64::EPSILON);
    }

    // --- Judge prompt selection ---

    fn make_instance(
        question_id: &str,
        question_type: dataset::QuestionType,
    ) -> dataset::LongMemEvalInstance {
        dataset::LongMemEvalInstance {
            question_id: question_id.into(),
            question_type,
            question: "What?".into(),
            answer: serde_json::json!("yes"),
            question_date: "2023/05/25 (Thu) 14:30".into(),
            haystack_dates: vec!["2023/05/20 (Mon) 10:15".into()],
            haystack_session_ids: vec!["s1".into()],
            haystack_sessions: vec![vec![dataset::Turn {
                role: "user".into(),
                content: "Hi".into(),
            }]],
            answer_session_ids: vec!["s1".into()],
        }
    }

    #[test]
    fn select_judge_prompt_factual() {
        let inst = make_instance("ssu_1", dataset::QuestionType::SingleSessionUser);
        assert_eq!(select_judge_prompt(&inst), JUDGE_FACTUAL);
    }

    #[test]
    fn select_judge_prompt_ssa() {
        let inst = make_instance("ssa_1", dataset::QuestionType::SingleSessionAssistant);
        assert_eq!(select_judge_prompt(&inst), JUDGE_FACTUAL);
    }

    #[test]
    fn select_judge_prompt_multi_session() {
        let inst = make_instance("ms_1", dataset::QuestionType::MultiSession);
        assert_eq!(select_judge_prompt(&inst), JUDGE_FACTUAL);
    }

    #[test]
    fn select_judge_prompt_temporal() {
        let inst = make_instance("tr_1", dataset::QuestionType::TemporalReasoning);
        assert_eq!(select_judge_prompt(&inst), JUDGE_TEMPORAL);
    }

    #[test]
    fn select_judge_prompt_knowledge_update() {
        let inst = make_instance("ku_1", dataset::QuestionType::KnowledgeUpdate);
        assert_eq!(select_judge_prompt(&inst), JUDGE_KNOWLEDGE_UPDATE);
    }

    #[test]
    fn select_judge_prompt_preference() {
        let inst = make_instance("ssp_1", dataset::QuestionType::SingleSessionPreference);
        assert_eq!(select_judge_prompt(&inst), JUDGE_PREFERENCE);
    }

    #[test]
    fn select_judge_prompt_abstention() {
        let inst = make_instance("ssu_1_abs", dataset::QuestionType::SingleSessionUser);
        assert_eq!(select_judge_prompt(&inst), JUDGE_ABSTENTION);
    }

    #[test]
    fn render_judge_prompt_replaces_placeholders() {
        let rendered = render_judge_prompt(
            JUDGE_FACTUAL,
            "What color?",
            "blue",
            "The color is blue.",
        );
        assert!(rendered.contains("What color?"));
        assert!(rendered.contains("blue"));
        assert!(rendered.contains("The color is blue."));
        assert!(!rendered.contains("{question}"));
        assert!(!rendered.contains("{answer}"));
        assert!(!rendered.contains("{response}"));
    }

    #[test]
    fn judge_prompt_hash_deterministic() {
        let h1 = judge_prompt_hash();
        let h2 = judge_prompt_hash();
        assert_eq!(h1, h2);
        assert_eq!(h1.len(), 16);
    }

    #[test]
    fn resolve_judge_model_with_config_override() {
        let config = RunConfig {
            judge_model: Some("claude-3".into()),
            ..Default::default()
        };
        assert_eq!(resolve_judge_model(&config), Some("claude-3".into()));
    }

    #[test]
    fn resolve_judge_model_default_gpt4o() {
        // Ensure JUDGE_MODEL is not set for this test
        let _guard = env_guard("JUDGE_MODEL");
        let config = RunConfig::default();
        assert_eq!(resolve_judge_model(&config), Some("gpt-4o".into()));
    }

    /// RAII guard that removes an env var for the test scope and restores it after.
    struct EnvGuard {
        key: &'static str,
        original: Option<String>,
    }

    impl Drop for EnvGuard {
        fn drop(&mut self) {
            match &self.original {
                Some(val) => unsafe { env::set_var(self.key, val) },
                None => unsafe { env::remove_var(self.key) },
            }
        }
    }

    fn env_guard(key: &'static str) -> EnvGuard {
        let original = env::var(key).ok();
        unsafe { env::remove_var(key) };
        EnvGuard { key, original }
    }

    // --- compute_accuracy ---

    fn make_qr(question_id: &str, correct: bool, status: &str) -> QuestionResult {
        QuestionResult {
            question_id: question_id.into(),
            category: "test".into(),
            judge_correct: correct,
            judge_reasoning: String::new(),
            hypothesis: String::new(),
            ground_truth: String::new(),
            bank_id: String::new(),
            elapsed_s: 0.0,
            status: status.into(),
            error: None,
            qa_stage_metrics: BTreeMap::new(),
        }
    }

    #[test]
    fn accuracy_all_correct() {
        let results = vec![
            make_qr("q1", true, "ok"),
            make_qr("q2", true, "ok"),
            make_qr("q3", true, "ok"),
        ];
        let acc = compute_accuracy(&results);
        assert!((acc - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn accuracy_mixed() {
        let results = vec![
            make_qr("q1", true, "ok"),
            make_qr("q2", true, "ok"),
            make_qr("q3", false, "ok"),
        ];
        let acc = compute_accuracy(&results);
        assert!((acc - 2.0 / 3.0).abs() < f64::EPSILON);
    }

    #[test]
    fn accuracy_all_wrong() {
        let results = vec![
            make_qr("q1", false, "ok"),
            make_qr("q2", false, "ok"),
            make_qr("q3", false, "ok"),
        ];
        let acc = compute_accuracy(&results);
        assert!((acc - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn accuracy_with_errors_counted_wrong() {
        // reflect_error results have judge_correct=false, counted in denominator
        let results = vec![
            make_qr("q1", true, "ok"),
            make_qr("q2", false, "reflect_error"),
            make_qr("q3", true, "ok"),
        ];
        let acc = compute_accuracy(&results);
        assert!((acc - 2.0 / 3.0).abs() < f64::EPSILON);
    }

    #[test]
    fn accuracy_empty() {
        let results: Vec<QuestionResult> = vec![];
        let acc = compute_accuracy(&results);
        assert!((acc - 0.0).abs() < f64::EPSILON);
    }
}
