//! LongMemEval benchmark harness for Elephant.

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::str::FromStr;

use serde::{Deserialize, Serialize};

#[path = "../common/mod.rs"]
mod common;
mod dataset;
mod ingest;

use ingest::{ConsolidationMode, IngestFormat};

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
    // For now, start with default config since we don't have BenchmarkOutput yet (Phase 4).
    // The qa subcommand will load the artifact and extract config from it.
    let mut config = RunConfig::default();

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

// --- Main ---

fn main() {
    let invocation = parse_args();
    let command = invocation.command;
    let artifact_path = invocation.artifact_path.clone();
    let config = invocation.config;
    let output_path = default_output_path(command, &config, artifact_path.as_deref());

    eprintln!("longmemeval-bench {}", command.as_str());
    eprintln!("  profile:       {}", config.profile.as_str());
    eprintln!("  dataset:       {}", config.dataset.display());
    eprintln!("  output:        {}", output_path.display());
    if let Some(ref tag) = config.tag {
        eprintln!("  tag:           {tag}");
    }
    eprintln!("  ingest_format: {}", config.ingest_format.as_str());
    eprintln!("  consolidation: {}", config.consolidation.as_str());
    eprintln!("  instance_jobs: {}", config.instance_jobs);
    if let Some(limit) = config.session_limit {
        eprintln!("  session_limit: {limit}");
    }
    if let Some(limit) = config.instance_limit {
        eprintln!("  instance_limit: {limit}");
    }
    if !config.instances.is_empty() {
        eprintln!("  instances:     {:?}", config.instances);
    }
    if let Some(ref judge) = config.judge_model {
        eprintln!("  judge_model:   {judge}");
    }

    eprintln!();
    eprintln!("pipeline not yet wired (Plan 02)");
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
        let raw = vec![s("bin"), s("qa"), s("path.json")];
        let inv = parse_args_from(&raw).unwrap().unwrap();
        assert_eq!(inv.command, BenchCommand::Qa);
        assert_eq!(inv.artifact_path, Some(PathBuf::from("path.json")));
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
}
