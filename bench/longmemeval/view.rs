//! View a single LongMemEval benchmark result or compare two side-by-side.
//!
//! Usage:
//!     cargo run --bin longmemeval-view -- <file.json>
//!     cargo run --bin longmemeval-view -- [--verbose] <file.json> [file2.json]

use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::path::Path;
use std::process;

use serde::Deserialize;
use tabled::settings::object::Columns;
use tabled::settings::style::Style;
use tabled::settings::{Alignment, Modify};
use tabled::{Table, Tabled};

#[path = "../common/mod.rs"]
mod common;

use common::io::sidecar_path;

// ---------------------------------------------------------------------------
// View-local deserialization types (reader resilience: #[serde(default)] on ALL)
// ---------------------------------------------------------------------------

#[derive(Debug, Default, Deserialize)]
struct ViewBenchmarkOutput {
    #[serde(default)]
    benchmark: String,
    #[serde(default)]
    timestamp: String,
    #[serde(default)]
    commit: Option<String>,
    #[serde(default)]
    tag: Option<String>,
    #[serde(default)]
    retain_model: String,
    #[serde(default)]
    reflect_model: String,
    #[serde(default)]
    embedding_model: String,
    #[serde(default)]
    reranker_model: String,
    #[serde(default)]
    judge_model: String,
    #[serde(default)]
    consolidation_strategy: String,
    #[serde(default)]
    total_questions: usize,
    #[serde(default)]
    accuracy: f64,
    #[serde(default)]
    per_category: BTreeMap<String, ViewCategoryResult>,
    #[serde(default)]
    banks: BTreeMap<String, String>,
    #[serde(default)]
    manifest: ViewManifest,
    #[serde(default)]
    artifacts: ViewArtifacts,
    #[serde(default)]
    stage_metrics: BTreeMap<String, ViewStageUsage>,
    #[serde(default)]
    cache_aware_stage_metrics: BTreeMap<String, ViewCacheAwareStageUsage>,
    #[serde(default)]
    total_time_s: f64,
}

#[derive(Debug, Default, Deserialize)]
struct ViewManifest {
    #[serde(default)]
    protocol_version: String,
    #[serde(default)]
    profile: String,
    #[serde(default)]
    mode: String,
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
    #[serde(default)]
    session_limit: Option<usize>,
    #[serde(default)]
    instance_limit: Option<usize>,
    #[serde(default)]
    dirty_worktree: Option<bool>,
    #[serde(default)]
    prompt_hashes: ViewPromptHashes,
    #[serde(default)]
    runtime_config: ViewRuntimeConfig,
    #[serde(default)]
    source_artifact: Option<ViewSourceArtifact>,
}

#[derive(Debug, Default, Deserialize)]
struct ViewPromptHashes {
    #[serde(default)]
    judge: String,
    #[serde(default, flatten)]
    other: BTreeMap<String, serde_json::Value>,
}

#[derive(Debug, Default, Deserialize)]
struct ViewRuntimeConfig {
    #[serde(default, flatten)]
    fields: BTreeMap<String, serde_json::Value>,
}

#[derive(Debug, Default, Deserialize)]
struct ViewSourceArtifact {
    #[serde(default)]
    path: String,
    #[serde(default)]
    tag: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
struct ViewArtifacts {
    #[serde(default)]
    questions_path: String,
    #[serde(default)]
    debug_path: String,
}

#[derive(Debug, Default, Deserialize)]
struct ViewCategoryResult {
    #[serde(default)]
    accuracy: f64,
    #[serde(default)]
    count: usize,
}

#[derive(Debug, Default, Deserialize)]
struct ViewStageUsage {
    #[serde(default)]
    input_tokens: u64,
    #[serde(default)]
    output_tokens: u64,
    #[serde(default)]
    requests: u64,
}

impl ViewStageUsage {
    fn total_tokens(&self) -> u64 {
        self.input_tokens + self.output_tokens
    }
}

#[derive(Debug, Clone, Default, Deserialize)]
struct ViewCacheAwareStageUsage {
    #[serde(default)]
    prompt_tokens: u64,
    #[serde(default)]
    uncached_prompt_tokens: u64,
    #[serde(default)]
    cache_hit_prompt_tokens: u64,
    #[serde(default)]
    cache_write_prompt_tokens: u64,
    #[serde(default)]
    completion_tokens: u64,
    #[serde(default)]
    calls: u64,
    #[serde(default)]
    errors: u64,
    #[serde(default)]
    latency_ms: u64,
    #[serde(default)]
    cache_supported_calls: u64,
    #[serde(default)]
    cache_hit_calls: u64,
    #[serde(default)]
    cache_write_calls: u64,
    #[serde(default)]
    cache_unsupported_calls: u64,
}

impl ViewCacheAwareStageUsage {
    fn merge_from(&mut self, other: &Self) {
        self.prompt_tokens += other.prompt_tokens;
        self.uncached_prompt_tokens += other.uncached_prompt_tokens;
        self.cache_hit_prompt_tokens += other.cache_hit_prompt_tokens;
        self.cache_write_prompt_tokens += other.cache_write_prompt_tokens;
        self.completion_tokens += other.completion_tokens;
        self.calls += other.calls;
        self.errors += other.errors;
        self.latency_ms += other.latency_ms;
        self.cache_supported_calls += other.cache_supported_calls;
        self.cache_hit_calls += other.cache_hit_calls;
        self.cache_write_calls += other.cache_write_calls;
        self.cache_unsupported_calls += other.cache_unsupported_calls;
    }
}

#[derive(Debug, Default, Deserialize)]
struct ViewQuestionResult {
    #[serde(default)]
    question_id: String,
    #[serde(default)]
    category: String,
    #[serde(default)]
    judge_correct: bool,
    #[serde(default)]
    hypothesis: String,
    #[serde(default)]
    ground_truth: String,
    #[serde(default)]
    bank_id: String,
    #[serde(default)]
    elapsed_s: f64,
    #[serde(default)]
    status: String,
    #[serde(default)]
    error: Option<String>,
}

// ---------------------------------------------------------------------------
// Tabled row types
// ---------------------------------------------------------------------------

#[derive(Tabled)]
struct SingleConfigRow {
    #[tabled(rename = "config")]
    key: String,
    value: String,
}

#[derive(Tabled)]
struct CompareConfigRow {
    #[tabled(rename = "config")]
    key: String,
    #[tabled(rename = "A")]
    val_a: String,
    #[tabled(rename = "B")]
    val_b: String,
}

#[derive(Tabled)]
struct CategoryRow {
    category: String,
    #[tabled(rename = "acc")]
    acc: String,
    #[tabled(rename = "n")]
    n: usize,
}

#[derive(Tabled)]
struct CompareCategoryRow {
    category: String,
    #[tabled(rename = "A")]
    acc_a: String,
    #[tabled(rename = "B")]
    acc_b: String,
    delta: String,
    #[tabled(rename = "n(A)")]
    n_a: usize,
    #[tabled(rename = "n(B)")]
    n_b: usize,
}

#[derive(Tabled)]
struct StageRow {
    stage: String,
    requests: u64,
    input_tok: u64,
    output_tok: u64,
}

#[derive(Tabled)]
struct CompareStageRow {
    stage: String,
    #[tabled(rename = "req(A)")]
    req_a: u64,
    #[tabled(rename = "req(B)")]
    req_b: u64,
    #[tabled(rename = "tok(A)")]
    tok_a: u64,
    #[tabled(rename = "tok(B)")]
    tok_b: u64,
    #[tabled(rename = "\u{0394} tok")]
    delta_tok: String,
}

#[derive(Tabled)]
struct CacheSummaryRow {
    metric: String,
    value: String,
}

#[derive(Tabled)]
struct CacheStageRow {
    stage: String,
    #[tabled(rename = "effective prompt tok")]
    effective_prompt_tok: u64,
    #[tabled(rename = "cache hit tok")]
    cache_hit_tok: u64,
    #[tabled(rename = "cache write tok")]
    cache_write_tok: u64,
    #[tabled(rename = "completion tok")]
    completion_tok: u64,
    calls: u64,
}

#[derive(Tabled)]
struct CacheCompareRow {
    metric: String,
    #[tabled(rename = "A")]
    val_a: String,
    #[tabled(rename = "B")]
    val_b: String,
    #[tabled(rename = "Δ")]
    delta: String,
}

#[derive(Tabled)]
struct CacheStageCompareRow {
    stage: String,
    #[tabled(rename = "eff(A)")]
    effective_prompt_a: u64,
    #[tabled(rename = "eff(B)")]
    effective_prompt_b: u64,
    #[tabled(rename = "Δ eff")]
    delta_effective_prompt: String,
    #[tabled(rename = "hit(A)")]
    cache_hit_a: u64,
    #[tabled(rename = "hit(B)")]
    cache_hit_b: u64,
    #[tabled(rename = "Δ hit")]
    delta_cache_hit: String,
}

#[derive(Tabled)]
struct QuestionRow {
    question_id: String,
    category: String,
    correct: String,
    status: String,
    #[tabled(rename = "elapsed")]
    elapsed_s: String,
}

#[derive(Tabled)]
struct CompareQuestionRow {
    question_id: String,
    category: String,
    #[tabled(rename = "A")]
    a: String,
    #[tabled(rename = "B")]
    b: String,
}

// ---------------------------------------------------------------------------
// Formatting helpers
// ---------------------------------------------------------------------------

fn fmt_pct(v: f64) -> String {
    format!("{:.1}%", v * 100.0)
}

fn fmt_pct_delta(a: f64, b: f64) -> String {
    let delta = (b - a) * 100.0;
    let rendered = format!("{delta:+.1}%");
    if delta.abs() < 0.05 {
        rendered
    } else if delta > 0.0 {
        format!("\x1b[32m{rendered}\x1b[0m")
    } else {
        format!("\x1b[31m{rendered}\x1b[0m")
    }
}

fn fmt_cost_delta_u64(a: u64, b: u64) -> String {
    let delta = b as i128 - a as i128;
    let rendered = format!("{delta:+}");
    if delta == 0 {
        rendered
    } else if delta < 0 {
        format!("\x1b[32m{rendered}\x1b[0m")
    } else {
        format!("\x1b[31m{rendered}\x1b[0m")
    }
}

fn fmt_time(seconds: f64) -> String {
    if seconds < 60.0 {
        format!("{seconds:.1}s")
    } else {
        let minutes = (seconds / 60.0).floor() as u64;
        let rem = seconds - minutes as f64 * 60.0;
        format!("{minutes}m{rem:.0}s")
    }
}

fn file_label(output: &ViewBenchmarkOutput, path: &str) -> String {
    if let Some(ref tag) = output.tag {
        tag.clone()
    } else {
        Path::new(path)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("?")
            .to_string()
    }
}

fn question_mark(correct: bool) -> String {
    if correct {
        "\x1b[32m\u{2713}\x1b[0m".into()
    } else {
        "\x1b[31m\u{2717}\x1b[0m".into()
    }
}

fn cache_usage_available(output: &ViewBenchmarkOutput) -> bool {
    !output.cache_aware_stage_metrics.is_empty()
}

fn sum_cache_usage(output: &ViewBenchmarkOutput) -> ViewCacheAwareStageUsage {
    let mut total = ViewCacheAwareStageUsage::default();
    for usage in output.cache_aware_stage_metrics.values() {
        total.merge_from(usage);
    }
    total
}

fn operator_stage_name(stage: &str) -> Option<&'static str> {
    match stage {
        "retain_extract" | "retain_resolve" | "retain_graph" | "retain_opinion" => Some("retain"),
        "reflect" => Some("reflect"),
        "consolidate" => Some("consolidate"),
        "opinion_merge" => Some("opinion_merge"),
        "judge" => Some("judge"),
        _ => None,
    }
}

fn rollup_operator_cache_metrics(
    output: &ViewBenchmarkOutput,
) -> BTreeMap<String, ViewCacheAwareStageUsage> {
    let mut rolled_up = BTreeMap::new();
    for (stage, usage) in &output.cache_aware_stage_metrics {
        if let Some(operator_stage) = operator_stage_name(stage) {
            rolled_up
                .entry(operator_stage.to_string())
                .or_insert_with(ViewCacheAwareStageUsage::default)
                .merge_from(usage);
        }
    }
    rolled_up
}

fn cache_summary_rows(total: &ViewCacheAwareStageUsage) -> Vec<CacheSummaryRow> {
    let effective_prompt_tokens =
        total.uncached_prompt_tokens + total.cache_write_prompt_tokens;
    let cache_hit_rate = if total.prompt_tokens == 0 {
        "0.0%".to_string()
    } else {
        fmt_pct(total.cache_hit_prompt_tokens as f64 / total.prompt_tokens as f64)
    };

    vec![
        CacheSummaryRow {
            metric: "effective prompt tok".into(),
            value: effective_prompt_tokens.to_string(),
        },
        CacheSummaryRow {
            metric: "cache hit tok".into(),
            value: total.cache_hit_prompt_tokens.to_string(),
        },
        CacheSummaryRow {
            metric: "cache write tok".into(),
            value: total.cache_write_prompt_tokens.to_string(),
        },
        CacheSummaryRow {
            metric: "cache supported".into(),
            value: total.cache_supported_calls.to_string(),
        },
        CacheSummaryRow {
            metric: "cache unsupported".into(),
            value: total.cache_unsupported_calls.to_string(),
        },
        CacheSummaryRow {
            metric: "cache hit rate".into(),
            value: cache_hit_rate,
        },
    ]
}

fn effective_prompt_tokens(usage: &ViewCacheAwareStageUsage) -> u64 {
    usage.uncached_prompt_tokens + usage.cache_write_prompt_tokens
}

fn cache_hit_rate(usage: &ViewCacheAwareStageUsage) -> f64 {
    if usage.prompt_tokens == 0 {
        0.0
    } else {
        usage.cache_hit_prompt_tokens as f64 / usage.prompt_tokens as f64
    }
}

fn cache_compare_rows(
    a: &ViewCacheAwareStageUsage,
    b: &ViewCacheAwareStageUsage,
) -> Vec<CacheCompareRow> {
    vec![
        CacheCompareRow {
            metric: "effective prompt tok".into(),
            val_a: effective_prompt_tokens(a).to_string(),
            val_b: effective_prompt_tokens(b).to_string(),
            delta: fmt_cost_delta_u64(effective_prompt_tokens(a), effective_prompt_tokens(b)),
        },
        CacheCompareRow {
            metric: "cache hit tok".into(),
            val_a: a.cache_hit_prompt_tokens.to_string(),
            val_b: b.cache_hit_prompt_tokens.to_string(),
            delta: fmt_cost_delta_u64(a.cache_hit_prompt_tokens, b.cache_hit_prompt_tokens),
        },
        CacheCompareRow {
            metric: "cache write tok".into(),
            val_a: a.cache_write_prompt_tokens.to_string(),
            val_b: b.cache_write_prompt_tokens.to_string(),
            delta: fmt_cost_delta_u64(a.cache_write_prompt_tokens, b.cache_write_prompt_tokens),
        },
        CacheCompareRow {
            metric: "cache supported".into(),
            val_a: a.cache_supported_calls.to_string(),
            val_b: b.cache_supported_calls.to_string(),
            delta: fmt_cost_delta_u64(a.cache_supported_calls, b.cache_supported_calls),
        },
        CacheCompareRow {
            metric: "cache unsupported".into(),
            val_a: a.cache_unsupported_calls.to_string(),
            val_b: b.cache_unsupported_calls.to_string(),
            delta: fmt_cost_delta_u64(a.cache_unsupported_calls, b.cache_unsupported_calls),
        },
        CacheCompareRow {
            metric: "cache hit rate".into(),
            val_a: fmt_pct(cache_hit_rate(a)),
            val_b: fmt_pct(cache_hit_rate(b)),
            delta: fmt_pct_delta(cache_hit_rate(a), cache_hit_rate(b)),
        },
    ]
}

const REQUIRED_PROMPT_HASH_KEYS: &[&str] = &[
    "retain_extract",
    "retain_resolve_system",
    "retain_resolve_user",
    "retain_graph_system",
    "retain_graph_user",
    "retain_opinion",
    "reflect_agent",
    "consolidate",
    "opinion_merge",
];

fn prompt_hashes_complete(hashes: &ViewPromptHashes) -> bool {
    if hashes.judge.is_empty() {
        return false;
    }
    REQUIRED_PROMPT_HASH_KEYS
        .iter()
        .all(|key| matches!(hashes.other.get(*key), Some(serde_json::Value::String(text)) if !text.is_empty()))
}

fn savings_comparable(a: &ViewBenchmarkOutput, b: &ViewBenchmarkOutput) -> bool {
    !a.manifest.dataset_fingerprint.is_empty()
        && a.manifest.dataset_fingerprint == b.manifest.dataset_fingerprint
        && prompt_hashes_complete(&a.manifest.prompt_hashes)
        && prompt_hashes_complete(&b.manifest.prompt_hashes)
        && a.manifest.prompt_hashes.judge == b.manifest.prompt_hashes.judge
        && a.manifest.prompt_hashes.other == b.manifest.prompt_hashes.other
        && !a.retain_model.is_empty()
        && a.retain_model == b.retain_model
        && !a.reflect_model.is_empty()
        && a.reflect_model == b.reflect_model
        && !a.judge_model.is_empty()
        && a.judge_model == b.judge_model
}

fn savings_signal(
    a: &ViewCacheAwareStageUsage,
    b: &ViewCacheAwareStageUsage,
    comparable: bool,
) -> &'static str {
    if !comparable {
        return "unavailable";
    }

    let effective_a = effective_prompt_tokens(a);
    let effective_b = effective_prompt_tokens(b);

    if a.cache_hit_prompt_tokens == 0 && b.cache_hit_prompt_tokens == 0 {
        "no cache-hit evidence"
    } else if effective_b < effective_a && b.cache_hit_prompt_tokens > a.cache_hit_prompt_tokens {
        "cache savings visible"
    } else if b.cache_hit_prompt_tokens > 0 && effective_b >= effective_a {
        "warm-up / write-heavy"
    } else {
        "no cache-hit evidence"
    }
}

fn savings_unavailable_reason(a: &ViewBenchmarkOutput, b: &ViewBenchmarkOutput) -> String {
    if a.manifest.dataset_fingerprint.is_empty() || b.manifest.dataset_fingerprint.is_empty() {
        return "dataset fingerprint missing".into();
    }
    if a.manifest.dataset_fingerprint != b.manifest.dataset_fingerprint {
        return "dataset fingerprint mismatch".into();
    }
    if !prompt_hashes_complete(&a.manifest.prompt_hashes)
        || !prompt_hashes_complete(&b.manifest.prompt_hashes)
    {
        return "prompt hashes missing".into();
    }
    if a.manifest.prompt_hashes.judge != b.manifest.prompt_hashes.judge
        || a.manifest.prompt_hashes.other != b.manifest.prompt_hashes.other
    {
        return "prompt hashes mismatch".into();
    }
    if a.retain_model.is_empty() || b.retain_model.is_empty() {
        return "retain model missing".into();
    }
    if a.retain_model != b.retain_model {
        return "retain model mismatch".into();
    }
    if a.reflect_model.is_empty() || b.reflect_model.is_empty() {
        return "reflect model missing".into();
    }
    if a.reflect_model != b.reflect_model {
        return "reflect model mismatch".into();
    }
    if a.judge_model.is_empty() || b.judge_model.is_empty() {
        return "judge model missing".into();
    }
    if a.judge_model != b.judge_model {
        return "judge model mismatch".into();
    }
    "comparison metadata missing".into()
}

fn render_table<T: Tabled>(rows: &[T], align_from: usize) -> String {
    Table::new(rows)
        .with(Style::rounded())
        .with(Modify::new(Columns::new(align_from..)).with(Alignment::right()))
        .to_string()
}

fn render_single_output(
    output: &ViewBenchmarkOutput,
    path: &str,
    verbose: bool,
    questions: &[ViewQuestionResult],
) -> String {
    let label = file_label(output, path);
    let mut rendered = String::new();

    rendered.push_str(&format!("LongMemEval: {label}\n\n"));

    let config_rows = build_config_rows(output);
    if !config_rows.is_empty() {
        rendered.push_str(&Table::new(&config_rows).with(Style::rounded()).to_string());
        rendered.push_str("\n\n");
    }

    let cat_rows = build_category_rows(output);
    rendered.push_str(&render_table(&cat_rows, 1));
    rendered.push_str("\n\n");

    let stage_rows = build_stage_rows(output);
    if !stage_rows.is_empty() {
        rendered.push_str(&render_table(&stage_rows, 1));
        rendered.push_str("\n\n");
    }

    if cache_usage_available(output) {
        let cache_rows = cache_summary_rows(&sum_cache_usage(output));
        rendered.push_str(&render_table(&cache_rows, 1));
        rendered.push_str("\n\n");

        let cache_stage_rows: Vec<CacheStageRow> = rollup_operator_cache_metrics(output)
            .into_iter()
            .map(|(stage, usage)| CacheStageRow {
                stage,
                effective_prompt_tok: usage.uncached_prompt_tokens + usage.cache_write_prompt_tokens,
                cache_hit_tok: usage.cache_hit_prompt_tokens,
                cache_write_tok: usage.cache_write_prompt_tokens,
                completion_tok: usage.completion_tokens,
                calls: usage.calls,
            })
            .collect();
        if !cache_stage_rows.is_empty() {
            rendered.push_str(&render_table(&cache_stage_rows, 1));
            rendered.push_str("\n\n");
        }
    } else {
        rendered.push_str("cache-aware metrics unavailable (legacy artifact)\n\n");
    }

    rendered.push_str(&format!("Total time: {}", fmt_time(output.total_time_s)));

    if verbose {
        if questions.is_empty() {
            rendered.push_str("\n\n(no question sidecar found)");
        } else {
            let mut sorted = questions.iter().collect::<Vec<_>>();
            sorted.sort_by(|a, b| a.category.cmp(&b.category).then(a.question_id.cmp(&b.question_id)));
            let q_rows: Vec<QuestionRow> = sorted
                .iter()
                .map(|q| QuestionRow {
                    question_id: q.question_id.clone(),
                    category: q.category.clone(),
                    correct: question_mark(q.judge_correct),
                    status: if q.status.is_empty() {
                        "ok".into()
                    } else {
                        q.status.clone()
                    },
                    elapsed_s: format!("{:.1}s", q.elapsed_s),
                })
                .collect();
            rendered.push('\n');
            rendered.push('\n');
            rendered.push_str(&render_table(&q_rows, 3));
        }
    }

    rendered
}

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

struct CliArgs {
    verbose: bool,
    files: Vec<String>,
}

fn parse_args() -> CliArgs {
    let raw = env::args().collect::<Vec<_>>();
    let mut verbose = false;
    let mut files = Vec::new();

    let mut i = 1;
    while i < raw.len() {
        match raw[i].as_str() {
            "--verbose" | "-v" => verbose = true,
            "--help" | "-h" => {
                print_usage();
                process::exit(0);
            }
            arg if arg.starts_with('-') => {
                eprintln!("Unknown flag: {arg}");
                print_usage();
                process::exit(1);
            }
            _ => files.push(raw[i].clone()),
        }
        i += 1;
    }

    if files.is_empty() || files.len() > 2 {
        print_usage();
        process::exit(1);
    }

    CliArgs { verbose, files }
}

fn print_usage() {
    eprintln!("Usage: longmemeval-view [--verbose] <file.json> [file2.json]");
    eprintln!();
    eprintln!("Flags:");
    eprintln!("  --verbose, -v  Show per-question table from JSONL sidecar");
    eprintln!("  --help, -h     Show this help");
}

// ---------------------------------------------------------------------------
// Load / parse
// ---------------------------------------------------------------------------

fn parse_output(raw: &str) -> Result<ViewBenchmarkOutput, serde_json::Error> {
    serde_json::from_str(raw)
}

fn load_file(path: &str) -> ViewBenchmarkOutput {
    let raw = fs::read_to_string(path).unwrap_or_else(|e| {
        eprintln!("Failed to read {path}: {e}");
        process::exit(1);
    });
    parse_output(&raw).unwrap_or_else(|e| {
        eprintln!("Failed to parse {path}: {e}");
        process::exit(1);
    })
}

pub fn parse_summary_artifact(raw: &str) -> Result<(), serde_json::Error> {
    parse_output(raw).map(|_| ())
}

pub fn render_compare_artifacts(
    raw_a: &str,
    raw_b: &str,
    path_a: &str,
    path_b: &str,
) -> Result<String, serde_json::Error> {
    let a = parse_output(raw_a)?;
    let b = parse_output(raw_b)?;
    Ok(render_compare_output(&a, &b, path_a, path_b, false))
}

fn load_questions(artifact_path: &Path) -> Vec<ViewQuestionResult> {
    let path = sidecar_path(artifact_path, "questions");
    if !path.exists() {
        return Vec::new();
    }
    let raw = fs::read_to_string(&path).unwrap_or_else(|e| {
        eprintln!("Failed to read {}: {e}", path.display());
        process::exit(1);
    });
    raw.lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| {
            serde_json::from_str::<ViewQuestionResult>(line).unwrap_or_else(|e| {
                eprintln!("Failed to parse question in {}: {e}", path.display());
                process::exit(1);
            })
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Build display rows
// ---------------------------------------------------------------------------

fn build_config_rows(output: &ViewBenchmarkOutput) -> Vec<SingleConfigRow> {
    let mut rows = Vec::new();

    let add = |rows: &mut Vec<SingleConfigRow>, key: &str, value: String| {
        if !value.is_empty() {
            rows.push(SingleConfigRow {
                key: key.into(),
                value,
            });
        }
    };

    add(&mut rows, "profile", output.manifest.profile.clone());
    add(
        &mut rows,
        "dataset",
        output.manifest.dataset_fingerprint.clone(),
    );
    add(
        &mut rows,
        "consolidation",
        output.consolidation_strategy.clone(),
    );
    if output.manifest.instance_concurrency > 0 {
        rows.push(SingleConfigRow {
            key: "instance_jobs".into(),
            value: output.manifest.instance_concurrency.to_string(),
        });
    }
    add(&mut rows, "retain_model", output.retain_model.clone());
    add(&mut rows, "reflect_model", output.reflect_model.clone());
    add(&mut rows, "embedding_model", output.embedding_model.clone());
    add(&mut rows, "reranker_model", output.reranker_model.clone());
    add(&mut rows, "judge_model", output.judge_model.clone());
    if let Some(ref tag) = output.tag {
        add(&mut rows, "tag", tag.clone());
    }
    if let Some(ref commit) = output.commit {
        add(&mut rows, "commit", commit.clone());
    }
    add(&mut rows, "timestamp", output.timestamp.clone());

    rows
}

fn build_category_rows(output: &ViewBenchmarkOutput) -> Vec<CategoryRow> {
    let mut rows: Vec<CategoryRow> = output
        .per_category
        .iter()
        .map(|(cat, result)| CategoryRow {
            category: cat.clone(),
            acc: fmt_pct(result.accuracy),
            n: result.count,
        })
        .collect();
    // alphabetical sort (BTreeMap already sorted, but let's be explicit)
    rows.sort_by(|a, b| a.category.cmp(&b.category));
    // overall row
    rows.push(CategoryRow {
        category: "overall".into(),
        acc: fmt_pct(output.accuracy),
        n: output.total_questions,
    });
    rows
}

fn build_stage_rows(output: &ViewBenchmarkOutput) -> Vec<StageRow> {
    let mut rows: Vec<StageRow> = output
        .stage_metrics
        .iter()
        .filter(|(_, usage)| usage.requests > 0)
        .map(|(stage, usage)| StageRow {
            stage: stage.clone(),
            requests: usage.requests,
            input_tok: usage.input_tokens,
            output_tok: usage.output_tokens,
        })
        .collect();
    rows.sort_by(|a, b| a.stage.cmp(&b.stage));
    rows
}

// ---------------------------------------------------------------------------
// Single-file mode
// ---------------------------------------------------------------------------

fn view_single(output: &ViewBenchmarkOutput, path: &str, verbose: bool) {
    let questions = if verbose {
        load_questions(Path::new(path))
    } else {
        Vec::new()
    };
    println!("{}", render_single_output(output, path, verbose, &questions));
}

// ---------------------------------------------------------------------------
// Comparison mode
// ---------------------------------------------------------------------------

fn render_compare_output(
    a: &ViewBenchmarkOutput,
    b: &ViewBenchmarkOutput,
    path_a: &str,
    path_b: &str,
    verbose: bool,
) -> String {
    let label_a = file_label(a, path_a);
    let label_b = file_label(b, path_b);
    let mut rendered = String::new();

    rendered.push_str(&format!("LongMemEval comparison: {label_a} vs {label_b}\n\n"));

    let config_pairs: Vec<(&str, String, String)> = vec![
        ("profile", a.manifest.profile.clone(), b.manifest.profile.clone()),
        (
            "dataset",
            a.manifest.dataset_fingerprint.clone(),
            b.manifest.dataset_fingerprint.clone(),
        ),
        (
            "consolidation",
            a.consolidation_strategy.clone(),
            b.consolidation_strategy.clone(),
        ),
        (
            "instance_jobs",
            a.manifest.instance_concurrency.to_string(),
            b.manifest.instance_concurrency.to_string(),
        ),
        ("retain_model", a.retain_model.clone(), b.retain_model.clone()),
        ("reflect_model", a.reflect_model.clone(), b.reflect_model.clone()),
        ("embedding_model", a.embedding_model.clone(), b.embedding_model.clone()),
        ("reranker_model", a.reranker_model.clone(), b.reranker_model.clone()),
        ("judge_model", a.judge_model.clone(), b.judge_model.clone()),
        ("tag", a.tag.clone().unwrap_or_default(), b.tag.clone().unwrap_or_default()),
        (
            "commit",
            a.commit.clone().unwrap_or_default(),
            b.commit.clone().unwrap_or_default(),
        ),
        ("timestamp", a.timestamp.clone(), b.timestamp.clone()),
    ];

    let config_rows: Vec<CompareConfigRow> = config_pairs
        .into_iter()
        .filter(|(_, va, vb)| !va.is_empty() || !vb.is_empty())
        .map(|(key, va, vb)| CompareConfigRow {
            key: key.into(),
            val_a: if va.is_empty() { "-".into() } else { va },
            val_b: if vb.is_empty() { "-".into() } else { vb },
        })
        .collect();

    if !config_rows.is_empty() {
        rendered.push_str(&Table::new(&config_rows).with(Style::rounded()).to_string());
        rendered.push_str("\n\n");
    }

    let mut all_categories: Vec<String> = a
        .per_category
        .keys()
        .chain(b.per_category.keys())
        .cloned()
        .collect();
    all_categories.sort();
    all_categories.dedup();
    all_categories.retain(|c| c != "overall");

    let default_cat = ViewCategoryResult::default();
    let mut cat_rows: Vec<CompareCategoryRow> = all_categories
        .iter()
        .map(|cat| {
            let ca = a.per_category.get(cat).unwrap_or(&default_cat);
            let cb = b.per_category.get(cat).unwrap_or(&default_cat);
            CompareCategoryRow {
                category: cat.clone(),
                acc_a: fmt_pct(ca.accuracy),
                acc_b: fmt_pct(cb.accuracy),
                delta: fmt_pct_delta(ca.accuracy, cb.accuracy),
                n_a: ca.count,
                n_b: cb.count,
            }
        })
        .collect();
    cat_rows.push(CompareCategoryRow {
        category: "overall".into(),
        acc_a: fmt_pct(a.accuracy),
        acc_b: fmt_pct(b.accuracy),
        delta: fmt_pct_delta(a.accuracy, b.accuracy),
        n_a: a.total_questions,
        n_b: b.total_questions,
    });
    rendered.push_str(&render_table(&cat_rows, 1));
    rendered.push_str("\n\n");

    let mut all_stages: Vec<String> = a
        .stage_metrics
        .keys()
        .chain(b.stage_metrics.keys())
        .cloned()
        .collect();
    all_stages.sort();
    all_stages.dedup();

    let default_stage = ViewStageUsage::default();
    let stage_rows: Vec<CompareStageRow> = all_stages
        .iter()
        .filter(|stage| {
            let sa = a.stage_metrics.get(*stage).unwrap_or(&default_stage);
            let sb = b.stage_metrics.get(*stage).unwrap_or(&default_stage);
            sa.requests > 0 || sb.requests > 0
        })
        .map(|stage| {
            let sa = a.stage_metrics.get(stage).unwrap_or(&default_stage);
            let sb = b.stage_metrics.get(stage).unwrap_or(&default_stage);
            CompareStageRow {
                stage: stage.clone(),
                req_a: sa.requests,
                req_b: sb.requests,
                tok_a: sa.total_tokens(),
                tok_b: sb.total_tokens(),
                delta_tok: fmt_cost_delta_u64(sa.total_tokens(), sb.total_tokens()),
            }
        })
        .collect();

    if !stage_rows.is_empty() {
        rendered.push_str(&render_table(&stage_rows, 1));
        rendered.push_str("\n\n");
    }

    let cache_total_a = sum_cache_usage(a);
    let cache_total_b = sum_cache_usage(b);
    rendered.push_str("cache-aware comparison\n");
    rendered.push_str(&render_table(&cache_compare_rows(&cache_total_a, &cache_total_b), 1));
    rendered.push_str("\n\n");

    let rolled_up_a = rollup_operator_cache_metrics(a);
    let rolled_up_b = rollup_operator_cache_metrics(b);
    let cache_stage_order = ["retain", "reflect", "consolidate", "opinion_merge", "judge"];
    let cache_stage_rows: Vec<CacheStageCompareRow> = cache_stage_order
        .into_iter()
        .filter_map(|stage| {
            let usage_a = rolled_up_a.get(stage).cloned().unwrap_or_default();
            let usage_b = rolled_up_b.get(stage).cloned().unwrap_or_default();
            if usage_a.calls > 0 || usage_b.calls > 0 || usage_a.prompt_tokens > 0 || usage_b.prompt_tokens > 0
            {
                Some(CacheStageCompareRow {
                    stage: stage.to_string(),
                    effective_prompt_a: effective_prompt_tokens(&usage_a),
                    effective_prompt_b: effective_prompt_tokens(&usage_b),
                    delta_effective_prompt: fmt_cost_delta_u64(
                        effective_prompt_tokens(&usage_a),
                        effective_prompt_tokens(&usage_b),
                    ),
                    cache_hit_a: usage_a.cache_hit_prompt_tokens,
                    cache_hit_b: usage_b.cache_hit_prompt_tokens,
                    delta_cache_hit: fmt_cost_delta_u64(
                        usage_a.cache_hit_prompt_tokens,
                        usage_b.cache_hit_prompt_tokens,
                    ),
                })
            } else {
                None
            }
        })
        .collect();
    if !cache_stage_rows.is_empty() {
        rendered.push_str(&render_table(&cache_stage_rows, 1));
        rendered.push_str("\n\n");
    }

    rendered.push_str("cache savings verification\n");
    if savings_comparable(a, b) {
        let signal = savings_signal(&cache_total_a, &cache_total_b, true);
        let savings_rows = vec![
            SingleConfigRow {
                key: "verdict".into(),
                value: signal.into(),
            },
            SingleConfigRow {
                key: "effective prompt tok".into(),
                value: format!(
                    "{} -> {} ({})",
                    effective_prompt_tokens(&cache_total_a),
                    effective_prompt_tokens(&cache_total_b),
                    fmt_cost_delta_u64(
                        effective_prompt_tokens(&cache_total_a),
                        effective_prompt_tokens(&cache_total_b),
                    )
                ),
            },
            SingleConfigRow {
                key: "cache hit tok".into(),
                value: format!(
                    "{} -> {} ({})",
                    cache_total_a.cache_hit_prompt_tokens,
                    cache_total_b.cache_hit_prompt_tokens,
                    fmt_cost_delta_u64(
                        cache_total_a.cache_hit_prompt_tokens,
                        cache_total_b.cache_hit_prompt_tokens,
                    )
                ),
            },
            SingleConfigRow {
                key: "cache write tok".into(),
                value: format!(
                    "{} -> {} ({})",
                    cache_total_a.cache_write_prompt_tokens,
                    cache_total_b.cache_write_prompt_tokens,
                    fmt_cost_delta_u64(
                        cache_total_a.cache_write_prompt_tokens,
                        cache_total_b.cache_write_prompt_tokens,
                    )
                ),
            },
        ];
        rendered.push_str(&Table::new(&savings_rows).with(Style::rounded()).to_string());
        rendered.push_str("\n\n");
    } else {
        rendered.push_str(&format!(
            "verification unavailable: {}\n\n",
            savings_unavailable_reason(a, b)
        ));
    }

    rendered.push_str(&format!(
        "Total time: {} vs {}",
        fmt_time(a.total_time_s),
        fmt_time(b.total_time_s)
    ));

    if verbose {
        let questions_a = load_questions(Path::new(path_a));
        let questions_b = load_questions(Path::new(path_b));

        if questions_a.is_empty() && questions_b.is_empty() {
            rendered.push_str("\n\n(no question sidecars found)");
        } else {
            let map_b: BTreeMap<&str, &ViewQuestionResult> =
                questions_b.iter().map(|q| (q.question_id.as_str(), q)).collect();

            let mut sorted_a = questions_a.iter().collect::<Vec<_>>();
            sorted_a.sort_by(|x, y| x.category.cmp(&y.category).then(x.question_id.cmp(&y.question_id)));

            let q_rows: Vec<CompareQuestionRow> = sorted_a
                .iter()
                .map(|qa| {
                    let qb = map_b.get(qa.question_id.as_str());
                    CompareQuestionRow {
                        question_id: qa.question_id.clone(),
                        category: qa.category.clone(),
                        a: question_mark(qa.judge_correct),
                        b: qb
                            .map(|q| question_mark(q.judge_correct))
                            .unwrap_or_else(|| "-".into()),
                    }
                })
                .collect();

            rendered.push_str("\n\n");
            rendered.push_str(
                &Table::new(&q_rows)
                    .with(Style::rounded())
                    .with(Modify::new(Columns::new(2..)).with(Alignment::center()))
                    .to_string(),
            );
        }
    }

    rendered
}

fn view_compare(
    a: &ViewBenchmarkOutput,
    b: &ViewBenchmarkOutput,
    path_a: &str,
    path_b: &str,
    verbose: bool,
) {
    println!(
        "{}",
        render_compare_output(a, b, path_a, path_b, verbose)
    );
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

fn main() {
    let args = parse_args();

    if args.files.len() == 1 {
        let output = load_file(&args.files[0]);
        view_single(&output, &args.files[0], args.verbose);
    } else {
        let a = load_file(&args.files[0]);
        let b = load_file(&args.files[1]);
        view_compare(&a, &b, &args.files[0], &args.files[1], args.verbose);
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_args_single_file() {
        // parse_args uses env::args, so test the logic directly
        let args = CliArgs {
            verbose: false,
            files: vec!["result.json".into()],
        };
        assert_eq!(args.files.len(), 1);
        assert!(!args.verbose);
    }

    #[test]
    fn test_parse_args_comparison() {
        let args = CliArgs {
            verbose: false,
            files: vec!["a.json".into(), "b.json".into()],
        };
        assert_eq!(args.files.len(), 2);
    }

    #[test]
    fn test_parse_args_verbose() {
        let args = CliArgs {
            verbose: true,
            files: vec!["result.json".into()],
        };
        assert!(args.verbose);
    }

    #[test]
    fn test_format_category_rows() {
        let mut per_category = BTreeMap::new();
        per_category.insert(
            "multi-session".into(),
            ViewCategoryResult {
                accuracy: 0.88,
                count: 50,
            },
        );
        per_category.insert(
            "abstention".into(),
            ViewCategoryResult {
                accuracy: 0.70,
                count: 30,
            },
        );

        let output = ViewBenchmarkOutput {
            accuracy: 0.80,
            total_questions: 80,
            per_category,
            ..Default::default()
        };

        let rows = build_category_rows(&output);
        // alphabetical: abstention, multi-session, then overall
        assert_eq!(rows.len(), 3);
        assert_eq!(rows[0].category, "abstention");
        assert_eq!(rows[0].acc, "70.0%");
        assert_eq!(rows[0].n, 30);
        assert_eq!(rows[1].category, "multi-session");
        assert_eq!(rows[1].acc, "88.0%");
        assert_eq!(rows[1].n, 50);
        assert_eq!(rows[2].category, "overall");
        assert_eq!(rows[2].acc, "80.0%");
        assert_eq!(rows[2].n, 80);
    }

    #[test]
    fn test_format_category_rows_comparison() {
        let mut per_a = BTreeMap::new();
        per_a.insert(
            "multi-session".into(),
            ViewCategoryResult {
                accuracy: 0.85,
                count: 50,
            },
        );

        let mut per_b = BTreeMap::new();
        per_b.insert(
            "multi-session".into(),
            ViewCategoryResult {
                accuracy: 0.90,
                count: 50,
            },
        );

        let a = ViewBenchmarkOutput {
            accuracy: 0.85,
            total_questions: 50,
            per_category: per_a,
            ..Default::default()
        };
        let b = ViewBenchmarkOutput {
            accuracy: 0.90,
            total_questions: 50,
            per_category: per_b,
            ..Default::default()
        };

        // Test delta computation
        let delta: f64 = (0.90 - 0.85) * 100.0;
        assert!((delta - 5.0).abs() < 0.01);

        // Test fmt_pct_delta
        let rendered = fmt_pct_delta(0.85, 0.90);
        // Contains +5.0% (possibly with ANSI coloring)
        assert!(rendered.contains("+5.0%"));

        // Negative delta
        let neg = fmt_pct_delta(0.90, 0.85);
        assert!(neg.contains("-5.0%"));

        // Zero delta
        let zero = fmt_pct_delta(0.85, 0.85);
        assert!(zero.contains("+0.0%"));
        // No ANSI codes for zero
        assert!(!zero.contains("\x1b["));

        // Verify category rows exist in both
        assert!(a.per_category.contains_key("multi-session"));
        assert!(b.per_category.contains_key("multi-session"));
    }

    #[test]
    fn test_build_stage_rows() {
        let mut stage_metrics = BTreeMap::new();
        stage_metrics.insert(
            "reflect".into(),
            ViewStageUsage {
                input_tokens: 600000,
                output_tokens: 25000,
                requests: 500,
            },
        );
        stage_metrics.insert(
            "judge".into(),
            ViewStageUsage {
                input_tokens: 200000,
                output_tokens: 10000,
                requests: 500,
            },
        );
        stage_metrics.insert(
            "unused".into(),
            ViewStageUsage {
                input_tokens: 0,
                output_tokens: 0,
                requests: 0,
            },
        );

        let output = ViewBenchmarkOutput {
            stage_metrics,
            ..Default::default()
        };

        let rows = build_stage_rows(&output);
        // "unused" has 0 requests, should be filtered
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].stage, "judge");
        assert_eq!(rows[1].stage, "reflect");
        assert_eq!(rows[1].input_tok, 600000);
        assert_eq!(rows[1].output_tok, 25000);
    }

    #[test]
    fn test_fmt_pct() {
        assert_eq!(fmt_pct(0.857), "85.7%");
        assert_eq!(fmt_pct(1.0), "100.0%");
        assert_eq!(fmt_pct(0.0), "0.0%");
    }

    #[test]
    fn test_fmt_time() {
        assert_eq!(fmt_time(45.3), "45.3s");
        assert_eq!(fmt_time(125.0), "2m5s");
    }

    #[test]
    fn test_view_stage_usage_total_tokens() {
        let usage = ViewStageUsage {
            input_tokens: 1000,
            output_tokens: 200,
            requests: 5,
        };
        assert_eq!(usage.total_tokens(), 1200);
    }

    #[test]
    fn test_file_label_with_tag() {
        let output = ViewBenchmarkOutput {
            tag: Some("my-run".into()),
            ..Default::default()
        };
        assert_eq!(file_label(&output, "some/path.json"), "my-run");
    }

    #[test]
    fn test_file_label_from_path() {
        let output = ViewBenchmarkOutput::default();
        assert_eq!(file_label(&output, "results/baseline.json"), "baseline");
    }

    #[test]
    fn test_deserialize_benchmark_output() {
        let json = r#"{
            "benchmark": "longmemeval",
            "timestamp": "2026-03-15T00:00:00Z",
            "retain_model": "claude-3.5-sonnet",
            "reflect_model": "claude-3.5-sonnet",
            "embedding_model": "text-embedding-3-small",
            "reranker_model": "local",
            "judge_model": "gpt-4o",
            "consolidation_strategy": "end",
            "total_questions": 100,
            "accuracy": 0.85,
            "per_category": {
                "multi-session": {"accuracy": 0.90, "count": 50},
                "abstention": {"accuracy": 0.80, "count": 50}
            },
            "stage_metrics": {
                "retain_extract": {"input_tokens": 1000, "output_tokens": 200, "requests": 10},
                "reflect": {"input_tokens": 500, "output_tokens": 100, "requests": 5}
            },
            "total_time_s": 120.5
        }"#;

        let output: ViewBenchmarkOutput = serde_json::from_str(json).unwrap();
        assert_eq!(output.benchmark, "longmemeval");
        assert_eq!(output.accuracy, 0.85);
        assert_eq!(output.per_category.len(), 2);
        assert_eq!(output.stage_metrics.len(), 2);
        assert_eq!(output.total_time_s, 120.5);
    }

    #[test]
    fn test_deserialize_with_missing_fields() {
        // Minimal JSON -- everything should default gracefully
        let json = r#"{}"#;
        let output: ViewBenchmarkOutput = serde_json::from_str(json).unwrap();
        assert_eq!(output.accuracy, 0.0);
        assert_eq!(output.total_questions, 0);
        assert!(output.per_category.is_empty());
        assert!(output.stage_metrics.is_empty());
    }

    #[test]
    fn test_deserialize_question_result() {
        let json = r#"{"question_id":"q1","category":"multi-session","judge_correct":true,"hypothesis":"yes","ground_truth":"yes","bank_id":"b1","elapsed_s":1.5,"status":"ok"}"#;
        let q: ViewQuestionResult = serde_json::from_str(json).unwrap();
        assert_eq!(q.question_id, "q1");
        assert!(q.judge_correct);
        assert_eq!(q.elapsed_s, 1.5);
    }

    #[test]
    fn test_fmt_cost_delta_u64() {
        let positive = fmt_cost_delta_u64(100, 200);
        assert!(positive.contains("+100"));

        let negative = fmt_cost_delta_u64(200, 100);
        assert!(negative.contains("-100"));

        let zero = fmt_cost_delta_u64(100, 100);
        assert_eq!(zero, "+0");
    }

    fn cache_usage(
        prompt_tokens: u64,
        uncached_prompt_tokens: u64,
        cache_hit_prompt_tokens: u64,
        cache_write_prompt_tokens: u64,
        completion_tokens: u64,
        calls: u64,
        cache_supported_calls: u64,
        cache_unsupported_calls: u64,
    ) -> ViewCacheAwareStageUsage {
        ViewCacheAwareStageUsage {
            prompt_tokens,
            uncached_prompt_tokens,
            cache_hit_prompt_tokens,
            cache_write_prompt_tokens,
            completion_tokens,
            calls,
            cache_supported_calls,
            cache_unsupported_calls,
            ..Default::default()
        }
    }

    fn sample_cache_aware_output() -> ViewBenchmarkOutput {
        let mut cache_aware_stage_metrics = BTreeMap::new();
        cache_aware_stage_metrics.insert(
            "retain_extract".into(),
            cache_usage(120, 70, 40, 10, 12, 2, 2, 0),
        );
        cache_aware_stage_metrics.insert(
            "retain_resolve".into(),
            cache_usage(80, 50, 20, 10, 8, 1, 1, 0),
        );
        cache_aware_stage_metrics.insert(
            "reflect".into(),
            cache_usage(60, 40, 15, 5, 6, 1, 1, 0),
        );
        cache_aware_stage_metrics.insert(
            "consolidate".into(),
            cache_usage(50, 25, 20, 5, 7, 1, 1, 0),
        );
        cache_aware_stage_metrics.insert(
            "opinion_merge".into(),
            cache_usage(40, 30, 5, 5, 4, 1, 1, 0),
        );
        cache_aware_stage_metrics.insert(
            "judge".into(),
            cache_usage(30, 30, 0, 0, 9, 1, 0, 1),
        );

        ViewBenchmarkOutput {
            benchmark: "longmemeval".into(),
            timestamp: "2026-03-16T00:00:00Z".into(),
            retain_model: "retain-model".into(),
            reflect_model: "reflect-model".into(),
            embedding_model: "embed-model".into(),
            reranker_model: "reranker".into(),
            judge_model: "judge-model".into(),
            consolidation_strategy: "end".into(),
            total_questions: 4,
            accuracy: 0.75,
            cache_aware_stage_metrics,
            ..Default::default()
        }
    }

    #[test]
    fn cache_aware_single_summary_reports_prompt_components() {
        let rendered = render_single_output(
            &sample_cache_aware_output(),
            "bench/longmemeval/results/local/cache-aware.json",
            false,
            &[],
        );

        assert!(rendered.contains("effective prompt tok"));
        assert!(rendered.contains("cache hit tok"));
        assert!(rendered.contains("cache write tok"));
        assert!(rendered.contains("cache supported"));
        assert!(rendered.contains("cache unsupported"));
        assert!(rendered.contains("cache hit rate"));
    }

    #[test]
    fn cache_aware_single_rolls_up_operator_stages() {
        let rendered = render_single_output(
            &sample_cache_aware_output(),
            "bench/longmemeval/results/local/cache-aware.json",
            false,
            &[],
        );

        assert!(rendered.contains("retain"));
        assert!(rendered.contains("reflect"));
        assert!(rendered.contains("consolidate"));
        assert!(rendered.contains("opinion_merge"));
        assert!(rendered.contains("judge"));
    }

    #[test]
    fn cache_aware_single_legacy_artifact_reports_unavailable() {
        let rendered = render_single_output(
            &ViewBenchmarkOutput {
                benchmark: "longmemeval".into(),
                ..Default::default()
            },
            "bench/longmemeval/results/local/legacy.json",
            false,
            &[],
        );

        assert!(rendered.contains("cache-aware metrics unavailable (legacy artifact)"));
        assert_eq!(
            rendered
                .matches("cache-aware metrics unavailable (legacy artifact)")
                .count(),
            1
        );
    }

    fn compare_output_with_metadata() -> ViewBenchmarkOutput {
        let mut output = sample_cache_aware_output();
        output.manifest.dataset_fingerprint = "dataset-v1".into();
        output.manifest.prompt_hashes.judge = "judge-hash".into();
        for (key, value) in [
            ("retain_extract", "retain-extract-hash"),
            ("retain_resolve_system", "retain-resolve-system-hash"),
            ("retain_resolve_user", "retain-resolve-user-hash"),
            ("retain_graph_system", "retain-graph-system-hash"),
            ("retain_graph_user", "retain-graph-user-hash"),
            ("retain_opinion", "retain-opinion-hash"),
            ("reflect_agent", "reflect-agent-hash"),
            ("consolidate", "consolidate-hash"),
            ("opinion_merge", "opinion-merge-hash"),
        ] {
            output
                .manifest
                .prompt_hashes
                .other
                .insert(key.into(), serde_json::json!(value));
        }
        output
    }

    #[test]
    fn cache_aware_compare_reports_same_model_savings_signal() {
        let baseline = compare_output_with_metadata();
        let mut warm = compare_output_with_metadata();
        warm.tag = Some("warm".into());
        warm.cache_aware_stage_metrics.insert(
            "reflect".into(),
            cache_usage(60, 20, 35, 5, 6, 1, 1, 0),
        );

        let rendered = render_compare_output(&baseline, &warm, "baseline.json", "warm.json", false);

        assert!(rendered.contains("cache-aware comparison"));
        assert!(rendered.contains("cache savings verification"));
        assert!(rendered.contains("effective prompt tok"));
        assert!(rendered.contains("cache hit tok"));
        assert!(rendered.contains("retain"));
        assert!(rendered.contains("reflect"));
        assert!(rendered.contains("cache savings visible"));
    }

    #[test]
    fn cache_aware_compare_blocks_incomplete_prompt_hashes() {
        let baseline = compare_output_with_metadata();
        let mut incomplete = compare_output_with_metadata();
        incomplete
            .manifest
            .prompt_hashes
            .other
            .remove("retain_graph_user");

        let rendered =
            render_compare_output(&baseline, &incomplete, "baseline.json", "incomplete.json", false);

        assert!(rendered.contains("cache-aware comparison"));
        assert!(rendered.contains("cache savings verification"));
        assert!(rendered.contains("verification unavailable: prompt hashes missing"));
        assert!(!rendered.contains("cache savings visible"));
    }
}
