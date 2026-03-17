//! View a single benchmark result or compare two side-by-side.
//!
//! Usage:
//!     cargo run --bin view -- <file.json>
//!     cargo run --bin view -- <file1.json> <file2.json>

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::env;
use std::fmt::Write as _;
use std::fs;
use std::process;

use serde::Deserialize;
use tabled::settings::object::Columns;
use tabled::settings::style::Style;
use tabled::settings::{Alignment, Modify};
use tabled::{Table, Tabled};

#[derive(Debug, Default, Deserialize)]
struct BenchmarkManifest {
    #[serde(default)]
    protocol_version: String,
    #[serde(default)]
    profile: String,
    #[serde(default)]
    mode: String,
    #[serde(default)]
    dataset_fingerprint: String,
    #[serde(default)]
    selected_conversations: Vec<String>,
    #[serde(default)]
    session_limit: Option<usize>,
    #[serde(default)]
    question_limit: Option<usize>,
    #[serde(default)]
    ingestion_granularity: String,
    #[serde(default)]
    question_concurrency: usize,
    #[serde(default)]
    conversation_concurrency: usize,
    #[serde(default)]
    raw_json: bool,
    #[serde(default)]
    dirty_worktree: Option<bool>,
    #[serde(default)]
    prompt_hashes: BenchmarkPromptHashes,
    #[serde(default)]
    runtime_config: BenchmarkRuntimeConfig,
    #[serde(default)]
    source_artifact: Option<SourceArtifact>,
    #[serde(default)]
    source_artifacts: Vec<SourceArtifact>,
}

#[derive(Debug, Default, Deserialize)]
struct BenchmarkArtifacts {
    #[serde(default)]
    questions_path: String,
    #[serde(default)]
    debug_path: String,
}

#[derive(Debug, Default, Deserialize)]
struct ConversationSummary {
    #[serde(default)]
    bank_id: String,
    #[serde(default)]
    ingest_time_s: f64,
    #[serde(default)]
    consolidation_time_s: f64,
    #[serde(default)]
    qa_time_s: f64,
    #[serde(default)]
    total_time_s: f64,
    #[serde(default)]
    bank_stats: ConversationBankStats,
    #[serde(default)]
    cache_aware_stage_metrics: BTreeMap<String, CacheAwareStageUsage>,
}

#[derive(Debug, Default, Deserialize)]
struct BenchmarkPromptHashes {
    #[serde(default)]
    judge: String,
    #[serde(default)]
    retain_extract: String,
    #[serde(default)]
    reflect_agent: String,
    #[serde(default)]
    consolidate: String,
}

#[derive(Debug, Default, Deserialize)]
#[allow(dead_code)]
struct BenchmarkRuntimeConfig {
    #[serde(default)]
    dedup_threshold: Option<f32>,
    #[serde(default)]
    retriever_limit: usize,
    #[serde(default)]
    rerank_top_n: usize,
    #[serde(default)]
    reflect_max_iterations: usize,
    #[serde(default)]
    reflect_max_tokens: Option<usize>,
    #[serde(default)]
    reflect_budget_tokens: usize,
    #[serde(default)]
    judge_temperature: f32,
    #[serde(default)]
    judge_max_tokens: usize,
    #[serde(default)]
    judge_max_attempts: usize,
    #[serde(default)]
    consolidation_batch_size: usize,
    #[serde(default)]
    consolidation_max_tokens: usize,
    #[serde(default)]
    consolidation_recall_budget: usize,
    #[serde(default)]
    qa_updates_memory: bool,
}

#[derive(Debug, Default, Deserialize)]
#[allow(dead_code)]
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

#[derive(Debug, Default, Deserialize)]
#[allow(dead_code)]
struct ConversationBankStats {
    #[serde(default)]
    sessions_ingested: usize,
    #[serde(default)]
    turns_ingested: usize,
    #[serde(default)]
    facts_stored: usize,
    #[serde(default)]
    entities_resolved: usize,
    #[serde(default)]
    links_created: usize,
    #[serde(default)]
    opinions_reinforced: usize,
    #[serde(default)]
    opinions_weakened: usize,
    #[serde(default)]
    observations_created: usize,
    #[serde(default)]
    observations_updated: usize,
    #[serde(default)]
    final_fact_count: usize,
    #[serde(default)]
    final_observation_count: usize,
    #[serde(default)]
    final_opinion_count: usize,
    #[serde(default)]
    final_entity_count: usize,
}

#[derive(Debug, Clone, Default, Deserialize)]
struct StageUsage {
    #[serde(default)]
    prompt_tokens: u64,
    #[serde(default)]
    completion_tokens: u64,
    #[serde(default)]
    calls: u64,
    #[serde(default)]
    errors: u64,
    #[serde(default)]
    latency_ms: u64,
}

impl StageUsage {
    fn total_tokens(&self) -> u64 {
        self.prompt_tokens + self.completion_tokens
    }
}

#[derive(Debug, Clone, Default, Deserialize)]
struct CacheAwareStageUsage {
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

#[derive(Debug, Deserialize)]
struct BenchmarkOutput {
    #[serde(default)]
    tag: Option<String>,
    #[serde(default)]
    commit: Option<String>,
    #[serde(default)]
    judge_model: String,
    #[serde(default)]
    retain_model: String,
    #[serde(default)]
    reflect_model: String,
    #[serde(default)]
    embedding_model: String,
    #[serde(default)]
    reranker_model: String,
    #[serde(default)]
    consolidation_strategy: String,
    total_questions: usize,
    #[serde(default)]
    accuracy: f64,
    #[serde(default)]
    mean_f1: f64,
    #[serde(default)]
    mean_evidence_recall: f64,
    #[serde(default)]
    manifest: BenchmarkManifest,
    #[serde(default)]
    artifacts: BenchmarkArtifacts,
    #[serde(default)]
    per_conversation: HashMap<String, ConversationSummary>,
    #[serde(default)]
    stage_metrics: BTreeMap<String, StageUsage>,
    #[serde(default)]
    total_stage_usage: StageUsage,
    #[serde(default)]
    cache_aware_stage_metrics: BTreeMap<String, CacheAwareStageUsage>,
    #[serde(default)]
    cache_aware_total_stage_usage: CacheAwareStageUsage,
    #[serde(default)]
    results: Vec<QuestionResult>,
    total_time_s: f64,
}

fn default_status() -> String {
    "ok".into()
}

#[derive(Debug, Deserialize)]
struct QuestionResult {
    question_id: String,
    sample_id: String,
    question: String,
    #[serde(default)]
    category_name: String,
    judge_correct: bool,
    #[allow(dead_code)]
    #[serde(default)]
    ground_truth: String,
    #[allow(dead_code)]
    #[serde(default)]
    hypothesis: String,
    #[serde(default)]
    f1: f64,
    #[serde(default)]
    elapsed_s: f64,
    #[serde(default = "default_status")]
    status: String,
    #[allow(dead_code)]
    #[serde(default)]
    error: Option<String>,
    #[serde(default)]
    evidence_refs: Vec<String>,
    #[serde(default)]
    retrieved_turn_refs: Vec<String>,
    #[serde(default)]
    evidence_hit: bool,
    #[serde(default)]
    evidence_recall: f64,
}

type QKey = (String, String);

fn qkey(r: &QuestionResult) -> QKey {
    (r.sample_id.clone(), r.question.clone())
}

fn file_label(output: &BenchmarkOutput, path: &str) -> String {
    if let Some(ref tag) = output.tag {
        tag.clone()
    } else {
        std::path::Path::new(path)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("?")
            .to_string()
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

fn fmt_ms(ms: u64) -> String {
    fmt_time(ms as f64 / 1000.0)
}

fn fmt_pct(v: f64) -> String {
    format!("{:.1}%", v * 100.0)
}

fn short_hash(value: &str) -> String {
    if value.is_empty() {
        "-".into()
    } else {
        value.chars().take(8).collect()
    }
}

fn fmt_optional_usize(value: Option<usize>) -> String {
    value.map(|v| v.to_string()).unwrap_or_else(|| "-".into())
}

fn source_artifact_label(source: &Option<SourceArtifact>) -> String {
    match source {
        Some(source) if !source.path.is_empty() => {
            let tag = source.tag.clone().unwrap_or_else(|| "-".into());
            format!("{} ({tag}, {})", source.mode, source.path)
        }
        _ => "-".into(),
    }
}

fn source_artifacts_label(sources: &[SourceArtifact]) -> String {
    if sources.is_empty() {
        return "-".into();
    }
    let rendered = sources
        .iter()
        .map(|source| {
            let tag = source.tag.clone().unwrap_or_else(|| "-".into());
            format!("{tag}:{}", source.path)
        })
        .collect::<Vec<_>>();
    format!("{} [{}]", sources.len(), rendered.join(", "))
}

fn has_bank_stats(output: &BenchmarkOutput) -> bool {
    output.per_conversation.values().any(|summary| {
        let stats = &summary.bank_stats;
        stats.sessions_ingested > 0
            || stats.turns_ingested > 0
            || stats.facts_stored > 0
            || stats.final_fact_count > 0
            || stats.final_entity_count > 0
    })
}

fn fmt_stage_value(usage: &StageUsage) -> String {
    if usage.calls == 0 {
        "-".into()
    } else {
        format!("{} tok / {} calls", usage.total_tokens(), usage.calls)
    }
}

fn cache_usage_available(usage: &CacheAwareStageUsage) -> bool {
    usage.prompt_tokens > 0
        || usage.uncached_prompt_tokens > 0
        || usage.cache_hit_prompt_tokens > 0
        || usage.cache_write_prompt_tokens > 0
        || usage.completion_tokens > 0
        || usage.calls > 0
        || usage.errors > 0
        || usage.latency_ms > 0
        || usage.cache_supported_calls > 0
        || usage.cache_hit_calls > 0
        || usage.cache_write_calls > 0
        || usage.cache_unsupported_calls > 0
}

fn merge_cache_usage(total: &mut CacheAwareStageUsage, usage: &CacheAwareStageUsage) {
    total.prompt_tokens += usage.prompt_tokens;
    total.uncached_prompt_tokens += usage.uncached_prompt_tokens;
    total.cache_hit_prompt_tokens += usage.cache_hit_prompt_tokens;
    total.cache_write_prompt_tokens += usage.cache_write_prompt_tokens;
    total.completion_tokens += usage.completion_tokens;
    total.calls += usage.calls;
    total.errors += usage.errors;
    total.latency_ms += usage.latency_ms;
    total.cache_supported_calls += usage.cache_supported_calls;
    total.cache_hit_calls += usage.cache_hit_calls;
    total.cache_write_calls += usage.cache_write_calls;
    total.cache_unsupported_calls += usage.cache_unsupported_calls;
}

fn fmt_float_delta(a: f64, b: f64, precision: usize, higher_is_better: bool) -> String {
    let delta = b - a;
    let threshold = 0.5 * 10f64.powi(-(precision as i32));
    let rendered = format!("{:+.*}", precision, delta);
    if delta.abs() < threshold {
        rendered
    } else if (delta > 0.0) == higher_is_better {
        format!("\x1b[32m{rendered}\x1b[0m")
    } else {
        format!("\x1b[31m{rendered}\x1b[0m")
    }
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

fn avg(v: &[f64]) -> f64 {
    if v.is_empty() {
        0.0
    } else {
        v.iter().sum::<f64>() / v.len() as f64
    }
}

fn has_evidence(output: &BenchmarkOutput) -> bool {
    output.mean_evidence_recall > 0.0
        || output
            .results
            .iter()
            .any(|r| !r.evidence_refs.is_empty() || !r.retrieved_turn_refs.is_empty())
}

fn evidence_precision(result: &QuestionResult) -> Option<f64> {
    let expected = result
        .evidence_refs
        .iter()
        .map(String::as_str)
        .collect::<BTreeSet<_>>();
    let retrieved = result
        .retrieved_turn_refs
        .iter()
        .map(String::as_str)
        .collect::<BTreeSet<_>>();
    if expected.is_empty() && retrieved.is_empty() {
        return None;
    }
    if retrieved.is_empty() {
        return Some(0.0);
    }
    let hits = retrieved
        .iter()
        .filter(|turn_ref| expected.contains(**turn_ref))
        .count();
    Some(hits as f64 / retrieved.len() as f64)
}

fn avg_option<I>(values: I) -> Option<f64>
where
    I: IntoIterator<Item = Option<f64>>,
{
    let mut sum = 0.0;
    let mut count = 0usize;
    for value in values.into_iter().flatten() {
        sum += value;
        count += 1;
    }
    if count == 0 {
        None
    } else {
        Some(sum / count as f64)
    }
}

fn fmt_metric(value: Option<f64>, precision: usize) -> String {
    match value {
        Some(value) => format!("{value:.precision$}"),
        None => "-".into(),
    }
}

fn correct_count(results: &[QuestionResult]) -> usize {
    results.iter().filter(|r| r.judge_correct).count()
}

fn manifest_scope(manifest: &BenchmarkManifest) -> Option<String> {
    if !manifest.selected_conversations.is_empty() {
        return Some(manifest.selected_conversations.join(","));
    }
    None
}

fn failed_count(results: &[QuestionResult]) -> usize {
    results.iter().filter(|r| r.status != "ok").count()
}

fn summed_phase_times(output: &BenchmarkOutput) -> Option<(f64, f64, f64, f64)> {
    if output.per_conversation.is_empty() {
        return None;
    }
    Some(
        output
            .per_conversation
            .values()
            .fold((0.0, 0.0, 0.0, 0.0), |acc, summary| {
                (
                    acc.0 + summary.ingest_time_s,
                    acc.1 + summary.consolidation_time_s,
                    acc.2 + summary.qa_time_s,
                    acc.3 + summary.total_time_s,
                )
            }),
    )
}

fn question_mark(r: &QuestionResult) -> String {
    if r.status != "ok" {
        "!".into()
    } else if r.judge_correct {
        "\x1b[32m✓\x1b[0m".into()
    } else {
        "\x1b[31m✗\x1b[0m".into()
    }
}

#[derive(Tabled)]
struct ConfigRow {
    #[tabled(rename = "config")]
    key: String,
    #[tabled(rename = "A")]
    val_a: String,
    #[tabled(rename = "B")]
    val_b: String,
}

#[derive(Tabled)]
struct SummaryRow {
    category: String,
    #[tabled(rename = "A")]
    acc_a: String,
    #[tabled(rename = "B")]
    acc_b: String,
    delta: String,
    #[tabled(rename = "n")]
    n: usize,
}

#[derive(Tabled)]
struct MetricsRow {
    metric: String,
    #[tabled(rename = "A")]
    val_a: String,
    #[tabled(rename = "B")]
    val_b: String,
    #[tabled(rename = "Δ")]
    delta: String,
}

#[derive(Tabled)]
struct QuestionRow {
    #[tabled(rename = "id")]
    qid: String,
    sample: String,
    category: String,
    #[tabled(rename = "A")]
    a: String,
    #[tabled(rename = "B")]
    b: String,
}

#[derive(Tabled)]
struct SingleConfigRow {
    #[tabled(rename = "config")]
    key: String,
    value: String,
}

#[derive(Tabled)]
struct SingleSummaryRow {
    category: String,
    #[tabled(rename = "acc")]
    acc: String,
    #[tabled(rename = "n")]
    n: usize,
}

#[derive(Tabled)]
struct ConversationRow {
    #[tabled(rename = "conversation")]
    sample_id: String,
    #[tabled(rename = "acc")]
    acc: String,
    #[tabled(rename = "F1")]
    f1: String,
    #[tabled(rename = "ER")]
    evidence_recall: String,
    #[tabled(rename = "EP")]
    evidence_precision: String,
    #[tabled(rename = "avg time")]
    avg_time: String,
    failed: usize,
    #[tabled(rename = "n")]
    n: usize,
}

#[derive(Tabled)]
struct BankStatsRow {
    #[tabled(rename = "conversation")]
    sample_id: String,
    bank: String,
    sessions: usize,
    turns: usize,
    stored: usize,
    #[tabled(rename = "final facts")]
    final_facts: usize,
    #[tabled(rename = "obs +upd")]
    observations: String,
    entities: usize,
}

#[derive(Tabled)]
struct SingleQuestionRow {
    #[tabled(rename = "id")]
    qid: String,
    sample: String,
    category: String,
    result: String,
    status: String,
    #[tabled(rename = "ER")]
    evidence_recall: String,
}

#[derive(Tabled)]
struct StageRow {
    stage: String,
    tokens: String,
    calls: u64,
    errors: u64,
    latency: String,
}

#[derive(Tabled)]
struct StageCompareRow {
    stage: String,
    #[tabled(rename = "A")]
    val_a: String,
    #[tabled(rename = "B")]
    val_b: String,
    #[tabled(rename = "Δ tokens")]
    delta: String,
}

#[derive(Tabled)]
struct CacheStageRow {
    stage: String,
    #[tabled(rename = "effective prompt tok")]
    effective_prompt_tokens: String,
    #[tabled(rename = "cache hit tok")]
    cache_hit_prompt_tokens: String,
    #[tabled(rename = "cache write tok")]
    cache_write_prompt_tokens: String,
    calls: u64,
    errors: u64,
    latency: String,
}

#[derive(Tabled)]
struct CacheStageCompareRow {
    stage: String,
    #[tabled(rename = "eff(A)")]
    effective_prompt_a: String,
    #[tabled(rename = "eff(B)")]
    effective_prompt_b: String,
    #[tabled(rename = "Δ eff")]
    delta_effective_prompt: String,
    #[tabled(rename = "hit(A)")]
    cache_hit_a: String,
    #[tabled(rename = "hit(B)")]
    cache_hit_b: String,
    #[tabled(rename = "Δ hit")]
    delta_cache_hit: String,
}

fn operator_stage_name(stage: &str) -> Option<&'static str> {
    match stage {
        "retain_extract" | "retain_resolve" | "retain_graph" | "retain_opinion" => {
            Some("retain")
        }
        "reflect" => Some("reflect"),
        "consolidate" => Some("consolidate"),
        "opinion_merge" => Some("opinion_merge"),
        "judge" => Some("judge"),
        _ => None,
    }
}

fn rollup_operator_cache_metrics(
    stage_metrics: &BTreeMap<String, CacheAwareStageUsage>,
) -> BTreeMap<String, CacheAwareStageUsage> {
    let mut rolled_up = BTreeMap::new();
    for (stage, usage) in stage_metrics {
        if let Some(operator_stage) = operator_stage_name(stage) {
            merge_cache_usage(rolled_up.entry(operator_stage.to_string()).or_default(), usage);
        }
    }
    rolled_up
}

fn cache_summary_rows(usage: &CacheAwareStageUsage) -> Vec<SingleConfigRow> {
    let effective_prompt_tokens = usage.uncached_prompt_tokens + usage.cache_write_prompt_tokens;
    vec![
        SingleConfigRow {
            key: "effective prompt tok".into(),
            value: effective_prompt_tokens.to_string(),
        },
        SingleConfigRow {
            key: "cache hit tok".into(),
            value: usage.cache_hit_prompt_tokens.to_string(),
        },
        SingleConfigRow {
            key: "cache write tok".into(),
            value: usage.cache_write_prompt_tokens.to_string(),
        },
        SingleConfigRow {
            key: "cache supported".into(),
            value: usage.cache_supported_calls.to_string(),
        },
        SingleConfigRow {
            key: "cache unsupported".into(),
            value: usage.cache_unsupported_calls.to_string(),
        },
        SingleConfigRow {
            key: "cache hit rate".into(),
            value: if usage.prompt_tokens > 0 {
                fmt_pct(usage.cache_hit_prompt_tokens as f64 / usage.prompt_tokens as f64)
            } else {
                "-".into()
            },
        },
    ]
}

fn cache_compare_total(output: &BenchmarkOutput) -> CacheAwareStageUsage {
    if cache_usage_available(&output.cache_aware_total_stage_usage) {
        return output.cache_aware_total_stage_usage.clone();
    }

    let mut total = CacheAwareStageUsage::default();
    for usage in output.cache_aware_stage_metrics.values() {
        merge_cache_usage(&mut total, usage);
    }
    total
}

fn cache_hit_rate(usage: &CacheAwareStageUsage) -> f64 {
    if usage.prompt_tokens == 0 {
        0.0
    } else {
        usage.cache_hit_prompt_tokens as f64 / usage.prompt_tokens as f64
    }
}

fn effective_prompt_tokens(usage: &CacheAwareStageUsage) -> u64 {
    usage.uncached_prompt_tokens + usage.cache_write_prompt_tokens
}

fn cache_compare_rows(a: &CacheAwareStageUsage, b: &CacheAwareStageUsage) -> Vec<MetricsRow> {
    vec![
        MetricsRow {
            metric: "effective prompt tok".into(),
            val_a: effective_prompt_tokens(a).to_string(),
            val_b: effective_prompt_tokens(b).to_string(),
            delta: fmt_cost_delta_u64(effective_prompt_tokens(a), effective_prompt_tokens(b)),
        },
        MetricsRow {
            metric: "cache hit tok".into(),
            val_a: a.cache_hit_prompt_tokens.to_string(),
            val_b: b.cache_hit_prompt_tokens.to_string(),
            delta: fmt_cost_delta_u64(a.cache_hit_prompt_tokens, b.cache_hit_prompt_tokens),
        },
        MetricsRow {
            metric: "cache write tok".into(),
            val_a: a.cache_write_prompt_tokens.to_string(),
            val_b: b.cache_write_prompt_tokens.to_string(),
            delta: fmt_cost_delta_u64(a.cache_write_prompt_tokens, b.cache_write_prompt_tokens),
        },
        MetricsRow {
            metric: "cache supported".into(),
            val_a: a.cache_supported_calls.to_string(),
            val_b: b.cache_supported_calls.to_string(),
            delta: fmt_cost_delta_u64(a.cache_supported_calls, b.cache_supported_calls),
        },
        MetricsRow {
            metric: "cache unsupported".into(),
            val_a: a.cache_unsupported_calls.to_string(),
            val_b: b.cache_unsupported_calls.to_string(),
            delta: fmt_cost_delta_u64(a.cache_unsupported_calls, b.cache_unsupported_calls),
        },
        MetricsRow {
            metric: "cache hit rate".into(),
            val_a: fmt_pct(cache_hit_rate(a)),
            val_b: fmt_pct(cache_hit_rate(b)),
            delta: fmt_pct_delta(cache_hit_rate(a), cache_hit_rate(b)),
        },
    ]
}

fn prompt_hashes_complete(hashes: &BenchmarkPromptHashes) -> bool {
    !hashes.judge.is_empty()
        && !hashes.retain_extract.is_empty()
        && !hashes.reflect_agent.is_empty()
        && !hashes.consolidate.is_empty()
}

fn prompt_hashes_match(a: &BenchmarkPromptHashes, b: &BenchmarkPromptHashes) -> bool {
    a.judge == b.judge
        && a.retain_extract == b.retain_extract
        && a.reflect_agent == b.reflect_agent
        && a.consolidate == b.consolidate
}

fn savings_comparable(a: &BenchmarkOutput, b: &BenchmarkOutput) -> bool {
    !a.manifest.dataset_fingerprint.is_empty()
        && a.manifest.dataset_fingerprint == b.manifest.dataset_fingerprint
        && prompt_hashes_complete(&a.manifest.prompt_hashes)
        && prompt_hashes_complete(&b.manifest.prompt_hashes)
        && prompt_hashes_match(&a.manifest.prompt_hashes, &b.manifest.prompt_hashes)
        && !a.retain_model.is_empty()
        && a.retain_model == b.retain_model
        && !a.reflect_model.is_empty()
        && a.reflect_model == b.reflect_model
        && !a.judge_model.is_empty()
        && a.judge_model == b.judge_model
}

fn savings_signal(
    a: &CacheAwareStageUsage,
    b: &CacheAwareStageUsage,
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

fn savings_unavailable_reason(a: &BenchmarkOutput, b: &BenchmarkOutput) -> String {
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
    if !prompt_hashes_match(&a.manifest.prompt_hashes, &b.manifest.prompt_hashes) {
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

fn render_single(output: &BenchmarkOutput, path: &str) -> String {
    let label = file_label(output, path);
    let evidence_available = has_evidence(output);
    let mut rendered = String::new();

    let mut config_rows = vec![
        SingleConfigRow {
            key: "tag".into(),
            value: label,
        },
        SingleConfigRow {
            key: "judge".into(),
            value: output.judge_model.clone(),
        },
        SingleConfigRow {
            key: "retain".into(),
            value: output.retain_model.clone(),
        },
        SingleConfigRow {
            key: "reflect".into(),
            value: output.reflect_model.clone(),
        },
        SingleConfigRow {
            key: "embedding".into(),
            value: output.embedding_model.clone(),
        },
        SingleConfigRow {
            key: "reranker".into(),
            value: output.reranker_model.clone(),
        },
        SingleConfigRow {
            key: "consolidation".into(),
            value: output.consolidation_strategy.clone(),
        },
        SingleConfigRow {
            key: "questions".into(),
            value: output.total_questions.to_string(),
        },
    ];

    if let Some(commit) = &output.commit {
        config_rows.push(SingleConfigRow {
            key: "commit".into(),
            value: commit.clone(),
        });
    }
    if !output.manifest.profile.is_empty() {
        config_rows.push(SingleConfigRow {
            key: "profile".into(),
            value: output.manifest.profile.clone(),
        });
    }
    if !output.manifest.mode.is_empty() {
        config_rows.push(SingleConfigRow {
            key: "mode".into(),
            value: output.manifest.mode.clone(),
        });
    }
    if !output.manifest.protocol_version.is_empty() {
        config_rows.push(SingleConfigRow {
            key: "protocol".into(),
            value: output.manifest.protocol_version.clone(),
        });
    }
    if !output.manifest.dataset_fingerprint.is_empty() {
        config_rows.push(SingleConfigRow {
            key: "dataset".into(),
            value: output.manifest.dataset_fingerprint.clone(),
        });
    }
    if !output.manifest.ingestion_granularity.is_empty() {
        config_rows.push(SingleConfigRow {
            key: "ingest".into(),
            value: output.manifest.ingestion_granularity.clone(),
        });
    }
    if let Some(scope) = manifest_scope(&output.manifest) {
        config_rows.push(SingleConfigRow {
            key: "scope".into(),
            value: scope,
        });
    }
    if output.manifest.question_concurrency > 0 {
        config_rows.push(SingleConfigRow {
            key: "q concurrency".into(),
            value: output.manifest.question_concurrency.to_string(),
        });
    }
    if output.manifest.conversation_concurrency > 0 {
        config_rows.push(SingleConfigRow {
            key: "conv concurrency".into(),
            value: output.manifest.conversation_concurrency.to_string(),
        });
    }
    if output.manifest.raw_json {
        config_rows.push(SingleConfigRow {
            key: "raw json".into(),
            value: "true".into(),
        });
    }
    if let Some(limit) = output.manifest.session_limit {
        config_rows.push(SingleConfigRow {
            key: "session limit".into(),
            value: limit.to_string(),
        });
    }
    if let Some(limit) = output.manifest.question_limit {
        config_rows.push(SingleConfigRow {
            key: "question limit".into(),
            value: limit.to_string(),
        });
    }
    if let Some(dirty) = output.manifest.dirty_worktree {
        config_rows.push(SingleConfigRow {
            key: "dirty tree".into(),
            value: dirty.to_string(),
        });
    }
    if output.manifest.runtime_config.retriever_limit > 0 {
        config_rows.push(SingleConfigRow {
            key: "retriever limit".into(),
            value: output.manifest.runtime_config.retriever_limit.to_string(),
        });
    }
    if output.manifest.runtime_config.rerank_top_n > 0 {
        config_rows.push(SingleConfigRow {
            key: "rerank top-n".into(),
            value: output.manifest.runtime_config.rerank_top_n.to_string(),
        });
    }
    if output.manifest.runtime_config.reflect_max_iterations > 0 {
        config_rows.push(SingleConfigRow {
            key: "reflect iter".into(),
            value: output
                .manifest
                .runtime_config
                .reflect_max_iterations
                .to_string(),
        });
    }
    if output.manifest.runtime_config.reflect_budget_tokens > 0 {
        config_rows.push(SingleConfigRow {
            key: "reflect budget".into(),
            value: output
                .manifest
                .runtime_config
                .reflect_budget_tokens
                .to_string(),
        });
    }
    if output.manifest.runtime_config.reflect_max_tokens.is_some() {
        config_rows.push(SingleConfigRow {
            key: "reflect cap".into(),
            value: fmt_optional_usize(output.manifest.runtime_config.reflect_max_tokens),
        });
    }
    if output.manifest.runtime_config.judge_max_tokens > 0 {
        config_rows.push(SingleConfigRow {
            key: "judge max tok".into(),
            value: output.manifest.runtime_config.judge_max_tokens.to_string(),
        });
    }
    if output.manifest.runtime_config.judge_max_attempts > 0 {
        config_rows.push(SingleConfigRow {
            key: "judge retries".into(),
            value: output
                .manifest
                .runtime_config
                .judge_max_attempts
                .to_string(),
        });
    }
    if let Some(threshold) = output.manifest.runtime_config.dedup_threshold {
        config_rows.push(SingleConfigRow {
            key: "dedup".into(),
            value: format!("{threshold:.2}"),
        });
    }
    if output.manifest.runtime_config.consolidation_batch_size > 0 {
        config_rows.push(SingleConfigRow {
            key: "cons batch".into(),
            value: output
                .manifest
                .runtime_config
                .consolidation_batch_size
                .to_string(),
        });
    }
    if output.manifest.runtime_config.consolidation_recall_budget > 0 {
        config_rows.push(SingleConfigRow {
            key: "cons recall".into(),
            value: output
                .manifest
                .runtime_config
                .consolidation_recall_budget
                .to_string(),
        });
    }
    if !output.manifest.prompt_hashes.judge.is_empty() {
        config_rows.push(SingleConfigRow {
            key: "judge hash".into(),
            value: short_hash(&output.manifest.prompt_hashes.judge),
        });
    }
    if !output.manifest.prompt_hashes.retain_extract.is_empty() {
        config_rows.push(SingleConfigRow {
            key: "extract hash".into(),
            value: short_hash(&output.manifest.prompt_hashes.retain_extract),
        });
    }
    if !output.manifest.prompt_hashes.reflect_agent.is_empty() {
        config_rows.push(SingleConfigRow {
            key: "reflect hash".into(),
            value: short_hash(&output.manifest.prompt_hashes.reflect_agent),
        });
    }
    if !output.manifest.prompt_hashes.consolidate.is_empty() {
        config_rows.push(SingleConfigRow {
            key: "cons hash".into(),
            value: short_hash(&output.manifest.prompt_hashes.consolidate),
        });
    }
    if output.manifest.source_artifact.is_some() {
        config_rows.push(SingleConfigRow {
            key: "source artifact".into(),
            value: source_artifact_label(&output.manifest.source_artifact),
        });
    }
    if !output.manifest.source_artifacts.is_empty() {
        config_rows.push(SingleConfigRow {
            key: "source runs".into(),
            value: source_artifacts_label(&output.manifest.source_artifacts),
        });
    }
    if output.total_time_s > 0.0 {
        config_rows.push(SingleConfigRow {
            key: "total time".into(),
            value: fmt_time(output.total_time_s),
        });
    }
    if let Some((ingest_time_s, consolidation_time_s, qa_time_s, conversation_total_s)) =
        summed_phase_times(output)
    {
        config_rows.push(SingleConfigRow {
            key: "ingest time".into(),
            value: fmt_time(ingest_time_s),
        });
        config_rows.push(SingleConfigRow {
            key: "consolidate time".into(),
            value: fmt_time(consolidation_time_s),
        });
        config_rows.push(SingleConfigRow {
            key: "qa time".into(),
            value: fmt_time(qa_time_s),
        });
        config_rows.push(SingleConfigRow {
            key: "sum conv time".into(),
            value: fmt_time(conversation_total_s),
        });
    }
    if !output.artifacts.questions_path.is_empty() {
        config_rows.push(SingleConfigRow {
            key: "questions file".into(),
            value: output.artifacts.questions_path.clone(),
        });
    }
    if !output.artifacts.debug_path.is_empty() {
        config_rows.push(SingleConfigRow {
            key: "debug file".into(),
            value: output.artifacts.debug_path.clone(),
        });
    }

    writeln!(
        rendered,
        "{}",
        Table::new(&config_rows).with(Style::rounded())
    )
    .expect("write config table");
    writeln!(rendered).expect("write config spacer");

    let mut cat_stats: BTreeMap<&str, (usize, usize)> = BTreeMap::new();
    for result in &output.results {
        if result.category_name.is_empty() {
            continue;
        }
        let entry = cat_stats.entry(&result.category_name).or_insert((0, 0));
        entry.1 += 1;
        if result.judge_correct {
            entry.0 += 1;
        }
    }

    let mut summary_rows = Vec::new();
    for (category, (correct, total)) in &cat_stats {
        summary_rows.push(SingleSummaryRow {
            category: (*category).to_string(),
            acc: fmt_pct(*correct as f64 / (*total).max(1) as f64),
            n: *total,
        });
    }
    summary_rows.push(SingleSummaryRow {
        category: "TOTAL".into(),
        acc: fmt_pct(output.accuracy),
        n: output.results.len(),
    });

    writeln!(
        rendered,
        "{}",
        Table::new(&summary_rows)
            .with(Style::rounded())
            .with(Modify::new(Columns::new(1..)).with(Alignment::right()))
    )
    .expect("write summary table");

    let mut metrics_rows = vec![
        SingleConfigRow {
            key: "accuracy".into(),
            value: fmt_pct(output.accuracy),
        },
        SingleConfigRow {
            key: "F1".into(),
            value: format!("{:.3}", output.mean_f1),
        },
    ];
    if evidence_available {
        metrics_rows.push(SingleConfigRow {
            key: "evidence recall".into(),
            value: format!("{:.3}", output.mean_evidence_recall),
        });
        metrics_rows.push(SingleConfigRow {
            key: "evidence precision".into(),
            value: fmt_metric(avg_option(output.results.iter().map(evidence_precision)), 3),
        });
    }
    metrics_rows.push(SingleConfigRow {
        key: "avg time".into(),
        value: fmt_time(avg(&output
            .results
            .iter()
            .map(|r| r.elapsed_s)
            .collect::<Vec<_>>())),
    });
    metrics_rows.push(SingleConfigRow {
        key: "failed".into(),
        value: failed_count(&output.results).to_string(),
    });
    if output.total_stage_usage.calls > 0 {
        metrics_rows.push(SingleConfigRow {
            key: "tokens".into(),
            value: output.total_stage_usage.total_tokens().to_string(),
        });
        metrics_rows.push(SingleConfigRow {
            key: "calls".into(),
            value: output.total_stage_usage.calls.to_string(),
        });
        metrics_rows.push(SingleConfigRow {
            key: "latency".into(),
            value: fmt_ms(output.total_stage_usage.latency_ms),
        });
        let correct = correct_count(&output.results);
        metrics_rows.push(SingleConfigRow {
            key: "tokens/correct".into(),
            value: if correct > 0 {
                format!(
                    "{:.1}",
                    output.total_stage_usage.total_tokens() as f64 / correct as f64
                )
            } else {
                "-".into()
            },
        });
    }

    writeln!(rendered).expect("write metrics spacer");
    writeln!(
        rendered,
        "{}",
        Table::new(&metrics_rows).with(Style::rounded())
    )
    .expect("write metrics table");

    if !output.stage_metrics.is_empty() {
        let stage_rows = output
            .stage_metrics
            .iter()
            .map(|(stage, usage)| StageRow {
                stage: stage.clone(),
                tokens: usage.total_tokens().to_string(),
                calls: usage.calls,
                errors: usage.errors,
                latency: fmt_ms(usage.latency_ms),
            })
            .collect::<Vec<_>>();

        writeln!(rendered).expect("write legacy stage spacer");
        writeln!(
            rendered,
            "{}",
            Table::new(&stage_rows)
                .with(Style::rounded())
                .with(Modify::new(Columns::new(1..)).with(Alignment::right()))
        )
        .expect("write legacy stage table");
    }

    let cache_available = cache_usage_available(&output.cache_aware_total_stage_usage)
        || output
            .cache_aware_stage_metrics
            .values()
            .any(cache_usage_available);
    if cache_available {
        let cache_summary = cache_summary_rows(&output.cache_aware_total_stage_usage);
        writeln!(rendered).expect("write cache summary spacer");
        writeln!(
            rendered,
            "{}",
            Table::new(&cache_summary).with(Style::rounded())
        )
        .expect("write cache summary table");

        let rolled_up = rollup_operator_cache_metrics(&output.cache_aware_stage_metrics);
        let stage_order = ["retain", "reflect", "consolidate", "opinion_merge", "judge"];
        let cache_stage_rows = stage_order
            .into_iter()
            .filter_map(|stage| {
                rolled_up.get(stage).and_then(|usage| {
                    if cache_usage_available(usage) {
                        Some(CacheStageRow {
                            stage: stage.to_string(),
                            effective_prompt_tokens: (
                                usage.uncached_prompt_tokens + usage.cache_write_prompt_tokens
                            )
                            .to_string(),
                            cache_hit_prompt_tokens: usage.cache_hit_prompt_tokens.to_string(),
                            cache_write_prompt_tokens: usage.cache_write_prompt_tokens.to_string(),
                            calls: usage.calls,
                            errors: usage.errors,
                            latency: fmt_ms(usage.latency_ms),
                        })
                    } else {
                        None
                    }
                })
            })
            .collect::<Vec<_>>();
        if !cache_stage_rows.is_empty() {
            writeln!(rendered).expect("write cache stage spacer");
            writeln!(
                rendered,
                "{}",
                Table::new(&cache_stage_rows)
                    .with(Style::rounded())
                    .with(Modify::new(Columns::new(1..)).with(Alignment::right()))
            )
            .expect("write cache stage table");
        }
    } else {
        writeln!(rendered).expect("write cache unavailable spacer");
        writeln!(
            rendered,
            "cache-aware metrics unavailable (legacy artifact)"
        )
        .expect("write cache unavailable line");
    }

    let conv_ids: BTreeSet<&str> = output
        .results
        .iter()
        .map(|r| r.sample_id.as_str())
        .collect();
    let mut conv_rows = Vec::new();
    for conv_id in conv_ids {
        let questions = output
            .results
            .iter()
            .filter(|r| r.sample_id == conv_id)
            .collect::<Vec<_>>();
        let correct = questions.iter().filter(|r| r.judge_correct).count();
        conv_rows.push(ConversationRow {
            sample_id: conv_id.to_string(),
            acc: fmt_pct(correct as f64 / questions.len().max(1) as f64),
            f1: format!(
                "{:.3}",
                avg(&questions.iter().map(|r| r.f1).collect::<Vec<_>>())
            ),
            evidence_recall: if evidence_available {
                format!(
                    "{:.3}",
                    avg(&questions
                        .iter()
                        .map(|r| r.evidence_recall)
                        .collect::<Vec<_>>())
                )
            } else {
                "-".into()
            },
            evidence_precision: if evidence_available {
                fmt_metric(
                    avg_option(questions.iter().map(|r| evidence_precision(r))),
                    3,
                )
            } else {
                "-".into()
            },
            avg_time: fmt_time(avg(&questions
                .iter()
                .map(|r| r.elapsed_s)
                .collect::<Vec<_>>())),
            failed: questions.iter().filter(|r| r.status != "ok").count(),
            n: questions.len(),
        });
    }
    conv_rows.push(ConversationRow {
        sample_id: "TOTAL".into(),
        acc: fmt_pct(output.accuracy),
        f1: format!("{:.3}", output.mean_f1),
        evidence_recall: if evidence_available {
            format!("{:.3}", output.mean_evidence_recall)
        } else {
            "-".into()
        },
        evidence_precision: if evidence_available {
            fmt_metric(avg_option(output.results.iter().map(evidence_precision)), 3)
        } else {
            "-".into()
        },
        avg_time: fmt_time(avg(&output
            .results
            .iter()
            .map(|r| r.elapsed_s)
            .collect::<Vec<_>>())),
        failed: failed_count(&output.results),
        n: output.results.len(),
    });

    writeln!(rendered).expect("write conversation spacer");
    writeln!(
        rendered,
        "{}",
        Table::new(&conv_rows)
            .with(Style::rounded())
            .with(Modify::new(Columns::new(1..)).with(Alignment::right()))
    )
    .expect("write conversation table");

    if has_bank_stats(output) {
        let mut bank_rows = output
            .per_conversation
            .iter()
            .map(|(sample_id, summary)| BankStatsRow {
                sample_id: sample_id.clone(),
                bank: summary.bank_id.clone(),
                sessions: summary.bank_stats.sessions_ingested,
                turns: summary.bank_stats.turns_ingested,
                stored: summary.bank_stats.facts_stored,
                final_facts: summary.bank_stats.final_fact_count,
                observations: format!(
                    "{}+{}",
                    summary.bank_stats.observations_created,
                    summary.bank_stats.observations_updated
                ),
                entities: summary.bank_stats.final_entity_count,
            })
            .collect::<Vec<_>>();
        bank_rows.sort_by(|a, b| a.sample_id.cmp(&b.sample_id));

        writeln!(rendered).expect("write bank spacer");
        writeln!(
            rendered,
            "{}",
            Table::new(&bank_rows)
                .with(Style::rounded())
                .with(Modify::new(Columns::new(1..)).with(Alignment::right()))
        )
        .expect("write bank table");
    }

    if output.results.is_empty() {
        return rendered;
    }

    let mut sorted = output.results.iter().collect::<Vec<_>>();
    sorted.sort_by(|a, b| {
        a.category_name
            .cmp(&b.category_name)
            .then(a.sample_id.cmp(&b.sample_id))
            .then(a.question_id.cmp(&b.question_id))
    });

    let question_rows = sorted
        .iter()
        .map(|result| SingleQuestionRow {
            qid: result.question_id.clone(),
            sample: result.sample_id.clone(),
            category: result.category_name.clone(),
            result: question_mark(result),
            status: result.status.clone(),
            evidence_recall: if evidence_available {
                format!("{:.2}", result.evidence_recall)
            } else {
                "-".into()
            },
        })
        .collect::<Vec<_>>();

    writeln!(rendered).expect("write question spacer");
    writeln!(
        rendered,
        "{}",
        Table::new(&question_rows)
            .with(Style::rounded())
            .with(Modify::new(Columns::new(3..=5)).with(Alignment::center()))
    )
    .expect("write question table");

    let correct = output.results.iter().filter(|r| r.judge_correct).count();
    let wrong = output.results.len().saturating_sub(correct);
    let failed = failed_count(&output.results);
    let evidence_hits = output.results.iter().filter(|r| r.evidence_hit).count();
    writeln!(rendered).expect("write footer spacer");
    if evidence_available {
        writeln!(
            rendered,
            "\x1b[32m{correct} correct\x1b[0m, \x1b[31m{wrong} wrong\x1b[0m, {failed} failed, {evidence_hits} evidence hits"
        )
        .expect("write footer");
    } else {
        writeln!(
            rendered,
            "\x1b[32m{correct} correct\x1b[0m, \x1b[31m{wrong} wrong\x1b[0m, {failed} failed"
        )
        .expect("write footer");
    }
    rendered
}

fn view_single(output: &BenchmarkOutput, path: &str) {
    print!("{}", render_single(output, path));
}

fn parse_output(raw: &str) -> Result<BenchmarkOutput, serde_json::Error> {
    serde_json::from_str(raw)
}

fn render_compare(a: &BenchmarkOutput, b: &BenchmarkOutput, path_a: &str, path_b: &str) -> String {
    let evidence_available = has_evidence(a) || has_evidence(b);
    let label_a = file_label(a, path_a);
    let label_b = file_label(b, path_b);

    let map_b = b
        .results
        .iter()
        .map(|result| (qkey(result), result))
        .collect::<HashMap<_, _>>();
    let matched = a
        .results
        .iter()
        .filter_map(|left| map_b.get(&qkey(left)).map(|right| (left, *right)))
        .collect::<Vec<_>>();

    let categories = {
        let mut cats = BTreeSet::new();
        for result in &a.results {
            cats.insert(result.category_name.clone());
        }
        for result in &b.results {
            cats.insert(result.category_name.clone());
        }
        cats.into_iter()
            .filter(|c| !c.is_empty())
            .collect::<Vec<_>>()
    };

    let mut cat_stats: BTreeMap<&str, (usize, usize, usize)> = BTreeMap::new();
    for (left, right) in &matched {
        let entry = cat_stats.entry(&left.category_name).or_insert((0, 0, 0));
        entry.2 += 1;
        if left.judge_correct {
            entry.0 += 1;
        }
        if right.judge_correct {
            entry.1 += 1;
        }
    }

    let mut rendered = String::new();
    let mut config_rows = vec![
        ConfigRow {
            key: "tag".into(),
            val_a: label_a.clone(),
            val_b: label_b.clone(),
        },
        ConfigRow {
            key: "judge".into(),
            val_a: a.judge_model.clone(),
            val_b: b.judge_model.clone(),
        },
        ConfigRow {
            key: "retain".into(),
            val_a: a.retain_model.clone(),
            val_b: b.retain_model.clone(),
        },
        ConfigRow {
            key: "reflect".into(),
            val_a: a.reflect_model.clone(),
            val_b: b.reflect_model.clone(),
        },
        ConfigRow {
            key: "embedding".into(),
            val_a: a.embedding_model.clone(),
            val_b: b.embedding_model.clone(),
        },
        ConfigRow {
            key: "reranker".into(),
            val_a: a.reranker_model.clone(),
            val_b: b.reranker_model.clone(),
        },
        ConfigRow {
            key: "consolidation".into(),
            val_a: a.consolidation_strategy.clone(),
            val_b: b.consolidation_strategy.clone(),
        },
        ConfigRow {
            key: "questions".into(),
            val_a: a.total_questions.to_string(),
            val_b: b.total_questions.to_string(),
        },
    ];

    if a.commit.is_some() || b.commit.is_some() {
        config_rows.push(ConfigRow {
            key: "commit".into(),
            val_a: a.commit.clone().unwrap_or_else(|| "-".into()),
            val_b: b.commit.clone().unwrap_or_else(|| "-".into()),
        });
    }
    if !a.manifest.profile.is_empty() || !b.manifest.profile.is_empty() {
        config_rows.push(ConfigRow {
            key: "profile".into(),
            val_a: a.manifest.profile.clone(),
            val_b: b.manifest.profile.clone(),
        });
    }
    if !a.manifest.mode.is_empty() || !b.manifest.mode.is_empty() {
        config_rows.push(ConfigRow {
            key: "mode".into(),
            val_a: a.manifest.mode.clone(),
            val_b: b.manifest.mode.clone(),
        });
    }
    if a.manifest.runtime_config.retriever_limit > 0
        || b.manifest.runtime_config.retriever_limit > 0
    {
        config_rows.push(ConfigRow {
            key: "retriever limit".into(),
            val_a: a.manifest.runtime_config.retriever_limit.to_string(),
            val_b: b.manifest.runtime_config.retriever_limit.to_string(),
        });
    }
    if a.manifest.runtime_config.rerank_top_n > 0 || b.manifest.runtime_config.rerank_top_n > 0 {
        config_rows.push(ConfigRow {
            key: "rerank top-n".into(),
            val_a: a.manifest.runtime_config.rerank_top_n.to_string(),
            val_b: b.manifest.runtime_config.rerank_top_n.to_string(),
        });
    }
    if a.manifest.runtime_config.reflect_max_iterations > 0
        || b.manifest.runtime_config.reflect_max_iterations > 0
    {
        config_rows.push(ConfigRow {
            key: "reflect iter".into(),
            val_a: a.manifest.runtime_config.reflect_max_iterations.to_string(),
            val_b: b.manifest.runtime_config.reflect_max_iterations.to_string(),
        });
    }
    if a.manifest.runtime_config.reflect_budget_tokens > 0
        || b.manifest.runtime_config.reflect_budget_tokens > 0
    {
        config_rows.push(ConfigRow {
            key: "reflect budget".into(),
            val_a: a.manifest.runtime_config.reflect_budget_tokens.to_string(),
            val_b: b.manifest.runtime_config.reflect_budget_tokens.to_string(),
        });
    }
    if a.manifest.runtime_config.reflect_max_tokens.is_some()
        || b.manifest.runtime_config.reflect_max_tokens.is_some()
    {
        config_rows.push(ConfigRow {
            key: "reflect cap".into(),
            val_a: fmt_optional_usize(a.manifest.runtime_config.reflect_max_tokens),
            val_b: fmt_optional_usize(b.manifest.runtime_config.reflect_max_tokens),
        });
    }
    if !a.manifest.prompt_hashes.judge.is_empty() || !b.manifest.prompt_hashes.judge.is_empty() {
        config_rows.push(ConfigRow {
            key: "judge hash".into(),
            val_a: short_hash(&a.manifest.prompt_hashes.judge),
            val_b: short_hash(&b.manifest.prompt_hashes.judge),
        });
    }
    if !a.manifest.prompt_hashes.reflect_agent.is_empty()
        || !b.manifest.prompt_hashes.reflect_agent.is_empty()
    {
        config_rows.push(ConfigRow {
            key: "reflect hash".into(),
            val_a: short_hash(&a.manifest.prompt_hashes.reflect_agent),
            val_b: short_hash(&b.manifest.prompt_hashes.reflect_agent),
        });
    }
    if a.manifest.source_artifact.is_some() || b.manifest.source_artifact.is_some() {
        config_rows.push(ConfigRow {
            key: "source artifact".into(),
            val_a: source_artifact_label(&a.manifest.source_artifact),
            val_b: source_artifact_label(&b.manifest.source_artifact),
        });
    }
    if !a.manifest.source_artifacts.is_empty() || !b.manifest.source_artifacts.is_empty() {
        config_rows.push(ConfigRow {
            key: "source runs".into(),
            val_a: source_artifacts_label(&a.manifest.source_artifacts),
            val_b: source_artifacts_label(&b.manifest.source_artifacts),
        });
    }
    if matched.len() != a.results.len() || matched.len() != b.results.len() {
        config_rows.push(ConfigRow {
            key: "matched".into(),
            val_a: matched.len().to_string(),
            val_b: matched.len().to_string(),
        });
    }
    if !a.manifest.ingestion_granularity.is_empty() || !b.manifest.ingestion_granularity.is_empty()
    {
        config_rows.push(ConfigRow {
            key: "ingest".into(),
            val_a: a.manifest.ingestion_granularity.clone(),
            val_b: b.manifest.ingestion_granularity.clone(),
        });
    }
    if manifest_scope(&a.manifest).is_some() || manifest_scope(&b.manifest).is_some() {
        config_rows.push(ConfigRow {
            key: "scope".into(),
            val_a: manifest_scope(&a.manifest).unwrap_or_else(|| "-".into()),
            val_b: manifest_scope(&b.manifest).unwrap_or_else(|| "-".into()),
        });
    }
    if a.total_time_s > 0.0 || b.total_time_s > 0.0 {
        config_rows.push(ConfigRow {
            key: "total time".into(),
            val_a: fmt_time(a.total_time_s),
            val_b: fmt_time(b.total_time_s),
        });
    }

    writeln!(
        rendered,
        "{}",
        Table::new(&config_rows).with(Style::rounded())
    )
    .expect("write config table");

    let mut summary_rows = categories
        .iter()
        .map(|category| {
            let (correct_a, correct_b, total) = cat_stats.get(category.as_str()).copied().unwrap_or_default();
            SummaryRow {
                category: category.clone(),
                acc_a: if total > 0 {
                    fmt_pct(correct_a as f64 / total as f64)
                } else {
                    "-".into()
                },
                acc_b: if total > 0 {
                    fmt_pct(correct_b as f64 / total as f64)
                } else {
                    "-".into()
                },
                delta: if total > 0 {
                    fmt_pct_delta(correct_a as f64 / total as f64, correct_b as f64 / total as f64)
                } else {
                    "-".into()
                },
                n: total,
            }
        })
        .collect::<Vec<_>>();

    summary_rows.push(SummaryRow {
        category: "TOTAL".into(),
        acc_a: fmt_pct(a.accuracy),
        acc_b: fmt_pct(b.accuracy),
        delta: fmt_pct_delta(a.accuracy, b.accuracy),
        n: matched.len(),
    });

    writeln!(rendered).expect("write summary spacer");
    writeln!(
        rendered,
        "{}",
        Table::new(&summary_rows)
            .with(Style::rounded())
            .with(Modify::new(Columns::new(1..)).with(Alignment::right()))
    )
    .expect("write summary table");

    let failed_a = failed_count(&a.results);
    let failed_b = failed_count(&b.results);
    let correct_a = correct_count(&a.results);
    let correct_b = correct_count(&b.results);
    let mut metrics_rows = vec![
        MetricsRow {
            metric: "accuracy".into(),
            val_a: fmt_pct(a.accuracy),
            val_b: fmt_pct(b.accuracy),
            delta: fmt_pct_delta(a.accuracy, b.accuracy),
        },
        MetricsRow {
            metric: "F1".into(),
            val_a: format!("{:.3}", a.mean_f1),
            val_b: format!("{:.3}", b.mean_f1),
            delta: fmt_float_delta(a.mean_f1, b.mean_f1, 3, true),
        },
        MetricsRow {
            metric: "avg time".into(),
            val_a: fmt_time(avg(&a.results.iter().map(|r| r.elapsed_s).collect::<Vec<_>>())),
            val_b: fmt_time(avg(&b.results.iter().map(|r| r.elapsed_s).collect::<Vec<_>>())),
            delta: fmt_float_delta(
                avg(&a.results.iter().map(|r| r.elapsed_s).collect::<Vec<_>>()),
                avg(&b.results.iter().map(|r| r.elapsed_s).collect::<Vec<_>>()),
                1,
                false,
            ),
        },
        MetricsRow {
            metric: "failed".into(),
            val_a: failed_a.to_string(),
            val_b: failed_b.to_string(),
            delta: fmt_cost_delta_u64(failed_a as u64, failed_b as u64),
        },
    ];

    if evidence_available {
        metrics_rows.insert(
            2,
            MetricsRow {
                metric: "evidence recall".into(),
                val_a: format!("{:.3}", a.mean_evidence_recall),
                val_b: format!("{:.3}", b.mean_evidence_recall),
                delta: fmt_float_delta(
                    a.mean_evidence_recall,
                    b.mean_evidence_recall,
                    3,
                    true,
                ),
            },
        );
        metrics_rows.insert(
            3,
            MetricsRow {
                metric: "evidence precision".into(),
                val_a: fmt_metric(avg_option(a.results.iter().map(evidence_precision)), 3),
                val_b: fmt_metric(avg_option(b.results.iter().map(evidence_precision)), 3),
                delta: match (
                    avg_option(a.results.iter().map(evidence_precision)),
                    avg_option(b.results.iter().map(evidence_precision)),
                ) {
                    (Some(left), Some(right)) => fmt_float_delta(left, right, 3, true),
                    _ => "-".into(),
                },
            },
        );
    }

    if a.total_stage_usage.calls > 0 || b.total_stage_usage.calls > 0 {
        metrics_rows.push(MetricsRow {
            metric: "tokens".into(),
            val_a: a.total_stage_usage.total_tokens().to_string(),
            val_b: b.total_stage_usage.total_tokens().to_string(),
            delta: fmt_cost_delta_u64(
                a.total_stage_usage.total_tokens(),
                b.total_stage_usage.total_tokens(),
            ),
        });
        metrics_rows.push(MetricsRow {
            metric: "calls".into(),
            val_a: a.total_stage_usage.calls.to_string(),
            val_b: b.total_stage_usage.calls.to_string(),
            delta: fmt_cost_delta_u64(a.total_stage_usage.calls, b.total_stage_usage.calls),
        });
        metrics_rows.push(MetricsRow {
            metric: "latency".into(),
            val_a: fmt_ms(a.total_stage_usage.latency_ms),
            val_b: fmt_ms(b.total_stage_usage.latency_ms),
            delta: fmt_cost_delta_u64(a.total_stage_usage.latency_ms, b.total_stage_usage.latency_ms),
        });
    }

    if correct_a > 0 || correct_b > 0 {
        metrics_rows.push(MetricsRow {
            metric: "tokens/correct".into(),
            val_a: if correct_a > 0 {
                format!("{:.1}", a.total_stage_usage.total_tokens() as f64 / correct_a as f64)
            } else {
                "-".into()
            },
            val_b: if correct_b > 0 {
                format!("{:.1}", b.total_stage_usage.total_tokens() as f64 / correct_b as f64)
            } else {
                "-".into()
            },
            delta: if correct_a > 0 && correct_b > 0 {
                fmt_float_delta(
                    a.total_stage_usage.total_tokens() as f64 / correct_a as f64,
                    b.total_stage_usage.total_tokens() as f64 / correct_b as f64,
                    1,
                    false,
                )
            } else {
                "-".into()
            },
        });
    }

    writeln!(rendered).expect("write metrics spacer");
    writeln!(
        rendered,
        "{}",
        Table::new(&metrics_rows)
            .with(Style::rounded())
            .with(Modify::new(Columns::new(1..)).with(Alignment::right()))
    )
    .expect("write metrics table");

    if !a.stage_metrics.is_empty() || !b.stage_metrics.is_empty() {
        let stages = a
            .stage_metrics
            .keys()
            .chain(b.stage_metrics.keys())
            .cloned()
            .collect::<BTreeSet<_>>();
        let stage_rows = stages
            .into_iter()
            .map(|stage| {
                let usage_a = a.stage_metrics.get(&stage).cloned().unwrap_or_default();
                let usage_b = b.stage_metrics.get(&stage).cloned().unwrap_or_default();
                StageCompareRow {
                    stage,
                    val_a: fmt_stage_value(&usage_a),
                    val_b: fmt_stage_value(&usage_b),
                    delta: fmt_cost_delta_u64(usage_a.total_tokens(), usage_b.total_tokens()),
                }
            })
            .collect::<Vec<_>>();

        writeln!(rendered).expect("write legacy stage spacer");
        writeln!(
            rendered,
            "{}",
            Table::new(&stage_rows)
                .with(Style::rounded())
                .with(Modify::new(Columns::new(1..)).with(Alignment::right()))
        )
        .expect("write legacy stage table");
    }

    let cache_total_a = cache_compare_total(a);
    let cache_total_b = cache_compare_total(b);
    let cache_compare = cache_compare_rows(&cache_total_a, &cache_total_b);
    writeln!(rendered).expect("write cache compare spacer");
    writeln!(rendered, "cache-aware comparison").expect("write cache compare header");
    writeln!(
        rendered,
        "{}",
        Table::new(&cache_compare)
            .with(Style::rounded())
            .with(Modify::new(Columns::new(1..)).with(Alignment::right()))
    )
    .expect("write cache compare table");

    let rolled_up_a = rollup_operator_cache_metrics(&a.cache_aware_stage_metrics);
    let rolled_up_b = rollup_operator_cache_metrics(&b.cache_aware_stage_metrics);
    let stage_order = ["retain", "reflect", "consolidate", "opinion_merge", "judge"];
    let cache_stage_rows = stage_order
        .into_iter()
        .filter_map(|stage| {
            let usage_a = rolled_up_a.get(stage).cloned().unwrap_or_default();
            let usage_b = rolled_up_b.get(stage).cloned().unwrap_or_default();
            if cache_usage_available(&usage_a) || cache_usage_available(&usage_b) {
                Some(CacheStageCompareRow {
                    stage: stage.to_string(),
                    effective_prompt_a: effective_prompt_tokens(&usage_a).to_string(),
                    effective_prompt_b: effective_prompt_tokens(&usage_b).to_string(),
                    delta_effective_prompt: fmt_cost_delta_u64(
                        effective_prompt_tokens(&usage_a),
                        effective_prompt_tokens(&usage_b),
                    ),
                    cache_hit_a: usage_a.cache_hit_prompt_tokens.to_string(),
                    cache_hit_b: usage_b.cache_hit_prompt_tokens.to_string(),
                    delta_cache_hit: fmt_cost_delta_u64(
                        usage_a.cache_hit_prompt_tokens,
                        usage_b.cache_hit_prompt_tokens,
                    ),
                })
            } else {
                None
            }
        })
        .collect::<Vec<_>>();
    if !cache_stage_rows.is_empty() {
        writeln!(rendered).expect("write cache stage spacer");
        writeln!(
            rendered,
            "{}",
            Table::new(&cache_stage_rows)
                .with(Style::rounded())
                .with(Modify::new(Columns::new(1..)).with(Alignment::right()))
        )
        .expect("write cache stage compare table");
    }

    writeln!(rendered).expect("write savings spacer");
    writeln!(rendered, "cache savings verification").expect("write savings header");
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
        writeln!(
            rendered,
            "{}",
            Table::new(&savings_rows).with(Style::rounded())
        )
        .expect("write savings table");
    } else {
        writeln!(
            rendered,
            "verification unavailable: {}",
            savings_unavailable_reason(a, b)
        )
        .expect("write savings unavailable");
    }

    if matched.is_empty() {
        return rendered;
    }

    let mut question_rows = Vec::new();
    for category in &categories {
        let mut per_category = matched
            .iter()
            .filter(|(left, _)| left.category_name == *category)
            .collect::<Vec<_>>();
        per_category.sort_by(|a, b| {
            a.0.sample_id
                .cmp(&b.0.sample_id)
                .then(a.0.question_id.cmp(&b.0.question_id))
        });
        for (left, right) in per_category {
            question_rows.push(QuestionRow {
                qid: left.question_id.clone(),
                sample: left.sample_id.clone(),
                category: left.category_name.clone(),
                a: question_mark(left),
                b: question_mark(right),
            });
        }
    }

    writeln!(rendered).expect("write question spacer");
    writeln!(
        rendered,
        "{}",
        Table::new(&question_rows)
            .with(Style::rounded())
            .with(Modify::new(Columns::new(3..=4)).with(Alignment::center()))
    )
    .expect("write question table");

    let regressions = matched
        .iter()
        .filter(|(left, right)| left.judge_correct && !right.judge_correct)
        .count();
    let improvements = matched
        .iter()
        .filter(|(left, right)| !left.judge_correct && right.judge_correct)
        .count();
    let unchanged = matched.len().saturating_sub(improvements + regressions);
    writeln!(rendered).expect("write footer spacer");
    writeln!(
        rendered,
        "\x1b[32m{improvements} improved\x1b[0m, \x1b[31m{regressions} regressed\x1b[0m, {unchanged} unchanged"
    )
    .expect("write footer");

    rendered
}

fn load_file(path: &str) -> BenchmarkOutput {
    let raw = fs::read_to_string(path).unwrap_or_else(|e| {
        eprintln!("Failed to read {path}: {e}");
        process::exit(1);
    });
    let mut output: BenchmarkOutput = parse_output(&raw).unwrap_or_else(|e| {
        eprintln!("Failed to parse {path}: {e}");
        process::exit(1);
    });
    if output.results.is_empty() && !output.artifacts.questions_path.is_empty() {
        let base = std::path::Path::new(path)
            .parent()
            .unwrap_or_else(|| std::path::Path::new("."));
        let questions_path = base.join(&output.artifacts.questions_path);
        if questions_path.exists() {
            let raw_questions = fs::read_to_string(&questions_path).unwrap_or_else(|e| {
                eprintln!("Failed to read {}: {e}", questions_path.display());
                process::exit(1);
            });
            output.results = raw_questions
                .lines()
                .filter(|line| !line.trim().is_empty())
                .map(|line| {
                    serde_json::from_str::<QuestionResult>(line).unwrap_or_else(|e| {
                        eprintln!(
                            "Failed to parse question record in {}: {e}",
                            questions_path.display()
                        );
                        process::exit(1);
                    })
                })
                .collect();
        }
    }
    output
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
    Ok(render_compare(&a, &b, path_a, path_b))
}

fn filter_conv(mut output: BenchmarkOutput, conv: &str) -> BenchmarkOutput {
    output.results.retain(|r| r.sample_id == conv);
    output.total_questions = output.results.len();
    output.accuracy = if output.results.is_empty() {
        0.0
    } else {
        output.results.iter().filter(|r| r.judge_correct).count() as f64
            / output.results.len() as f64
    };
    output.mean_f1 = avg(&output.results.iter().map(|r| r.f1).collect::<Vec<_>>());
    output.mean_evidence_recall = avg(&output
        .results
        .iter()
        .map(|r| r.evidence_recall)
        .collect::<Vec<_>>());
    output
}

fn main() {
    let raw_args = env::args().collect::<Vec<_>>();

    let mut conv_filter: Option<String> = None;
    let mut files = Vec::new();
    let mut index = 1;
    while index < raw_args.len() {
        if raw_args[index] == "--conv" {
            index += 1;
            if index >= raw_args.len() {
                eprintln!("--conv requires a conversation ID");
                process::exit(1);
            }
            conv_filter = Some(raw_args[index].clone());
        } else {
            files.push(raw_args[index].clone());
        }
        index += 1;
    }

    if files.len() == 1 {
        let mut output = load_file(&files[0]);
        if let Some(ref conv) = conv_filter {
            output = filter_conv(output, conv);
        }
        view_single(&output, &files[0]);
        return;
    }
    if files.len() != 2 {
        eprintln!("Usage: view [--conv <id>] <file.json> [file2.json]");
        process::exit(1);
    }

    let mut a = load_file(&files[0]);
    let mut b = load_file(&files[1]);
    if let Some(ref conv) = conv_filter {
        a = filter_conv(a, conv);
        b = filter_conv(b, conv);
    }
    print!("{}", render_compare(&a, &b, &files[0], &files[1]));
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn cache_usage(
        prompt_tokens: u64,
        uncached_prompt_tokens: u64,
        cache_hit_prompt_tokens: u64,
        cache_write_prompt_tokens: u64,
        completion_tokens: u64,
        calls: u64,
    ) -> CacheAwareStageUsage {
        CacheAwareStageUsage {
            prompt_tokens,
            uncached_prompt_tokens,
            cache_hit_prompt_tokens,
            cache_write_prompt_tokens,
            completion_tokens,
            calls,
            errors: 0,
            latency_ms: 0,
            cache_supported_calls: calls,
            cache_hit_calls: u64::from(cache_hit_prompt_tokens > 0),
            cache_write_calls: u64::from(cache_write_prompt_tokens > 0),
            cache_unsupported_calls: 0,
        }
    }

    fn sample_output_with_cache() -> BenchmarkOutput {
        serde_json::from_value(json!({
            "tag": "cachey",
            "judge_model": "judge",
            "retain_model": "retain",
            "reflect_model": "reflect",
            "embedding_model": "embed",
            "reranker_model": "rerank",
            "consolidation_strategy": "end",
            "total_questions": 2,
            "accuracy": 1.0,
            "mean_f1": 1.0,
            "total_time_s": 2.0,
            "stage_metrics": {
                "retain_extract": {
                    "prompt_tokens": 12,
                    "completion_tokens": 4,
                    "calls": 2,
                    "errors": 0,
                    "latency_ms": 20
                },
                "retain_resolve": {
                    "prompt_tokens": 8,
                    "completion_tokens": 2,
                    "calls": 1,
                    "errors": 0,
                    "latency_ms": 10
                }
            },
            "total_stage_usage": {
                "prompt_tokens": 20,
                "completion_tokens": 6,
                "calls": 3,
                "errors": 0,
                "latency_ms": 30
            },
            "cache_aware_stage_metrics": {
                "retain_extract": {
                    "prompt_tokens": 12,
                    "uncached_prompt_tokens": 8,
                    "cache_hit_prompt_tokens": 3,
                    "cache_write_prompt_tokens": 1,
                    "completion_tokens": 4,
                    "calls": 2,
                    "errors": 0,
                    "latency_ms": 20,
                    "cache_supported_calls": 2,
                    "cache_hit_calls": 1,
                    "cache_write_calls": 1,
                    "cache_unsupported_calls": 0
                },
                "retain_resolve": {
                    "prompt_tokens": 8,
                    "uncached_prompt_tokens": 7,
                    "cache_hit_prompt_tokens": 1,
                    "cache_write_prompt_tokens": 0,
                    "completion_tokens": 2,
                    "calls": 1,
                    "errors": 0,
                    "latency_ms": 10,
                    "cache_supported_calls": 1,
                    "cache_hit_calls": 1,
                    "cache_write_calls": 0,
                    "cache_unsupported_calls": 0
                },
                "reflect": {
                    "prompt_tokens": 9,
                    "uncached_prompt_tokens": 7,
                    "cache_hit_prompt_tokens": 1,
                    "cache_write_prompt_tokens": 1,
                    "completion_tokens": 3,
                    "calls": 1,
                    "errors": 0,
                    "latency_ms": 11,
                    "cache_supported_calls": 1,
                    "cache_hit_calls": 1,
                    "cache_write_calls": 1,
                    "cache_unsupported_calls": 0
                },
                "consolidate": {
                    "prompt_tokens": 10,
                    "uncached_prompt_tokens": 8,
                    "cache_hit_prompt_tokens": 1,
                    "cache_write_prompt_tokens": 1,
                    "completion_tokens": 4,
                    "calls": 1,
                    "errors": 0,
                    "latency_ms": 12,
                    "cache_supported_calls": 1,
                    "cache_hit_calls": 1,
                    "cache_write_calls": 1,
                    "cache_unsupported_calls": 0
                },
                "opinion_merge": {
                    "prompt_tokens": 11,
                    "uncached_prompt_tokens": 9,
                    "cache_hit_prompt_tokens": 1,
                    "cache_write_prompt_tokens": 1,
                    "completion_tokens": 5,
                    "calls": 1,
                    "errors": 0,
                    "latency_ms": 13,
                    "cache_supported_calls": 1,
                    "cache_hit_calls": 1,
                    "cache_write_calls": 1,
                    "cache_unsupported_calls": 0
                },
                "judge": {
                    "prompt_tokens": 7,
                    "uncached_prompt_tokens": 5,
                    "cache_hit_prompt_tokens": 1,
                    "cache_write_prompt_tokens": 1,
                    "completion_tokens": 2,
                    "calls": 1,
                    "errors": 0,
                    "latency_ms": 9,
                    "cache_supported_calls": 1,
                    "cache_hit_calls": 1,
                    "cache_write_calls": 1,
                    "cache_unsupported_calls": 0
                }
            },
            "cache_aware_total_stage_usage": {
                "prompt_tokens": 57,
                "uncached_prompt_tokens": 44,
                "cache_hit_prompt_tokens": 8,
                "cache_write_prompt_tokens": 5,
                "completion_tokens": 20,
                "calls": 7,
                "errors": 0,
                "latency_ms": 75,
                "cache_supported_calls": 7,
                "cache_hit_calls": 6,
                "cache_write_calls": 5,
                "cache_unsupported_calls": 0
            }
        }))
        .expect("fixture should deserialize")
    }

    fn legacy_output() -> BenchmarkOutput {
        serde_json::from_value(json!({
            "tag": "legacy",
            "judge_model": "judge",
            "retain_model": "retain",
            "reflect_model": "reflect",
            "embedding_model": "embed",
            "reranker_model": "rerank",
            "consolidation_strategy": "end",
            "total_questions": 1,
            "accuracy": 1.0,
            "mean_f1": 1.0,
            "total_time_s": 1.0,
            "stage_metrics": {
                "retain_extract": {
                    "prompt_tokens": 12,
                    "completion_tokens": 4,
                    "calls": 2,
                    "errors": 0,
                    "latency_ms": 20
                }
            },
            "total_stage_usage": {
                "prompt_tokens": 12,
                "completion_tokens": 4,
                "calls": 2,
                "errors": 0,
                "latency_ms": 20
            }
        }))
        .expect("legacy fixture should deserialize")
    }

    #[test]
    fn cache_aware_single_summary_deserializes_additive_fields() {
        let output: BenchmarkOutput = serde_json::from_value(json!({
            "judge_model": "judge",
            "retain_model": "retain",
            "reflect_model": "reflect",
            "embedding_model": "embed",
            "reranker_model": "rerank",
            "consolidation_strategy": "end",
            "total_questions": 1,
            "total_time_s": 1.0,
            "cache_aware_stage_metrics": {
                "retain_extract": {
                    "prompt_tokens": 12,
                    "uncached_prompt_tokens": 8,
                    "cache_hit_prompt_tokens": 3,
                    "cache_write_prompt_tokens": 1,
                    "completion_tokens": 5,
                    "calls": 2,
                    "cache_supported_calls": 2,
                    "cache_hit_calls": 1,
                    "cache_write_calls": 1
                }
            },
            "cache_aware_total_stage_usage": {
                "prompt_tokens": 50,
                "uncached_prompt_tokens": 30,
                "cache_hit_prompt_tokens": 15,
                "cache_write_prompt_tokens": 5,
                "completion_tokens": 9,
                "calls": 4,
                "cache_supported_calls": 4,
                "cache_hit_calls": 2,
                "cache_write_calls": 1
            },
            "per_conversation": {
                "conv-1": {
                    "bank_id": "bank-1",
                    "cache_aware_stage_metrics": {
                        "judge": {
                            "prompt_tokens": 9,
                            "uncached_prompt_tokens": 6,
                            "cache_hit_prompt_tokens": 2,
                            "cache_write_prompt_tokens": 1,
                            "completion_tokens": 3,
                            "calls": 1,
                            "cache_supported_calls": 1,
                            "cache_hit_calls": 1,
                            "cache_write_calls": 1
                        }
                    }
                }
            }
        }))
        .expect("cache-aware artifact should deserialize");

        assert_eq!(output.cache_aware_total_stage_usage.cache_hit_prompt_tokens, 15);
        assert_eq!(
            output
                .cache_aware_stage_metrics
                .get("retain_extract")
                .expect("retain_extract present")
                .cache_write_prompt_tokens,
            1
        );
        assert_eq!(
            output
                .per_conversation
                .get("conv-1")
                .expect("conv-1 present")
                .cache_aware_stage_metrics
                .get("judge")
                .expect("judge present")
                .cache_hit_prompt_tokens,
            2
        );
        assert!(cache_usage_available(&output.cache_aware_total_stage_usage));
    }

    #[test]
    fn cache_aware_single_summary_rolls_up_operator_cache_metrics() {
        let stage_metrics = BTreeMap::from([
            (
                "retain_extract".to_string(),
                cache_usage(12, 8, 3, 1, 5, 2),
            ),
            (
                "retain_resolve".to_string(),
                cache_usage(9, 7, 1, 1, 2, 1),
            ),
            ("reflect".to_string(), cache_usage(11, 9, 1, 1, 4, 1)),
        ]);

        let rollup = rollup_operator_cache_metrics(&stage_metrics);

        assert_eq!(rollup.len(), 2);
        assert_eq!(
            rollup.get("retain").expect("retain row").prompt_tokens,
            21
        );
        assert_eq!(
            rollup
                .get("retain")
                .expect("retain row")
                .uncached_prompt_tokens,
            15
        );
        assert_eq!(
            rollup
                .get("retain")
                .expect("retain row")
                .cache_hit_prompt_tokens,
            4
        );
        assert_eq!(
            rollup.get("reflect").expect("reflect row").completion_tokens,
            4
        );
    }

    #[test]
    fn cache_aware_single_summary_legacy_defaults_to_unavailable() {
        let output: BenchmarkOutput = serde_json::from_value(json!({
            "judge_model": "judge",
            "retain_model": "retain",
            "reflect_model": "reflect",
            "embedding_model": "embed",
            "reranker_model": "rerank",
            "consolidation_strategy": "end",
            "total_questions": 1,
            "total_time_s": 1.0,
            "per_conversation": {
                "conv-1": {
                    "bank_id": "bank-1"
                }
            }
        }))
        .expect("legacy artifact should deserialize");

        assert!(!cache_usage_available(&output.cache_aware_total_stage_usage));
        assert!(output.cache_aware_stage_metrics.is_empty());
        assert!(
            output
                .per_conversation
                .get("conv-1")
                .expect("conv-1 present")
                .cache_aware_stage_metrics
                .is_empty()
        );
    }

    #[test]
    fn cache_aware_single_summary_reports_prompt_components() {
        let rendered = render_single(&sample_output_with_cache(), "cachey.json");

        assert!(rendered.contains("effective prompt tok"));
        assert!(rendered.contains("cache hit tok"));
        assert!(rendered.contains("cache write tok"));
        assert!(rendered.contains("cache supported"));
        assert!(rendered.contains("cache unsupported"));
        assert!(rendered.contains("cache hit rate"));
        assert!(rendered.contains("49"));
        assert!(rendered.contains("8"));
        assert!(rendered.contains("5"));
    }

    #[test]
    fn cache_aware_single_rolls_up_operator_stages() {
        let rendered = render_single(&sample_output_with_cache(), "cachey.json");

        assert!(rendered.contains("retain"));
        assert!(rendered.contains("reflect"));
        assert!(rendered.contains("consolidate"));
        assert!(rendered.contains("opinion_merge"));
        assert!(rendered.contains("judge"));
    }

    #[test]
    fn cache_aware_single_legacy_artifact_reports_unavailable() {
        let rendered = render_single(&legacy_output(), "legacy.json");

        assert!(rendered.contains("cache-aware metrics unavailable (legacy artifact)"));
    }

    fn compare_output_with_metadata() -> BenchmarkOutput {
        let mut output = sample_output_with_cache();
        output.manifest.dataset_fingerprint = "dataset-v1".into();
        output.manifest.prompt_hashes = BenchmarkPromptHashes {
            judge: "judge-hash".into(),
            retain_extract: "retain-hash".into(),
            reflect_agent: "reflect-hash".into(),
            consolidate: "consolidate-hash".into(),
        };
        output.results = vec![QuestionResult {
            question_id: "q1".into(),
            sample_id: "conv-1".into(),
            question: "Question?".into(),
            category_name: "fact".into(),
            judge_correct: true,
            ground_truth: "yes".into(),
            hypothesis: "yes".into(),
            f1: 1.0,
            evidence_recall: 1.0,
            evidence_refs: vec![],
            retrieved_turn_refs: vec![],
            evidence_hit: false,
            elapsed_s: 1.0,
            status: "ok".into(),
            error: None,
        }];
        output
    }

    #[test]
    fn cache_aware_compare_reports_same_model_savings_signal() {
        let baseline = compare_output_with_metadata();
        let mut warm = compare_output_with_metadata();
        warm.tag = Some("warm".into());
        warm.cache_aware_total_stage_usage = CacheAwareStageUsage {
            prompt_tokens: 57,
            uncached_prompt_tokens: 30,
            cache_hit_prompt_tokens: 20,
            cache_write_prompt_tokens: 3,
            completion_tokens: 20,
            calls: 7,
            errors: 0,
            latency_ms: 70,
            cache_supported_calls: 7,
            cache_hit_calls: 7,
            cache_write_calls: 3,
            cache_unsupported_calls: 0,
        };
        warm.cache_aware_stage_metrics.insert(
            "reflect".into(),
            cache_usage(9, 3, 5, 1, 3, 1),
        );

        let rendered = render_compare(&baseline, &warm, "baseline.json", "warm.json");

        assert!(rendered.contains("cache-aware comparison"));
        assert!(rendered.contains("cache savings verification"));
        assert!(rendered.contains("effective prompt tok"));
        assert!(rendered.contains("cache hit tok"));
        assert!(rendered.contains("retain"));
        assert!(rendered.contains("reflect"));
        assert!(rendered.contains("cache savings visible"));
    }

    #[test]
    fn cache_aware_compare_blocks_mismatched_metadata() {
        let baseline = compare_output_with_metadata();
        let mut mismatch = compare_output_with_metadata();
        mismatch.manifest.prompt_hashes.reflect_agent = "different-hash".into();

        let rendered = render_compare(&baseline, &mismatch, "baseline.json", "mismatch.json");

        assert!(rendered.contains("cache-aware comparison"));
        assert!(rendered.contains("cache savings verification"));
        assert!(rendered.contains("verification unavailable:"));
    }
}
