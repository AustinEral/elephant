//! LoCoMo benchmark harness for Elephant.
//!
//! This runner executes Elephant in-process so benchmark artifacts can include
//! stage-level usage, dataset evidence refs, and stable run provenance.

#[path = "../common/mod.rs"]
mod common;

use std::collections::hash_map::DefaultHasher;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::env;
use std::fs;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::str::FromStr;
use std::sync::Arc;
use std::time::Instant;

use chrono::{DateTime, NaiveDateTime, Utc};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use tokio::sync::{Mutex, Semaphore, mpsc};

use elephant::MemoryStore;
use elephant::consolidation::{ConsolidationProgress, observation};
use elephant::llm::LlmClient;
use elephant::metrics::{
    CacheAwareStageUsage, LlmStage, MetricsCollector, StageUsage, with_scoped_collector,
};
use elephant::runtime::{
    BuildRuntimeOptions, ElephantRuntime, RuntimePromptHashes as ElephantPromptHashes,
    RuntimeTuning as ElephantRuntimeTuning, build_runtime_from_env,
};
use elephant::types::{
    BankId, Disposition, MemoryBank, NetworkType, ReflectDoneTrace, ReflectQuery, RetainInput,
    RetrievalSource, TurnId,
};

// --- LoCoMo dataset types ---

#[derive(Debug, Deserialize)]
struct LocomoEntry {
    sample_id: String,
    conversation: Conversation,
    qa: Vec<QaPair>,
}

#[derive(Debug, Deserialize)]
struct Conversation {
    speaker_a: String,
    speaker_b: String,
    #[serde(flatten)]
    sessions: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Deserialize)]
struct Turn {
    speaker: String,
    text: String,
    #[serde(default)]
    dia_id: Option<String>,
    blip_caption: Option<String>,
}

#[derive(Debug, Deserialize)]
struct QaPair {
    question: String,
    /// Usually missing for category 5 (adversarial), though the upstream dataset
    /// currently contains two category-5 rows with populated `answer` values.
    answer: Option<serde_json::Value>,
    category: u8,
    #[serde(default)]
    evidence: Vec<String>,
}

// --- Judge types (delegated to common::judge) ---

// --- Result types ---

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct BenchmarkManifest {
    protocol_version: String,
    profile: String,
    mode: String,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    config_path: Option<String>,
    dataset_path: String,
    dataset_fingerprint: String,
    command: String,
    category_filter: Vec<u8>,
    #[serde(default)]
    selected_conversations: Vec<String>,
    image_policy: String,
    ingestion_granularity: String,
    question_concurrency: usize,
    conversation_concurrency: usize,
    consolidation_strategy: String,
    session_limit: Option<usize>,
    question_limit: Option<usize>,
    raw_json: bool,
    dirty_worktree: Option<bool>,
    #[serde(default)]
    prompt_hashes: BenchmarkPromptHashes,
    #[serde(default)]
    runtime_config: BenchmarkRuntimeConfig,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    source_artifact: Option<SourceArtifact>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    source_artifacts: Vec<SourceArtifact>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct ConversationSummary {
    bank_id: String,
    accuracy: f64,
    mean_f1: f64,
    mean_evidence_recall: f64,
    count: usize,
    #[serde(default)]
    ingest_time_s: f64,
    #[serde(default)]
    consolidation_time_s: f64,
    #[serde(default)]
    qa_time_s: f64,
    #[serde(default)]
    total_time_s: f64,
    #[serde(default)]
    stage_metrics: BTreeMap<LlmStage, StageUsage>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    cache_aware_stage_metrics: BTreeMap<LlmStage, CacheAwareStageUsage>,
    #[serde(default)]
    bank_stats: ConversationBankStats,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct BenchmarkArtifacts {
    #[serde(default)]
    questions_path: String,
    #[serde(default)]
    debug_path: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct BenchmarkOutput {
    benchmark: String,
    timestamp: String,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    commit: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    tag: Option<String>,
    judge_model: String,
    retain_model: String,
    reflect_model: String,
    embedding_model: String,
    reranker_model: String,
    consolidation_strategy: String,
    total_questions: usize,
    accuracy: f64,
    mean_f1: f64,
    mean_evidence_recall: f64,
    per_category: HashMap<String, CategoryResult>,
    #[serde(default)]
    per_conversation: HashMap<String, ConversationSummary>,
    /// Maps sample_id → bank_id for resuming without re-ingestion.
    #[serde(rename = "bank_ids", alias = "banks", default)]
    banks: HashMap<String, String>,
    /// Maps benchmark turn ids back to LoCoMo refs like D1:3.
    #[serde(default)]
    turn_refs: HashMap<String, String>,
    #[serde(default)]
    manifest: BenchmarkManifest,
    #[serde(default)]
    artifacts: BenchmarkArtifacts,
    #[serde(default)]
    stage_metrics: BTreeMap<LlmStage, StageUsage>,
    #[serde(default)]
    total_stage_usage: StageUsage,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    cache_aware_stage_metrics: BTreeMap<LlmStage, CacheAwareStageUsage>,
    #[serde(default, skip_serializing_if = "cache_aware_stage_usage_is_empty")]
    cache_aware_total_stage_usage: CacheAwareStageUsage,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    results: Vec<QuestionResult>,
    total_time_s: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CategoryResult {
    accuracy: f64,
    mean_f1: f64,
    mean_evidence_recall: f64,
    count: usize,
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
    #[serde(default)]
    reflect_budget_tokens: usize,
    #[serde(default)]
    judge_temperature: f32,
    #[serde(default)]
    judge_max_tokens: usize,
    #[serde(default)]
    judge_max_attempts: usize,
    #[serde(default)]
    qa_updates_memory: bool,
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

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
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

#[derive(Debug, Clone, Serialize, Deserialize)]
struct RetrievedFactEntry {
    id: String,
    content: String,
    score: f32,
    network: String,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    source_turn_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    source_turn_ref: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    evidence_ids: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    retrieval_sources: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    support_turn_ids: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    support_turn_refs: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct RetrievedSourceEntry {
    id: String,
    fact_id: String,
    timestamp: String,
    content: String,
    #[serde(default)]
    truncated: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ReflectTraceEntry {
    iteration: usize,
    tool_name: String,
    query: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    returned_fact_ids: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    requested_fact_ids: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    new_fact_ids: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    returned_source_ids: Vec<String>,
    facts_returned: usize,
    total_tokens: usize,
    latency_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct QuestionResult {
    /// Stable hash of (sample_id, question) for cross-run traceability.
    question_id: String,
    sample_id: String,
    question: String,
    ground_truth: String,
    hypothesis: String,
    category_name: String,
    f1: f64,
    judge_correct: bool,
    judge_reasoning: String,
    elapsed_s: f64,
    status: String,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    error: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    final_source_ids: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    evidence_refs: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    retrieved_turn_refs: Vec<String>,
    evidence_hit: bool,
    evidence_recall: f64,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    qa_stage_metrics: BTreeMap<LlmStage, StageUsage>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    cache_aware_qa_stage_metrics: BTreeMap<LlmStage, CacheAwareStageUsage>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct QuestionDebugRecord {
    question_id: String,
    sample_id: String,
    question: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    reflect_trace: Vec<ReflectTraceEntry>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    final_done: Option<ReflectDoneTrace>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    retrieved_context: Vec<RetrievedFactEntry>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    retrieved_sources: Vec<RetrievedSourceEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct ConversationPhaseTimings {
    ingest_time_s: f64,
    consolidation_time_s: f64,
    qa_time_s: f64,
    total_time_s: f64,
}

// --- Category names ---

fn category_name(cat: u8) -> &'static str {
    match cat {
        1 => "single-hop",
        2 => "multi-hop",
        3 => "temporal",
        4 => "open-domain",
        5 => "unanswerable",
        _ => "unknown",
    }
}

fn should_score_question(qa: &QaPair) -> bool {
    matches!(qa.category, 1..=4) && qa.answer.is_some()
}

fn network_name(network: NetworkType) -> &'static str {
    match network {
        NetworkType::World => "world",
        NetworkType::Experience => "experience",
        NetworkType::Observation => "observation",
        NetworkType::Opinion => "opinion",
    }
}

fn retrieval_source_name(source: RetrievalSource) -> &'static str {
    match source {
        RetrievalSource::Semantic => "semantic",
        RetrievalSource::Keyword => "keyword",
        RetrievalSource::Graph => "graph",
        RetrievalSource::Temporal => "temporal",
    }
}

fn preserve_stage_for_qa(stage: LlmStage) -> bool {
    !matches!(stage, LlmStage::Reflect | LlmStage::Judge)
}

fn cache_aware_stage_usage_is_empty(usage: &CacheAwareStageUsage) -> bool {
    usage == &CacheAwareStageUsage::default()
}

fn filter_stage_snapshot<T: Clone>(
    command: BenchCommand,
    snapshot: &BTreeMap<LlmStage, T>,
) -> BTreeMap<LlmStage, T> {
    if matches!(command, BenchCommand::Qa) {
        snapshot
            .iter()
            .filter(|(stage, _)| preserve_stage_for_qa(**stage))
            .map(|(stage, usage)| (*stage, usage.clone()))
            .collect()
    } else {
        snapshot.clone()
    }
}

fn hybrid_stage_snapshots(
    cache_aware: &BTreeMap<LlmStage, CacheAwareStageUsage>,
    legacy: &BTreeMap<LlmStage, StageUsage>,
) -> (
    BTreeMap<LlmStage, StageUsage>,
    BTreeMap<LlmStage, CacheAwareStageUsage>,
) {
    let metrics = MetricsCollector::new();
    metrics.extend_cache_aware_or_legacy_snapshot(cache_aware, legacy);
    (metrics.snapshot(), metrics.cache_aware_snapshot())
}

fn seeded_hybrid_stage_snapshots(
    command: BenchCommand,
    cache_aware: &BTreeMap<LlmStage, CacheAwareStageUsage>,
    legacy: &BTreeMap<LlmStage, StageUsage>,
) -> (
    BTreeMap<LlmStage, StageUsage>,
    BTreeMap<LlmStage, CacheAwareStageUsage>,
) {
    let (legacy_snapshot, cache_aware_snapshot) = hybrid_stage_snapshots(cache_aware, legacy);
    (
        filter_stage_snapshot(command, &legacy_snapshot),
        filter_stage_snapshot(command, &cache_aware_snapshot),
    )
}

// --- Token F1 ---

fn normalize_answer(s: &str) -> String {
    let lower = s.to_lowercase();
    let no_punct: String = lower
        .chars()
        .map(|c| if c.is_ascii_punctuation() { ' ' } else { c })
        .collect();
    let words: Vec<&str> = no_punct
        .split_whitespace()
        .filter(|w| !matches!(*w, "a" | "an" | "the"))
        .collect();
    words.join(" ")
}

fn token_f1(prediction: &str, ground_truth: &str) -> f64 {
    let pred_norm = normalize_answer(prediction);
    let gold_norm = normalize_answer(ground_truth);
    let pred_tokens: Vec<&str> = pred_norm.split_whitespace().collect();
    let gold_tokens: Vec<&str> = gold_norm.split_whitespace().collect();

    if gold_tokens.is_empty() {
        return if pred_tokens.is_empty() { 1.0 } else { 0.0 };
    }
    if pred_tokens.is_empty() {
        return 0.0;
    }

    let mut gold_counts: HashMap<&str, usize> = HashMap::new();
    for t in &gold_tokens {
        *gold_counts.entry(t).or_default() += 1;
    }
    let mut pred_counts: HashMap<&str, usize> = HashMap::new();
    for t in &pred_tokens {
        *pred_counts.entry(t).or_default() += 1;
    }

    let mut common = 0usize;
    for (token, &gold_count) in &gold_counts {
        if let Some(&pred_count) = pred_counts.get(token) {
            common += gold_count.min(pred_count);
        }
    }

    if common == 0 {
        return 0.0;
    }

    let precision = common as f64 / pred_tokens.len() as f64;
    let recall = common as f64 / gold_tokens.len() as f64;
    2.0 * precision * recall / (precision + recall)
}

// --- Session parsing ---

fn parse_session_date(date_str: &str) -> DateTime<Utc> {
    let cleaned = date_str.trim();
    if let Ok(dt) = NaiveDateTime::parse_from_str(cleaned, "%I:%M %p on %-d %B, %Y") {
        return dt.and_utc();
    }
    if let Ok(dt) = NaiveDateTime::parse_from_str(cleaned, "%I:%M %P on %-d %B, %Y") {
        return dt.and_utc();
    }
    Utc::now()
}

fn session_count(conv: &Conversation) -> usize {
    conv.sessions
        .keys()
        .filter_map(|k| {
            let rest = k.strip_prefix("session_")?;
            if let Ok(n) = rest.parse::<usize>() {
                return Some(n);
            }
            rest.strip_suffix("_dialogue")
                .or_else(|| rest.strip_suffix("_dialog"))
                .and_then(|n| n.parse::<usize>().ok())
        })
        .max()
        .unwrap_or(0)
}

fn get_session_turns(conv: &Conversation, idx: usize) -> Vec<Turn> {
    let plain_key = format!("session_{idx}");
    if let Some(v) = conv.sessions.get(&plain_key)
        && let Ok(turns) = serde_json::from_value::<Vec<Turn>>(v.clone())
    {
        return turns;
    }
    for suffix in ["dialogue", "dialog"] {
        let key = format!("session_{idx}_{suffix}");
        if let Some(v) = conv.sessions.get(&key) {
            if let Ok(turns) = serde_json::from_value::<Vec<Turn>>(v.clone()) {
                return turns;
            }
        }
    }
    Vec::new()
}

fn get_session_date(conv: &Conversation, idx: usize) -> String {
    let key = format!("session_{idx}_date_time");
    conv.sessions
        .get(&key)
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string()
}

fn format_turn(turn: &Turn) -> String {
    match turn.blip_caption.as_deref() {
        Some(caption) if !caption.trim().is_empty() => {
            format!(
                "{}: {}\n[Image caption: {}]",
                turn.speaker, turn.text, caption
            )
        }
        _ => format!("{}: {}", turn.speaker, turn.text),
    }
}

fn format_session(turns: &[Turn], date_str: &str) -> String {
    let dialogue = turns.iter().map(format_turn).collect::<Vec<_>>().join("\n");
    format!("Date: {date_str}\n\n{dialogue}")
}

fn format_session_raw(conv: &Conversation, idx: usize) -> String {
    let plain_key = format!("session_{idx}");
    let dialogue_key = format!("session_{idx}_dialogue");
    let dialog_key = format!("session_{idx}_dialog");
    let date_key = format!("session_{idx}_date_time");
    let obj = serde_json::json!({
        "sample_session": idx,
        "date_time": conv.sessions.get(&date_key).cloned().unwrap_or(serde_json::Value::Null),
        "dialogue": conv.sessions.get(&dialogue_key)
            .cloned()
            .or_else(|| conv.sessions.get(&dialog_key).cloned())
            .or_else(|| conv.sessions.get(&plain_key).cloned())
            .unwrap_or(serde_json::Value::Null),
    });
    serde_json::to_string_pretty(&obj).unwrap_or_default()
}

fn answer_to_string(val: &serde_json::Value) -> String {
    match val {
        serde_json::Value::String(s) => s.clone(),
        serde_json::Value::Number(n) => n.to_string(),
        other => other.to_string(),
    }
}

fn evidence_recall(expected: &[String], retrieved: &[String]) -> (bool, f64) {
    let expected_set: BTreeSet<&str> = expected.iter().map(String::as_str).collect();
    if expected_set.is_empty() {
        return (true, 1.0);
    }
    let retrieved_set: BTreeSet<&str> = retrieved.iter().map(String::as_str).collect();
    let hits = expected_set
        .iter()
        .filter(|e| retrieved_set.contains(**e))
        .count();
    (hits > 0, hits as f64 / expected_set.len() as f64)
}

fn evidence_precision(expected: &[String], retrieved: &[String]) -> Option<f64> {
    let expected_set: BTreeSet<&str> = expected.iter().map(String::as_str).collect();
    let retrieved_set: BTreeSet<&str> = retrieved.iter().map(String::as_str).collect();
    if expected_set.is_empty() && retrieved_set.is_empty() {
        return None;
    }
    if retrieved_set.is_empty() {
        return Some(0.0);
    }
    let hits = retrieved_set
        .iter()
        .filter(|turn_ref| expected_set.contains(**turn_ref))
        .count();
    Some(hits as f64 / retrieved_set.len() as f64)
}

fn fnv1a64(data: &[u8]) -> u64 {
    common::fnv1a64(data)
}

fn fnv1a64_hex(data: &str) -> String {
    common::fnv1a64_hex(data)
}

fn fmt_elapsed(seconds: f64) -> String {
    let m = (seconds as u64) / 60;
    let s = (seconds as u64) % 60;
    format!("{m}:{s:02}")
}

fn benchmark_prompt_hashes(runtime: &ElephantRuntime) -> BenchmarkPromptHashes {
    BenchmarkPromptHashes {
        judge: fnv1a64_hex(JUDGE_PROMPT),
        elephant: runtime.info.prompt_hashes.clone(),
    }
}

fn benchmark_runtime_config(runtime: &ElephantRuntime) -> BenchmarkRuntimeConfig {
    BenchmarkRuntimeConfig {
        elephant: runtime.info.tuning.clone(),
        reflect_budget_tokens: REFLECT_BUDGET_TOKENS,
        judge_temperature: common::judge::JUDGE_TEMPERATURE,
        judge_max_tokens: common::judge::JUDGE_MAX_TOKENS,
        judge_max_attempts: common::judge::JUDGE_MAX_ATTEMPTS,
        qa_updates_memory: false,
    }
}

async fn finalize_bank_stats(
    store: &Arc<dyn MemoryStore>,
    bank_id: BankId,
    stats: &mut ConversationBankStats,
) -> Result<(), String> {
    let all_facts = store
        .get_facts_by_bank(bank_id, Default::default())
        .await
        .map_err(|e| format!("failed to read final bank facts: {e}"))?;
    stats.final_fact_count = all_facts.len();
    stats.final_observation_count = all_facts
        .iter()
        .filter(|fact| fact.network == NetworkType::Observation)
        .count();
    stats.final_opinion_count = all_facts
        .iter()
        .filter(|fact| fact.network == NetworkType::Opinion)
        .count();
    stats.final_entity_count = store
        .list_entities(bank_id)
        .await
        .map_err(|e| format!("failed to read final bank entities: {e}"))?
        .len();
    Ok(())
}

// --- LLM Judge (delegated to common::judge) ---

const JUDGE_PROMPT: &str = include_str!("judge_answer.txt");
const REFLECT_BUDGET_TOKENS: usize = 4096;

fn build_judge_client(
    metrics: Arc<MetricsCollector>,
    override_model: Option<String>,
) -> Arc<dyn LlmClient> {
    common::judge::build_judge_client(metrics, override_model)
}

fn judge_label(override_model: &Option<String>) -> String {
    common::judge::judge_label(override_model)
}

async fn llm_judge(
    judge: &dyn LlmClient,
    question: &str,
    gold_answer: &str,
    generated_answer: &str,
) -> Result<(bool, String), String> {
    let rendered = JUDGE_PROMPT
        .replace("{question}", question)
        .replace("{gold_answer}", gold_answer)
        .replace("{generated_answer}", generated_answer);
    common::judge::llm_judge(judge, &rendered).await
}

// --- Result flushing ---

fn sidecar_path(output_path: &Path, suffix: &str) -> PathBuf {
    common::sidecar_path(output_path, suffix)
}

fn relative_artifact_path(base: &Path, target: &Path) -> String {
    target
        .strip_prefix(
            base.parent()
                .map(Path::to_path_buf)
                .unwrap_or_default()
                .as_path(),
        )
        .unwrap_or(target)
        .display()
        .to_string()
}

fn append_jsonl<T: Serialize>(path: &Path, value: &T) {
    common::append_jsonl(path, value);
}

#[allow(clippy::too_many_arguments)]
fn flush_results(
    results: &[QuestionResult],
    banks: &HashMap<String, String>,
    conversation_stage_metrics: &HashMap<String, BTreeMap<LlmStage, StageUsage>>,
    conversation_cache_aware_stage_metrics: &HashMap<
        String,
        BTreeMap<LlmStage, CacheAwareStageUsage>,
    >,
    conversation_timings: &HashMap<String, ConversationPhaseTimings>,
    conversation_bank_stats: &HashMap<String, ConversationBankStats>,
    turn_refs: &HashMap<String, String>,
    output_path: &Path,
    questions_path: &Path,
    debug_path: &Path,
    judge_label: &str,
    tag: &Option<String>,
    retain_model: &str,
    reflect_model: &str,
    embedding_model: &str,
    reranker_model: &str,
    consolidation_strategy: &str,
    manifest: &BenchmarkManifest,
    metrics: &MetricsCollector,
    run_timestamp: &str,
    commit: Option<&str>,
    bench_elapsed: f64,
) {
    let mut sorted_results = results.to_vec();
    sorted_results.sort_by(|a, b| {
        a.sample_id
            .cmp(&b.sample_id)
            .then(a.category_name.cmp(&b.category_name))
            .then(a.question_id.cmp(&b.question_id))
            .then(a.question.cmp(&b.question))
    });
    let total_questions = sorted_results.len();

    let mut category_results: HashMap<String, Vec<&QuestionResult>> = HashMap::new();
    let mut conversation_results: HashMap<String, Vec<&QuestionResult>> = HashMap::new();
    for r in &sorted_results {
        category_results
            .entry(r.category_name.clone())
            .or_default()
            .push(r);
        conversation_results
            .entry(r.sample_id.clone())
            .or_default()
            .push(r);
    }

    let mean_f1 = if total_questions > 0 {
        sorted_results.iter().map(|r| r.f1).sum::<f64>() / total_questions as f64
    } else {
        0.0
    };
    let mean_evidence_recall = if total_questions > 0 {
        sorted_results
            .iter()
            .map(|r| r.evidence_recall)
            .sum::<f64>()
            / total_questions as f64
    } else {
        0.0
    };
    let total_correct = sorted_results.iter().filter(|r| r.judge_correct).count();
    let accuracy = if total_questions > 0 {
        total_correct as f64 / total_questions as f64
    } else {
        0.0
    };

    let mut per_category = HashMap::new();
    for (name, rows) in &category_results {
        let n = rows.len();
        per_category.insert(
            name.clone(),
            CategoryResult {
                accuracy: rows.iter().filter(|r| r.judge_correct).count() as f64 / n as f64,
                mean_f1: rows.iter().map(|r| r.f1).sum::<f64>() / n as f64,
                mean_evidence_recall: rows.iter().map(|r| r.evidence_recall).sum::<f64>()
                    / n as f64,
                count: n,
            },
        );
    }

    let conversation_ids = conversation_results
        .keys()
        .cloned()
        .chain(conversation_stage_metrics.keys().cloned())
        .chain(conversation_cache_aware_stage_metrics.keys().cloned())
        .chain(banks.keys().cloned())
        .chain(conversation_timings.keys().cloned())
        .chain(conversation_bank_stats.keys().cloned())
        .collect::<BTreeSet<_>>();

    let mut per_conversation = HashMap::new();
    for sample_id in conversation_ids {
        let rows = conversation_results
            .get(&sample_id)
            .cloned()
            .unwrap_or_default();
        let n = rows.len();
        per_conversation.insert(
            sample_id.clone(),
            ConversationSummary {
                bank_id: banks.get(&sample_id).cloned().unwrap_or_default(),
                accuracy: if n > 0 {
                    rows.iter().filter(|r| r.judge_correct).count() as f64 / n as f64
                } else {
                    0.0
                },
                mean_f1: if n > 0 {
                    rows.iter().map(|r| r.f1).sum::<f64>() / n as f64
                } else {
                    0.0
                },
                mean_evidence_recall: if n > 0 {
                    rows.iter().map(|r| r.evidence_recall).sum::<f64>() / n as f64
                } else {
                    0.0
                },
                count: n,
                ingest_time_s: conversation_timings
                    .get(&sample_id)
                    .map(|timings| timings.ingest_time_s)
                    .unwrap_or_default(),
                consolidation_time_s: conversation_timings
                    .get(&sample_id)
                    .map(|timings| timings.consolidation_time_s)
                    .unwrap_or_default(),
                qa_time_s: conversation_timings
                    .get(&sample_id)
                    .map(|timings| timings.qa_time_s)
                    .unwrap_or_default(),
                total_time_s: conversation_timings
                    .get(&sample_id)
                    .map(|timings| timings.total_time_s)
                    .unwrap_or_default(),
                stage_metrics: conversation_stage_metrics
                    .get(&sample_id)
                    .cloned()
                    .unwrap_or_default(),
                cache_aware_stage_metrics: conversation_cache_aware_stage_metrics
                    .get(&sample_id)
                    .cloned()
                    .unwrap_or_default(),
                bank_stats: conversation_bank_stats
                    .get(&sample_id)
                    .cloned()
                    .unwrap_or_default(),
            },
        );
    }

    let stage_metrics = metrics.snapshot();
    let total_stage_usage = metrics.total_usage();
    let cache_aware_stage_metrics = metrics.cache_aware_snapshot();
    let cache_aware_total_stage_usage = metrics.cache_aware_total_usage();

    let output = BenchmarkOutput {
        benchmark: "locomo".into(),
        timestamp: run_timestamp.to_string(),
        commit: commit.map(str::to_owned),
        tag: tag.clone(),
        judge_model: judge_label.to_string(),
        retain_model: retain_model.to_string(),
        reflect_model: reflect_model.to_string(),
        embedding_model: embedding_model.to_string(),
        reranker_model: reranker_model.to_string(),
        consolidation_strategy: consolidation_strategy.to_string(),
        total_questions,
        accuracy,
        mean_f1,
        mean_evidence_recall,
        per_category,
        per_conversation,
        banks: banks.clone(),
        turn_refs: turn_refs.clone(),
        manifest: manifest.clone(),
        artifacts: BenchmarkArtifacts {
            questions_path: relative_artifact_path(output_path, questions_path),
            debug_path: relative_artifact_path(output_path, debug_path),
        },
        stage_metrics,
        total_stage_usage,
        cache_aware_stage_metrics,
        cache_aware_total_stage_usage,
        results: Vec::new(),
        total_time_s: bench_elapsed,
    };

    if let Ok(json) = serde_json::to_string_pretty(&output) {
        let _ = fs::write(output_path, &json);
    }
}

// --- CLI ---

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum RunProfile {
    Full,
    Smoke,
    LegacyRaw,
}

impl Default for RunProfile {
    fn default() -> Self {
        Self::Full
    }
}

impl RunProfile {
    fn as_str(self) -> &'static str {
        match self {
            Self::Full => "full",
            Self::Smoke => "smoke",
            Self::LegacyRaw => "legacy-raw",
        }
    }

    fn config_path(self) -> PathBuf {
        PathBuf::from(format!("bench/locomo/profiles/{}.json", self.as_str()))
    }
}

impl FromStr for RunProfile {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "full" => Ok(Self::Full),
            "smoke" => Ok(Self::Smoke),
            "legacy-raw" => Ok(Self::LegacyRaw),
            other => Err(format!(
                "invalid --profile value: {other} (expected one of: full, smoke, legacy-raw)"
            )),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BenchCommand {
    Run,
    Ingest,
    Qa,
    Merge,
}

impl BenchCommand {
    fn as_str(self) -> &'static str {
        match self {
            Self::Run => "run",
            Self::Ingest => "ingest",
            Self::Qa => "qa",
            Self::Merge => "merge",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum IngestMode {
    Turn,
    Session,
    RawJson,
}

impl Default for IngestMode {
    fn default() -> Self {
        Self::Turn
    }
}

impl IngestMode {
    fn as_str(self) -> &'static str {
        match self {
            Self::Turn => "turn",
            Self::Session => "session",
            Self::RawJson => "raw-json",
        }
    }

    fn ingestion_granularity(self) -> &'static str {
        match self {
            Self::Turn => "turn",
            Self::Session | Self::RawJson => "session",
        }
    }

    fn image_policy(self) -> &'static str {
        match self {
            Self::RawJson => "raw_json_session_payload",
            Self::Turn | Self::Session => "blip_caption_inline",
        }
    }

    fn ingest_per_session(self) -> bool {
        !matches!(self, Self::Turn)
    }

    fn raw_json(self) -> bool {
        matches!(self, Self::RawJson)
    }
}

impl FromStr for IngestMode {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "turn" => Ok(Self::Turn),
            "session" => Ok(Self::Session),
            "raw-json" => Ok(Self::RawJson),
            other => Err(format!(
                "invalid --ingest value: {other} (expected one of: turn, session, raw-json)"
            )),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum ConsolidationMode {
    End,
    PerSession,
    Off,
}

impl Default for ConsolidationMode {
    fn default() -> Self {
        Self::End
    }
}

impl ConsolidationMode {
    fn as_str(self) -> &'static str {
        match self {
            Self::End => "end",
            Self::PerSession => "per-session",
            Self::Off => "off",
        }
    }

    fn enabled(self) -> bool {
        !matches!(self, Self::Off)
    }

    fn per_session(self) -> bool {
        matches!(self, Self::PerSession)
    }
}

impl FromStr for ConsolidationMode {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "end" => Ok(Self::End),
            "per-session" => Ok(Self::PerSession),
            "off" => Ok(Self::Off),
            other => Err(format!(
                "invalid --consolidation value: {other} (expected one of: end, per-session, off)"
            )),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct FileRunConfig {
    #[serde(default)]
    dataset: Option<PathBuf>,
    #[serde(default)]
    output: Option<PathBuf>,
    #[serde(default)]
    tag: Option<String>,
    #[serde(default)]
    conversations: Vec<String>,
    #[serde(default)]
    session_limit: Option<usize>,
    #[serde(default)]
    question_limit: Option<usize>,
    #[serde(default)]
    ingest: Option<IngestMode>,
    #[serde(default)]
    consolidation: Option<ConsolidationMode>,
    #[serde(default)]
    conversation_jobs: Option<usize>,
    #[serde(default)]
    question_jobs: Option<usize>,
    #[serde(default)]
    judge_model: Option<String>,
}

#[derive(Debug, Clone)]
struct RunConfig {
    profile: RunProfile,
    config_path: Option<PathBuf>,
    dataset: PathBuf,
    output: Option<PathBuf>,
    tag: Option<String>,
    conversations: Vec<String>,
    session_limit: Option<usize>,
    question_limit: Option<usize>,
    ingest: IngestMode,
    consolidation: ConsolidationMode,
    conversation_jobs: usize,
    question_jobs: usize,
    judge_model: Option<String>,
    allow_overwrite: bool,
}

impl Default for RunConfig {
    fn default() -> Self {
        Self {
            profile: RunProfile::Full,
            config_path: None,
            dataset: PathBuf::from("data/locomo10.json"),
            output: None,
            tag: None,
            conversations: Vec::new(),
            session_limit: None,
            question_limit: None,
            ingest: IngestMode::Session,
            consolidation: ConsolidationMode::End,
            conversation_jobs: 1,
            question_jobs: 1,
            judge_model: None,
            allow_overwrite: false,
        }
    }
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
        if !self.conversations.is_empty() {
            config.conversations = self.conversations;
        }
        if let Some(limit) = self.session_limit {
            config.session_limit = Some(limit);
        }
        if let Some(limit) = self.question_limit {
            config.question_limit = Some(limit);
        }
        if let Some(ingest) = self.ingest {
            config.ingest = ingest;
        }
        if let Some(consolidation) = self.consolidation {
            config.consolidation = consolidation;
        }
        if let Some(jobs) = self.conversation_jobs {
            config.conversation_jobs = jobs;
        }
        if let Some(jobs) = self.question_jobs {
            config.question_jobs = jobs;
        }
        if let Some(judge_model) = self.judge_model {
            config.judge_model = Some(judge_model);
        }
    }
}

#[derive(Debug, Default)]
struct CliOverrides {
    help: bool,
    profile: Option<RunProfile>,
    config_path: Option<PathBuf>,
    dataset: Option<PathBuf>,
    output: Option<PathBuf>,
    tag: Option<String>,
    conversations: Vec<String>,
    session_limit: Option<usize>,
    question_limit: Option<usize>,
    ingest: Option<IngestMode>,
    consolidation: Option<ConsolidationMode>,
    conversation_jobs: Option<usize>,
    question_jobs: Option<usize>,
    judge_model: Option<String>,
    allow_overwrite: bool,
}

#[derive(Debug, Clone)]
struct BenchInvocation {
    command: BenchCommand,
    artifact_path: Option<PathBuf>,
    merge_artifacts: Vec<PathBuf>,
    config: RunConfig,
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
        if !self.conversations.is_empty() {
            config.conversations = self.conversations;
        }
        if let Some(limit) = self.session_limit {
            config.session_limit = Some(limit);
        }
        if let Some(limit) = self.question_limit {
            config.question_limit = Some(limit);
        }
        if let Some(ingest) = self.ingest {
            config.ingest = ingest;
        }
        if let Some(consolidation) = self.consolidation {
            config.consolidation = consolidation;
        }
        if let Some(jobs) = self.conversation_jobs {
            config.conversation_jobs = jobs;
        }
        if let Some(jobs) = self.question_jobs {
            config.question_jobs = jobs;
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
    merge_artifacts: Vec<PathBuf>,
    overrides: CliOverrides,
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

fn parse_args_from(raw: &[String]) -> Result<Option<BenchInvocation>, String> {
    let ParsedCli {
        command,
        artifact_path,
        merge_artifacts,
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
        BenchCommand::Merge => resolve_merge_config(overrides)?,
    };
    validate_run_config(command, &config)?;
    Ok(Some(BenchInvocation {
        command,
        artifact_path,
        merge_artifacts,
        config,
    }))
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

fn resolve_merge_config(overrides: CliOverrides) -> Result<RunConfig, String> {
    validate_merge_overrides(&overrides)?;
    let mut config = RunConfig::default();
    if let Some(output) = overrides.output {
        config.output = Some(output);
    }
    if let Some(tag) = overrides.tag {
        config.tag = Some(tag);
    }
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
    if !overrides.conversations.is_empty() {
        config.conversations = overrides.conversations;
    }
    if let Some(jobs) = overrides.conversation_jobs {
        config.conversation_jobs = jobs;
    }
    if let Some(jobs) = overrides.question_jobs {
        config.question_jobs = jobs;
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
    if overrides.question_limit.is_some() {
        unsupported.push("--question-limit");
    }
    if overrides.ingest.is_some() {
        unsupported.push("--ingest");
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

fn validate_merge_overrides(overrides: &CliOverrides) -> Result<(), String> {
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
    if !overrides.conversations.is_empty() {
        unsupported.push("--conversation");
    }
    if overrides.session_limit.is_some() {
        unsupported.push("--session-limit");
    }
    if overrides.question_limit.is_some() {
        unsupported.push("--question-limit");
    }
    if overrides.ingest.is_some() {
        unsupported.push("--ingest");
    }
    if overrides.consolidation.is_some() {
        unsupported.push("--consolidation");
    }
    if overrides.conversation_jobs.is_some() {
        unsupported.push("--conversation-jobs");
    }
    if overrides.question_jobs.is_some() {
        unsupported.push("--question-jobs");
    }
    if overrides.judge_model.is_some() {
        unsupported.push("--judge-model");
    }

    if unsupported.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "`merge` does not accept {}; it only supports input artifacts plus --out/--tag",
            unsupported.join(", ")
        ))
    }
}

fn parse_cli_overrides(raw: &[String]) -> Result<ParsedCli, String> {
    let mut overrides = CliOverrides::default();
    let command = match raw.get(1).map(String::as_str) {
        None => return Err("expected subcommand: run, ingest, qa, or merge".into()),
        Some("--help") | Some("-h") => {
            overrides.help = true;
            return Ok(ParsedCli {
                command: BenchCommand::Run,
                artifact_path: None,
                merge_artifacts: Vec::new(),
                overrides,
            });
        }
        Some("run") => BenchCommand::Run,
        Some("ingest") => BenchCommand::Ingest,
        Some("qa") => BenchCommand::Qa,
        Some("merge") => BenchCommand::Merge,
        Some(other) => {
            return Err(format!(
                "unknown subcommand: {other} (expected one of: run, ingest, qa, merge)"
            ));
        }
    };

    let mut artifact_path = None;
    let mut merge_artifacts = Vec::new();
    let mut i = 2;
    if matches!(command, BenchCommand::Qa) {
        match raw.get(2).map(String::as_str) {
            None => return Err("expected artifact path after `qa`".into()),
            Some("--help") | Some("-h") => {
                overrides.help = true;
                return Ok(ParsedCli {
                    command,
                    artifact_path: None,
                    merge_artifacts,
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
            "--conversation" => {
                i += 1;
                overrides.conversations.push(
                    raw.get(i)
                        .ok_or_else(|| "--conversation requires a value".to_string())?
                        .clone(),
                );
            }
            "--session-limit" => {
                i += 1;
                overrides.session_limit = Some(parse_usize_arg(raw.get(i), "--session-limit")?);
            }
            "--question-limit" => {
                i += 1;
                overrides.question_limit = Some(parse_usize_arg(raw.get(i), "--question-limit")?);
            }
            "--ingest" => {
                i += 1;
                overrides.ingest = Some(
                    raw.get(i)
                        .ok_or_else(|| "--ingest requires a value".to_string())?
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
            "--conversation-jobs" => {
                i += 1;
                overrides.conversation_jobs =
                    Some(parse_usize_arg(raw.get(i), "--conversation-jobs")?);
            }
            "--question-jobs" => {
                i += 1;
                overrides.question_jobs = Some(parse_usize_arg(raw.get(i), "--question-jobs")?);
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
            value if matches!(command, BenchCommand::Merge) && !value.starts_with('-') => {
                merge_artifacts.push(PathBuf::from(value));
            }
            other => return Err(format!("Unknown argument: {other}")),
        }
        i += 1;
    }

    if matches!(command, BenchCommand::Merge) && merge_artifacts.len() < 2 {
        return Err("`merge` requires at least two input artifacts".into());
    }
    Ok(ParsedCli {
        command,
        artifact_path,
        merge_artifacts,
        overrides,
    })
}

fn parse_usize_arg(raw: Option<&String>, flag: &str) -> Result<usize, String> {
    raw.ok_or_else(|| format!("{flag} requires a value"))?
        .parse()
        .map_err(|_| format!("invalid numeric value for {flag}"))
}

fn load_json_config(path: &Path) -> Result<FileRunConfig, String> {
    let raw =
        fs::read_to_string(path).map_err(|e| format!("failed to read {}: {e}", path.display()))?;
    serde_json::from_str(&raw).map_err(|e| format!("failed to parse {}: {e}", path.display()))
}

fn load_benchmark_output(path: &Path) -> Result<BenchmarkOutput, String> {
    let raw =
        fs::read_to_string(path).map_err(|e| format!("failed to read {}: {e}", path.display()))?;
    serde_json::from_str(&raw).map_err(|e| format!("failed to parse {}: {e}", path.display()))
}

#[derive(Debug)]
struct LoadedArtifactBundle {
    path: PathBuf,
    fingerprint: String,
    output: BenchmarkOutput,
    questions: Vec<QuestionResult>,
    debug_records: Vec<QuestionDebugRecord>,
}

fn read_jsonl_records<T: DeserializeOwned>(path: &Path) -> Result<Vec<T>, String> {
    let raw =
        fs::read_to_string(path).map_err(|e| format!("failed to read {}: {e}", path.display()))?;
    raw.lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| {
            serde_json::from_str::<T>(line)
                .map_err(|e| format!("failed to parse record in {}: {e}", path.display()))
        })
        .collect()
}

fn artifact_relative_path(summary_path: &Path, rel: &str) -> PathBuf {
    summary_path
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join(rel)
}

fn load_artifact_bundle(path: &Path) -> Result<LoadedArtifactBundle, String> {
    let artifact_bytes =
        fs::read(path).map_err(|e| format!("failed to read {}: {e}", path.display()))?;
    let output: BenchmarkOutput = serde_json::from_slice(&artifact_bytes)
        .map_err(|e| format!("failed to parse {}: {e}", path.display()))?;

    let questions = if output.results.is_empty() {
        if output.artifacts.questions_path.is_empty() {
            return Err(format!(
                "artifact {} is missing question sidecars; merge requires new-style artifacts",
                path.display()
            ));
        }
        read_jsonl_records::<QuestionResult>(&artifact_relative_path(
            path,
            &output.artifacts.questions_path,
        ))?
    } else {
        output.results.clone()
    };

    if output.total_questions > 0 && output.total_questions != questions.len() {
        return Err(format!(
            "artifact {} summary says {} questions but loaded {} question records",
            path.display(),
            output.total_questions,
            questions.len()
        ));
    }

    let debug_records = if output.artifacts.debug_path.is_empty() {
        Vec::new()
    } else {
        let debug_path = artifact_relative_path(path, &output.artifacts.debug_path);
        if !debug_path.exists() {
            return Err(format!(
                "artifact {} is missing debug sidecar {}",
                path.display(),
                debug_path.display()
            ));
        }
        read_jsonl_records::<QuestionDebugRecord>(&debug_path)?
    };

    if !debug_records.is_empty() && debug_records.len() != questions.len() {
        return Err(format!(
            "artifact {} has {} question records but {} debug records",
            path.display(),
            questions.len(),
            debug_records.len()
        ));
    }

    Ok(LoadedArtifactBundle {
        path: path.to_path_buf(),
        fingerprint: format!("{:016x}", fnv1a64(&artifact_bytes)),
        output,
        questions,
        debug_records,
    })
}

fn ingest_mode_from_manifest(manifest: &BenchmarkManifest) -> Result<IngestMode, String> {
    if manifest.raw_json {
        return Ok(IngestMode::RawJson);
    }
    match manifest.ingestion_granularity.as_str() {
        "turn" => Ok(IngestMode::Turn),
        "session" => Ok(IngestMode::Session),
        "" => Err("artifact manifest is missing ingestion_granularity".into()),
        other => Err(format!(
            "unsupported artifact ingestion_granularity: {other}"
        )),
    }
}

fn run_config_from_artifact(artifact: &BenchmarkOutput) -> Result<RunConfig, String> {
    let manifest = &artifact.manifest;
    let profile = if manifest.profile.is_empty() || manifest.profile == "mixed" {
        RunProfile::Full
    } else {
        manifest.profile.parse()?
    };
    if manifest.dataset_path.is_empty() {
        return Err("artifact manifest is missing dataset_path".into());
    }

    let consolidation = if !manifest.consolidation_strategy.is_empty() {
        manifest.consolidation_strategy.parse()?
    } else if !artifact.consolidation_strategy.is_empty() {
        artifact.consolidation_strategy.parse()?
    } else {
        ConsolidationMode::End
    };

    Ok(RunConfig {
        profile,
        config_path: manifest.config_path.as_ref().map(PathBuf::from),
        dataset: PathBuf::from(&manifest.dataset_path),
        output: None,
        tag: None,
        conversations: manifest.selected_conversations.clone(),
        session_limit: manifest.session_limit,
        question_limit: manifest.question_limit,
        ingest: ingest_mode_from_manifest(manifest)?,
        consolidation,
        conversation_jobs: manifest.conversation_concurrency.max(1),
        question_jobs: manifest.question_concurrency.max(1),
        judge_model: None,
        allow_overwrite: false,
    })
}

fn merge_source_artifact(bundle: &LoadedArtifactBundle) -> SourceArtifact {
    SourceArtifact {
        path: bundle.path.display().to_string(),
        fingerprint: bundle.fingerprint.clone(),
        mode: bundle.output.manifest.mode.clone(),
        tag: bundle.output.tag.clone(),
        commit: bundle.output.commit.clone(),
    }
}

fn artifact_scope_ids(bundle: &LoadedArtifactBundle) -> BTreeSet<String> {
    bundle
        .output
        .per_conversation
        .keys()
        .cloned()
        .chain(bundle.output.banks.keys().cloned())
        .chain(
            bundle
                .questions
                .iter()
                .map(|result| result.sample_id.clone()),
        )
        .collect()
}

fn ensure_merge_compatible(
    base: &LoadedArtifactBundle,
    other: &LoadedArtifactBundle,
) -> Result<(), String> {
    let base_manifest = &base.output.manifest;
    let other_manifest = &other.output.manifest;

    macro_rules! ensure_same {
        ($left:expr, $right:expr, $label:literal) => {
            if $left != $right {
                return Err(format!(
                    "cannot merge {} and {}: mismatched {}",
                    base.path.display(),
                    other.path.display(),
                    $label
                ));
            }
        };
    }

    ensure_same!(base.output.benchmark, other.output.benchmark, "benchmark");
    ensure_same!(
        base.output.judge_model,
        other.output.judge_model,
        "judge model"
    );
    ensure_same!(
        base.output.retain_model,
        other.output.retain_model,
        "retain model"
    );
    ensure_same!(
        base.output.reflect_model,
        other.output.reflect_model,
        "reflect model"
    );
    ensure_same!(
        base.output.embedding_model,
        other.output.embedding_model,
        "embedding model"
    );
    ensure_same!(
        base.output.reranker_model,
        other.output.reranker_model,
        "reranker model"
    );
    ensure_same!(
        base.output.consolidation_strategy,
        other.output.consolidation_strategy,
        "consolidation strategy"
    );
    ensure_same!(
        base_manifest.protocol_version,
        other_manifest.protocol_version,
        "protocol version"
    );
    ensure_same!(base_manifest.mode, other_manifest.mode, "mode");
    ensure_same!(
        base_manifest.dataset_fingerprint,
        other_manifest.dataset_fingerprint,
        "dataset fingerprint"
    );
    ensure_same!(
        base_manifest.category_filter,
        other_manifest.category_filter,
        "category filter"
    );
    ensure_same!(
        base_manifest.image_policy,
        other_manifest.image_policy,
        "image policy"
    );
    ensure_same!(
        base_manifest.ingestion_granularity,
        other_manifest.ingestion_granularity,
        "ingestion granularity"
    );
    ensure_same!(
        base_manifest.consolidation_strategy,
        other_manifest.consolidation_strategy,
        "manifest consolidation strategy"
    );
    ensure_same!(
        base_manifest.session_limit,
        other_manifest.session_limit,
        "session limit"
    );
    ensure_same!(
        base_manifest.question_limit,
        other_manifest.question_limit,
        "question limit"
    );
    ensure_same!(
        base_manifest.raw_json,
        other_manifest.raw_json,
        "raw-json mode"
    );
    if serde_json::to_string(&base_manifest.prompt_hashes).ok()
        != serde_json::to_string(&other_manifest.prompt_hashes).ok()
    {
        return Err(format!(
            "cannot merge {} and {}: mismatched prompt hashes",
            base.path.display(),
            other.path.display()
        ));
    }
    if serde_json::to_string(&base_manifest.runtime_config).ok()
        != serde_json::to_string(&other_manifest.runtime_config).ok()
    {
        return Err(format!(
            "cannot merge {} and {}: mismatched runtime config",
            base.path.display(),
            other.path.display()
        ));
    }

    Ok(())
}

fn write_jsonl_records<T: Serialize>(path: &Path, values: &[T]) -> Result<(), String> {
    fs::write(path, "").map_err(|e| format!("failed to initialize {}: {e}", path.display()))?;
    for value in values {
        append_jsonl(path, value);
    }
    Ok(())
}

fn merge_profile_value(bundles: &[LoadedArtifactBundle]) -> String {
    let profiles = bundles
        .iter()
        .map(|bundle| bundle.output.manifest.profile.clone())
        .collect::<BTreeSet<_>>();
    if profiles.len() == 1 {
        profiles.into_iter().next().unwrap_or_default()
    } else {
        "mixed".into()
    }
}

fn merge_concurrency_value<F>(bundles: &[LoadedArtifactBundle], select: F) -> usize
where
    F: Fn(&LoadedArtifactBundle) -> usize,
{
    let values = bundles.iter().map(select).collect::<BTreeSet<_>>();
    if values.len() == 1 {
        values.into_iter().next().unwrap_or_default()
    } else {
        0
    }
}

fn warn_if_mixed<F>(bundles: &[LoadedArtifactBundle], label: &str, select: F)
where
    F: Fn(&LoadedArtifactBundle) -> String,
{
    let values = bundles.iter().map(select).collect::<BTreeSet<_>>();
    if values.len() > 1 {
        eprintln!(
            "merge note: mixed {label}: {}",
            values.into_iter().collect::<Vec<_>>().join(", ")
        );
    }
}

fn merge_artifacts(
    input_paths: &[PathBuf],
    output_path: &Path,
    tag: Option<String>,
) -> Result<(), String> {
    let bundles = input_paths
        .iter()
        .map(|path| load_artifact_bundle(path))
        .collect::<Result<Vec<_>, _>>()?;
    let base = bundles
        .first()
        .ok_or_else(|| "`merge` requires at least two input artifacts".to_string())?;

    for bundle in bundles.iter().skip(1) {
        ensure_merge_compatible(base, bundle)?;
    }

    warn_if_mixed(&bundles, "profiles", |bundle| {
        bundle.output.manifest.profile.clone()
    });
    warn_if_mixed(&bundles, "question concurrency", |bundle| {
        bundle.output.manifest.question_concurrency.to_string()
    });
    warn_if_mixed(&bundles, "conversation concurrency", |bundle| {
        bundle.output.manifest.conversation_concurrency.to_string()
    });
    warn_if_mixed(&bundles, "source commits", |bundle| {
        bundle
            .output
            .commit
            .clone()
            .unwrap_or_else(|| "<none>".into())
    });
    warn_if_mixed(&bundles, "source dirty-tree state", |bundle| {
        bundle
            .output
            .manifest
            .dirty_worktree
            .map(|dirty| dirty.to_string())
            .unwrap_or_else(|| "<unknown>".into())
    });

    let mut merged_results = Vec::new();
    let mut merged_debug = Vec::new();
    let mut banks = HashMap::new();
    let mut turn_refs = HashMap::new();
    let mut conversation_stage_metrics = HashMap::new();
    let mut conversation_cache_aware_stage_metrics = HashMap::new();
    let mut conversation_timings = HashMap::new();
    let mut conversation_bank_stats = HashMap::new();
    let metrics = MetricsCollector::new();
    let mut total_time_s = 0.0;
    let mut seen_conversations = BTreeSet::new();
    let mut seen_question_ids = BTreeSet::new();
    let mut seen_debug_ids = BTreeSet::new();

    for bundle in &bundles {
        let scope_ids = artifact_scope_ids(bundle);
        let overlaps = scope_ids
            .iter()
            .filter(|sample_id| seen_conversations.contains(*sample_id))
            .cloned()
            .collect::<Vec<_>>();
        if !overlaps.is_empty() {
            return Err(format!(
                "cannot merge {}: overlapping conversations: {}",
                bundle.path.display(),
                overlaps.join(", ")
            ));
        }
        seen_conversations.extend(scope_ids);

        metrics.extend_cache_aware_or_legacy_snapshot(
            &bundle.output.cache_aware_stage_metrics,
            &bundle.output.stage_metrics,
        );
        total_time_s += bundle.output.total_time_s;

        for (sample_id, bank_id) in &bundle.output.banks {
            if banks.insert(sample_id.clone(), bank_id.clone()).is_some() {
                return Err(format!(
                    "cannot merge {}: duplicate bank for conversation {}",
                    bundle.path.display(),
                    sample_id
                ));
            }
        }
        for (turn_id, turn_ref) in &bundle.output.turn_refs {
            if let Some(existing) = turn_refs.insert(turn_id.clone(), turn_ref.clone())
                && existing != *turn_ref
            {
                return Err(format!(
                    "cannot merge {}: conflicting turn ref for {}",
                    bundle.path.display(),
                    turn_id
                ));
            }
        }
        for (sample_id, summary) in &bundle.output.per_conversation {
            let (stage_metrics, cache_aware_stage_metrics) =
                hybrid_stage_snapshots(&summary.cache_aware_stage_metrics, &summary.stage_metrics);
            conversation_stage_metrics.insert(sample_id.clone(), stage_metrics);
            conversation_cache_aware_stage_metrics
                .insert(sample_id.clone(), cache_aware_stage_metrics);
            conversation_timings.insert(
                sample_id.clone(),
                ConversationPhaseTimings {
                    ingest_time_s: summary.ingest_time_s,
                    consolidation_time_s: summary.consolidation_time_s,
                    qa_time_s: summary.qa_time_s,
                    total_time_s: summary.total_time_s,
                },
            );
            conversation_bank_stats.insert(sample_id.clone(), summary.bank_stats.clone());
        }
        for result in &bundle.questions {
            if !seen_question_ids.insert(result.question_id.clone()) {
                return Err(format!(
                    "cannot merge {}: duplicate question id {}",
                    bundle.path.display(),
                    result.question_id
                ));
            }
            merged_results.push(result.clone());
        }
        for record in &bundle.debug_records {
            if !seen_debug_ids.insert(record.question_id.clone()) {
                return Err(format!(
                    "cannot merge {}: duplicate debug record for question {}",
                    bundle.path.display(),
                    record.question_id
                ));
            }
            merged_debug.push(record.clone());
        }
    }

    merged_results.sort_by(|a, b| {
        a.sample_id
            .cmp(&b.sample_id)
            .then(a.category_name.cmp(&b.category_name))
            .then(a.question_id.cmp(&b.question_id))
            .then(a.question.cmp(&b.question))
    });
    merged_debug.sort_by(|a, b| {
        a.sample_id
            .cmp(&b.sample_id)
            .then(a.question_id.cmp(&b.question_id))
            .then(a.question.cmp(&b.question))
    });

    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent)
            .map_err(|e| format!("failed to create {}: {e}", parent.display()))?;
    }
    let questions_path = sidecar_path(output_path, "questions");
    let debug_path = sidecar_path(output_path, "debug");
    write_jsonl_records(&questions_path, &merged_results)?;
    write_jsonl_records(&debug_path, &merged_debug)?;

    let mut manifest = base.output.manifest.clone();
    manifest.mode = BenchCommand::Merge.as_str().into();
    manifest.profile = merge_profile_value(&bundles);
    manifest.command = env::args().collect::<Vec<_>>().join(" ");
    manifest.selected_conversations = seen_conversations.into_iter().collect();
    manifest.question_concurrency = merge_concurrency_value(&bundles, |bundle| {
        bundle.output.manifest.question_concurrency
    });
    manifest.conversation_concurrency = merge_concurrency_value(&bundles, |bundle| {
        bundle.output.manifest.conversation_concurrency
    });
    manifest.dirty_worktree = git_dirty_worktree();
    manifest.source_artifact = None;
    manifest.source_artifacts = bundles.iter().map(merge_source_artifact).collect();

    let run_timestamp = Utc::now().to_rfc3339();
    let commit = git_commit_sha();
    flush_results(
        &merged_results,
        &banks,
        &conversation_stage_metrics,
        &conversation_cache_aware_stage_metrics,
        &conversation_timings,
        &conversation_bank_stats,
        &turn_refs,
        output_path,
        &questions_path,
        &debug_path,
        &base.output.judge_model,
        &tag,
        &base.output.retain_model,
        &base.output.reflect_model,
        &base.output.embedding_model,
        &base.output.reranker_model,
        &base.output.consolidation_strategy,
        &manifest,
        &metrics,
        &run_timestamp,
        commit.as_deref(),
        total_time_s,
    );

    println!(
        "Merged {} artifacts into {}",
        bundles.len(),
        output_path.display()
    );
    println!("Questions saved to {}", questions_path.display());
    println!("Debug saved to {}", debug_path.display());
    Ok(())
}

fn validate_run_config(command: BenchCommand, config: &RunConfig) -> Result<(), String> {
    if config.conversation_jobs == 0 {
        return Err("--conversation-jobs must be >= 1".into());
    }
    if config.question_jobs == 0 {
        return Err("--question-jobs must be >= 1".into());
    }
    if matches!(command, BenchCommand::Qa) && config.dataset.as_os_str().is_empty() {
        return Err("artifact-backed `qa` requires a dataset path".into());
    }
    Ok(())
}

fn print_help() {
    eprintln!("Usage:");
    eprintln!("  locomo-bench run [OPTIONS]");
    eprintln!("  locomo-bench ingest [OPTIONS]");
    eprintln!("  locomo-bench qa <RESULTS.json> [OPTIONS]");
    eprintln!("  locomo-bench merge <RESULTS.json> <RESULTS.json>... [--out PATH|--tag NAME]");
    eprintln!();
    eprintln!("Subcommands:");
    eprintln!("  run                              Fresh ingest, consolidate, then score QA");
    eprintln!("  ingest                           Ingest and consolidate only; do not run QA");
    eprintln!(
        "  qa <RESULTS.json>                Score QA against existing banks; skip ingest and consolidation"
    );
    eprintln!(
        "  merge <RESULTS.json>...          Combine compatible subset artifacts into one canonical result"
    );
    eprintln!();
    eprintln!("Options:");
    eprintln!("  --profile <NAME>                Named profile for `run`/`ingest` [default: full]");
    eprintln!("                                  Profiles: full, smoke, legacy-raw");
    eprintln!("  --config <PATH>                 JSON config file for `run`/`ingest`");
    eprintln!(
        "  --dataset <PATH>                Dataset path for `run`/`ingest` [default: data/locomo10.json]"
    );
    eprintln!("  --tag <NAME>                    Save to results/local/<tag>.json by default");
    eprintln!("  --out <PATH>                    Output results path (overrides --tag)");
    eprintln!("  --conversation <ID>             Run one specific conversation (repeatable)");
    eprintln!("  --ingest <MODE>                 turn | session | raw-json (`run`/`ingest` only)");
    eprintln!("  --consolidation <MODE>          end | per-session | off (`run`/`ingest` only)");
    eprintln!("  --conversation-jobs <N>         Parallel conversations");
    eprintln!("  --question-jobs <N>             Parallel questions per conversation");
    eprintln!("  --judge-model <MODEL>           Override judge model");
    eprintln!("  --force                         Allow overwriting existing output files");
    eprintln!();
    eprintln!("Debug slice options:");
    eprintln!(
        "  --session-limit <N>             Limit sessions per conversation (`run`/`ingest` only)"
    );
    eprintln!(
        "  --question-limit <N>            Limit questions per conversation (`run`/`ingest` only)"
    );
}

// --- Shared state for incremental writes ---

struct SharedResults {
    results: Vec<QuestionResult>,
    banks: HashMap<String, String>,
    conversation_stage_metrics: HashMap<String, BTreeMap<LlmStage, StageUsage>>,
    conversation_cache_aware_stage_metrics:
        HashMap<String, BTreeMap<LlmStage, CacheAwareStageUsage>>,
    conversation_timings: HashMap<String, ConversationPhaseTimings>,
    conversation_bank_stats: HashMap<String, ConversationBankStats>,
    turn_refs: HashMap<String, String>,
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
}

impl SharedResults {
    fn push_and_flush(
        &mut self,
        result: QuestionResult,
        debug: QuestionDebugRecord,
        sample_id: String,
        stage_metrics: BTreeMap<LlmStage, StageUsage>,
        cache_aware_stage_metrics: BTreeMap<LlmStage, CacheAwareStageUsage>,
    ) {
        append_jsonl(&self.questions_path, &result);
        append_jsonl(&self.debug_path, &debug);
        self.results.push(result);
        self.conversation_stage_metrics
            .insert(sample_id.clone(), stage_metrics);
        self.conversation_cache_aware_stage_metrics
            .insert(sample_id, cache_aware_stage_metrics);
        self.flush();
    }

    fn record_bank(&mut self, sample_id: String, bank_id: String) {
        self.banks.insert(sample_id, bank_id);
        self.flush();
    }

    fn record_conversation_metrics(
        &mut self,
        sample_id: String,
        stage_metrics: BTreeMap<LlmStage, StageUsage>,
        cache_aware_stage_metrics: BTreeMap<LlmStage, CacheAwareStageUsage>,
    ) {
        self.conversation_stage_metrics
            .insert(sample_id.clone(), stage_metrics);
        self.conversation_cache_aware_stage_metrics
            .insert(sample_id, cache_aware_stage_metrics);
        self.flush();
    }

    fn record_conversation_timings(
        &mut self,
        sample_id: String,
        timings: ConversationPhaseTimings,
    ) {
        self.conversation_timings.insert(sample_id, timings);
        self.flush();
    }

    fn record_conversation_bank_stats(&mut self, sample_id: String, stats: ConversationBankStats) {
        self.conversation_bank_stats.insert(sample_id, stats);
        self.flush();
    }

    fn record_turn_ref(&mut self, turn_id: TurnId, turn_ref: String) {
        self.turn_refs.insert(turn_id.to_string(), turn_ref);
    }

    fn flush(&self) {
        flush_results(
            &self.results,
            &self.banks,
            &self.conversation_stage_metrics,
            &self.conversation_cache_aware_stage_metrics,
            &self.conversation_timings,
            &self.conversation_bank_stats,
            &self.turn_refs,
            &self.output_path,
            &self.questions_path,
            &self.debug_path,
            &self.judge_label,
            &self.tag,
            &self.retain_model,
            &self.reflect_model,
            &self.embedding_model,
            &self.reranker_model,
            &self.consolidation_strategy,
            &self.manifest,
            &self.metrics,
            &self.run_timestamp,
            self.commit.as_deref(),
            self.bench_start.elapsed().as_secs_f64(),
        );
    }
}

// --- Per-conversation worker ---

#[derive(Debug, Clone)]
struct ConversationRunOptions {
    max_sessions: Option<usize>,
    max_questions: Option<usize>,
    question_concurrency: usize,
    ingest_mode: IngestMode,
    consolidation: ConsolidationMode,
    seed_stage_metrics: BTreeMap<LlmStage, StageUsage>,
    seed_cache_aware_stage_metrics: BTreeMap<LlmStage, CacheAwareStageUsage>,
    seed_timings: ConversationPhaseTimings,
    seed_bank_stats: ConversationBankStats,
    existing_bank: Option<String>,
    require_existing_bank: bool,
    ingest_only: bool,
}

fn question_id_for(sample_id: &str, question: &str) -> String {
    let mut h = DefaultHasher::new();
    sample_id.hash(&mut h);
    question.hash(&mut h);
    format!("{:06x}", h.finish() & 0xFFFFFF)
}

async fn count_unconsolidated_facts(
    runtime: &ElephantRuntime,
    bank_id: BankId,
) -> Result<usize, String> {
    runtime
        .store
        .get_facts_by_bank(
            bank_id,
            elephant::types::FactFilter {
                network: Some(vec![NetworkType::World, NetworkType::Experience]),
                unconsolidated_only: true,
                ..Default::default()
            },
        )
        .await
        .map(|facts| facts.len())
        .map_err(|e| format!("failed to count unconsolidated facts: {e}"))
}

fn should_log_consolidation_progress(progress: &ConsolidationProgress) -> bool {
    progress.batch_index == 1
        || progress.batch_index == progress.total_batches
        || progress.batch_index % 10 == 0
}

async fn consolidate_with_bench_progress(
    tag: &str,
    runtime: Arc<ElephantRuntime>,
    bank_id: BankId,
    conversation_metrics: Arc<MetricsCollector>,
) -> Result<elephant::types::ConsolidationReport, String> {
    let total_facts = count_unconsolidated_facts(runtime.as_ref(), bank_id).await?;
    let total_batches = if total_facts == 0 {
        0
    } else {
        total_facts.div_ceil(observation::batch_size())
    };
    println!(
        "[{tag}] Consolidating {total_facts} fact{} in {total_batches} batch{}...",
        if total_facts == 1 { "" } else { "s" },
        if total_batches == 1 { "" } else { "es" },
    );

    let started = Instant::now();
    let (tx, mut rx) = mpsc::unbounded_channel();
    let consolidator = runtime.consolidator.clone();
    let task = tokio::spawn(async move {
        with_scoped_collector(
            conversation_metrics,
            consolidator.consolidate_with_progress(bank_id, Some(tx)),
        )
        .await
    });

    while let Some(progress) = rx.recv().await {
        if should_log_consolidation_progress(&progress) {
            println!(
                "[{tag}]   consolidate [{}/{}] | {} facts | {} created | {} updated | elapsed: {}",
                progress.batch_index,
                progress.total_batches,
                progress.batch_facts,
                progress.observations_created,
                progress.observations_updated,
                fmt_elapsed(started.elapsed().as_secs_f64()),
            );
        }
    }

    task.await
        .map_err(|e| format!("[{tag}] consolidate task failed: {e}"))?
        .map_err(|e| format!("[{tag}] consolidate failed: {e}"))
}

async fn run_conversation(
    tag: String,
    runtime: Arc<ElephantRuntime>,
    entry: LocomoEntry,
    judge: Arc<dyn LlmClient>,
    options: ConversationRunOptions,
    shared: Arc<Mutex<SharedResults>>,
) -> Result<(), String> {
    let conv = &entry.conversation;
    let sample_id = entry.sample_id.clone();
    let total_sessions = session_count(conv);
    let conversation_metrics = Arc::new(MetricsCollector::new());
    conversation_metrics.extend_cache_aware_or_legacy_snapshot(
        &options.seed_cache_aware_stage_metrics,
        &options.seed_stage_metrics,
    );
    let mut ingest_time_s = options.seed_timings.ingest_time_s;
    let mut consolidation_time_s = options.seed_timings.consolidation_time_s;
    let mut bank_stats = options.seed_bank_stats.clone();

    println!(
        "[{tag}] {} & {} ({})",
        conv.speaker_a, conv.speaker_b, sample_id
    );

    let reused_bank = options.existing_bank.clone();
    let bank_id = if let Some(id) = reused_bank {
        println!("[{tag}] Using existing bank: {id} (skipping ingestion)");
        BankId::from_str(&id).map_err(|e| format!("[{tag}] invalid bank id: {e}"))?
    } else {
        if options.require_existing_bank {
            return Err(format!(
                "[{tag}] missing bank for {} in the source artifact",
                sample_id
            ));
        }

        let bank = MemoryBank {
            id: BankId::new(),
            name: format!("locomo-{sample_id}"),
            mission: "Long-term conversational memory benchmark".into(),
            directives: vec![],
            disposition: Disposition::default(),
            embedding_model: runtime.embeddings.model_name().to_string(),
            embedding_dimensions: runtime.embeddings.dimensions() as u16,
        };
        runtime
            .store
            .create_bank(&bank)
            .await
            .map_err(|e| format!("[{tag}] failed to create bank: {e}"))?;

        let ingest_sessions = options
            .max_sessions
            .map(|m| m.min(total_sessions))
            .unwrap_or(total_sessions);
        let selected_turns = (1..=ingest_sessions)
            .map(|idx| get_session_turns(conv, idx).len())
            .sum::<usize>();
        println!(
            "[{tag}] Bank: {} | Ingesting {ingest_sessions} session{} / {selected_turns} turn{} ({total_sessions} sessions available)...",
            bank.id,
            if ingest_sessions == 1 { "" } else { "s" },
            if selected_turns == 1 { "" } else { "s" }
        );

        let ingest_start = Instant::now();
        let mut stored_facts = 0usize;

        for idx in 1..=ingest_sessions {
            let turns = get_session_turns(conv, idx);
            let session_turns = turns.len();
            let date_str = get_session_date(conv, idx);
            let timestamp = parse_session_date(&date_str);
            let session_start = Instant::now();
            let mut session_facts = 0usize;
            let mut session_failures = 0usize;
            let mut session_failed = false;
            bank_stats.sessions_ingested += 1;
            bank_stats.turns_ingested += session_turns;

            if options.ingest_mode.ingest_per_session() {
                let content = if options.ingest_mode.raw_json() {
                    format_session_raw(conv, idx)
                } else {
                    format_session(&turns, &date_str)
                };
                match with_scoped_collector(
                    conversation_metrics.clone(),
                    runtime.retain.retain(&RetainInput {
                        bank_id: bank.id,
                        content,
                        timestamp,
                        turn_id: None,
                        context: None,
                        custom_instructions: None,
                        speaker: None,
                    }),
                )
                .await
                {
                    Ok(resp) => {
                        stored_facts += resp.facts_stored;
                        session_facts += resp.facts_stored;
                        bank_stats.facts_stored += resp.facts_stored;
                        bank_stats.entities_resolved += resp.entities_resolved;
                        bank_stats.links_created += resp.links_created;
                        bank_stats.opinions_reinforced += resp.opinions_reinforced;
                        bank_stats.opinions_weakened += resp.opinions_weakened;
                    }
                    Err(e) => {
                        eprintln!("[{tag}] ingest [{idx}/{ingest_sessions}] FAILED: {e}");
                        session_failures += 1;
                        session_failed = true;
                    }
                }
            } else {
                let mut prior_turns: Vec<String> = Vec::new();
                for (turn_idx, turn) in turns.iter().enumerate() {
                    let turn_text = format_turn(turn);
                    let turn_id = TurnId::new();
                    let turn_ref = turn
                        .dia_id
                        .clone()
                        .unwrap_or_else(|| format!("D{idx}:{}", turn_idx + 1));

                    let resp = with_scoped_collector(
                        conversation_metrics.clone(),
                        runtime.retain.retain(&RetainInput {
                            bank_id: bank.id,
                            content: turn_text.clone(),
                            timestamp,
                            turn_id: Some(turn_id),
                            context: if prior_turns.is_empty() {
                                None
                            } else {
                                Some(prior_turns.join("\n"))
                            },
                            custom_instructions: None,
                            speaker: Some(turn.speaker.clone()),
                        }),
                    )
                    .await;

                    match resp {
                        Ok(retain_resp) => {
                            stored_facts += retain_resp.facts_stored;
                            session_facts += retain_resp.facts_stored;
                            bank_stats.facts_stored += retain_resp.facts_stored;
                            bank_stats.entities_resolved += retain_resp.entities_resolved;
                            bank_stats.links_created += retain_resp.links_created;
                            bank_stats.opinions_reinforced += retain_resp.opinions_reinforced;
                            bank_stats.opinions_weakened += retain_resp.opinions_weakened;
                            shared.lock().await.record_turn_ref(turn_id, turn_ref);
                            prior_turns.push(turn_text);
                        }
                        Err(e) => {
                            eprintln!(
                                "[{tag}] ingest [{idx}/{} turn {}] FAILED: {e}",
                                ingest_sessions,
                                turn_idx + 1
                            );
                            session_failures += 1;
                        }
                    }
                }
            }
            let session_elapsed = session_start.elapsed().as_secs_f64();
            let ingest_elapsed = ingest_start.elapsed().as_secs_f64();
            let mode_label = if options.ingest_mode.ingest_per_session() {
                if options.ingest_mode.raw_json() {
                    "raw-json"
                } else {
                    "session"
                }
            } else {
                "turn"
            };
            let failure_suffix = if session_failures == 0 {
                String::new()
            } else {
                format!(
                    " | {} failure{}",
                    session_failures,
                    if session_failures == 1 { "" } else { "s" }
                )
            };
            println!(
                "[{tag}] ingest [{idx}/{ingest_sessions}] {mode_label}-level complete | {session_turns} turns | {session_facts} facts | session: {} | total: {}{}",
                fmt_elapsed(session_elapsed),
                fmt_elapsed(ingest_elapsed),
                failure_suffix,
            );

            if session_failed {
                continue;
            }

            if options.consolidation.per_session() {
                let t0 = Instant::now();
                match consolidate_with_bench_progress(
                    &tag,
                    runtime.clone(),
                    bank.id,
                    conversation_metrics.clone(),
                )
                .await
                {
                    Ok(cr) => {
                        bank_stats.observations_created += cr.observations_created;
                        bank_stats.observations_updated += cr.observations_updated;
                    }
                    Err(e) => {
                        eprintln!("{e}");
                    }
                }
                consolidation_time_s += t0.elapsed().as_secs_f64();
            }
        }

        println!(
            "[{tag}] Ingestion complete: {stored_facts} facts in {}",
            fmt_elapsed(ingest_start.elapsed().as_secs_f64())
        );
        ingest_time_s = ingest_start.elapsed().as_secs_f64();

        bank.id
    };

    shared
        .lock()
        .await
        .record_bank(sample_id.clone(), bank_id.to_string());

    if options.consolidation.enabled()
        && !options.consolidation.per_session()
        && options.existing_bank.is_none()
    {
        let t0 = Instant::now();
        match consolidate_with_bench_progress(
            &tag,
            runtime.clone(),
            bank_id,
            conversation_metrics.clone(),
        )
        .await
        {
            Ok(resp) => {
                let elapsed = t0.elapsed().as_secs_f64();
                bank_stats.observations_created += resp.observations_created;
                bank_stats.observations_updated += resp.observations_updated;
                println!(
                    "[{tag}] Consolidation done in {}: {} observations created, {} updated",
                    fmt_elapsed(elapsed),
                    resp.observations_created,
                    resp.observations_updated,
                );
            }
            Err(e) => {
                eprintln!("{e}");
            }
        }
        consolidation_time_s += t0.elapsed().as_secs_f64();
    }

    shared.lock().await.record_conversation_metrics(
        sample_id.clone(),
        conversation_metrics.snapshot(),
        conversation_metrics.cache_aware_snapshot(),
    );
    finalize_bank_stats(&runtime.store, bank_id, &mut bank_stats).await?;
    shared
        .lock()
        .await
        .record_conversation_bank_stats(sample_id.clone(), bank_stats.clone());

    if options.ingest_only {
        shared.lock().await.record_conversation_timings(
            sample_id,
            ConversationPhaseTimings {
                ingest_time_s,
                consolidation_time_s,
                qa_time_s: 0.0,
                total_time_s: ingest_time_s + consolidation_time_s,
            },
        );
        return Ok(());
    }

    let qa_list: &[QaPair] = if let Some(max) = options.max_questions {
        &entry.qa[..max.min(entry.qa.len())]
    } else {
        &entry.qa
    };

    let scored_questions = qa_list
        .iter()
        .filter(|qa| should_score_question(qa))
        .count();
    println!(
        "[{tag}] Asking {scored_questions} questions (concurrency: {})...",
        options.question_concurrency
    );
    let qa_start = Instant::now();
    let local_correct = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let local_total = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let completed = Arc::new(std::sync::atomic::AtomicUsize::new(0));

    let turn_refs = Arc::new(shared.lock().await.turn_refs.clone());
    let qa_sem = Arc::new(Semaphore::new(options.question_concurrency));
    let mut qa_handles = Vec::new();

    for qa in qa_list.iter().filter(|qa| should_score_question(qa)) {
        let sem = qa_sem.clone();
        let runtime = runtime.clone();
        let judge = judge.clone();
        let shared = shared.clone();
        let tag = tag.clone();
        let sample_id = sample_id.clone();
        let question = qa.question.clone();
        let gold = answer_to_string(
            qa.answer
                .as_ref()
                .expect("filtered questions always have an answer"),
        );
        let category = qa.category;
        let qa_len = scored_questions;
        let local_correct = local_correct.clone();
        let local_total = local_total.clone();
        let completed = completed.clone();
        let evidence_refs = qa.evidence.clone();
        let turn_refs = turn_refs.clone();
        let bank_id = bank_id;
        let conversation_metrics = conversation_metrics.clone();
        let question_metrics = Arc::new(MetricsCollector::new());

        qa_handles.push(tokio::spawn(async move {
            let _permit = sem.acquire().await.expect("semaphore closed");
            let cat_name = category_name(category);
            let question_started = Instant::now();

            let question_metrics_for_scope = question_metrics.clone();
            let conversation_metrics_for_scope = conversation_metrics.clone();
            let (
                hypothesis,
                _reflect_confidence,
                retrieved_context,
                retrieved_sources,
                final_source_ids,
                reflect_trace,
                final_done,
                status,
                error,
                _reflect_elapsed,
            ) = with_scoped_collector(
                conversation_metrics_for_scope,
                with_scoped_collector(question_metrics_for_scope, async {
                    let t0 = Instant::now();
                    let reflect_result = runtime
                        .reflect
                        .reflect(&ReflectQuery {
                            bank_id,
                            question: question.clone(),
                            budget_tokens: REFLECT_BUDGET_TOKENS,
                            temporal_context: None,
                        })
                        .await;
                    let elapsed = t0.elapsed().as_secs_f64();

                    match reflect_result {
                        Ok(resp) => {
                            let retrieved_context = resp
                                .retrieved_context
                                .into_iter()
                                .map(|fact| {
                                    let source_turn_id =
                                        fact.source_turn_id.map(|id| id.to_string());
                                    let source_turn_ref = source_turn_id
                                        .as_ref()
                                        .and_then(|id| turn_refs.get(id).cloned());
                                    let support_turn_ids = fact
                                        .support_turn_ids
                                        .iter()
                                        .map(ToString::to_string)
                                        .collect::<Vec<_>>();
                                    let support_turn_refs = support_turn_ids
                                        .iter()
                                        .filter_map(|id| turn_refs.get(id).cloned())
                                        .collect::<Vec<_>>();
                                    RetrievedFactEntry {
                                        id: fact.id.to_string(),
                                        content: fact.content,
                                        score: fact.score,
                                        network: network_name(fact.network).to_string(),
                                        source_turn_id,
                                        source_turn_ref,
                                        evidence_ids: fact
                                            .evidence_ids
                                            .into_iter()
                                            .map(|id| id.to_string())
                                            .collect(),
                                        retrieval_sources: fact
                                            .retrieval_sources
                                            .into_iter()
                                            .map(|source| retrieval_source_name(source).to_string())
                                            .collect(),
                                        support_turn_ids,
                                        support_turn_refs,
                                    }
                                })
                                .collect::<Vec<_>>();
                            let retrieved_sources = resp
                                .retrieved_sources
                                .into_iter()
                                .map(|source| RetrievedSourceEntry {
                                    id: source.id.to_string(),
                                    fact_id: source.fact_id.to_string(),
                                    timestamp: source.timestamp.to_rfc3339(),
                                    content: source.content,
                                    truncated: source.truncated,
                                })
                                .collect::<Vec<_>>();
                            let final_source_ids = resp
                                .sources
                                .into_iter()
                                .map(|id| id.to_string())
                                .collect::<Vec<_>>();
                            let reflect_trace = resp
                                .trace
                                .into_iter()
                                .map(|step| ReflectTraceEntry {
                                    iteration: step.iteration,
                                    tool_name: step.tool_name,
                                    query: step.query,
                                    returned_fact_ids: step
                                        .returned_fact_ids
                                        .into_iter()
                                        .map(|id| id.to_string())
                                        .collect(),
                                    requested_fact_ids: step
                                        .requested_fact_ids
                                        .into_iter()
                                        .map(|id| id.to_string())
                                        .collect(),
                                    new_fact_ids: step
                                        .new_fact_ids
                                        .into_iter()
                                        .map(|id| id.to_string())
                                        .collect(),
                                    returned_source_ids: step
                                        .returned_source_ids
                                        .into_iter()
                                        .map(|id| id.to_string())
                                        .collect(),
                                    facts_returned: step.facts_returned,
                                    total_tokens: step.total_tokens,
                                    latency_ms: step.latency_ms,
                                })
                                .collect::<Vec<_>>();
                            (
                                resp.response,
                                resp.confidence,
                                retrieved_context,
                                retrieved_sources,
                                final_source_ids,
                                reflect_trace,
                                resp.final_done,
                                "ok".to_string(),
                                None,
                                elapsed,
                            )
                        }
                        Err(e) => (
                            String::new(),
                            0.0,
                            Vec::new(),
                            Vec::new(),
                            Vec::new(),
                            Vec::new(),
                            None,
                            "reflect_error".to_string(),
                            Some(e.to_string()),
                            elapsed,
                        ),
                    }
                }),
            )
            .await;

            let f1 = token_f1(&hypothesis, &gold);
            let retrieved_turn_refs = retrieved_context
                .iter()
                .flat_map(|fact| {
                    fact.source_turn_ref
                        .iter()
                        .cloned()
                        .chain(fact.support_turn_refs.iter().cloned())
                })
                .collect::<BTreeSet<_>>()
                .into_iter()
                .collect::<Vec<_>>();
            let (evidence_hit, evidence_recall) =
                evidence_recall(&evidence_refs, &retrieved_turn_refs);

            let (judge_correct, judge_reasoning, status, error) = if hypothesis.is_empty() {
                (
                    false,
                    error.clone().unwrap_or_else(|| "empty response".into()),
                    status,
                    error,
                )
            } else {
                match with_scoped_collector(
                    conversation_metrics.clone(),
                    with_scoped_collector(
                        question_metrics.clone(),
                        llm_judge(judge.as_ref(), &question, &gold, &hypothesis),
                    ),
                )
                .await
                {
                    Ok((correct, reasoning)) => (correct, reasoning, status, error),
                    Err(judge_error) => (
                        false,
                        judge_error.clone(),
                        "judge_error".into(),
                        Some(judge_error),
                    ),
                }
            };
            let elapsed = question_started.elapsed().as_secs_f64();

            local_total.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if judge_correct {
                local_correct.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
            let done = completed.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;

            let result = QuestionResult {
                question_id: question_id_for(&sample_id, &question),
                sample_id: sample_id.clone(),
                question,
                ground_truth: gold,
                hypothesis,
                category_name: cat_name.into(),
                f1,
                judge_correct,
                judge_reasoning,
                elapsed_s: elapsed,
                status,
                error,
                final_source_ids,
                evidence_refs,
                retrieved_turn_refs,
                evidence_hit,
                evidence_recall,
                qa_stage_metrics: question_metrics.snapshot(),
                cache_aware_qa_stage_metrics: question_metrics.cache_aware_snapshot(),
            };
            let debug = QuestionDebugRecord {
                question_id: result.question_id.clone(),
                sample_id: result.sample_id.clone(),
                question: result.question.clone(),
                reflect_trace,
                final_done,
                retrieved_context,
                retrieved_sources,
            };

            shared.lock().await.push_and_flush(
                result,
                debug,
                sample_id.clone(),
                conversation_metrics.snapshot(),
                conversation_metrics.cache_aware_snapshot(),
            );

            let total = local_total.load(std::sync::atomic::Ordering::Relaxed);
            let correct = local_correct.load(std::sync::atomic::Ordering::Relaxed);
            let running_acc = if total > 0 {
                correct as f64 / total as f64
            } else {
                0.0
            };
            let qa_elapsed = qa_start.elapsed().as_secs_f64();
            let avg_time = if done > 0 {
                qa_elapsed / done as f64
            } else {
                0.0
            };
            let remaining = qa_len.saturating_sub(done);
            let eta = avg_time * remaining as f64;

            let label = if judge_correct { "CORRECT" } else { "WRONG  " };
            println!(
                "[{tag}] [{done}/{}] {label} F1={f1:.2} ER={:.2} ({cat_name}) | acc: {:.1}% | elapsed: {} | ETA: {}",
                qa_len,
                evidence_recall,
                running_acc * 100.0,
                fmt_elapsed(qa_elapsed),
                fmt_elapsed(eta),
            );
        }));
    }

    for handle in qa_handles {
        if let Err(e) = handle.await {
            eprintln!("[{tag}] question task failed: {e}");
        }
    }

    let qa_time_s = qa_start.elapsed().as_secs_f64();

    shared.lock().await.record_conversation_metrics(
        sample_id.clone(),
        conversation_metrics.snapshot(),
        conversation_metrics.cache_aware_snapshot(),
    );
    shared
        .lock()
        .await
        .record_conversation_bank_stats(sample_id.clone(), bank_stats);
    shared.lock().await.record_conversation_timings(
        sample_id,
        ConversationPhaseTimings {
            ingest_time_s,
            consolidation_time_s,
            qa_time_s,
            total_time_s: ingest_time_s + consolidation_time_s + qa_time_s,
        },
    );

    Ok(())
}

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

fn normalize_status_path(path: &str) -> &str {
    path.trim().trim_matches('"')
}

fn is_generated_bench_artifact_path(path: &str) -> bool {
    normalize_status_path(path).starts_with("bench/locomo/results/")
}

fn status_line_is_ignored(line: &str) -> bool {
    if line.len() < 4 {
        return false;
    }
    let payload = &line[3..];
    if let Some((from, to)) = payload.split_once(" -> ") {
        is_generated_bench_artifact_path(from) && is_generated_bench_artifact_path(to)
    } else {
        is_generated_bench_artifact_path(payload)
    }
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
    Some(
        stdout
            .lines()
            .any(|line| !line.trim().is_empty() && !status_line_is_ignored(line)),
    )
}

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
    merge_inputs: &[PathBuf],
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

    if matches!(command, BenchCommand::Merge)
        && merge_inputs
            .iter()
            .any(|input| same_path(output_path, input))
    {
        return Err(format!(
            "refusing to overwrite merge input {}; choose a different --out/--tag or pass --force",
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
                PathBuf::from(format!("bench/locomo/results/local/{tag}.json"))
            } else {
                artifact_path
                    .map(Path::to_path_buf)
                    .unwrap_or_else(|| PathBuf::from("bench/locomo/results/local/qa.json"))
            }
        }
        BenchCommand::Merge => {
            let stem = config.tag.as_deref().unwrap_or("merged");
            PathBuf::from(format!("bench/locomo/results/local/{stem}.json"))
        }
        BenchCommand::Run | BenchCommand::Ingest => {
            let stem = if let Some(ref tag) = config.tag {
                tag.clone()
            } else {
                format!("{}-{}", config.profile.as_str(), command.as_str())
            };
            PathBuf::from(format!("bench/locomo/results/local/{stem}.json"))
        }
    }
}

// --- Main ---

#[tokio::main]
async fn main() {
    let _ = dotenvy::dotenv();

    let invocation = parse_args();
    let command = invocation.command;
    let artifact_path = invocation.artifact_path.clone();
    let merge_inputs = invocation.merge_artifacts.clone();
    let config = invocation.config;
    let output_path = default_output_path(command, &config, artifact_path.as_deref());

    if let Err(err) = ensure_output_paths_are_safe(
        command,
        &output_path,
        artifact_path.as_deref(),
        &merge_inputs,
        config.allow_overwrite,
    ) {
        eprintln!("{err}");
        std::process::exit(1);
    }

    if matches!(command, BenchCommand::Merge) {
        if let Err(err) = merge_artifacts(&merge_inputs, &output_path, config.tag.clone()) {
            eprintln!("{err}");
            std::process::exit(1);
        }
        return;
    }

    let artifact_state = artifact_path.as_ref().map(|path| {
        let artifact_bytes = fs::read(path).unwrap_or_else(|e| {
            eprintln!("failed to read {}: {e}", path.display());
            std::process::exit(1);
        });
        let prev = load_benchmark_output(path).unwrap_or_else(|e| {
            eprintln!("{e}");
            std::process::exit(1);
        });
        println!(
            "Using artifact {} ({} banks, {} turn refs)",
            path.display(),
            prev.banks.len(),
            prev.turn_refs.len()
        );
        let source_artifact = SourceArtifact {
            path: path.display().to_string(),
            fingerprint: format!("{:016x}", fnv1a64(&artifact_bytes)),
            mode: prev.manifest.mode.clone(),
            tag: prev.tag.clone(),
            commit: prev.commit.clone(),
        };
        (prev, source_artifact)
    });
    let (
        existing_banks,
        existing_turn_refs,
        existing_stage_metrics,
        existing_cache_aware_stage_metrics,
        existing_conversation_metrics,
        existing_conversation_cache_aware_metrics,
        existing_conversation_timings,
        existing_conversation_bank_stats,
        source_artifact,
    ) = if let Some((prev, source_artifact)) = artifact_state.as_ref() {
        let (existing_stage_metrics, existing_cache_aware_stage_metrics) =
            seeded_hybrid_stage_snapshots(
                command,
                &prev.cache_aware_stage_metrics,
                &prev.stage_metrics,
            );
        (
            prev.banks.clone(),
            prev.turn_refs.clone(),
            existing_stage_metrics,
            existing_cache_aware_stage_metrics,
            prev.per_conversation
                .iter()
                .map(|(sample_id, summary)| {
                    let (stage_metrics, _) = seeded_hybrid_stage_snapshots(
                        command,
                        &summary.cache_aware_stage_metrics,
                        &summary.stage_metrics,
                    );
                    (sample_id.clone(), stage_metrics)
                })
                .collect(),
            prev.per_conversation
                .iter()
                .map(|(sample_id, summary)| {
                    let (_, cache_aware_stage_metrics) = seeded_hybrid_stage_snapshots(
                        command,
                        &summary.cache_aware_stage_metrics,
                        &summary.stage_metrics,
                    );
                    (sample_id.clone(), cache_aware_stage_metrics)
                })
                .collect(),
            prev.per_conversation
                .iter()
                .map(|(sample_id, summary)| {
                    (
                        sample_id.clone(),
                        ConversationPhaseTimings {
                            ingest_time_s: summary.ingest_time_s,
                            consolidation_time_s: summary.consolidation_time_s,
                            qa_time_s: 0.0,
                            total_time_s: summary.ingest_time_s + summary.consolidation_time_s,
                        },
                    )
                })
                .collect(),
            prev.per_conversation
                .iter()
                .map(|(sample_id, summary)| (sample_id.clone(), summary.bank_stats.clone()))
                .collect(),
            Some(source_artifact.clone()),
        )
    } else {
        (
            HashMap::new(),
            HashMap::new(),
            BTreeMap::new(),
            BTreeMap::new(),
            HashMap::new(),
            HashMap::new(),
            HashMap::new(),
            HashMap::new(),
            None,
        )
    };

    if !config.dataset.exists() {
        eprintln!("Dataset not found: {}", config.dataset.display());
        eprintln!("Download it:");
        eprintln!("  mkdir -p data");
        eprintln!(
            "  curl -o data/locomo10.json https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"
        );
        std::process::exit(1);
    }

    let raw_bytes = fs::read(&config.dataset).expect("failed to read dataset");
    let mut dataset: Vec<LocomoEntry> =
        serde_json::from_slice(&raw_bytes).expect("failed to parse dataset");

    if !config.conversations.is_empty() {
        let mut by_id = dataset
            .into_iter()
            .map(|entry| (entry.sample_id.clone(), entry))
            .collect::<HashMap<_, _>>();
        let mut selected = Vec::with_capacity(config.conversations.len());
        for sample_id in &config.conversations {
            let entry = by_id.remove(sample_id).unwrap_or_else(|| {
                eprintln!("Conversation not found in dataset: {sample_id}");
                std::process::exit(1);
            });
            selected.push(entry);
        }
        dataset = selected;
    }

    if matches!(command, BenchCommand::Qa) {
        let missing_banks = dataset
            .iter()
            .filter(|entry| !existing_banks.contains_key(&entry.sample_id))
            .map(|entry| entry.sample_id.clone())
            .collect::<Vec<_>>();
        if !missing_banks.is_empty() {
            eprintln!(
                "`qa` requires bank ids for every selected conversation; missing: {}",
                missing_banks.join(", ")
            );
            std::process::exit(1);
        }
    }

    let metrics = Arc::new(MetricsCollector::new());
    metrics.extend_cache_aware_or_legacy_snapshot(
        &existing_cache_aware_stage_metrics,
        &existing_stage_metrics,
    );
    let runtime = Arc::new(
        build_runtime_from_env(BuildRuntimeOptions {
            metrics: Some(metrics.clone()),
            max_pool_connections: None,
        })
        .await
        .expect("failed to build Elephant runtime"),
    );

    let judge = build_judge_client(metrics.clone(), config.judge_model.clone());
    let judge_label = judge_label(&config.judge_model);

    println!("retain_model: {}", runtime.info.retain_model);
    println!("reflect_model: {}", runtime.info.reflect_model);
    println!("reranker_model: {}", runtime.info.reranker_model);
    println!("embedding_model: {}", runtime.info.embedding_model);
    println!("LLM judge: {judge_label}");
    println!("Profile: {}", config.profile.as_str());
    println!("Mode: {}", command.as_str());
    println!("Ingest: {}", config.ingest.as_str());
    println!("Consolidation: {}", config.consolidation.as_str());
    println!("Conversation concurrency: {}", config.conversation_jobs);
    println!("Question concurrency: {}", config.question_jobs);
    println!(
        "Worktree: {}",
        match git_dirty_worktree() {
            Some(true) => "dirty",
            Some(false) => "clean",
            None => "unknown",
        }
    );

    let run_timestamp = Utc::now().to_rfc3339();
    let commit = git_commit_sha();
    let bench_start = Instant::now();
    let total_convs = dataset.len();

    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent).ok();
    }
    let questions_path = sidecar_path(&output_path, "questions");
    let debug_path = sidecar_path(&output_path, "debug");
    let _ = fs::write(&questions_path, "");
    let _ = fs::write(&debug_path, "");

    let consolidation_strategy = config.consolidation.as_str();

    let cli_command = env::args().collect::<Vec<_>>().join(" ");
    let manifest = BenchmarkManifest {
        protocol_version: "2026-03-10-config-v1".into(),
        profile: config.profile.as_str().into(),
        mode: command.as_str().into(),
        config_path: config
            .config_path
            .as_ref()
            .map(|path| path.display().to_string()),
        dataset_path: config.dataset.display().to_string(),
        dataset_fingerprint: format!("{:016x}", fnv1a64(&raw_bytes)),
        command: cli_command,
        category_filter: vec![1, 2, 3, 4],
        selected_conversations: config.conversations.clone(),
        image_policy: config.ingest.image_policy().into(),
        ingestion_granularity: config.ingest.ingestion_granularity().into(),
        question_concurrency: config.question_jobs,
        conversation_concurrency: config.conversation_jobs,
        consolidation_strategy: consolidation_strategy.into(),
        session_limit: config.session_limit,
        question_limit: config.question_limit,
        raw_json: config.ingest.raw_json(),
        dirty_worktree: git_dirty_worktree(),
        prompt_hashes: benchmark_prompt_hashes(&runtime),
        runtime_config: benchmark_runtime_config(&runtime),
        source_artifact,
        source_artifacts: Vec::new(),
    };

    let shared = Arc::new(Mutex::new(SharedResults {
        results: Vec::new(),
        banks: existing_banks.clone(),
        conversation_stage_metrics: existing_conversation_metrics.clone(),
        conversation_cache_aware_stage_metrics: existing_conversation_cache_aware_metrics.clone(),
        conversation_timings: existing_conversation_timings.clone(),
        conversation_bank_stats: existing_conversation_bank_stats.clone(),
        turn_refs: existing_turn_refs,
        output_path: output_path.clone(),
        questions_path: questions_path.clone(),
        debug_path: debug_path.clone(),
        judge_label: judge_label.clone(),
        tag: config.tag.clone(),
        retain_model: runtime.info.retain_model.clone(),
        reflect_model: runtime.info.reflect_model.clone(),
        embedding_model: runtime.info.embedding_model.clone(),
        reranker_model: runtime.info.reranker_model.clone(),
        consolidation_strategy: consolidation_strategy.into(),
        manifest,
        metrics: metrics.clone(),
        run_timestamp,
        commit,
        bench_start,
    }));

    let semaphore = Arc::new(Semaphore::new(config.conversation_jobs));
    let mut handles = Vec::new();

    for (conv_idx, entry) in dataset.into_iter().enumerate() {
        let sem = semaphore.clone();
        let runtime = runtime.clone();
        let judge = judge.clone();
        let options = ConversationRunOptions {
            max_sessions: config.session_limit,
            max_questions: config.question_limit,
            question_concurrency: config.question_jobs,
            ingest_mode: config.ingest,
            consolidation: config.consolidation,
            seed_stage_metrics: existing_conversation_metrics
                .get(&entry.sample_id)
                .cloned()
                .unwrap_or_default(),
            seed_cache_aware_stage_metrics: existing_conversation_cache_aware_metrics
                .get(&entry.sample_id)
                .cloned()
                .unwrap_or_default(),
            seed_timings: existing_conversation_timings
                .get(&entry.sample_id)
                .cloned()
                .unwrap_or_default(),
            seed_bank_stats: existing_conversation_bank_stats
                .get(&entry.sample_id)
                .cloned()
                .unwrap_or_default(),
            existing_bank: existing_banks.get(&entry.sample_id).cloned(),
            require_existing_bank: matches!(command, BenchCommand::Qa),
            ingest_only: matches!(command, BenchCommand::Ingest),
        };
        let tag = format!("conv {}/{total_convs}", conv_idx + 1);
        let shared = shared.clone();

        handles.push(tokio::spawn(async move {
            let _permit = sem
                .acquire()
                .await
                .map_err(|e| format!("semaphore closed: {e}"))?;
            run_conversation(tag, runtime, entry, judge, options, shared).await
        }));
    }

    for handle in handles {
        match handle.await {
            Ok(Ok(())) => {}
            Ok(Err(e)) => eprintln!("conversation failed: {e}"),
            Err(e) => eprintln!("conversation task panicked: {e}"),
        }
    }

    let shared_snapshot = shared.lock().await;
    let total_questions = shared_snapshot.results.len();
    let total_correct = shared_snapshot
        .results
        .iter()
        .filter(|r| r.judge_correct)
        .count();
    let accuracy = if total_questions > 0 {
        total_correct as f64 / total_questions as f64
    } else {
        0.0
    };
    let mean_f1 = if total_questions > 0 {
        shared_snapshot.results.iter().map(|r| r.f1).sum::<f64>() / total_questions as f64
    } else {
        0.0
    };
    let mean_evidence_recall = if total_questions > 0 {
        shared_snapshot
            .results
            .iter()
            .map(|r| r.evidence_recall)
            .sum::<f64>()
            / total_questions as f64
    } else {
        0.0
    };
    let evidence_precision_values = shared_snapshot
        .results
        .iter()
        .filter_map(|result| evidence_precision(&result.evidence_refs, &result.retrieved_turn_refs))
        .collect::<Vec<_>>();
    let mean_evidence_precision = if evidence_precision_values.is_empty() {
        0.0
    } else {
        evidence_precision_values.iter().sum::<f64>() / evidence_precision_values.len() as f64
    };
    let total_stage_usage = metrics.total_usage();
    let total_ingest_time_s = shared_snapshot
        .conversation_timings
        .values()
        .map(|timings| timings.ingest_time_s)
        .sum::<f64>();
    let total_consolidation_time_s = shared_snapshot
        .conversation_timings
        .values()
        .map(|timings| timings.consolidation_time_s)
        .sum::<f64>();
    let total_qa_time_s = shared_snapshot
        .conversation_timings
        .values()
        .map(|timings| timings.qa_time_s)
        .sum::<f64>();

    println!();
    println!("{}", "=".repeat(60));
    println!("LOCOMO BENCHMARK RESULTS");
    println!("{}", "=".repeat(60));
    println!("Total questions: {total_questions}");
    println!(
        "Total time: {}",
        fmt_elapsed(bench_start.elapsed().as_secs_f64())
    );
    println!("Judge model: {judge_label}");
    println!();
    println!(
        "Accuracy (LLM judge): {:.1}% ({}/{})",
        accuracy * 100.0,
        total_correct,
        total_questions
    );
    println!("Token F1 (reference): {mean_f1:.4}");
    println!("Evidence recall: {mean_evidence_recall:.4}");
    if !evidence_precision_values.is_empty() {
        println!("Evidence precision: {mean_evidence_precision:.4}");
    }
    println!(
        "Phase times: ingest {} | consolidate {} | qa {}",
        fmt_elapsed(total_ingest_time_s),
        fmt_elapsed(total_consolidation_time_s),
        fmt_elapsed(total_qa_time_s),
    );
    println!(
        "Stage usage: {} prompt + {} completion = {} total tokens across {} calls",
        total_stage_usage.prompt_tokens,
        total_stage_usage.completion_tokens,
        total_stage_usage.total_tokens(),
        total_stage_usage.calls,
    );
    if total_correct > 0 {
        println!(
            "Tokens per correct: {:.1}",
            total_stage_usage.total_tokens() as f64 / total_correct as f64
        );
    }

    shared_snapshot.flush();
    println!();
    println!("Summary saved to {}", output_path.display());
    println!("Questions saved to {}", questions_path.display());
    println!("Debug saved to {}", debug_path.display());
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn test_cache_aware_usage() -> CacheAwareStageUsage {
        CacheAwareStageUsage {
            prompt_tokens: 10,
            uncached_prompt_tokens: 6,
            cache_hit_prompt_tokens: 2,
            cache_write_prompt_tokens: 2,
            completion_tokens: 5,
            calls: 1,
            errors: 0,
            latency_ms: 100,
            cache_supported_calls: 1,
            cache_hit_calls: 1,
            cache_write_calls: 1,
            cache_unsupported_calls: 0,
        }
    }

    fn test_question_result(
        sample_id: &str,
        question_id: &str,
        category_name: &str,
        judge_correct: bool,
    ) -> QuestionResult {
        QuestionResult {
            question_id: question_id.into(),
            sample_id: sample_id.into(),
            question: format!("Question for {sample_id}"),
            ground_truth: "gold".into(),
            hypothesis: "hypothesis".into(),
            category_name: category_name.into(),
            f1: if judge_correct { 1.0 } else { 0.0 },
            judge_correct,
            judge_reasoning: "reason".into(),
            elapsed_s: 1.5,
            status: "ok".into(),
            error: None,
            final_source_ids: vec!["fact-1".into()],
            evidence_refs: vec!["D1:1".into()],
            retrieved_turn_refs: if judge_correct {
                vec!["D1:1".into()]
            } else {
                vec!["D9:9".into()]
            },
            evidence_hit: judge_correct,
            evidence_recall: if judge_correct { 1.0 } else { 0.0 },
            qa_stage_metrics: BTreeMap::from([(
                LlmStage::Reflect,
                StageUsage {
                    prompt_tokens: 10,
                    completion_tokens: 5,
                    calls: 1,
                    errors: 0,
                    latency_ms: 100,
                },
            )]),
            cache_aware_qa_stage_metrics: BTreeMap::from([(
                LlmStage::Reflect,
                test_cache_aware_usage(),
            )]),
        }
    }

    fn write_test_artifact(
        dir: &Path,
        stem: &str,
        sample_id: &str,
        question_id: &str,
        judge_correct: bool,
    ) -> PathBuf {
        let output_path = dir.join(format!("{stem}.json"));
        let questions_path = sidecar_path(&output_path, "questions");
        let debug_path = sidecar_path(&output_path, "debug");
        let question = test_question_result(sample_id, question_id, "single-hop", judge_correct);
        let debug = QuestionDebugRecord {
            question_id: question.question_id.clone(),
            sample_id: question.sample_id.clone(),
            question: question.question.clone(),
            reflect_trace: Vec::new(),
            final_done: None,
            retrieved_context: Vec::new(),
            retrieved_sources: Vec::new(),
        };
        write_jsonl_records(&questions_path, std::slice::from_ref(&question))
            .expect("write question sidecar");
        write_jsonl_records(&debug_path, std::slice::from_ref(&debug))
            .expect("write debug sidecar");

        let mut per_category = HashMap::new();
        per_category.insert(
            "single-hop".into(),
            CategoryResult {
                accuracy: if judge_correct { 1.0 } else { 0.0 },
                mean_f1: if judge_correct { 1.0 } else { 0.0 },
                mean_evidence_recall: if judge_correct { 1.0 } else { 0.0 },
                count: 1,
            },
        );

        let stage_usage = StageUsage {
            prompt_tokens: 10,
            completion_tokens: 5,
            calls: 1,
            errors: 0,
            latency_ms: 100,
        };

        let output = BenchmarkOutput {
            benchmark: "locomo".into(),
            timestamp: "2026-03-10T00:00:00Z".into(),
            commit: Some("deadbeefcafe".into()),
            tag: Some(stem.into()),
            judge_model: "anthropic/test-judge".into(),
            retain_model: "anthropic/test-main".into(),
            reflect_model: "anthropic/test-main".into(),
            embedding_model: "local/test-embedding".into(),
            reranker_model: "local/test-reranker".into(),
            consolidation_strategy: "end".into(),
            total_questions: 1,
            accuracy: if judge_correct { 1.0 } else { 0.0 },
            mean_f1: if judge_correct { 1.0 } else { 0.0 },
            mean_evidence_recall: if judge_correct { 1.0 } else { 0.0 },
            per_category,
            per_conversation: HashMap::from([(
                sample_id.into(),
                ConversationSummary {
                    bank_id: format!("bank-{sample_id}"),
                    accuracy: if judge_correct { 1.0 } else { 0.0 },
                    mean_f1: if judge_correct { 1.0 } else { 0.0 },
                    mean_evidence_recall: if judge_correct { 1.0 } else { 0.0 },
                    count: 1,
                    ingest_time_s: 2.0,
                    consolidation_time_s: 1.0,
                    qa_time_s: 3.0,
                    total_time_s: 6.0,
                    stage_metrics: BTreeMap::from([(LlmStage::Reflect, stage_usage.clone())]),
                    cache_aware_stage_metrics: BTreeMap::from([(
                        LlmStage::Reflect,
                        test_cache_aware_usage(),
                    )]),
                    bank_stats: ConversationBankStats {
                        sessions_ingested: 1,
                        turns_ingested: 2,
                        facts_stored: 3,
                        entities_resolved: 1,
                        links_created: 0,
                        opinions_reinforced: 0,
                        opinions_weakened: 0,
                        observations_created: 1,
                        observations_updated: 0,
                        final_fact_count: 4,
                        final_observation_count: 1,
                        final_opinion_count: 0,
                        final_entity_count: 1,
                    },
                },
            )]),
            banks: HashMap::from([(sample_id.into(), format!("bank-{sample_id}"))]),
            turn_refs: HashMap::from([(format!("{sample_id}-turn-1"), "D1:1".into())]),
            manifest: BenchmarkManifest {
                protocol_version: "2026-03-10-config-v1".into(),
                profile: "full".into(),
                mode: "run".into(),
                config_path: None,
                dataset_path: "data/locomo10.json".into(),
                dataset_fingerprint: "dataset123".into(),
                command: format!("locomo-bench run --conversation {sample_id}"),
                category_filter: vec![1, 2, 3, 4],
                selected_conversations: vec![sample_id.into()],
                image_policy: "blip_caption_inline".into(),
                ingestion_granularity: "session".into(),
                question_concurrency: 1,
                conversation_concurrency: 1,
                consolidation_strategy: "end".into(),
                session_limit: None,
                question_limit: None,
                raw_json: false,
                dirty_worktree: Some(false),
                prompt_hashes: BenchmarkPromptHashes::default(),
                runtime_config: BenchmarkRuntimeConfig::default(),
                source_artifact: None,
                source_artifacts: Vec::new(),
            },
            artifacts: BenchmarkArtifacts {
                questions_path: relative_artifact_path(&output_path, &questions_path),
                debug_path: relative_artifact_path(&output_path, &debug_path),
            },
            stage_metrics: BTreeMap::from([(LlmStage::Reflect, stage_usage.clone())]),
            total_stage_usage: stage_usage,
            cache_aware_stage_metrics: BTreeMap::from([(
                LlmStage::Reflect,
                test_cache_aware_usage(),
            )]),
            cache_aware_total_stage_usage: test_cache_aware_usage(),
            results: Vec::new(),
            total_time_s: 6.0,
        };

        fs::write(
            &output_path,
            serde_json::to_string_pretty(&output).expect("serialize output"),
        )
        .expect("write summary");
        output_path
    }

    fn strip_cache_aware_fields_from_artifact(summary_path: &Path) {
        let mut summary: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(summary_path).expect("read summary artifact"))
                .expect("parse summary artifact");
        summary
            .as_object_mut()
            .expect("summary is object")
            .remove("cache_aware_stage_metrics");
        summary
            .as_object_mut()
            .expect("summary is object")
            .remove("cache_aware_total_stage_usage");
        if let Some(per_conversation) = summary
            .get_mut("per_conversation")
            .and_then(serde_json::Value::as_object_mut)
        {
            for conversation in per_conversation.values_mut() {
                if let Some(conversation) = conversation.as_object_mut() {
                    conversation.remove("cache_aware_stage_metrics");
                }
            }
        }
        fs::write(
            summary_path,
            serde_json::to_string_pretty(&summary).expect("serialize legacy summary"),
        )
        .expect("rewrite summary without cache-aware fields");

        let questions_path = sidecar_path(summary_path, "questions");
        let rewritten_questions = fs::read_to_string(&questions_path)
            .expect("read question sidecar")
            .lines()
            .filter(|line| !line.trim().is_empty())
            .map(|line| {
                let mut value: serde_json::Value =
                    serde_json::from_str(line).expect("parse question sidecar record");
                value
                    .as_object_mut()
                    .expect("question record is object")
                    .remove("cache_aware_qa_stage_metrics");
                serde_json::to_string(&value).expect("serialize legacy question record")
            })
            .collect::<Vec<_>>()
            .join("\n");
        fs::write(&questions_path, format!("{rewritten_questions}\n"))
            .expect("rewrite question sidecar without cache-aware fields");
    }

    #[test]
    fn excludes_category_five_even_when_answer_is_present() {
        let qa = QaPair {
            question: "Is Oscar Melanie's pet?".into(),
            answer: Some(json!("No")),
            category: 5,
            evidence: vec![],
        };
        assert!(!should_score_question(&qa));
    }

    #[test]
    fn includes_answered_category_four_questions() {
        let qa = QaPair {
            question: "Who painted the bowl?".into(),
            answer: Some(json!("Melanie")),
            category: 4,
            evidence: vec!["D1:3".into()],
        };
        assert!(should_score_question(&qa));
    }

    #[test]
    fn computes_evidence_recall_on_unique_refs() {
        let expected = vec!["D1:3".into(), "D2:1".into()];
        let retrieved = vec!["D1:3".into(), "D1:3".into()];
        let (hit, recall) = evidence_recall(&expected, &retrieved);
        assert!(hit);
        assert!((recall - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn ignores_generated_benchmark_result_paths_in_dirty_check() {
        assert!(status_line_is_ignored(
            "?? bench/locomo/results/series1-conv-26.json"
        ));
        assert!(status_line_is_ignored(" M bench/locomo/results/quick.json"));
        assert!(status_line_is_ignored(
            "R  bench/locomo/results/old.json -> bench/locomo/results/new.json"
        ));
        assert!(!status_line_is_ignored(" M src/runtime.rs"));
        assert!(!status_line_is_ignored(
            "R  bench/locomo/results/old.json -> src/runtime.rs"
        ));
    }

    #[test]
    fn blocks_overwriting_existing_fresh_output_without_force() {
        let test_dir =
            env::temp_dir().join(format!("locomo-overwrite-test-{}", std::process::id()));
        fs::create_dir_all(&test_dir).expect("create overwrite test dir");
        let output = test_dir.join("run.json");
        fs::write(&output, "{}").expect("write existing output");

        let err = ensure_output_paths_are_safe(BenchCommand::Run, &output, None, &[], false)
            .expect_err("fresh overwrite should be blocked");
        assert!(err.contains("refusing to overwrite"));

        fs::remove_dir_all(&test_dir).ok();
    }

    #[test]
    fn qa_blocks_in_place_artifact_update_without_force() {
        let test_dir = env::temp_dir().join(format!("locomo-qa-safe-test-{}", std::process::id()));
        fs::create_dir_all(&test_dir).expect("create qa safe test dir");
        let artifact = test_dir.join("artifact.json");
        fs::write(&artifact, "{}").expect("write artifact");

        let err =
            ensure_output_paths_are_safe(BenchCommand::Qa, &artifact, Some(&artifact), &[], false)
                .expect_err("qa should block in-place update without force");
        assert!(err.contains("refusing to overwrite source artifact"));

        fs::remove_dir_all(&test_dir).ok();
    }

    #[test]
    fn merge_output_must_differ_from_inputs_without_force() {
        let test_dir =
            env::temp_dir().join(format!("locomo-merge-safe-test-{}", std::process::id()));
        fs::create_dir_all(&test_dir).expect("create merge safe test dir");
        let input = test_dir.join("merged.json");
        fs::write(&input, "{}").expect("write merge input");

        let err = ensure_output_paths_are_safe(
            BenchCommand::Merge,
            &input,
            None,
            std::slice::from_ref(&input),
            false,
        )
        .expect_err("merge should reject overwriting its input");
        assert!(err.contains("refusing to overwrite merge input"));

        fs::remove_dir_all(&test_dir).ok();
    }

    #[test]
    fn reads_plain_session_keys_from_locomo_variant() {
        let conversation = Conversation {
            speaker_a: "Caroline".into(),
            speaker_b: "Melanie".into(),
            sessions: HashMap::from([(
                "session_1".into(),
                json!([
                    {
                        "speaker": "Caroline",
                        "dia_id": "D1:1",
                        "text": "Hey Mel!"
                    },
                    {
                        "speaker": "Melanie",
                        "dia_id": "D1:2",
                        "text": "Hey Caroline!"
                    }
                ]),
            )]),
        };

        let turns = get_session_turns(&conversation, 1);
        assert_eq!(turns.len(), 2);
        assert_eq!(turns[0].dia_id.as_deref(), Some("D1:1"));
        assert_eq!(turns[1].dia_id.as_deref(), Some("D1:2"));
    }

    #[test]
    fn session_count_ignores_summary_and_observation_keys() {
        let conversation = Conversation {
            speaker_a: "Caroline".into(),
            speaker_b: "Melanie".into(),
            sessions: HashMap::from([
                (
                    "session_1".into(),
                    json!([
                        {
                            "speaker": "Caroline",
                            "dia_id": "D1:1",
                            "text": "Hey Mel!"
                        }
                    ]),
                ),
                (
                    "session_1_date_time".into(),
                    json!("1:56 pm on 8 May, 2023"),
                ),
                ("session_1_summary".into(), json!("summary text")),
                ("session_1_observation".into(), json!({ "facts": [] })),
                ("session_35_summary".into(), json!("later summary text")),
                ("session_35_observation".into(), json!({ "facts": [] })),
            ]),
        };

        assert_eq!(session_count(&conversation), 1);
    }

    #[test]
    fn smoke_profile_loads_expected_defaults() {
        let raw = vec![
            "locomo-bench".to_string(),
            "run".to_string(),
            "--profile".to_string(),
            "smoke".to_string(),
        ];
        let invocation = parse_args_from(&raw).unwrap().unwrap();
        assert_eq!(invocation.command, BenchCommand::Run);
        let config = invocation.config;
        assert_eq!(config.profile, RunProfile::Smoke);
        assert_eq!(config.ingest, IngestMode::Session);
        assert_eq!(config.consolidation, ConsolidationMode::End);
        assert_eq!(config.conversations, vec!["conv-26".to_string()]);
        assert_eq!(config.session_limit, Some(1));
        assert_eq!(config.question_limit, Some(5));
    }

    #[test]
    fn unknown_subcommand_is_rejected() {
        let raw = vec!["locomo-bench".to_string(), "resume".to_string()];
        let err = parse_args_from(&raw).unwrap_err();
        assert!(err.contains("unknown subcommand"));
    }

    #[test]
    fn merge_loads_input_artifacts_and_output_override() {
        let raw = vec![
            "locomo-bench".to_string(),
            "merge".to_string(),
            "a.json".to_string(),
            "b.json".to_string(),
            "--out".to_string(),
            "merged.json".to_string(),
        ];
        let invocation = parse_args_from(&raw).unwrap().unwrap();
        assert_eq!(invocation.command, BenchCommand::Merge);
        assert_eq!(
            invocation.merge_artifacts,
            vec![PathBuf::from("a.json"), PathBuf::from("b.json")]
        );
        assert_eq!(invocation.config.output, Some(PathBuf::from("merged.json")));
    }

    #[test]
    fn default_run_output_uses_local_directory() {
        let config = RunConfig {
            profile: RunProfile::Full,
            ..RunConfig::default()
        };
        assert_eq!(
            default_output_path(BenchCommand::Run, &config, None),
            PathBuf::from("bench/locomo/results/local/full-run.json")
        );
    }

    #[test]
    fn default_tagged_run_output_uses_local_directory() {
        let config = RunConfig {
            profile: RunProfile::Full,
            tag: Some("series1-conv-26".into()),
            ..RunConfig::default()
        };
        assert_eq!(
            default_output_path(BenchCommand::Run, &config, None),
            PathBuf::from("bench/locomo/results/local/series1-conv-26.json")
        );
    }

    #[test]
    fn default_merge_output_uses_local_directory() {
        let config = RunConfig {
            profile: RunProfile::Full,
            ..RunConfig::default()
        };
        assert_eq!(
            default_output_path(BenchCommand::Merge, &config, None),
            PathBuf::from("bench/locomo/results/local/merged.json")
        );
    }

    #[test]
    fn qa_defaults_to_source_artifact_path() {
        let config = RunConfig::default();
        let artifact = PathBuf::from("bench/locomo/results/local/ingest.json");
        assert_eq!(
            default_output_path(BenchCommand::Qa, &config, Some(&artifact)),
            artifact
        );
    }

    #[test]
    fn qa_loads_scope_from_artifact_manifest() {
        let artifact_path =
            env::temp_dir().join(format!("locomo-qa-test-{}.json", std::process::id()));
        let artifact = json!({
            "benchmark": "locomo",
            "timestamp": "2026-03-10T00:00:00Z",
            "judge_model": "anthropic/test-judge",
            "retain_model": "anthropic/test-main",
            "reflect_model": "anthropic/test-main",
            "embedding_model": "local/test-embedding",
            "reranker_model": "local/test-reranker",
            "consolidation_strategy": "end",
            "total_questions": 0,
            "accuracy": 0.0,
            "mean_f1": 0.0,
            "mean_evidence_recall": 0.0,
            "per_category": {},
            "per_conversation": {},
            "bank_ids": {
                "conv-26": "01KK623GTJJB2WW3RKHSDSCDT6"
            },
            "turn_refs": {},
            "manifest": {
                "protocol_version": "2026-03-10-config-v1",
                "profile": "smoke",
                "mode": "ingest",
                "config_path": null,
                "dataset_path": "data/locomo10.json",
                "dataset_fingerprint": "9f7f4c0a5fbb2df2",
                "command": "locomo-bench ingest --profile smoke",
                "category_filter": [1, 2, 3, 4],
                "selected_conversations": ["conv-26"],
                "image_policy": "blip_caption_inline",
                "ingestion_granularity": "session",
                "question_concurrency": 1,
                "conversation_concurrency": 1,
                "consolidation_strategy": "end",
                "session_limit": 1,
                "question_limit": 5,
                "raw_json": false,
                "dirty_worktree": false
            },
            "stage_metrics": {},
            "total_stage_usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "calls": 0,
                "errors": 0,
                "latency_ms": 0
            },
            "results": [],
            "total_time_s": 0.0
        });
        fs::write(
            &artifact_path,
            serde_json::to_string(&artifact).expect("serialize test artifact"),
        )
        .expect("write test artifact");

        let raw = vec![
            "locomo-bench".to_string(),
            "qa".to_string(),
            artifact_path.display().to_string(),
            "--question-jobs".to_string(),
            "4".to_string(),
        ];
        let invocation = parse_args_from(&raw).unwrap().unwrap();
        assert_eq!(invocation.command, BenchCommand::Qa);
        assert_eq!(invocation.artifact_path, Some(artifact_path.clone()));
        assert_eq!(invocation.config.profile, RunProfile::Smoke);
        assert_eq!(
            invocation.config.dataset,
            PathBuf::from("data/locomo10.json")
        );
        assert_eq!(invocation.config.conversations, vec!["conv-26".to_string()]);
        assert_eq!(invocation.config.session_limit, Some(1));
        assert_eq!(invocation.config.question_limit, Some(5));
        assert_eq!(invocation.config.question_jobs, 4);

        fs::remove_file(&artifact_path).ok();
    }

    #[test]
    fn qa_rejects_protocol_overrides() {
        let artifact_path = env::temp_dir().join(format!(
            "locomo-qa-test-override-{}.json",
            std::process::id()
        ));
        let artifact = json!({
            "benchmark": "locomo",
            "timestamp": "2026-03-10T00:00:00Z",
            "judge_model": "anthropic/test-judge",
            "retain_model": "anthropic/test-main",
            "reflect_model": "anthropic/test-main",
            "embedding_model": "local/test-embedding",
            "reranker_model": "local/test-reranker",
            "consolidation_strategy": "end",
            "total_questions": 0,
            "accuracy": 0.0,
            "mean_f1": 0.0,
            "mean_evidence_recall": 0.0,
            "per_category": {},
            "per_conversation": {},
            "bank_ids": {
                "conv-26": "01KK623GTJJB2WW3RKHSDSCDT6"
            },
            "turn_refs": {},
            "manifest": {
                "protocol_version": "2026-03-10-config-v1",
                "profile": "full",
                "mode": "ingest",
                "config_path": null,
                "dataset_path": "data/locomo10.json",
                "dataset_fingerprint": "9f7f4c0a5fbb2df2",
                "command": "locomo-bench ingest --profile full",
                "category_filter": [1, 2, 3, 4],
                "selected_conversations": [],
                "image_policy": "blip_caption_inline",
                "ingestion_granularity": "session",
                "question_concurrency": 1,
                "conversation_concurrency": 1,
                "consolidation_strategy": "end",
                "session_limit": null,
                "question_limit": null,
                "raw_json": false,
                "dirty_worktree": false
            },
            "stage_metrics": {},
            "total_stage_usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "calls": 0,
                "errors": 0,
                "latency_ms": 0
            },
            "results": [],
            "total_time_s": 0.0
        });
        fs::write(
            &artifact_path,
            serde_json::to_string(&artifact).expect("serialize test artifact"),
        )
        .expect("write test artifact");

        let raw = vec![
            "locomo-bench".to_string(),
            "qa".to_string(),
            artifact_path.display().to_string(),
            "--profile".to_string(),
            "smoke".to_string(),
        ];
        let err = parse_args_from(&raw).unwrap_err();
        assert!(err.contains("`qa` does not accept --profile"));

        fs::remove_file(&artifact_path).ok();
    }

    #[test]
    fn removed_alias_falls_back_to_unknown_argument() {
        let raw = vec![
            "locomo-bench".to_string(),
            "run".to_string(),
            "--question-concurrency".to_string(),
            "4".to_string(),
        ];
        let err = parse_args_from(&raw).unwrap_err();
        assert!(err.contains("Unknown argument"));
    }

    #[test]
    fn merge_combines_disjoint_subset_artifacts() {
        let test_dir = env::temp_dir().join(format!("locomo-merge-test-{}", std::process::id()));
        fs::create_dir_all(&test_dir).expect("create merge test dir");

        let left = write_test_artifact(&test_dir, "left", "conv-01", "q-left", true);
        let right = write_test_artifact(&test_dir, "right", "conv-02", "q-right", false);
        let merged = test_dir.join("merged.json");

        merge_artifacts(
            &[left.clone(), right.clone()],
            &merged,
            Some("merged".into()),
        )
        .expect("merge succeeds");

        let merged_bundle = load_artifact_bundle(&merged).expect("load merged artifact");
        assert_eq!(merged_bundle.output.manifest.mode, "merge");
        assert_eq!(merged_bundle.output.tag.as_deref(), Some("merged"));
        assert_eq!(merged_bundle.output.total_questions, 2);
        assert_eq!(merged_bundle.questions.len(), 2);
        assert_eq!(merged_bundle.debug_records.len(), 2);
        assert_eq!(merged_bundle.output.per_conversation.len(), 2);
        assert_eq!(merged_bundle.output.banks.len(), 2);
        assert_eq!(merged_bundle.output.manifest.source_artifacts.len(), 2);
        assert!((merged_bundle.output.accuracy - 0.5).abs() < f64::EPSILON);
        assert_eq!(merged_bundle.output.total_stage_usage.total_tokens(), 30);

        fs::remove_dir_all(&test_dir).ok();
    }

    #[test]
    fn merge_allows_provenance_field_differences() {
        let test_dir = env::temp_dir().join(format!(
            "locomo-merge-provenance-test-{}",
            std::process::id()
        ));
        fs::create_dir_all(&test_dir).expect("create merge provenance test dir");

        let left = write_test_artifact(&test_dir, "left-prov", "conv-01", "q-left", true);
        let right = write_test_artifact(&test_dir, "right-prov", "conv-02", "q-right", true);

        let mut right_output = load_benchmark_output(&right).expect("load right artifact");
        right_output.commit = Some("feedfacebeef".into());
        right_output.manifest.profile = "smoke".into();
        right_output.manifest.question_concurrency = 4;
        right_output.manifest.conversation_concurrency = 2;
        right_output.manifest.dirty_worktree = Some(true);
        fs::write(
            &right,
            serde_json::to_string_pretty(&right_output).expect("serialize right artifact"),
        )
        .expect("rewrite right artifact");

        let merged = test_dir.join("merged-prov.json");
        merge_artifacts(
            &[left.clone(), right.clone()],
            &merged,
            Some("merged-prov".into()),
        )
        .expect("merge succeeds with provenance differences");

        let merged_output = load_benchmark_output(&merged).expect("load merged artifact");
        assert_eq!(merged_output.manifest.profile, "mixed");
        assert_eq!(merged_output.manifest.question_concurrency, 0);
        assert_eq!(merged_output.manifest.conversation_concurrency, 0);
        assert_eq!(merged_output.total_questions, 2);

        fs::remove_dir_all(&test_dir).ok();
    }

    #[test]
    fn cache_artifact_locomo_summary_serializes_cache_aware_fields() {
        let stage_usage = StageUsage {
            prompt_tokens: 10,
            completion_tokens: 5,
            calls: 1,
            errors: 0,
            latency_ms: 100,
        };
        let cache_aware_usage = test_cache_aware_usage();
        let output = BenchmarkOutput {
            benchmark: "locomo".into(),
            timestamp: "2026-03-10T00:00:00Z".into(),
            commit: Some("deadbeefcafe".into()),
            tag: Some("cache-aware".into()),
            judge_model: "anthropic/test-judge".into(),
            retain_model: "anthropic/test-main".into(),
            reflect_model: "anthropic/test-main".into(),
            embedding_model: "local/test-embedding".into(),
            reranker_model: "local/test-reranker".into(),
            consolidation_strategy: "end".into(),
            total_questions: 1,
            accuracy: 1.0,
            mean_f1: 1.0,
            mean_evidence_recall: 1.0,
            per_category: HashMap::new(),
            per_conversation: HashMap::from([(
                "conv-01".into(),
                ConversationSummary {
                    bank_id: "bank-conv-01".into(),
                    accuracy: 1.0,
                    mean_f1: 1.0,
                    mean_evidence_recall: 1.0,
                    count: 1,
                    ingest_time_s: 1.0,
                    consolidation_time_s: 2.0,
                    qa_time_s: 3.0,
                    total_time_s: 6.0,
                    stage_metrics: BTreeMap::from([(LlmStage::Reflect, stage_usage.clone())]),
                    cache_aware_stage_metrics: BTreeMap::from([(
                        LlmStage::Reflect,
                        cache_aware_usage.clone(),
                    )]),
                    bank_stats: ConversationBankStats::default(),
                },
            )]),
            banks: HashMap::from([("conv-01".into(), "bank-conv-01".into())]),
            turn_refs: HashMap::new(),
            manifest: BenchmarkManifest::default(),
            artifacts: BenchmarkArtifacts::default(),
            stage_metrics: BTreeMap::from([(LlmStage::Reflect, stage_usage.clone())]),
            total_stage_usage: stage_usage,
            cache_aware_stage_metrics: BTreeMap::from([(
                LlmStage::Reflect,
                cache_aware_usage.clone(),
            )]),
            cache_aware_total_stage_usage: cache_aware_usage,
            results: Vec::new(),
            total_time_s: 6.0,
        };

        let value = serde_json::to_value(&output).expect("serialize summary output");
        assert!(value.get("cache_aware_stage_metrics").is_some());
        assert!(value.get("cache_aware_total_stage_usage").is_some());
        assert!(
            value
                .get("per_conversation")
                .and_then(|per_conversation| per_conversation.get("conv-01"))
                .and_then(|summary| summary.get("cache_aware_stage_metrics"))
                .is_some()
        );
    }

    #[test]
    fn cache_artifact_locomo_question_sidecar_serializes_cache_aware_fields() {
        let result = test_question_result("conv-01", "q-01", "single-hop", true);
        let value = serde_json::to_value(&result).expect("serialize question result");

        assert!(value.get("qa_stage_metrics").is_some());
        assert!(value.get("cache_aware_qa_stage_metrics").is_some());
    }

    #[test]
    fn cache_artifact_locomo_merge_prefers_cache_aware_metrics() {
        let test_dir = env::temp_dir().join(format!(
            "locomo-cache-aware-merge-test-{}",
            std::process::id()
        ));
        fs::create_dir_all(&test_dir).expect("create cache-aware merge test dir");

        let new_style = write_test_artifact(&test_dir, "new-style", "conv-01", "q-left", true);
        let legacy = write_test_artifact(&test_dir, "legacy", "conv-02", "q-right", false);
        strip_cache_aware_fields_from_artifact(&legacy);
        let merged = test_dir.join("merged-cache-aware.json");

        merge_artifacts(
            &[new_style.clone(), legacy.clone()],
            &merged,
            Some("merged-cache-aware".into()),
        )
        .expect("merge succeeds");

        let merged_output = load_benchmark_output(&merged).expect("load merged artifact");
        let usage = merged_output
            .cache_aware_stage_metrics
            .get(&LlmStage::Reflect)
            .expect("reflect metrics");
        assert_eq!(usage.prompt_tokens, 20);
        assert_eq!(usage.cache_hit_prompt_tokens, 2);
        assert_eq!(usage.cache_write_prompt_tokens, 2);
        assert_eq!(usage.cache_supported_calls, 1);
        assert_eq!(usage.cache_unsupported_calls, 1);

        fs::remove_dir_all(&test_dir).ok();
    }

    #[test]
    fn cache_artifact_locomo_merge_falls_back_to_legacy_metrics() {
        let test_dir = env::temp_dir().join(format!(
            "locomo-cache-aware-merge-legacy-test-{}",
            std::process::id()
        ));
        fs::create_dir_all(&test_dir).expect("create legacy merge test dir");

        let new_style = write_test_artifact(&test_dir, "new-style", "conv-01", "q-left", true);
        let legacy = write_test_artifact(&test_dir, "legacy", "conv-02", "q-right", false);
        strip_cache_aware_fields_from_artifact(&legacy);
        let merged = test_dir.join("merged-legacy-fallback.json");

        merge_artifacts(
            &[new_style.clone(), legacy.clone()],
            &merged,
            Some("merged-legacy-fallback".into()),
        )
        .expect("merge succeeds");

        let merged_output = load_benchmark_output(&merged).expect("load merged artifact");
        let usage = merged_output
            .per_conversation
            .get("conv-02")
            .expect("legacy conversation summary")
            .cache_aware_stage_metrics
            .get(&LlmStage::Reflect)
            .expect("legacy reflect metrics");
        assert_eq!(usage.prompt_tokens, 10);
        assert_eq!(usage.cache_hit_prompt_tokens, 0);
        assert_eq!(usage.cache_write_prompt_tokens, 0);
        assert_eq!(usage.cache_unsupported_calls, 1);

        fs::remove_dir_all(&test_dir).ok();
    }
}
