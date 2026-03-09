//! LoCoMo benchmark harness for Elephant.
//!
//! This runner executes Elephant in-process so benchmark artifacts can include
//! stage-level usage, dataset evidence refs, and stable run provenance.

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
use serde::{Deserialize, Serialize};
use tokio::sync::{Mutex, Semaphore};

use elephant::llm::retry::{RetryPolicy, RetryingLlmClient};
use elephant::llm::{self, LlmClient, Provider, ProviderConfig};
use elephant::metrics::{LlmStage, MeteredLlmClient, MetricsCollector, StageUsage};
use elephant::runtime::{BuildRuntimeOptions, ElephantRuntime, build_runtime_from_env};
use elephant::types::{
    BankId, CompletionRequest, Disposition, MemoryBank, Message, NetworkType, ReflectQuery,
    RetainInput, TurnId,
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

// --- Judge types ---

#[derive(Debug, Deserialize)]
struct JudgeResponse {
    reasoning: String,
    label: String,
}

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
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct ConversationSummary {
    bank_id: String,
    accuracy: f64,
    mean_f1: f64,
    mean_evidence_recall: f64,
    count: usize,
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
    stage_metrics: BTreeMap<LlmStage, StageUsage>,
    #[serde(default)]
    total_stage_usage: StageUsage,
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
    confidence: f32,
    elapsed_s: f64,
    status: String,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    error: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    evidence_refs: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    retrieved_turn_refs: Vec<String>,
    evidence_hit: bool,
    evidence_recall: f64,
    /// All facts retrieved during reflect, in ranked order with scores.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    retrieved_context: Vec<RetrievedFactEntry>,
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
            k.strip_prefix("session_")
                .and_then(|rest| rest.strip_suffix("_date_time"))
                .or_else(|| {
                    k.strip_prefix("session_").and_then(|rest| {
                        rest.strip_suffix("_dialogue")
                            .or_else(|| rest.strip_suffix("_dialog"))
                    })
                })
                .and_then(|n| n.parse::<usize>().ok())
        })
        .max()
        .unwrap_or(0)
}

fn get_session_turns(conv: &Conversation, idx: usize) -> Vec<Turn> {
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
    let dialogue_key = format!("session_{idx}_dialogue");
    let dialog_key = format!("session_{idx}_dialog");
    let date_key = format!("session_{idx}_date_time");
    let obj = serde_json::json!({
        "sample_session": idx,
        "date_time": conv.sessions.get(&date_key).cloned().unwrap_or(serde_json::Value::Null),
        "dialogue": conv.sessions.get(&dialogue_key)
            .cloned()
            .or_else(|| conv.sessions.get(&dialog_key).cloned())
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

fn fnv1a64(data: &[u8]) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for byte in data {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

fn fmt_elapsed(seconds: f64) -> String {
    let m = (seconds as u64) / 60;
    let s = (seconds as u64) % 60;
    format!("{m}:{s:02}")
}

// --- LLM Judge ---

const JUDGE_PROMPT: &str = include_str!("judge_answer.txt");

async fn llm_judge(
    judge: &dyn LlmClient,
    question: &str,
    gold_answer: &str,
    generated_answer: &str,
) -> Result<(bool, String), String> {
    let prompt = JUDGE_PROMPT
        .replace("{question}", question)
        .replace("{gold_answer}", gold_answer)
        .replace("{generated_answer}", generated_answer);

    let request = CompletionRequest {
        model: String::new(),
        messages: vec![Message::text("user", prompt)],
        max_tokens: Some(200),
        temperature: Some(0.0),
        system: None,
        ..Default::default()
    };

    for attempt in 0..3 {
        let result = judge.complete(request.clone()).await;
        match result {
            Ok(resp) => {
                if let Ok(parsed) = serde_json::from_str::<JudgeResponse>(&resp.content) {
                    let correct = parsed.label.eq_ignore_ascii_case("CORRECT");
                    return Ok((correct, parsed.reasoning));
                }
                if let Ok(json_str) = llm::extract_json(&resp.content)
                    && let Ok(parsed) = serde_json::from_str::<JudgeResponse>(&json_str)
                {
                    let correct = parsed.label.eq_ignore_ascii_case("CORRECT");
                    return Ok((correct, parsed.reasoning));
                }
                if attempt == 2 {
                    return Err(format!(
                        "could not parse judge response: {}",
                        &resp.content[..resp.content.len().min(120)]
                    ));
                }
            }
            Err(e) => {
                return Err(format!("judge error: {e}"));
            }
        }
    }
    Err("judge failed after retries".into())
}

fn build_judge_client(
    metrics: Arc<MetricsCollector>,
    override_model: Option<String>,
) -> Arc<dyn LlmClient> {
    let judge_provider_str = env::var("JUDGE_PROVIDER")
        .or_else(|_| env::var("LLM_PROVIDER"))
        .expect("JUDGE_PROVIDER or LLM_PROVIDER must be set");
    let judge_api_key = env::var("JUDGE_API_KEY")
        .or_else(|_| env::var("LLM_API_KEY"))
        .expect("JUDGE_API_KEY or LLM_API_KEY must be set");
    let judge_model = override_model.unwrap_or_else(|| {
        env::var("JUDGE_MODEL")
            .or_else(|_| env::var("LLM_MODEL"))
            .expect("JUDGE_MODEL or LLM_MODEL must be set")
    });
    let provider = match judge_provider_str.as_str() {
        "openai" => Provider::OpenAi,
        _ => Provider::Anthropic,
    };
    let judge_config = ProviderConfig {
        provider,
        api_key: judge_api_key,
        model: judge_model,
        base_url: env::var("JUDGE_BASE_URL")
            .ok()
            .or_else(|| env::var("LLM_BASE_URL").ok()),
    };
    let inner: Arc<dyn LlmClient> = Arc::from(llm::build_client(&judge_config).unwrap());
    let metered: Arc<dyn LlmClient> =
        Arc::new(MeteredLlmClient::new(inner, metrics, LlmStage::Judge));
    Arc::new(RetryingLlmClient::new(metered, RetryPolicy::default()))
}

fn judge_label(override_model: &Option<String>) -> String {
    let provider = env::var("JUDGE_PROVIDER")
        .or_else(|_| env::var("LLM_PROVIDER"))
        .expect("JUDGE_PROVIDER or LLM_PROVIDER must be set");
    let model = override_model.clone().unwrap_or_else(|| {
        env::var("JUDGE_MODEL")
            .or_else(|_| env::var("LLM_MODEL"))
            .expect("JUDGE_MODEL or LLM_MODEL must be set")
    });
    format!("{provider}/{model}")
}

// --- Result flushing ---

#[allow(clippy::too_many_arguments)]
fn flush_results(
    results: &[QuestionResult],
    banks: &HashMap<String, String>,
    turn_refs: &HashMap<String, String>,
    output_path: &Path,
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
    bench_start: Instant,
) {
    let bench_elapsed = bench_start.elapsed().as_secs_f64();
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

    let mut per_conversation = HashMap::new();
    for (sample_id, rows) in &conversation_results {
        let n = rows.len();
        per_conversation.insert(
            sample_id.clone(),
            ConversationSummary {
                bank_id: banks.get(sample_id).cloned().unwrap_or_default(),
                accuracy: rows.iter().filter(|r| r.judge_correct).count() as f64 / n as f64,
                mean_f1: rows.iter().map(|r| r.f1).sum::<f64>() / n as f64,
                mean_evidence_recall: rows.iter().map(|r| r.evidence_recall).sum::<f64>()
                    / n as f64,
                count: n,
            },
        );
    }

    let stage_metrics = metrics.snapshot();
    let total_stage_usage = metrics.total_usage();

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
        stage_metrics,
        total_stage_usage,
        results: sorted_results,
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
            ingest: IngestMode::Turn,
            consolidation: ConsolidationMode::End,
            conversation_jobs: 1,
            question_jobs: 1,
            judge_model: None,
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
}

#[derive(Debug, Clone)]
struct BenchInvocation {
    command: BenchCommand,
    artifact_path: Option<PathBuf>,
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
    validate_run_config(command, &config)?;
    Ok(Some(BenchInvocation {
        command,
        artifact_path,
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
    let profile = if manifest.profile.is_empty() {
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
    })
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
    eprintln!();
    eprintln!("Subcommands:");
    eprintln!("  run                              Fresh ingest, consolidate, then score QA");
    eprintln!("  ingest                           Ingest and consolidate only; do not run QA");
    eprintln!(
        "  qa <RESULTS.json>                Score QA against existing banks; skip ingest and consolidation"
    );
    eprintln!();
    eprintln!("Options:");
    eprintln!("  --profile <NAME>                Named profile for `run`/`ingest` [default: full]");
    eprintln!("                                  Profiles: full, smoke, legacy-raw");
    eprintln!("  --config <PATH>                 JSON config file for `run`/`ingest`");
    eprintln!(
        "  --dataset <PATH>                Dataset path for `run`/`ingest` [default: data/locomo10.json]"
    );
    eprintln!("  --tag <NAME>                    Save to bench/locomo/results/<tag>.json");
    eprintln!("  --out <PATH>                    Output results path (overrides --tag)");
    eprintln!("  --conversation <ID>             Run one specific conversation (repeatable)");
    eprintln!("  --ingest <MODE>                 turn | session | raw-json (`run`/`ingest` only)");
    eprintln!("  --consolidation <MODE>          end | per-session | off (`run`/`ingest` only)");
    eprintln!("  --conversation-jobs <N>         Parallel conversations");
    eprintln!("  --question-jobs <N>             Parallel questions per conversation");
    eprintln!("  --judge-model <MODEL>           Override judge model");
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
    turn_refs: HashMap<String, String>,
    output_path: PathBuf,
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
    fn push_and_flush(&mut self, result: QuestionResult) {
        self.results.push(result);
        self.flush();
    }

    fn record_bank(&mut self, sample_id: String, bank_id: String) {
        self.banks.insert(sample_id, bank_id);
        self.flush();
    }

    fn record_turn_ref(&mut self, turn_id: TurnId, turn_ref: String) {
        self.turn_refs.insert(turn_id.to_string(), turn_ref);
    }

    fn flush(&self) {
        flush_results(
            &self.results,
            &self.banks,
            &self.turn_refs,
            &self.output_path,
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
            self.bench_start,
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

async fn run_conversation(
    tag: String,
    runtime: Arc<ElephantRuntime>,
    entry: LocomoEntry,
    judge: Arc<dyn LlmClient>,
    options: ConversationRunOptions,
    shared: Arc<Mutex<SharedResults>>,
) -> Result<(), String> {
    let conv = &entry.conversation;
    let total_sessions = session_count(conv);

    println!(
        "[{tag}] {} & {} ({})",
        conv.speaker_a, conv.speaker_b, entry.sample_id
    );

    let reused_bank = options.existing_bank.clone();
    let bank_id = if let Some(id) = reused_bank {
        println!("[{tag}] Using existing bank: {id} (skipping ingestion)");
        BankId::from_str(&id).map_err(|e| format!("[{tag}] invalid bank id: {e}"))?
    } else {
        if options.require_existing_bank {
            return Err(format!(
                "[{tag}] missing bank for {} in the source artifact",
                entry.sample_id
            ));
        }

        let bank = MemoryBank {
            id: BankId::new(),
            name: format!("locomo-{}", entry.sample_id),
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
        println!(
            "[{tag}] Bank: {} | Ingesting {ingest_sessions}/{total_sessions} sessions...",
            bank.id
        );

        let ingest_start = Instant::now();
        let mut stored_facts = 0usize;

        for idx in 1..=ingest_sessions {
            let turns = get_session_turns(conv, idx);
            let date_str = get_session_date(conv, idx);
            let timestamp = parse_session_date(&date_str);

            if options.ingest_mode.ingest_per_session() {
                let content = if options.ingest_mode.raw_json() {
                    format_session_raw(conv, idx)
                } else {
                    format_session(&turns, &date_str)
                };
                match runtime
                    .retain
                    .retain(&RetainInput {
                        bank_id: bank.id,
                        content,
                        timestamp,
                        turn_id: None,
                        context: None,
                        custom_instructions: None,
                        speaker: None,
                    })
                    .await
                {
                    Ok(resp) => {
                        stored_facts += resp.facts_stored;
                        println!(
                            "[{tag}] ingest [{idx}/{ingest_sessions}] {} facts | elapsed: {}",
                            resp.facts_stored,
                            fmt_elapsed(ingest_start.elapsed().as_secs_f64()),
                        );
                    }
                    Err(e) => {
                        eprintln!("[{tag}] ingest [{idx}/{ingest_sessions}] FAILED: {e}");
                        continue;
                    }
                }
            } else {
                let mut prior_turns: Vec<String> = Vec::new();
                for (turn_idx, turn) in turns.iter().enumerate() {
                    let turn_text = format_turn(turn);
                    let turn_id = TurnId::new();
                    let turn_ref = format!("D{idx}:{}", turn_idx + 1);

                    let resp = runtime
                        .retain
                        .retain(&RetainInput {
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
                        })
                        .await;

                    match resp {
                        Ok(retain_resp) => {
                            stored_facts += retain_resp.facts_stored;
                            shared.lock().await.record_turn_ref(turn_id, turn_ref);
                            prior_turns.push(turn_text);
                        }
                        Err(e) => {
                            eprintln!(
                                "[{tag}] ingest [{idx}/{} turn {}] FAILED: {e}",
                                ingest_sessions,
                                turn_idx + 1
                            );
                        }
                    }
                }
                println!(
                    "[{tag}] ingest [{idx}/{ingest_sessions}] turn-level complete | elapsed: {}",
                    fmt_elapsed(ingest_start.elapsed().as_secs_f64()),
                );
            }

            if options.consolidation.per_session() {
                match runtime.consolidator.consolidate(bank.id).await {
                    Ok(cr) => {
                        println!(
                            "[{tag}]   consolidate: {} created, {} updated",
                            cr.observations_created, cr.observations_updated,
                        );
                    }
                    Err(e) => {
                        eprintln!("[{tag}]   consolidate failed: {e}");
                    }
                }
            }
        }

        println!(
            "[{tag}] Ingestion complete: {stored_facts} facts in {}",
            fmt_elapsed(ingest_start.elapsed().as_secs_f64())
        );

        bank.id
    };

    shared
        .lock()
        .await
        .record_bank(entry.sample_id.clone(), bank_id.to_string());

    if options.consolidation.enabled()
        && !options.consolidation.per_session()
        && options.existing_bank.is_none()
    {
        println!("[{tag}] Consolidating...");
        let t0 = Instant::now();
        match runtime.consolidator.consolidate(bank_id).await {
            Ok(resp) => {
                let elapsed = t0.elapsed().as_secs_f64();
                println!(
                    "[{tag}] Consolidation done in {}: {} observations created, {} updated",
                    fmt_elapsed(elapsed),
                    resp.observations_created,
                    resp.observations_updated,
                );
            }
            Err(e) => {
                eprintln!("[{tag}] consolidate FAILED: {e}");
            }
        }
    }

    if options.ingest_only {
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
        let sample_id = entry.sample_id.clone();
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

        qa_handles.push(tokio::spawn(async move {
            let _permit = sem.acquire().await.expect("semaphore closed");
            let cat_name = category_name(category);

            let t0 = Instant::now();
            let reflect_result = runtime
                .reflect
                .reflect(&ReflectQuery {
                    bank_id,
                    question: question.clone(),
                    budget_tokens: 4096,
                })
                .await;
            let elapsed = t0.elapsed().as_secs_f64();

            let (hypothesis, confidence, retrieved_context, status, error) = match reflect_result {
                Ok(resp) => {
                    let retrieved_context = resp
                        .retrieved_context
                        .into_iter()
                        .map(|fact| RetrievedFactEntry {
                            id: fact.id.to_string(),
                            content: fact.content,
                            score: fact.score,
                            network: network_name(fact.network).to_string(),
                            source_turn_id: fact.source_turn_id.map(|id| id.to_string()),
                            source_turn_ref: fact
                                .source_turn_id
                                .and_then(|id| turn_refs.get(&id.to_string()).cloned()),
                        })
                        .collect::<Vec<_>>();
                    (
                        resp.response,
                        resp.confidence,
                        retrieved_context,
                        "ok".to_string(),
                        None,
                    )
                }
                Err(e) => (
                    String::new(),
                    0.0,
                    Vec::new(),
                    "reflect_error".to_string(),
                    Some(e.to_string()),
                ),
            };

            let f1 = token_f1(&hypothesis, &gold);
            let retrieved_turn_refs = retrieved_context
                .iter()
                .filter_map(|fact| fact.source_turn_ref.clone())
                .collect::<Vec<_>>();
            let (evidence_hit, evidence_recall) = evidence_recall(&evidence_refs, &retrieved_turn_refs);

            let (judge_correct, judge_reasoning, status, error) = if hypothesis.is_empty() {
                (
                    false,
                    error.clone().unwrap_or_else(|| "empty response".into()),
                    status,
                    error,
                )
            } else {
                match llm_judge(judge.as_ref(), &question, &gold, &hypothesis).await {
                    Ok((correct, reasoning)) => (correct, reasoning, status, error),
                    Err(judge_error) => (
                        false,
                        judge_error.clone(),
                        "judge_error".into(),
                        Some(judge_error),
                    ),
                }
            };

            local_total.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if judge_correct {
                local_correct.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
            let done = completed.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;

            let result = QuestionResult {
                question_id: question_id_for(&sample_id, &question),
                sample_id,
                question,
                ground_truth: gold,
                hypothesis,
                category_name: cat_name.into(),
                f1,
                judge_correct,
                judge_reasoning,
                confidence,
                elapsed_s: elapsed,
                status,
                error,
                evidence_refs,
                retrieved_turn_refs,
                evidence_hit,
                evidence_recall,
                retrieved_context,
            };

            shared.lock().await.push_and_flush(result);

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

fn git_dirty_worktree() -> Option<bool> {
    let output = Command::new("git")
        .args(["status", "--porcelain"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    Some(!output.stdout.is_empty())
}

// --- Main ---

#[tokio::main]
async fn main() {
    let _ = dotenvy::dotenv();

    let invocation = parse_args();
    let command = invocation.command;
    let artifact_path = invocation.artifact_path.clone();
    let config = invocation.config;
    let output_path = if let Some(ref p) = config.output {
        p.clone()
    } else if let Some(ref tag) = config.tag {
        PathBuf::from(format!("bench/locomo/results/{tag}.json"))
    } else if let Some(ref artifact) = artifact_path {
        artifact.clone()
    } else {
        PathBuf::from(format!(
            "bench/locomo/results/{}-{}.json",
            config.profile.as_str(),
            command.as_str()
        ))
    };

    let artifact_state = artifact_path.as_ref().map(|path| {
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
        prev
    });
    let (existing_banks, existing_turn_refs) = if let Some(prev) = artifact_state.as_ref() {
        (prev.banks.clone(), prev.turn_refs.clone())
    } else {
        (HashMap::new(), HashMap::new())
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
    let runtime = Arc::new(
        build_runtime_from_env(BuildRuntimeOptions {
            metrics: Some(metrics.clone()),
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

    let run_timestamp = Utc::now().to_rfc3339();
    let commit = git_commit_sha();
    let bench_start = Instant::now();
    let total_convs = dataset.len();

    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent).ok();
    }

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
    };

    let shared = Arc::new(Mutex::new(SharedResults {
        results: Vec::new(),
        banks: existing_banks.clone(),
        turn_refs: existing_turn_refs,
        output_path: output_path.clone(),
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
    let total_stage_usage = metrics.total_usage();

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
    println!(
        "Stage usage: {} prompt + {} completion = {} total tokens across {} calls",
        total_stage_usage.prompt_tokens,
        total_stage_usage.completion_tokens,
        total_stage_usage.total_tokens(),
        total_stage_usage.calls,
    );

    shared_snapshot.flush();
    println!();
    println!("Results saved to {}", output_path.display());
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

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
        assert_eq!(config.ingest, IngestMode::Turn);
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
                "ingestion_granularity": "turn",
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
                "ingestion_granularity": "turn",
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
}
