//! LoCoMo benchmark harness for Elephant.
//!
//! Ingests conversations via the retain API, asks questions via reflect,
//! and scores answers with token-level F1 + LLM-as-judge.
//!
//! Usage:
//!     cargo run --bin locomo-bench -- --data data/locomo10.json
//!
//! Config from .env: LLM_PROVIDER, LLM_API_KEY, LLM_MODEL, ELEPHANT_URL

use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::env;
use std::fs;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use chrono::{DateTime, NaiveDateTime, Utc};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use tokio::sync::{Mutex, Semaphore};

use elephant::llm::retry::{RetryPolicy, RetryingLlmClient};
use elephant::llm::{self, LlmClient, Provider, ProviderConfig};
use elephant::types::llm::{CompletionRequest, Message};

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
}

#[derive(Debug, Deserialize)]
struct QaPair {
    question: String,
    /// Missing for category 5 (adversarial), which uses `adversarial_answer` instead.
    answer: Option<serde_json::Value>,
    category: u8,
}

// --- API types (match Elephant's HTTP interface) ---

#[derive(Debug, Serialize)]
struct RetainRequest {
    bank_id: String,
    content: String,
    timestamp: String,
}

#[derive(Debug, Deserialize)]
struct RetainResponse {
    facts_stored: usize,
}

#[derive(Debug, Serialize)]
struct ReflectRequest {
    bank_id: String,
    question: String,
    budget_tokens: usize,
}

#[derive(Debug, Deserialize)]
struct ReflectResponse {
    response: String,
    confidence: f32,
}

#[derive(Debug, Serialize)]
struct CreateBankRequest {
    name: String,
    mission: String,
}

#[derive(Debug, Deserialize)]
struct CreateBankResponse {
    id: String,
}

#[derive(Debug, Deserialize)]
struct ServerInfoResponse {
    retain_model: String,
    reflect_model: String,
    embedding_model: String,
    reranker_model: String,
}

#[derive(Debug, Serialize)]
struct ConsolidateRequest {}

#[derive(Debug, Deserialize)]
struct ConsolidateResponse {
    observations_created: usize,
    observations_updated: usize,
}

// --- Judge types ---

#[derive(Debug, Deserialize)]
struct JudgeResponse {
    reasoning: String,
    label: String,
}

// --- Result types ---

#[derive(Debug, Clone, Serialize, Deserialize)]
struct BenchmarkOutput {
    benchmark: String,
    timestamp: String,
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
    per_category: HashMap<String, CategoryResult>,
    /// Maps sample_id → bank_id for resuming without re-ingestion.
    #[serde(default)]
    banks: HashMap<String, String>,
    results: Vec<QuestionResult>,
    total_time_s: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CategoryResult {
    accuracy: f64,
    mean_f1: f64,
    count: usize,
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

// --- Token F1 ---

fn normalize_answer(s: &str) -> String {
    let lower = s.to_lowercase();
    let no_punct: String = lower
        .chars()
        .map(|c| if c.is_ascii_punctuation() { ' ' } else { c })
        .collect();
    // Remove articles
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

    // Count common tokens (multiset intersection)
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

fn parse_session_date(date_str: &str) -> String {
    // Format: "1:56 pm on 8 May, 2023"
    let cleaned = date_str.trim();
    if let Ok(dt) = NaiveDateTime::parse_from_str(cleaned, "%I:%M %p on %-d %B, %Y") {
        let utc: DateTime<Utc> = dt.and_utc();
        return utc.to_rfc3339();
    }
    // Fallback: try without comma
    if let Ok(dt) = NaiveDateTime::parse_from_str(cleaned, "%I:%M %p on %-d %B %Y") {
        let utc: DateTime<Utc> = dt.and_utc();
        return utc.to_rfc3339();
    }
    Utc::now().to_rfc3339()
}

fn format_session(turns: &[Turn], session_date: &str) -> String {
    let mut lines = vec![format!("[{session_date}]")];
    for turn in turns {
        lines.push(format!("{}: {}", turn.speaker, turn.text));
    }
    lines.join("\n")
}

fn session_count(conv: &Conversation) -> usize {
    let mut n = 0;
    while conv.sessions.contains_key(&format!("session_{}", n + 1)) {
        n += 1;
    }
    n
}

fn get_session_turns(conv: &Conversation, idx: usize) -> Vec<Turn> {
    let key = format!("session_{idx}");
    conv.sessions
        .get(&key)
        .and_then(|v| serde_json::from_value::<Vec<Turn>>(v.clone()).ok())
        .unwrap_or_default()
}

fn get_session_date(conv: &Conversation, idx: usize) -> String {
    let key = format!("session_{idx}_date_time");
    conv.sessions
        .get(&key)
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string()
}

fn answer_to_string(val: &serde_json::Value) -> String {
    match val {
        serde_json::Value::String(s) => s.clone(),
        serde_json::Value::Number(n) => n.to_string(),
        other => other.to_string(),
    }
}

// --- LLM Judge ---

const JUDGE_PROMPT: &str = include_str!("judge_answer.txt");

async fn llm_judge(
    judge: &dyn LlmClient,
    question: &str,
    gold_answer: &str,
    generated_answer: &str,
) -> (bool, String) {
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

    // RetryingLlmClient handles rate-limit/server-error retries.
    // We only retry here on parse failures (LLM returned unparseable text).
    for attempt in 0..3 {
        let result = judge.complete(request.clone()).await;
        match result {
            Ok(resp) => {
                // Try to parse as structured JSON
                if let Ok(parsed) = serde_json::from_str::<JudgeResponse>(&resp.content) {
                    let correct = parsed.label.eq_ignore_ascii_case("CORRECT");
                    return (correct, parsed.reasoning);
                }
                // Try extracting JSON from prose
                if let Ok(json_str) = llm::extract_json(&resp.content)
                    && let Ok(parsed) = serde_json::from_str::<JudgeResponse>(&json_str)
                {
                    let correct = parsed.label.eq_ignore_ascii_case("CORRECT");
                    return (correct, parsed.reasoning);
                }
                if attempt == 2 {
                    return (
                        false,
                        format!(
                            "could not parse judge response: {}",
                            &resp.content[..resp.content.len().min(100)]
                        ),
                    );
                }
            }
            Err(e) => {
                return (false, format!("judge error: {e}"));
            }
        }
    }
    (false, "judge failed after retries".into())
}

// --- HTTP helpers ---

async fn api_post<T: serde::de::DeserializeOwned>(
    client: &Client,
    url: &str,
    body: &impl Serialize,
) -> Result<T, String> {
    let resp = client
        .post(url)
        .json(body)
        .send()
        .await
        .map_err(|e| format!("request failed: {e}"))?;
    let status = resp.status();
    let text = resp.text().await.map_err(|e| format!("read error: {e}"))?;
    if !status.is_success() {
        return Err(format!("HTTP {status}: {text}"));
    }
    serde_json::from_str(&text).map_err(|e| format!("parse error: {e}\nbody: {text}"))
}

// --- Time formatting ---

fn fmt_elapsed(seconds: f64) -> String {
    let m = (seconds as u64) / 60;
    let s = (seconds as u64) % 60;
    format!("{m}:{s:02}")
}

// --- Incremental results flushing ---

fn flush_results(results: &[QuestionResult], banks: &HashMap<String, String>, output_path: &Path, judge_label: &str, tag: &Option<String>, retain_model: &str, reflect_model: &str, embedding_model: &str, reranker_model: &str, consolidation_strategy: &str, bench_start: Instant) {
    let bench_elapsed = bench_start.elapsed().as_secs_f64();
    let total_questions = results.len();

    let mut category_f1: HashMap<String, Vec<f64>> = HashMap::new();
    let mut category_judge: HashMap<String, Vec<f64>> = HashMap::new();
    for r in results {
        category_f1.entry(r.category_name.clone()).or_default().push(r.f1);
        let score = if r.judge_correct { 1.0 } else { 0.0 };
        category_judge.entry(r.category_name.clone()).or_default().push(score);
    }

    let mean_f1 = if total_questions > 0 {
        results.iter().map(|r| r.f1).sum::<f64>() / total_questions as f64
    } else {
        0.0
    };
    let total_correct: f64 = category_judge.values().map(|v| v.iter().sum::<f64>()).sum();
    let total_judged: usize = category_judge.values().map(|v| v.len()).sum();
    let accuracy = if total_judged > 0 {
        total_correct / total_judged as f64
    } else {
        0.0
    };

    let mut per_category = HashMap::new();
    for (name, f1_scores) in &category_f1 {
        let j_scores = category_judge.get(name);
        let n = f1_scores.len();
        per_category.insert(
            name.clone(),
            CategoryResult {
                accuracy: j_scores
                    .map(|v| v.iter().sum::<f64>() / v.len() as f64)
                    .unwrap_or(0.0),
                mean_f1: f1_scores.iter().sum::<f64>() / n as f64,
                count: n,
            },
        );
    }

    let output = BenchmarkOutput {
        benchmark: "locomo".into(),
        timestamp: Utc::now().to_rfc3339(),
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
        per_category,
        banks: banks.clone(),
        results: results.to_vec(),
        total_time_s: bench_elapsed,
    };

    if let Ok(json) = serde_json::to_string_pretty(&output) {
        let _ = fs::write(output_path, &json);
    }
}

// --- CLI ---

struct Args {
    data: PathBuf,
    api_url: String,
    output: Option<PathBuf>,
    tag: Option<String>,
    max_conversations: Option<usize>,
    max_sessions: Option<usize>,
    max_questions: Option<usize>,
    bank_id: Option<String>,
    judge_model: Option<String>,
    concurrency: usize,
    question_concurrency: usize,
    consolidate: bool,
    consolidate_per_session: bool,
    /// Resume from a previous results file — reuses bank IDs to skip ingestion.
    resume: Option<PathBuf>,
    /// Ingest only — skip question phase (useful for separating ingestion from evaluation).
    ingest_only: bool,
}

fn parse_args() -> Args {
    let mut args = Args {
        data: PathBuf::from("data/locomo10.json"),
        api_url: "http://localhost:3001".into(),
        output: None,
        tag: None,
        max_conversations: None,
        max_sessions: None,
        max_questions: None,
        bank_id: None,
        judge_model: None,
        concurrency: 1,
        question_concurrency: 1,
        consolidate: true,
        consolidate_per_session: false,
        resume: None,
        ingest_only: false,
    };

    let raw: Vec<String> = env::args().collect();
    let mut i = 1;
    while i < raw.len() {
        match raw[i].as_str() {
            "--data" => {
                i += 1;
                args.data = PathBuf::from(&raw[i]);
            }
            "--api-url" => {
                i += 1;
                args.api_url = raw[i].clone();
            }
            "--output" => {
                i += 1;
                args.output = Some(PathBuf::from(&raw[i]));
            }
            "--tag" => {
                i += 1;
                args.tag = Some(raw[i].clone());
            }
            "--max-conversations" => {
                i += 1;
                args.max_conversations = Some(raw[i].parse().expect("invalid --max-conversations"));
            }
            "--max-sessions" => {
                i += 1;
                args.max_sessions = Some(raw[i].parse().expect("invalid --max-sessions"));
            }
            "--max-questions" => {
                i += 1;
                args.max_questions = Some(raw[i].parse().expect("invalid --max-questions"));
            }
            "--bank-id" => {
                i += 1;
                args.bank_id = Some(raw[i].clone());
            }
            "--judge-model" => {
                i += 1;
                args.judge_model = Some(raw[i].clone());
            }
            "--conversation-concurrency" => {
                i += 1;
                args.concurrency = raw[i].parse().expect("invalid --conversation-concurrency");
            }
            "--question-concurrency" => {
                i += 1;
                args.question_concurrency = raw[i].parse().expect("invalid --question-concurrency");
            }
            "--no-consolidate" => {
                args.consolidate = false;
            }
            "--consolidate-per-session" => {
                args.consolidate_per_session = true;
            }
            "--resume" => {
                i += 1;
                args.resume = Some(PathBuf::from(&raw[i]));
            }
            "--ingest-only" => {
                args.ingest_only = true;
            }
            "--help" | "-h" => {
                eprintln!("Usage: locomo-bench [OPTIONS]");
                eprintln!();
                eprintln!("Options:");
                eprintln!(
                    "  --data <PATH>              Dataset path [default: data/locomo10.json]"
                );
                eprintln!(
                    "  --api-url <URL>            Elephant API URL [default: http://localhost:3001]"
                );
                eprintln!(
                    "  --output <PATH>            Output results path (overrides --tag)"
                );
                eprintln!(
                    "  --tag <NAME>               Save to bench/locomo/results/<tag>.json"
                );
                eprintln!("  --max-conversations <N>    Limit conversations");
                eprintln!("  --max-sessions <N>         Limit sessions ingested per conversation");
                eprintln!("  --max-questions <N>        Limit questions per conversation");
                eprintln!("  --bank-id <ID>             Reuse existing bank (skip ingestion)");
                eprintln!("  --judge-model <MODEL>      Override LLM_MODEL for judge");
                eprintln!(
                    "  --conversation-concurrency <N>  Max parallel conversations [default: 1]"
                );
                eprintln!(
                    "  --question-concurrency <N>      Max parallel questions per conversation [default: 1]"
                );
                eprintln!(
                    "  --no-consolidate                Skip consolidation after ingestion"
                );
                eprintln!(
                    "  --consolidate-per-session       Consolidate after each session (incremental)"
                );
                eprintln!(
                    "  --resume <PATH>                 Resume from previous results (reuse bank IDs)"
                );
                eprintln!(
                    "  --ingest-only                   Ingest only, skip question phase"
                );
                std::process::exit(0);
            }
            other => {
                eprintln!("Unknown argument: {other}");
                std::process::exit(1);
            }
        }
        i += 1;
    }
    args
}

// --- Shared state for incremental writes ---

struct SharedResults {
    results: Vec<QuestionResult>,
    banks: HashMap<String, String>,
    output_path: PathBuf,
    judge_label: String,
    tag: Option<String>,
    retain_model: String,
    reflect_model: String,
    embedding_model: String,
    reranker_model: String,
    consolidation_strategy: String,
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

    fn flush(&self) {
        flush_results(&self.results, &self.banks, &self.output_path, &self.judge_label, &self.tag, &self.retain_model, &self.reflect_model, &self.embedding_model, &self.reranker_model, &self.consolidation_strategy, self.bench_start);
    }
}

// --- Per-conversation worker ---

async fn run_conversation(
    tag: String,
    http: Client,
    api_url: String,
    entry: LocomoEntry,
    judge: Arc<dyn LlmClient>,
    max_sessions: Option<usize>,
    max_questions: Option<usize>,
    question_concurrency: usize,
    consolidate: bool,
    consolidate_per_session: bool,
    reuse_bank: Option<String>,
    ingest_only: bool,
    shared: Arc<Mutex<SharedResults>>,
) {
    let conv = &entry.conversation;
    let total_sessions = session_count(conv);

    println!("[{tag}] {} & {} ({})", conv.speaker_a, conv.speaker_b, entry.sample_id);

    // Ingestion phase
    let bank_id = if let Some(id) = reuse_bank {
        println!("[{tag}] Reusing bank: {id} (skipping ingestion)");
        id
    } else {
        let bank: CreateBankResponse = api_post(
            &http,
            &format!("{api_url}/v1/banks"),
            &CreateBankRequest {
                name: format!("locomo-{}", entry.sample_id),
                mission: "Long-term conversational memory benchmark".into(),
            },
        )
        .await
        .expect("failed to create bank");

        let ingest_sessions = max_sessions.map(|m| m.min(total_sessions)).unwrap_or(total_sessions);
        println!("[{tag}] Bank: {} | Ingesting {ingest_sessions}/{total_sessions} sessions...", bank.id);

        let ingest_start = Instant::now();
        let mut total_facts = 0usize;
        let mut session_times: Vec<f64> = Vec::new();

        for idx in 1..=ingest_sessions {
            let turns = get_session_turns(conv, idx);
            let date_str = get_session_date(conv, idx);
            let timestamp = parse_session_date(&date_str);
            let text = format_session(&turns, &date_str);

            let t0 = Instant::now();
            let resp: RetainResponse = api_post(
                &http,
                &format!("{api_url}/v1/banks/{}/retain", bank.id),
                &RetainRequest {
                    bank_id: bank.id.clone(),
                    content: text,
                    timestamp,
                },
            )
            .await
            .expect("retain failed");

            let elapsed = t0.elapsed().as_secs_f64();
            session_times.push(elapsed);
            total_facts += resp.facts_stored;

            let avg_time = session_times.iter().sum::<f64>() / session_times.len() as f64;
            let remaining = ingest_sessions - idx;
            let eta = avg_time * remaining as f64;
            let total_elapsed = ingest_start.elapsed().as_secs_f64();

            println!(
                "[{tag}] ingest [{idx}/{ingest_sessions}] {} facts ({elapsed:.0}s) | total: {total_facts} | elapsed: {} | ETA: {}",
                resp.facts_stored,
                fmt_elapsed(total_elapsed),
                fmt_elapsed(eta),
            );

            // Per-session consolidation (incremental edit path)
            if consolidate_per_session && resp.facts_stored > 0 {
                let consolidate_url = format!("{api_url}/v1/banks/{}/consolidate", bank.id);
                match api_post::<ConsolidateResponse>(
                    &http,
                    &consolidate_url,
                    &ConsolidateRequest {},
                )
                .await
                {
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

        let total_elapsed = ingest_start.elapsed().as_secs_f64();
        println!(
            "[{tag}] Ingestion complete: {total_facts} facts in {}",
            fmt_elapsed(total_elapsed)
        );

        bank.id
    };

    // Record bank ID for resume
    shared.lock().await.record_bank(entry.sample_id.clone(), bank_id.clone());

    // Consolidation phase (optional) — skip if already consolidated per-session
    if consolidate && !consolidate_per_session {
        println!("[{tag}] Consolidating...");
        let t0 = Instant::now();
        let consolidate_url = format!("{api_url}/v1/banks/{bank_id}/consolidate");
        match api_post::<ConsolidateResponse>(
            &http,
            &consolidate_url,
            &ConsolidateRequest {},
        )
        .await
        {
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
                eprintln!("[{tag}] Consolidation failed: {e}");
            }
        }
    }

    // Skip questions if ingest-only mode
    if ingest_only {
        println!("[{tag}] Ingest-only mode — skipping questions");
        return;
    }

    // Question phase
    let qa_list: &[QaPair] = if let Some(max) = max_questions {
        &entry.qa[..max.min(entry.qa.len())]
    } else {
        &entry.qa
    };

    println!("[{tag}] Asking {} questions (concurrency: {question_concurrency})...", qa_list.len());
    let qa_start = Instant::now();
    let local_correct = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let local_total = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let completed = Arc::new(std::sync::atomic::AtomicUsize::new(0));

    let qa_sem = Arc::new(Semaphore::new(question_concurrency));
    let mut qa_handles = Vec::new();

    for (qa_idx, qa) in qa_list.iter().enumerate() {
        // Skip category 5 (adversarial/unanswerable) — no ground truth answer.
        let gold = match &qa.answer {
            Some(val) => answer_to_string(val),
            None => continue,
        };

        let sem = qa_sem.clone();
        let http = http.clone();
        let api_url = api_url.clone();
        let bank_id = bank_id.clone();
        let judge = judge.clone();
        let shared = shared.clone();
        let tag = tag.clone();
        let sample_id = entry.sample_id.clone();
        let question = qa.question.clone();
        let category = qa.category;
        let qa_len = qa_list.len();
        let local_correct = local_correct.clone();
        let local_total = local_total.clone();
        let completed = completed.clone();
        let qa_start = qa_start;

        qa_handles.push(tokio::spawn(async move {
            let _permit = sem.acquire().await.expect("semaphore closed");
            let cat_name = category_name(category);

            let t0 = Instant::now();
            let (hypothesis, confidence) = match api_post::<ReflectResponse>(
                &http,
                &format!("{api_url}/v1/banks/{bank_id}/reflect"),
                &ReflectRequest {
                    bank_id: bank_id.clone(),
                    question: question.clone(),
                    budget_tokens: 4096,
                },
            )
            .await
            {
                Ok(resp) => (resp.response, resp.confidence),
                Err(e) => {
                    eprintln!("[{tag}] [{}/{}] ERROR: {e}", qa_idx + 1, qa_len);
                    (String::new(), 0.0)
                }
            };
            let elapsed = t0.elapsed().as_secs_f64();

            let f1 = token_f1(&hypothesis, &gold);

            // LLM judge
            let (judge_correct, judge_reasoning) = if hypothesis.is_empty() {
                (false, "empty response".into())
            } else {
                llm_judge(judge.as_ref(), &question, &gold, &hypothesis).await
            };

            local_total.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if judge_correct {
                local_correct.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
            let done = completed.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;

            let qid = {
                let mut h = DefaultHasher::new();
                sample_id.hash(&mut h);
                question.hash(&mut h);
                format!("{:06x}", h.finish() & 0xFFFFFF)
            };
            let result = QuestionResult {
                question_id: qid,
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
            };

            // Push to shared results and flush to disk
            shared.lock().await.push_and_flush(result);

            // Progress
            let total = local_total.load(std::sync::atomic::Ordering::Relaxed);
            let correct = local_correct.load(std::sync::atomic::Ordering::Relaxed);
            let running_acc = if total > 0 { correct as f64 / total as f64 } else { 0.0 };
            let qa_elapsed = qa_start.elapsed().as_secs_f64();
            let avg_time = if done > 0 { qa_elapsed / done as f64 } else { 0.0 };
            let remaining = qa_len.saturating_sub(done);
            let eta = avg_time * remaining as f64;

            let label = if judge_correct { "CORRECT" } else { "WRONG  " };
            println!(
                "[{tag}] [{done}/{}] {label} F1={f1:.2} ({cat_name}) | acc: {:.1}% | elapsed: {} | ETA: {}",
                qa_len,
                running_acc * 100.0,
                fmt_elapsed(qa_elapsed),
                fmt_elapsed(eta),
            );
        }));
    }

    for handle in qa_handles {
        handle.await.expect("question task panicked");
    }
}

// --- Main ---

#[tokio::main]
async fn main() {
    let _ = dotenvy::dotenv();

    let args = parse_args();

    // Resolve output path: --output overrides --tag, --tag maps to results/<tag>.json, default is locomo.json
    let output_path = if let Some(ref p) = args.output {
        p.clone()
    } else if let Some(ref tag) = args.tag {
        PathBuf::from(format!("bench/locomo/results/{tag}.json"))
    } else {
        PathBuf::from("bench/locomo/results/locomo.json")
    };

    // Validate dataset exists
    if !args.data.exists() {
        eprintln!("Dataset not found: {}", args.data.display());
        eprintln!("Download it:");
        eprintln!("  mkdir -p data");
        eprintln!(
            "  curl -o data/locomo10.json https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"
        );
        std::process::exit(1);
    }

    // Check Elephant is reachable
    let http = Client::builder()
        .timeout(std::time::Duration::from_secs(300))
        .build()
        .expect("failed to build HTTP client");
    // Fetch server info (model config)
    let server_info: ServerInfoResponse = match http.get(format!("{}/v1/info", args.api_url)).send().await {
        Ok(resp) if resp.status().is_success() => {
            resp.json().await.unwrap_or_else(|e| {
                eprintln!("Failed to parse /v1/info: {e}");
                std::process::exit(1);
            })
        }
        Ok(resp) => {
            eprintln!(
                "Elephant returned HTTP {}: is the server running?",
                resp.status()
            );
            eprintln!("  Start it with: docker compose up -d");
            std::process::exit(1);
        }
        Err(e) => {
            eprintln!("Cannot reach Elephant at {}: {e}", args.api_url);
            eprintln!("  Start it with: docker compose up -d");
            std::process::exit(1);
        }
    };
    println!("retain_model: {}", server_info.retain_model);
    println!("reflect_model: {}", server_info.reflect_model);
    println!("reranker_model: {}", server_info.reranker_model);
    println!("embedding_model: {}", server_info.embedding_model);

    // Build LLM judge client.
    // JUDGE_* env vars override LLM_* so you can use a different model for judging.
    let judge_provider_str = env::var("JUDGE_PROVIDER")
        .or_else(|_| env::var("LLM_PROVIDER"))
        .expect("JUDGE_PROVIDER or LLM_PROVIDER must be set");
    let judge_api_key = env::var("JUDGE_API_KEY")
        .or_else(|_| env::var("LLM_API_KEY"))
        .expect("JUDGE_API_KEY or LLM_API_KEY must be set");
    let judge_model = args.judge_model.clone().unwrap_or_else(|| {
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
        model: judge_model.clone(),
        base_url: None,
    };
    let judge: Arc<dyn LlmClient> = Arc::new(RetryingLlmClient::new(
        Arc::from(llm::build_client(&judge_config)),
        RetryPolicy::default(),
    ));

    let judge_label = format!("{judge_provider_str}/{judge_model}");
    println!("LLM judge: {judge_label}");
    // Load bank IDs from a previous run (--resume)
    let resume_banks: HashMap<String, String> = if let Some(ref path) = args.resume {
        let raw = fs::read_to_string(path).expect("failed to read resume file");
        let prev: BenchmarkOutput = serde_json::from_str(&raw).expect("failed to parse resume file");
        println!("Resuming from {} ({} banks)", path.display(), prev.banks.len());
        prev.banks
    } else {
        HashMap::new()
    };

    println!("Conversation concurrency: {}", args.concurrency);
    println!("Question concurrency: {}", args.question_concurrency);

    // Load dataset
    let raw = fs::read_to_string(&args.data).expect("failed to read dataset");
    let mut dataset: Vec<LocomoEntry> =
        serde_json::from_str(&raw).expect("failed to parse dataset");

    if let Some(max) = args.max_conversations {
        dataset.truncate(max);
    }

    let total_convs = dataset.len();
    let bench_start = Instant::now();

    // Ensure output directory exists
    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent).ok();
    }

    // Shared results — flushed to disk after each question
    let shared = Arc::new(Mutex::new(SharedResults {
        results: Vec::new(),
        banks: HashMap::new(),
        output_path: output_path.clone(),
        judge_label: judge_label.clone(),
        tag: args.tag.clone(),
        retain_model: server_info.retain_model.clone(),
        reflect_model: server_info.reflect_model.clone(),
        embedding_model: server_info.embedding_model.clone(),
        reranker_model: server_info.reranker_model.clone(),
        consolidation_strategy: if args.consolidate_per_session {
            "per-session".into()
        } else if args.consolidate {
            "end".into()
        } else {
            "none".into()
        },
        bench_start,
    }));

    // Spawn conversations concurrently, capped by semaphore.
    let semaphore = Arc::new(Semaphore::new(args.concurrency));
    let mut handles = Vec::new();

    for (conv_idx, entry) in dataset.into_iter().enumerate() {
        let sem = semaphore.clone();
        let http = http.clone();
        let api_url = args.api_url.clone();
        let judge = judge.clone();
        let max_sessions = args.max_sessions;
        let max_questions = args.max_questions;
        let question_concurrency = args.question_concurrency;
        let consolidate = args.consolidate;
        let consolidate_per_session = args.consolidate_per_session;
        let reuse_bank = resume_banks.get(&entry.sample_id).cloned()
            .or_else(|| if conv_idx == 0 { args.bank_id.clone() } else { None });
        let tag = format!("conv {}/{total_convs}", conv_idx + 1);
        let shared = shared.clone();
        let ingest_only = args.ingest_only;

        handles.push(tokio::spawn(async move {
            let _permit = sem.acquire().await.expect("semaphore closed");
            run_conversation(tag, http, api_url, entry, judge, max_sessions, max_questions, question_concurrency, consolidate, consolidate_per_session, reuse_bank, ingest_only, shared).await
        }));
    }

    // Wait for all conversations.
    for handle in handles {
        handle.await.expect("task panicked");
    }

    // Final aggregation from shared results.
    let all_results = {
        let lock = shared.lock().await;
        lock.results.clone()
    };

    let bench_elapsed = bench_start.elapsed().as_secs_f64();
    let total_questions = all_results.len();

    let mut category_f1: HashMap<String, Vec<f64>> = HashMap::new();
    let mut category_judge: HashMap<String, Vec<f64>> = HashMap::new();
    for r in &all_results {
        category_f1.entry(r.category_name.clone()).or_default().push(r.f1);
        let score = if r.judge_correct { 1.0 } else { 0.0 };
        category_judge.entry(r.category_name.clone()).or_default().push(score);
    }

    let mean_f1 = if total_questions > 0 {
        all_results.iter().map(|r| r.f1).sum::<f64>() / total_questions as f64
    } else {
        0.0
    };
    let total_correct: f64 = category_judge.values().map(|v| v.iter().sum::<f64>()).sum();
    let total_judged: usize = category_judge.values().map(|v| v.len()).sum();
    let accuracy = if total_judged > 0 {
        total_correct / total_judged as f64
    } else {
        0.0
    };

    println!();
    println!("{}", "=".repeat(60));
    println!("LOCOMO BENCHMARK RESULTS");
    println!("{}", "=".repeat(60));
    println!("Total questions: {total_questions}");
    println!("Total time: {}", fmt_elapsed(bench_elapsed));
    println!("Judge model: {judge_label}");
    println!();
    println!(
        "Accuracy (LLM judge): {:.1}% ({}/{})",
        accuracy * 100.0,
        total_correct as usize,
        total_judged
    );
    println!("Token F1 (reference): {mean_f1:.4}");
    println!();
    println!("Per-category breakdown:");
    println!("  {:15}  {:>6}  {:>6}  {:>4}", "category", "acc", "F1", "n");
    println!(
        "  {:15}  {:>6}  {:>6}  {:>4}",
        "-".repeat(15),
        "-".repeat(6),
        "-".repeat(6),
        "-".repeat(4)
    );

    let mut cat_names: Vec<&String> = category_f1.keys().collect();
    cat_names.sort();
    for name in cat_names {
        let f1_scores = &category_f1[name];
        let j_scores = category_judge.get(name);
        let n = f1_scores.len();
        let f1_avg = f1_scores.iter().sum::<f64>() / n as f64;
        let j_avg = j_scores
            .map(|v| v.iter().sum::<f64>() / v.len() as f64)
            .unwrap_or(0.0);
        println!("  {name:15}  {:.1}%  {f1_avg:.3}  {n:>4}", j_avg * 100.0);
    }

    // Final flush
    let all_banks = {
        let lock = shared.lock().await;
        lock.banks.clone()
    };
    let consolidation_strategy = if args.consolidate_per_session {
        "per-session"
    } else if args.consolidate {
        "end"
    } else {
        "none"
    };
    flush_results(&all_results, &all_banks, &output_path, &judge_label, &args.tag, &server_info.retain_model, &server_info.reflect_model, &server_info.embedding_model, &server_info.reranker_model, consolidation_strategy, bench_start);
    println!();
    println!("Results saved to {}", output_path.display());
}
