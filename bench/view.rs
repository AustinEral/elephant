//! View a single benchmark result or compare two side-by-side.
//!
//! Usage:
//!     cargo run --bin view -- <file.json>
//!     cargo run --bin view -- <file1.json> <file2.json>

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::env;
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
    stage_metrics: BTreeMap<String, StageUsage>,
    #[serde(default)]
    total_stage_usage: StageUsage,
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
    confidence: f32,
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

fn fmt_stage_value(usage: &StageUsage) -> String {
    if usage.calls == 0 {
        "-".into()
    } else {
        format!("{} tok / {} calls", usage.total_tokens(), usage.calls)
    }
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

fn manifest_scope(manifest: &BenchmarkManifest) -> Option<String> {
    if !manifest.selected_conversations.is_empty() {
        return Some(manifest.selected_conversations.join(","));
    }
    None
}

fn failed_count(results: &[QuestionResult]) -> usize {
    results.iter().filter(|r| r.status != "ok").count()
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
    #[tabled(rename = "conf")]
    conf: String,
    #[tabled(rename = "avg time")]
    avg_time: String,
    failed: usize,
    #[tabled(rename = "n")]
    n: usize,
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

fn view_single(output: &BenchmarkOutput, path: &str) {
    let label = file_label(output, path);
    let evidence_available = has_evidence(output);

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
    if output.total_time_s > 0.0 {
        config_rows.push(SingleConfigRow {
            key: "total time".into(),
            value: fmt_time(output.total_time_s),
        });
    }

    println!("{}", Table::new(&config_rows).with(Style::rounded()));
    println!();

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

    println!(
        "{}",
        Table::new(&summary_rows)
            .with(Style::rounded())
            .with(Modify::new(Columns::new(1..)).with(Alignment::right()))
    );

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
    }
    metrics_rows.push(SingleConfigRow {
        key: "confidence".into(),
        value: format!(
            "{:.3}",
            avg(&output
                .results
                .iter()
                .map(|r| r.confidence as f64)
                .collect::<Vec<_>>())
        ),
    });
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
    }

    println!();
    println!("{}", Table::new(&metrics_rows).with(Style::rounded()));

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

        println!();
        println!(
            "{}",
            Table::new(&stage_rows)
                .with(Style::rounded())
                .with(Modify::new(Columns::new(1..)).with(Alignment::right()))
        );
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
            conf: format!(
                "{:.3}",
                avg(&questions
                    .iter()
                    .map(|r| r.confidence as f64)
                    .collect::<Vec<_>>())
            ),
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
        conf: format!(
            "{:.3}",
            avg(&output
                .results
                .iter()
                .map(|r| r.confidence as f64)
                .collect::<Vec<_>>())
        ),
        avg_time: fmt_time(avg(&output
            .results
            .iter()
            .map(|r| r.elapsed_s)
            .collect::<Vec<_>>())),
        failed: failed_count(&output.results),
        n: output.results.len(),
    });

    println!();
    println!(
        "{}",
        Table::new(&conv_rows)
            .with(Style::rounded())
            .with(Modify::new(Columns::new(1..)).with(Alignment::right()))
    );

    if output.results.is_empty() {
        return;
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

    println!();
    println!(
        "{}",
        Table::new(&question_rows)
            .with(Style::rounded())
            .with(Modify::new(Columns::new(3..=5)).with(Alignment::center()))
    );

    let correct = output.results.iter().filter(|r| r.judge_correct).count();
    let wrong = output.results.len().saturating_sub(correct);
    let failed = failed_count(&output.results);
    let evidence_hits = output.results.iter().filter(|r| r.evidence_hit).count();
    println!();
    if evidence_available {
        println!(
            "\x1b[32m{correct} correct\x1b[0m, \x1b[31m{wrong} wrong\x1b[0m, {failed} failed, {evidence_hits} evidence hits"
        );
    } else {
        println!("\x1b[32m{correct} correct\x1b[0m, \x1b[31m{wrong} wrong\x1b[0m, {failed} failed");
    }
}

fn load_file(path: &str) -> BenchmarkOutput {
    let raw = fs::read_to_string(path).unwrap_or_else(|e| {
        eprintln!("Failed to read {path}: {e}");
        process::exit(1);
    });
    serde_json::from_str(&raw).unwrap_or_else(|e| {
        eprintln!("Failed to parse {path}: {e}");
        process::exit(1);
    })
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

    let evidence_available = has_evidence(&a) || has_evidence(&b);
    let label_a = file_label(&a, &files[0]);
    let label_b = file_label(&b, &files[1]);

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

    println!("{}", Table::new(&config_rows).with(Style::rounded()));
    println!();

    let mut summary_rows = Vec::new();
    for category in &categories {
        if let Some(&(correct_a, correct_b, total)) = cat_stats.get(category.as_str()) {
            let acc_a = correct_a as f64 / total.max(1) as f64;
            let acc_b = correct_b as f64 / total.max(1) as f64;
            summary_rows.push(SummaryRow {
                category: category.clone(),
                acc_a: fmt_pct(acc_a),
                acc_b: fmt_pct(acc_b),
                delta: fmt_pct_delta(acc_a, acc_b),
                n: total,
            });
        }
    }
    let total_a = matched
        .iter()
        .filter(|(left, _)| left.judge_correct)
        .count();
    let total_b = matched
        .iter()
        .filter(|(_, right)| right.judge_correct)
        .count();
    let total_n = matched.len();
    let acc_a = total_a as f64 / total_n.max(1) as f64;
    let acc_b = total_b as f64 / total_n.max(1) as f64;
    summary_rows.push(SummaryRow {
        category: "TOTAL".into(),
        acc_a: fmt_pct(acc_a),
        acc_b: fmt_pct(acc_b),
        delta: fmt_pct_delta(acc_a, acc_b),
        n: total_n,
    });

    println!(
        "{}",
        Table::new(&summary_rows)
            .with(Style::rounded())
            .with(Modify::new(Columns::new(1..)).with(Alignment::right()))
    );

    let mut metrics_rows = vec![
        MetricsRow {
            metric: "accuracy".into(),
            val_a: fmt_pct(acc_a),
            val_b: fmt_pct(acc_b),
            delta: fmt_pct_delta(acc_a, acc_b),
        },
        MetricsRow {
            metric: "F1".into(),
            val_a: format!(
                "{:.3}",
                avg(&matched.iter().map(|(left, _)| left.f1).collect::<Vec<_>>())
            ),
            val_b: format!(
                "{:.3}",
                avg(&matched
                    .iter()
                    .map(|(_, right)| right.f1)
                    .collect::<Vec<_>>())
            ),
            delta: fmt_float_delta(
                avg(&matched.iter().map(|(left, _)| left.f1).collect::<Vec<_>>()),
                avg(&matched
                    .iter()
                    .map(|(_, right)| right.f1)
                    .collect::<Vec<_>>()),
                3,
                true,
            ),
        },
    ];

    if evidence_available {
        let evidence_a = avg(&matched
            .iter()
            .map(|(left, _)| left.evidence_recall)
            .collect::<Vec<_>>());
        let evidence_b = avg(&matched
            .iter()
            .map(|(_, right)| right.evidence_recall)
            .collect::<Vec<_>>());
        metrics_rows.push(MetricsRow {
            metric: "evidence recall".into(),
            val_a: format!("{evidence_a:.3}"),
            val_b: format!("{evidence_b:.3}"),
            delta: fmt_float_delta(evidence_a, evidence_b, 3, true),
        });
    }

    let conf_a = avg(&matched
        .iter()
        .map(|(left, _)| left.confidence as f64)
        .collect::<Vec<_>>());
    let conf_b = avg(&matched
        .iter()
        .map(|(_, right)| right.confidence as f64)
        .collect::<Vec<_>>());
    metrics_rows.push(MetricsRow {
        metric: "confidence".into(),
        val_a: format!("{conf_a:.3}"),
        val_b: format!("{conf_b:.3}"),
        delta: fmt_float_delta(conf_a, conf_b, 3, true),
    });

    let time_a = avg(&matched
        .iter()
        .map(|(left, _)| left.elapsed_s)
        .collect::<Vec<_>>());
    let time_b = avg(&matched
        .iter()
        .map(|(_, right)| right.elapsed_s)
        .collect::<Vec<_>>());
    metrics_rows.push(MetricsRow {
        metric: "avg time".into(),
        val_a: fmt_time(time_a),
        val_b: fmt_time(time_b),
        delta: fmt_float_delta(time_a, time_b, 1, false),
    });

    let failed_a = matched
        .iter()
        .filter(|(left, _)| left.status != "ok")
        .count() as u64;
    let failed_b = matched
        .iter()
        .filter(|(_, right)| right.status != "ok")
        .count() as u64;
    metrics_rows.push(MetricsRow {
        metric: "failed".into(),
        val_a: failed_a.to_string(),
        val_b: failed_b.to_string(),
        delta: fmt_cost_delta_u64(failed_a, failed_b),
    });

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
            delta: fmt_cost_delta_u64(
                a.total_stage_usage.latency_ms,
                b.total_stage_usage.latency_ms,
            ),
        });
    }

    println!();
    println!(
        "{}",
        Table::new(&metrics_rows)
            .with(Style::rounded())
            .with(Modify::new(Columns::new(1..)).with(Alignment::right()))
    );

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

        println!();
        println!(
            "{}",
            Table::new(&stage_rows)
                .with(Style::rounded())
                .with(Modify::new(Columns::new(1..)).with(Alignment::right()))
        );
    }

    if matched.is_empty() {
        return;
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

    println!();
    println!(
        "{}",
        Table::new(&question_rows)
            .with(Style::rounded())
            .with(Modify::new(Columns::new(3..=4)).with(Alignment::center()))
    );

    let regressions = matched
        .iter()
        .filter(|(left, right)| left.judge_correct && !right.judge_correct)
        .count();
    let improvements = matched
        .iter()
        .filter(|(left, right)| !left.judge_correct && right.judge_correct)
        .count();
    let unchanged = matched.len().saturating_sub(improvements + regressions);
    println!();
    println!(
        "\x1b[32m{improvements} improved\x1b[0m, \x1b[31m{regressions} regressed\x1b[0m, {unchanged} unchanged"
    );
}
