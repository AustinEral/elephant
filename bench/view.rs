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
use tabled::settings::style::Style;
use tabled::settings::{Alignment, Modify};
use tabled::settings::object::Columns;
use tabled::{Table, Tabled};

#[derive(Debug, Deserialize)]
struct BenchmarkOutput {
    #[serde(default)]
    tag: Option<String>,
    judge_model: String,
    #[serde(default)]
    retain_model: String,
    #[serde(default)]
    reflect_model: String,
    #[serde(default)]
    embedding_model: String,
    #[serde(default)]
    consolidation_strategy: String,
    total_questions: usize,
    accuracy: f64,
    #[serde(default)]
    results: Vec<QuestionResult>,
    total_time_s: f64,
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
    #[serde(default)]
    f1: f64,
    #[serde(default)]
    confidence: f32,
    #[serde(default)]
    elapsed_s: f64,
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

fn fmt_time(s: f64) -> String {
    if s < 60.0 {
        format!("{s:.1}s")
    } else {
        let m = (s / 60.0).floor() as u64;
        let rem = s - m as f64 * 60.0;
        format!("{m}m{rem:.0}s")
    }
}

fn fmt_pct(v: f64) -> String {
    format!("{:.1}%", v * 100.0)
}

fn fmt_delta(a: f64, b: f64) -> String {
    let d = (b - a) * 100.0;
    if d.abs() < 0.05 {
        "0.0%".to_string()
    } else if d > 0.0 {
        format!("\x1b[32m+{d:.1}%\x1b[0m")
    } else {
        format!("\x1b[31m{d:.1}%\x1b[0m")
    }
}

fn fmt_num_delta(a: f64, b: f64, precision: usize) -> String {
    let d = b - a;
    if d.abs() < 0.005 {
        format!("{d:+.0$}", precision)
    } else if d > 0.0 {
        format!("\x1b[32m{d:+.0$}\x1b[0m", precision)
    } else {
        format!("\x1b[31m{d:+.0$}\x1b[0m", precision)
    }
}

fn fmt_time_delta(a: f64, b: f64) -> String {
    let d = b - a;
    if d.abs() < 0.05 {
        format!("{d:+.1}s")
    } else if d > 0.0 {
        format!("\x1b[31m{d:+.1}s\x1b[0m") // slower = red
    } else {
        format!("\x1b[32m{d:+.1}s\x1b[0m") // faster = green
    }
}

fn avg(v: &[f64]) -> f64 {
    if v.is_empty() { 0.0 } else { v.iter().sum::<f64>() / v.len() as f64 }
}

// --- Config row for tabled ---

#[derive(Tabled)]
struct ConfigRow {
    #[tabled(rename = "config")]
    key: String,
    #[tabled(rename = "A")]
    val_a: String,
    #[tabled(rename = "B")]
    val_b: String,
}

// --- Summary row for tabled ---

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

// --- Metrics row for tabled ---

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

// --- Per-question row for tabled ---

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

// --- Single-file view types ---

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
struct SingleMetricsRow {
    metric: String,
    value: String,
}

#[derive(Tabled)]
struct SingleQuestionRow {
    #[tabled(rename = "id")]
    qid: String,
    sample: String,
    category: String,
    result: String,
}

fn view_single(a: &BenchmarkOutput, path: &str) {
    let label = file_label(a, path);

    // --- Config table ---
    let mut config_rows = vec![
        SingleConfigRow { key: "tag".into(), value: label },
        SingleConfigRow { key: "judge".into(), value: a.judge_model.clone() },
    ];
    if !a.retain_model.is_empty() {
        config_rows.push(SingleConfigRow { key: "retain".into(), value: a.retain_model.clone() });
    }
    if !a.reflect_model.is_empty() {
        config_rows.push(SingleConfigRow { key: "reflect".into(), value: a.reflect_model.clone() });
    }
    if !a.embedding_model.is_empty() {
        config_rows.push(SingleConfigRow { key: "embedding".into(), value: a.embedding_model.clone() });
    }
    if !a.consolidation_strategy.is_empty() {
        config_rows.push(SingleConfigRow { key: "consolidation".into(), value: a.consolidation_strategy.clone() });
    }
    config_rows.push(SingleConfigRow { key: "questions".into(), value: a.total_questions.to_string() });
    if a.total_time_s > 0.0 {
        config_rows.push(SingleConfigRow { key: "total time".into(), value: fmt_time(a.total_time_s) });
    }

    let config_table = Table::new(&config_rows)
        .with(Style::rounded())
        .to_string();
    println!("{config_table}");
    println!();

    // --- Per-category accuracy ---
    let mut cat_stats: BTreeMap<&str, (usize, usize)> = BTreeMap::new(); // (correct, total)
    for r in &a.results {
        if r.category_name.is_empty() { continue; }
        let e = cat_stats.entry(&r.category_name).or_insert((0, 0));
        e.1 += 1;
        if r.judge_correct { e.0 += 1; }
    }

    let categories: Vec<&str> = cat_stats.keys().copied().collect();
    let mut summary_rows = Vec::new();
    for cat in &categories {
        if let Some(&(correct, total)) = cat_stats.get(cat) {
            summary_rows.push(SingleSummaryRow {
                category: cat.to_string(),
                acc: fmt_pct(correct as f64 / total.max(1) as f64),
                n: total,
            });
        }
    }
    summary_rows.push(SingleSummaryRow {
        category: "TOTAL".into(),
        acc: fmt_pct(a.accuracy),
        n: a.results.len(),
    });

    let summary_table = Table::new(&summary_rows)
        .with(Style::rounded())
        .with(Modify::new(Columns::new(1..)).with(Alignment::right()))
        .to_string();
    println!("{summary_table}");

    // --- Metrics table ---
    let avg_f1 = avg(&a.results.iter().map(|r| r.f1).collect::<Vec<_>>());
    let avg_conf = avg(&a.results.iter().map(|r| r.confidence as f64).collect::<Vec<_>>());
    let avg_time = avg(&a.results.iter().map(|r| r.elapsed_s).collect::<Vec<_>>());

    let metrics_rows = vec![
        SingleMetricsRow { metric: "F1".into(), value: format!("{avg_f1:.3}") },
        SingleMetricsRow { metric: "confidence".into(), value: format!("{avg_conf:.3}") },
        SingleMetricsRow { metric: "avg time".into(), value: fmt_time(avg_time) },
    ];

    println!();
    let metrics_table = Table::new(&metrics_rows)
        .with(Style::rounded())
        .with(Modify::new(Columns::new(1..)).with(Alignment::right()))
        .to_string();
    println!("{metrics_table}");

    // --- Per-question table ---
    if a.results.is_empty() { return; }
    println!();

    let mut question_rows: Vec<SingleQuestionRow> = Vec::new();
    let mut sorted: Vec<&QuestionResult> = a.results.iter().collect();
    sorted.sort_by(|a, b| a.category_name.cmp(&b.category_name)
        .then(a.sample_id.cmp(&b.sample_id))
        .then(a.question_id.cmp(&b.question_id)));

    for r in &sorted {
        let mark = if r.judge_correct { "\x1b[32m✓\x1b[0m".to_string() } else { "\x1b[31m✗\x1b[0m".to_string() };
        question_rows.push(SingleQuestionRow {
            qid: r.question_id.clone(),
            sample: r.sample_id.clone(),
            category: r.category_name.clone(),
            result: mark,
        });
    }

    let q_table = Table::new(&question_rows)
        .with(Style::rounded())
        .with(Modify::new(Columns::new(3..=3)).with(Alignment::center()))
        .to_string();
    println!("{q_table}");

    // Summary line
    let correct = a.results.iter().filter(|r| r.judge_correct).count();
    let wrong = a.results.len() - correct;
    println!();
    println!("\x1b[32m{correct} correct\x1b[0m, \x1b[31m{wrong} wrong\x1b[0m");
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

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() == 2 {
        view_single(&load_file(&args[1]), &args[1]);
        return;
    }
    if args.len() != 3 {
        eprintln!("Usage: view <file.json> [file2.json]");
        process::exit(1);
    }

    let a = load_file(&args[1]);
    let b = load_file(&args[2]);

    let label_a = file_label(&a, &args[1]);
    let label_b = file_label(&b, &args[2]);

    // Match questions across runs
    let map_b: HashMap<QKey, &QuestionResult> = b.results.iter().map(|r| (qkey(r), r)).collect();

    let matched: Vec<(&QuestionResult, &QuestionResult)> = a.results
        .iter()
        .filter_map(|ra| map_b.get(&qkey(ra)).map(|rb| (ra, *rb)))
        .collect();

    // Discover categories & compute per-category stats
    let categories: Vec<String> = {
        let mut cats = BTreeSet::new();
        for r in &a.results { cats.insert(r.category_name.clone()); }
        for r in &b.results { cats.insert(r.category_name.clone()); }
        cats.into_iter().filter(|c| !c.is_empty()).collect()
    };

    let mut cat_stats: BTreeMap<&str, (usize, usize, usize)> = BTreeMap::new(); // (correct_a, correct_b, total)
    for (ra, rb) in &matched {
        let e = cat_stats.entry(&ra.category_name).or_insert((0, 0, 0));
        e.2 += 1;
        if ra.judge_correct { e.0 += 1; }
        if rb.judge_correct { e.1 += 1; }
    }

    // --- Config table ---
    let mut config_rows = vec![
        ConfigRow { key: "tag".into(), val_a: label_a.clone(), val_b: label_b.clone() },
        ConfigRow { key: "judge".into(), val_a: a.judge_model.clone(), val_b: b.judge_model.clone() },
    ];
    if !a.retain_model.is_empty() || !b.retain_model.is_empty() {
        config_rows.push(ConfigRow { key: "retain".into(), val_a: a.retain_model.clone(), val_b: b.retain_model.clone() });
    }
    if !a.reflect_model.is_empty() || !b.reflect_model.is_empty() {
        config_rows.push(ConfigRow { key: "reflect".into(), val_a: a.reflect_model.clone(), val_b: b.reflect_model.clone() });
    }
    if !a.embedding_model.is_empty() || !b.embedding_model.is_empty() {
        config_rows.push(ConfigRow { key: "embedding".into(), val_a: a.embedding_model.clone(), val_b: b.embedding_model.clone() });
    }
    if !a.consolidation_strategy.is_empty() || !b.consolidation_strategy.is_empty() {
        config_rows.push(ConfigRow { key: "consolidation".into(), val_a: a.consolidation_strategy.clone(), val_b: b.consolidation_strategy.clone() });
    }
    config_rows.push(ConfigRow {
        key: "questions".into(),
        val_a: a.total_questions.to_string(),
        val_b: b.total_questions.to_string(),
    });
    if matched.len() != a.results.len() || matched.len() != b.results.len() {
        config_rows.push(ConfigRow { key: "matched".into(), val_a: matched.len().to_string(), val_b: matched.len().to_string() });
    }
    if a.total_time_s > 0.0 || b.total_time_s > 0.0 {
        config_rows.push(ConfigRow {
            key: "total time".into(),
            val_a: fmt_time(a.total_time_s),
            val_b: fmt_time(b.total_time_s),
        });
    }

    let config_table = Table::new(&config_rows)
        .with(Style::rounded())
        .to_string();
    println!("{config_table}");
    println!();

    // --- Summary table ---
    let mut summary_rows = Vec::new();

    for cat in &categories {
        if let Some(&(ca, cb, n)) = cat_stats.get(cat.as_str()) {
            let aa = ca as f64 / n.max(1) as f64;
            let ab = cb as f64 / n.max(1) as f64;
            summary_rows.push(SummaryRow {
                category: cat.clone(),
                acc_a: fmt_pct(aa),
                acc_b: fmt_pct(ab),
                delta: fmt_delta(aa, ab),
                n,
            });
        }
    }

    summary_rows.push(SummaryRow {
        category: "TOTAL".into(),
        acc_a: fmt_pct(a.accuracy),
        acc_b: fmt_pct(b.accuracy),
        delta: fmt_delta(a.accuracy, b.accuracy),
        n: matched.len(),
    });

    let summary_table = Table::new(&summary_rows)
        .with(Style::rounded())
        .with(Modify::new(Columns::new(1..)).with(Alignment::right()))
        .to_string();
    println!("{summary_table}");

    // --- Metrics table ---
    let avg_f1_a = avg(&matched.iter().map(|(ra, _)| ra.f1).collect::<Vec<_>>());
    let avg_f1_b = avg(&matched.iter().map(|(_, rb)| rb.f1).collect::<Vec<_>>());
    let avg_conf_a = avg(&matched.iter().map(|(ra, _)| ra.confidence as f64).collect::<Vec<_>>());
    let avg_conf_b = avg(&matched.iter().map(|(_, rb)| rb.confidence as f64).collect::<Vec<_>>());
    let avg_time_a = avg(&matched.iter().map(|(ra, _)| ra.elapsed_s).collect::<Vec<_>>());
    let avg_time_b = avg(&matched.iter().map(|(_, rb)| rb.elapsed_s).collect::<Vec<_>>());

    let metrics_rows = vec![
        MetricsRow {
            metric: "F1".into(),
            val_a: format!("{avg_f1_a:.3}"),
            val_b: format!("{avg_f1_b:.3}"),
            delta: fmt_num_delta(avg_f1_a, avg_f1_b, 3),
        },
        MetricsRow {
            metric: "confidence".into(),
            val_a: format!("{avg_conf_a:.3}"),
            val_b: format!("{avg_conf_b:.3}"),
            delta: fmt_num_delta(avg_conf_a, avg_conf_b, 3),
        },
        MetricsRow {
            metric: "avg time".into(),
            val_a: fmt_time(avg_time_a),
            val_b: fmt_time(avg_time_b),
            delta: fmt_time_delta(avg_time_a, avg_time_b),
        },
    ];

    println!();
    let metrics_table = Table::new(&metrics_rows)
        .with(Style::rounded())
        .with(Modify::new(Columns::new(1..)).with(Alignment::right()))
        .to_string();
    println!("{metrics_table}");

    // --- Per-question table ---
    if matched.is_empty() {
        return;
    }

    println!();

    let mut question_rows: Vec<QuestionRow> = Vec::new();

    for cat in &categories {
        let mut cat_qs: Vec<&(&QuestionResult, &QuestionResult)> = matched
            .iter()
            .filter(|(ra, _rb)| ra.category_name == *cat)
            .collect();
        cat_qs.sort_by(|a, b| a.0.sample_id.cmp(&b.0.sample_id).then(a.0.question_id.cmp(&b.0.question_id)));

        for (ra, rb) in &cat_qs {
            let mark_a = if ra.judge_correct { "\x1b[32m✓\x1b[0m".to_string() } else { "\x1b[31m✗\x1b[0m".to_string() };
            let mark_b = if rb.judge_correct { "\x1b[32m✓\x1b[0m".to_string() } else { "\x1b[31m✗\x1b[0m".to_string() };

            question_rows.push(QuestionRow {
                qid: ra.question_id.clone(),
                sample: ra.sample_id.clone(),
                category: ra.category_name.clone(),
                a: mark_a,
                b: mark_b,
            });
        }
    }

    let q_table = Table::new(&question_rows)
        .with(Style::rounded())
        .with(Modify::new(Columns::new(3..=4)).with(Alignment::center()))
        .to_string();
    println!("{q_table}");

    // Summary line
    let regressions = matched.iter().filter(|(ra, rb)| ra.judge_correct && !rb.judge_correct).count();
    let improvements = matched.iter().filter(|(ra, rb)| !ra.judge_correct && rb.judge_correct).count();
    let unchanged = matched.len() - improvements - regressions;
    println!();
    println!(
        "\x1b[32m{improvements} improved\x1b[0m, \x1b[31m{regressions} regressed\x1b[0m, {unchanged} unchanged"
    );
}
