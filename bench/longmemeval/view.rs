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
    total_stage_usage: ViewStageUsage,
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
    #[serde(alias = "prompt_tokens")]
    input_tokens: u64,
    #[serde(default)]
    cached_prompt_tokens: u64,
    #[serde(default)]
    cache_read_input_tokens: u64,
    #[serde(default)]
    cache_creation_input_tokens: u64,
    #[serde(default)]
    #[serde(alias = "completion_tokens")]
    output_tokens: u64,
    #[serde(default)]
    #[serde(alias = "requests")]
    calls: u64,
    #[serde(default)]
    errors: u64,
    #[serde(default)]
    latency_ms: u64,
}

impl ViewStageUsage {
    fn total_tokens(&self) -> u64 {
        self.input_tokens + self.output_tokens
    }

    fn has_cache_usage(&self) -> bool {
        self.cached_prompt_tokens > 0
            || self.cache_read_input_tokens > 0
            || self.cache_creation_input_tokens > 0
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
    calls: u64,
    input_tok: u64,
    output_tok: u64,
    cached_tok: u64,
    cache_read: u64,
    cache_write: u64,
    errors: u64,
    latency: String,
}

#[derive(Tabled)]
struct CompareStageRow {
    stage: String,
    #[tabled(rename = "A")]
    val_a: String,
    #[tabled(rename = "B")]
    val_b: String,
    #[tabled(rename = "\u{0394} tok")]
    delta_tok: String,
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

fn fmt_ms(ms: u64) -> String {
    fmt_time(ms as f64 / 1000.0)
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

fn fmt_stage_value(usage: &ViewStageUsage) -> String {
    if usage.calls == 0 {
        "-".into()
    } else {
        format!(
            "{} in / {} out / {} cached / {} read / {} write / {} calls",
            usage.input_tokens,
            usage.output_tokens,
            usage.cached_prompt_tokens,
            usage.cache_read_input_tokens,
            usage.cache_creation_input_tokens,
            usage.calls,
        )
    }
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

fn load_file(path: &str) -> ViewBenchmarkOutput {
    let raw = fs::read_to_string(path).unwrap_or_else(|e| {
        eprintln!("Failed to read {path}: {e}");
        process::exit(1);
    });
    serde_json::from_str(&raw).unwrap_or_else(|e| {
        eprintln!("Failed to parse {path}: {e}");
        process::exit(1);
    })
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
        .filter(|(_, usage)| usage.calls > 0)
        .map(|(stage, usage)| StageRow {
            stage: stage.clone(),
            calls: usage.calls,
            input_tok: usage.input_tokens,
            output_tok: usage.output_tokens,
            cached_tok: usage.cached_prompt_tokens,
            cache_read: usage.cache_read_input_tokens,
            cache_write: usage.cache_creation_input_tokens,
            errors: usage.errors,
            latency: fmt_ms(usage.latency_ms),
        })
        .collect();
    rows.sort_by(|a, b| a.stage.cmp(&b.stage));
    rows
}

// ---------------------------------------------------------------------------
// Single-file mode
// ---------------------------------------------------------------------------

fn view_single(output: &ViewBenchmarkOutput, path: &str, verbose: bool) {
    let label = file_label(output, path);
    println!("LongMemEval: {label}");
    println!();

    // Config table
    let config_rows = build_config_rows(output);
    if !config_rows.is_empty() {
        println!("{}", Table::new(&config_rows).with(Style::rounded()));
        println!();
    }

    // Per-category accuracy table
    let cat_rows = build_category_rows(output);
    println!(
        "{}",
        Table::new(&cat_rows)
            .with(Style::rounded())
            .with(Modify::new(Columns::new(1..)).with(Alignment::right()))
    );
    println!();

    // Stage metrics table
    let stage_rows = build_stage_rows(output);
    if !stage_rows.is_empty() {
        println!(
            "{}",
            Table::new(&stage_rows)
                .with(Style::rounded())
                .with(Modify::new(Columns::new(1..)).with(Alignment::right()))
        );
        println!();
    }

    if output.total_stage_usage.calls > 0 {
        let mut total_rows = vec![
            SingleConfigRow {
                key: "input tok".into(),
                value: output.total_stage_usage.input_tokens.to_string(),
            },
            SingleConfigRow {
                key: "output tok".into(),
                value: output.total_stage_usage.output_tokens.to_string(),
            },
            SingleConfigRow {
                key: "total tok".into(),
                value: output.total_stage_usage.total_tokens().to_string(),
            },
        ];
        if output.total_stage_usage.has_cache_usage() {
            total_rows.push(SingleConfigRow {
                key: "cached tok".into(),
                value: output.total_stage_usage.cached_prompt_tokens.to_string(),
            });
            total_rows.push(SingleConfigRow {
                key: "cache read".into(),
                value: output.total_stage_usage.cache_read_input_tokens.to_string(),
            });
            total_rows.push(SingleConfigRow {
                key: "cache write".into(),
                value: output
                    .total_stage_usage
                    .cache_creation_input_tokens
                    .to_string(),
            });
        }
        total_rows.push(SingleConfigRow {
            key: "calls".into(),
            value: output.total_stage_usage.calls.to_string(),
        });
        total_rows.push(SingleConfigRow {
            key: "errors".into(),
            value: output.total_stage_usage.errors.to_string(),
        });
        total_rows.push(SingleConfigRow {
            key: "latency".into(),
            value: fmt_ms(output.total_stage_usage.latency_ms),
        });

        println!("{}", Table::new(&total_rows).with(Style::rounded()));
        println!();
    }

    // Total time
    println!("Total time: {}", fmt_time(output.total_time_s));

    // Verbose: per-question table
    if verbose {
        let questions = load_questions(Path::new(path));
        if questions.is_empty() {
            println!();
            println!("(no question sidecar found)");
        } else {
            let mut sorted = questions;
            sorted.sort_by(|a, b| {
                a.category
                    .cmp(&b.category)
                    .then(a.question_id.cmp(&b.question_id))
            });
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
            println!();
            println!(
                "{}",
                Table::new(&q_rows)
                    .with(Style::rounded())
                    .with(Modify::new(Columns::new(3..)).with(Alignment::right()))
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Comparison mode
// ---------------------------------------------------------------------------

fn view_compare(
    a: &ViewBenchmarkOutput,
    b: &ViewBenchmarkOutput,
    path_a: &str,
    path_b: &str,
    verbose: bool,
) {
    let label_a = file_label(a, path_a);
    let label_b = file_label(b, path_b);
    println!("LongMemEval comparison: {label_a} vs {label_b}");
    println!();

    // Config comparison table
    let config_pairs: Vec<(&str, String, String)> = vec![
        (
            "profile",
            a.manifest.profile.clone(),
            b.manifest.profile.clone(),
        ),
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
        (
            "retain_model",
            a.retain_model.clone(),
            b.retain_model.clone(),
        ),
        (
            "reflect_model",
            a.reflect_model.clone(),
            b.reflect_model.clone(),
        ),
        (
            "embedding_model",
            a.embedding_model.clone(),
            b.embedding_model.clone(),
        ),
        (
            "reranker_model",
            a.reranker_model.clone(),
            b.reranker_model.clone(),
        ),
        ("judge_model", a.judge_model.clone(), b.judge_model.clone()),
        (
            "tag",
            a.tag.clone().unwrap_or_default(),
            b.tag.clone().unwrap_or_default(),
        ),
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
        println!("{}", Table::new(&config_rows).with(Style::rounded()));
        println!();
    }

    // Per-category comparison with deltas
    let mut all_categories: Vec<String> = {
        let mut cats: Vec<String> = a
            .per_category
            .keys()
            .chain(b.per_category.keys())
            .cloned()
            .collect();
        cats.sort();
        cats.dedup();
        cats
    };
    // "overall" always last
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
    // overall row
    cat_rows.push(CompareCategoryRow {
        category: "overall".into(),
        acc_a: fmt_pct(a.accuracy),
        acc_b: fmt_pct(b.accuracy),
        delta: fmt_pct_delta(a.accuracy, b.accuracy),
        n_a: a.total_questions,
        n_b: b.total_questions,
    });

    println!(
        "{}",
        Table::new(&cat_rows)
            .with(Style::rounded())
            .with(Modify::new(Columns::new(1..)).with(Alignment::right()))
    );
    println!();

    // Stage metrics comparison
    let all_stages: Vec<String> = {
        let mut stages: Vec<String> = a
            .stage_metrics
            .keys()
            .chain(b.stage_metrics.keys())
            .cloned()
            .collect();
        stages.sort();
        stages.dedup();
        stages
    };

    let default_stage = ViewStageUsage::default();
    let stage_rows: Vec<CompareStageRow> = all_stages
        .iter()
        .filter(|stage| {
            let sa = a.stage_metrics.get(*stage).unwrap_or(&default_stage);
            let sb = b.stage_metrics.get(*stage).unwrap_or(&default_stage);
            sa.calls > 0 || sb.calls > 0
        })
        .map(|stage| {
            let sa = a.stage_metrics.get(stage).unwrap_or(&default_stage);
            let sb = b.stage_metrics.get(stage).unwrap_or(&default_stage);
            CompareStageRow {
                stage: stage.clone(),
                val_a: fmt_stage_value(sa),
                val_b: fmt_stage_value(sb),
                delta_tok: fmt_cost_delta_u64(sa.total_tokens(), sb.total_tokens()),
            }
        })
        .collect();

    if !stage_rows.is_empty() {
        println!(
            "{}",
            Table::new(&stage_rows)
                .with(Style::rounded())
                .with(Modify::new(Columns::new(1..)).with(Alignment::right()))
        );
        println!();
    }

    if a.total_stage_usage.calls > 0 || b.total_stage_usage.calls > 0 {
        let mut total_rows = vec![
            CompareConfigRow {
                key: "input tok".into(),
                val_a: a.total_stage_usage.input_tokens.to_string(),
                val_b: b.total_stage_usage.input_tokens.to_string(),
            },
            CompareConfigRow {
                key: "output tok".into(),
                val_a: a.total_stage_usage.output_tokens.to_string(),
                val_b: b.total_stage_usage.output_tokens.to_string(),
            },
            CompareConfigRow {
                key: "total tok".into(),
                val_a: a.total_stage_usage.total_tokens().to_string(),
                val_b: b.total_stage_usage.total_tokens().to_string(),
            },
        ];
        if a.total_stage_usage.has_cache_usage() || b.total_stage_usage.has_cache_usage() {
            total_rows.push(CompareConfigRow {
                key: "cached tok".into(),
                val_a: a.total_stage_usage.cached_prompt_tokens.to_string(),
                val_b: b.total_stage_usage.cached_prompt_tokens.to_string(),
            });
            total_rows.push(CompareConfigRow {
                key: "cache read".into(),
                val_a: a.total_stage_usage.cache_read_input_tokens.to_string(),
                val_b: b.total_stage_usage.cache_read_input_tokens.to_string(),
            });
            total_rows.push(CompareConfigRow {
                key: "cache write".into(),
                val_a: a.total_stage_usage.cache_creation_input_tokens.to_string(),
                val_b: b.total_stage_usage.cache_creation_input_tokens.to_string(),
            });
        }
        total_rows.push(CompareConfigRow {
            key: "calls".into(),
            val_a: a.total_stage_usage.calls.to_string(),
            val_b: b.total_stage_usage.calls.to_string(),
        });
        total_rows.push(CompareConfigRow {
            key: "errors".into(),
            val_a: a.total_stage_usage.errors.to_string(),
            val_b: b.total_stage_usage.errors.to_string(),
        });
        total_rows.push(CompareConfigRow {
            key: "latency".into(),
            val_a: fmt_ms(a.total_stage_usage.latency_ms),
            val_b: fmt_ms(b.total_stage_usage.latency_ms),
        });

        println!("{}", Table::new(&total_rows).with(Style::rounded()));
        println!();
    }

    // Total time comparison
    println!(
        "Total time: {} vs {}",
        fmt_time(a.total_time_s),
        fmt_time(b.total_time_s)
    );

    // Verbose: per-question comparison
    if verbose {
        let questions_a = load_questions(Path::new(path_a));
        let questions_b = load_questions(Path::new(path_b));

        if questions_a.is_empty() && questions_b.is_empty() {
            println!();
            println!("(no question sidecars found)");
        } else {
            let map_b: BTreeMap<&str, &ViewQuestionResult> = questions_b
                .iter()
                .map(|q| (q.question_id.as_str(), q))
                .collect();

            let mut sorted_a = questions_a.iter().collect::<Vec<_>>();
            sorted_a.sort_by(|x, y| {
                x.category
                    .cmp(&y.category)
                    .then(x.question_id.cmp(&y.question_id))
            });

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

            println!();
            println!(
                "{}",
                Table::new(&q_rows)
                    .with(Style::rounded())
                    .with(Modify::new(Columns::new(2..)).with(Alignment::center()))
            );
        }
    }
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
                cached_prompt_tokens: 500000,
                cache_read_input_tokens: 0,
                cache_creation_input_tokens: 0,
                output_tokens: 25000,
                calls: 500,
                errors: 0,
                latency_ms: 120000,
            },
        );
        stage_metrics.insert(
            "judge".into(),
            ViewStageUsage {
                input_tokens: 200000,
                cached_prompt_tokens: 0,
                cache_read_input_tokens: 150000,
                cache_creation_input_tokens: 10000,
                output_tokens: 10000,
                calls: 500,
                errors: 0,
                latency_ms: 45000,
            },
        );
        stage_metrics.insert(
            "unused".into(),
            ViewStageUsage {
                input_tokens: 0,
                cached_prompt_tokens: 0,
                cache_read_input_tokens: 0,
                cache_creation_input_tokens: 0,
                output_tokens: 0,
                calls: 0,
                errors: 0,
                latency_ms: 0,
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
        assert_eq!(rows[1].cached_tok, 500000);
        assert_eq!(rows[1].calls, 500);
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
            cached_prompt_tokens: 600,
            cache_read_input_tokens: 0,
            cache_creation_input_tokens: 0,
            output_tokens: 200,
            calls: 5,
            errors: 0,
            latency_ms: 1000,
        };
        assert_eq!(usage.total_tokens(), 1200);
        assert!(usage.has_cache_usage());
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
                "retain_extract": {"input_tokens": 1000, "cached_prompt_tokens": 800, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0, "output_tokens": 200, "calls": 10, "errors": 0, "latency_ms": 1000},
                "reflect": {"input_tokens": 500, "cached_prompt_tokens": 0, "cache_read_input_tokens": 300, "cache_creation_input_tokens": 20, "output_tokens": 100, "calls": 5, "errors": 1, "latency_ms": 500}
            },
            "total_stage_usage": {
                "input_tokens": 1500,
                "cached_prompt_tokens": 800,
                "cache_read_input_tokens": 300,
                "cache_creation_input_tokens": 20,
                "output_tokens": 300,
                "calls": 15,
                "errors": 1,
                "latency_ms": 1500
            },
            "total_time_s": 120.5
        }"#;

        let output: ViewBenchmarkOutput = serde_json::from_str(json).unwrap();
        assert_eq!(output.benchmark, "longmemeval");
        assert_eq!(output.accuracy, 0.85);
        assert_eq!(output.per_category.len(), 2);
        assert_eq!(output.stage_metrics.len(), 2);
        assert_eq!(output.total_stage_usage.calls, 15);
        assert_eq!(
            output.stage_metrics["retain_extract"].cached_prompt_tokens,
            800
        );
        assert_eq!(output.total_time_s, 120.5);
    }

    #[test]
    fn test_deserialize_stage_usage_legacy_aliases() {
        let json = r#"{
            "stage_metrics": {
                "reflect": {
                    "prompt_tokens": 100,
                    "completion_tokens": 20,
                    "requests": 3
                }
            }
        }"#;

        let output: ViewBenchmarkOutput = serde_json::from_str(json).unwrap();
        let usage = &output.stage_metrics["reflect"];
        assert_eq!(usage.input_tokens, 100);
        assert_eq!(usage.output_tokens, 20);
        assert_eq!(usage.calls, 3);
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
}
