#![allow(dead_code)]

use std::env;
use std::fmt::Write as _;
use std::fs;
use std::path::PathBuf;

use elephant::retain::chunker::{Chunker, SimpleChunker};
use elephant::retain::extractor::LlmFactExtractor;
use elephant::types::{BankId, ChunkConfig, ExtractionInput};
use elephant_bench::resolve_longmemeval_bench_config;
use serde::Serialize;

#[path = "common/mod.rs"]
mod common;

#[path = "longmemeval/mod.rs"]
mod longmemeval;

use longmemeval::dataset::{LongMemEvalInstance, answer_to_string, load_dataset};
use longmemeval::ingest::{
    format_round_text, format_session_json, format_session_text, parse_date_prefix,
    parse_haystack_date, round_slices,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OutputFormat {
    Text,
    Json,
    Html,
}

impl OutputFormat {
    fn as_str(self) -> &'static str {
        match self {
            Self::Text => "text",
            Self::Json => "json",
            Self::Html => "html",
        }
    }
}

struct Cli {
    profile: String,
    profile_path: PathBuf,
    config_path: Option<PathBuf>,
    secrets_env_file: Option<PathBuf>,
    question_id: String,
    session: usize,
    round: Option<usize>,
    chunk: Option<usize>,
    output_format: OutputFormat,
    out: Option<PathBuf>,
}

#[derive(Debug, Serialize)]
struct InspectionBundle {
    profile: String,
    profile_path: String,
    config_path: Option<String>,
    dataset: String,
    contract_hash: String,
    ingest_format: String,
    session_limit: Option<usize>,
    item_id: String,
    session: usize,
    session_id: Option<String>,
    session_date: String,
    timestamp: String,
    question: String,
    answer: String,
    question_type: String,
    question_date: String,
    chunk_config: ChunkConfig,
    total_units: usize,
    displayed_units: usize,
    units: Vec<InspectionUnit>,
}

#[derive(Debug, Serialize)]
struct InspectionUnit {
    mode: String,
    round: Option<usize>,
    round_count: Option<usize>,
    turn_count: usize,
    retain_input: String,
    retain_input_chars: usize,
    retain_input_est_tokens: usize,
    total_chunk_count: usize,
    displayed_chunk_count: usize,
    chunks: Vec<InspectionChunk>,
}

#[derive(Debug, Serialize)]
struct InspectionChunk {
    index: usize,
    start_byte: usize,
    end_byte: usize,
    content_chars: usize,
    content_est_tokens: usize,
    context_chars: usize,
    context_est_tokens: usize,
    speaker: Option<String>,
    content: String,
    context: Option<String>,
    extractor_system_prompt: String,
    extractor_user_message: String,
}

#[derive(Debug, Clone)]
struct ChunkSpan {
    start: usize,
    end: usize,
}

fn print_help() {
    eprintln!("Usage:");
    eprintln!(
        "  cargo run -p elephant-bench --bin inspect_retain_source -- --profile <name> --question <ID> --session <N> [options]"
    );
    eprintln!();
    eprintln!("Required:");
    eprintln!("  --profile <NAME>          LongMemEval profile name or path");
    eprintln!("  --question <ID>           Question id, e.g. e47becba");
    eprintln!("  --session <N>             Original haystack session number");
    eprintln!();
    eprintln!("Optional:");
    eprintln!("  --config <PATH>           Execution overlay path");
    eprintln!("  --secrets-env-file <PATH> Benchmark secrets env file");
    eprintln!("  --round <N>               Optional round filter within the selected session");
    eprintln!("  --chunk <N>               Show only one extractor chunk per retained unit");
    eprintln!("  --output <FORMAT>         text | json | html [default: text]");
    eprintln!("  --out <PATH>              Write output to a file instead of stdout");
    eprintln!("  --help                    Show this help");
}

fn parse_args() -> Result<Cli, String> {
    let mut profile = None;
    let mut config_path = None;
    let mut secrets_env_file = None;
    let mut question_id = None;
    let mut session = None;
    let mut round = None;
    let mut chunk = None;
    let mut output_format = OutputFormat::Text;
    let mut out = None;

    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--profile" => {
                profile = Some(
                    args.next()
                        .ok_or_else(|| "--profile requires a value".to_string())?,
                );
            }
            "--config" => {
                config_path = Some(PathBuf::from(
                    args.next()
                        .ok_or_else(|| "--config requires a path".to_string())?,
                ));
            }
            "--secrets-env-file" => {
                secrets_env_file =
                    Some(PathBuf::from(args.next().ok_or_else(|| {
                        "--secrets-env-file requires a path".to_string()
                    })?));
            }
            "--question" => {
                question_id = Some(
                    args.next()
                        .ok_or_else(|| "--question requires a value".to_string())?,
                );
            }
            "--session" => {
                session = Some(
                    args.next()
                        .ok_or_else(|| "--session requires a value".to_string())?
                        .parse()
                        .map_err(|_| "--session must be an integer".to_string())?,
                );
            }
            "--round" => {
                round = Some(
                    args.next()
                        .ok_or_else(|| "--round requires a value".to_string())?
                        .parse()
                        .map_err(|_| "--round must be an integer".to_string())?,
                );
            }
            "--chunk" => {
                chunk = Some(
                    args.next()
                        .ok_or_else(|| "--chunk requires a value".to_string())?
                        .parse()
                        .map_err(|_| "--chunk must be an integer".to_string())?,
                );
            }
            "--output" => {
                output_format = match args
                    .next()
                    .ok_or_else(|| "--output requires a value".to_string())?
                    .as_str()
                {
                    "text" => OutputFormat::Text,
                    "json" => OutputFormat::Json,
                    "html" => OutputFormat::Html,
                    other => {
                        return Err(format!(
                            "invalid --output value: {other} (expected text, json, or html)"
                        ));
                    }
                };
            }
            "--out" => {
                out = Some(PathBuf::from(
                    args.next()
                        .ok_or_else(|| "--out requires a path".to_string())?,
                ));
            }
            "--help" | "-h" => {
                print_help();
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument: {other}")),
        }
    }

    let profile = profile.ok_or_else(|| "--profile is required".to_string())?;
    let question_id = question_id.ok_or_else(|| "--question is required".to_string())?;
    let session = session.ok_or_else(|| "--session is required".to_string())?;
    let profile_path = resolve_profile_path(&profile);

    Ok(Cli {
        profile,
        profile_path,
        config_path,
        secrets_env_file,
        question_id,
        session,
        round,
        chunk,
        output_format,
        out,
    })
}

fn resolve_profile_path(profile: &str) -> PathBuf {
    if profile.ends_with(".toml") || profile.contains('/') {
        PathBuf::from(profile)
    } else {
        PathBuf::from("bench/longmemeval/profiles").join(format!("{profile}.toml"))
    }
}

fn estimate_tokens(text: &str) -> usize {
    text.len().div_ceil(4)
}

fn extractor_inputs(base: &ExtractionInput, chunk_config: &ChunkConfig) -> Vec<ExtractionInput> {
    let chunker = SimpleChunker;
    chunker
        .chunk(&base.content, chunk_config)
        .into_iter()
        .map(|chunk| ExtractionInput {
            content: chunk.content,
            bank_id: base.bank_id,
            context: chunk.context.or_else(|| base.context.clone()),
            timestamp: base.timestamp,
            turn_id: base.turn_id,
            custom_instructions: base.custom_instructions.clone(),
            speaker: base.speaker.clone(),
        })
        .collect()
}

fn find_split_point(text: &str, target_byte: usize, preserve_turns: bool) -> usize {
    let search_start = target_byte.saturating_sub(200);
    let search_region = &text[search_start..target_byte.min(text.len())];

    if preserve_turns && let Some(pos) = search_region.rfind("\n\n") {
        return search_start + pos + 2;
    }
    if let Some(pos) = search_region.rfind("\n\n") {
        return search_start + pos + 2;
    }
    if let Some(pos) = search_region.rfind('\n') {
        return search_start + pos + 1;
    }
    if let Some(pos) = search_region.rfind(". ") {
        return search_start + pos + 2;
    }
    if let Some(pos) = search_region.rfind(' ') {
        return search_start + pos + 1;
    }
    target_byte.min(text.len())
}

fn reconstruct_chunk_spans(input: &str, config: &ChunkConfig) -> Vec<ChunkSpan> {
    let total_tokens = estimate_tokens(input);
    if total_tokens <= config.max_tokens {
        return vec![ChunkSpan {
            start: 0,
            end: input.len(),
        }];
    }

    let bytes_per_token = input.len() as f64 / total_tokens as f64;
    let max_bytes = (config.max_tokens as f64 * bytes_per_token) as usize;
    let overlap_bytes = (config.overlap_tokens as f64 * bytes_per_token) as usize;
    let mut spans = Vec::new();
    let mut byte_offset = 0;

    while byte_offset < input.len() {
        let target_end = (byte_offset + max_bytes).min(input.len());
        let chunk_end = if target_end >= input.len() {
            input.len()
        } else {
            find_split_point(input, target_end, config.preserve_turns)
        };

        spans.push(ChunkSpan {
            start: byte_offset,
            end: chunk_end,
        });

        let advance = chunk_end - byte_offset;
        if advance == 0 {
            break;
        }

        let previous_offset = byte_offset;
        byte_offset = chunk_end.saturating_sub(overlap_bytes);
        if byte_offset <= previous_offset {
            byte_offset = chunk_end;
        }
    }

    spans
}

fn build_unit(
    mode: String,
    round: Option<usize>,
    round_count: Option<usize>,
    turn_count: usize,
    chunk_config: &ChunkConfig,
    base_input: &ExtractionInput,
    selected_chunk: Option<usize>,
) -> Result<InspectionUnit, String> {
    let spans = reconstruct_chunk_spans(&base_input.content, chunk_config);
    let inputs = extractor_inputs(base_input, chunk_config);

    if spans.len() != inputs.len() {
        return Err(format!(
            "chunk span reconstruction mismatch: spans={} inputs={}",
            spans.len(),
            inputs.len()
        ));
    }

    let mut chunks = Vec::new();
    for (idx, (span, input)) in spans.iter().zip(inputs.iter()).enumerate() {
        if selected_chunk.is_some_and(|selected| selected != idx) {
            continue;
        }

        let actual = &base_input.content[span.start..span.end];
        if actual != input.content {
            return Err(format!(
                "chunk {} content mismatch while reconstructing spans",
                idx
            ));
        }

        let (system_prompt, user_message) = LlmFactExtractor::render_prompt(input);
        chunks.push(InspectionChunk {
            index: idx,
            start_byte: span.start,
            end_byte: span.end,
            content_chars: input.content.len(),
            content_est_tokens: estimate_tokens(&input.content),
            context_chars: input.context.as_ref().map(|ctx| ctx.len()).unwrap_or(0),
            context_est_tokens: input
                .context
                .as_ref()
                .map(|ctx| estimate_tokens(ctx))
                .unwrap_or(0),
            speaker: input.speaker.clone(),
            content: input.content.clone(),
            context: input.context.clone(),
            extractor_system_prompt: system_prompt,
            extractor_user_message: user_message,
        });
    }

    Ok(InspectionUnit {
        mode,
        round,
        round_count,
        turn_count,
        retain_input: base_input.content.clone(),
        retain_input_chars: base_input.content.len(),
        retain_input_est_tokens: estimate_tokens(&base_input.content),
        total_chunk_count: inputs.len(),
        displayed_chunk_count: chunks.len(),
        chunks,
    })
}

fn validate_selection(
    resolved: &elephant_bench::ResolvedLongMemEvalBenchConfig,
    entry: &LongMemEvalInstance,
    session: usize,
) -> Result<(), std::io::Error> {
    let selected_instances = resolved.selected_instances();
    if !selected_instances.is_empty()
        && !selected_instances.iter().any(|id| id == &entry.question_id)
    {
        return Err(std::io::Error::other(format!(
            "question {} is outside the resolved profile slice",
            entry.question_id
        )));
    }

    if session == 0 || session > entry.haystack_sessions.len() {
        return Err(std::io::Error::other(format!(
            "session {} is out of range for question {} ({} sessions)",
            session,
            entry.question_id,
            entry.haystack_sessions.len()
        )));
    }

    if let Some(limit) = resolved.session_limit()
        && session > limit
    {
        return Err(std::io::Error::other(format!(
            "session {} is outside the resolved profile session_limit={} for question {}",
            session, limit, entry.question_id
        )));
    }

    Ok(())
}

fn inspect_longmemeval(cli: &Cli) -> Result<InspectionBundle, Box<dyn std::error::Error>> {
    let resolved = resolve_longmemeval_bench_config(
        &cli.profile_path,
        cli.config_path.as_deref(),
        cli.secrets_env_file.as_deref(),
    )?;
    let (entries, _) = load_dataset(resolved.dataset_path()).map_err(std::io::Error::other)?;
    let entry = entries
        .iter()
        .find(|entry| entry.question_id == cli.question_id)
        .ok_or_else(|| std::io::Error::other(format!("question {} not found", cli.question_id)))?;

    validate_selection(&resolved, entry, cli.session)?;

    let index = cli.session - 1;
    let turns = &entry.haystack_sessions[index];
    let date_str = entry.haystack_dates[index].clone();
    let timestamp = parse_haystack_date(&date_str);
    let chunk_config = resolved.retain_chunk_config();

    let units = match resolved.ingest_format() {
        "text" => {
            if cli.round.is_some() {
                return Err(std::io::Error::other(
                    "--round is only valid when the resolved profile uses round ingest",
                )
                .into());
            }
            vec![
                build_unit(
                    "session-text".to_string(),
                    None,
                    None,
                    turns.len(),
                    &chunk_config,
                    &ExtractionInput {
                        content: format_session_text(turns, &date_str),
                        bank_id: BankId::new(),
                        context: None,
                        timestamp,
                        turn_id: None,
                        custom_instructions: None,
                        speaker: None,
                    },
                    cli.chunk,
                )
                .map_err(std::io::Error::other)?,
            ]
        }
        "json" => {
            if cli.round.is_some() {
                return Err(std::io::Error::other(
                    "--round is only valid when the resolved profile uses round ingest",
                )
                .into());
            }
            vec![
                build_unit(
                    "session-json".to_string(),
                    None,
                    None,
                    turns.len(),
                    &chunk_config,
                    &ExtractionInput {
                        content: format!(
                            "{}\n\n{}",
                            parse_date_prefix(&date_str),
                            format_session_json(turns)
                        ),
                        bank_id: BankId::new(),
                        context: None,
                        timestamp,
                        turn_id: None,
                        custom_instructions: None,
                        speaker: None,
                    },
                    cli.chunk,
                )
                .map_err(std::io::Error::other)?,
            ]
        }
        "round" => {
            let rounds = round_slices(turns);
            if let Some(round) = cli.round
                && (round == 0 || round > rounds.len())
            {
                return Err(std::io::Error::other(format!(
                    "--round {} is out of range for session {} ({} rounds)",
                    round,
                    cli.session,
                    rounds.len()
                ))
                .into());
            }

            rounds
                .iter()
                .enumerate()
                .filter(|(idx, _)| cli.round.is_none_or(|round| round == idx + 1))
                .map(|(idx, round_turns)| {
                    build_unit(
                        "round-text".to_string(),
                        Some(idx + 1),
                        Some(rounds.len()),
                        round_turns.len(),
                        &chunk_config,
                        &ExtractionInput {
                            content: format_round_text(round_turns, &date_str),
                            bank_id: BankId::new(),
                            context: None,
                            timestamp,
                            turn_id: None,
                            custom_instructions: None,
                            speaker: None,
                        },
                        cli.chunk,
                    )
                })
                .collect::<Result<Vec<_>, _>>()
                .map_err(std::io::Error::other)?
        }
        other => {
            return Err(std::io::Error::other(format!(
                "unsupported resolved ingest format: {other}"
            ))
            .into());
        }
    };

    Ok(InspectionBundle {
        profile: cli.profile.clone(),
        profile_path: cli.profile_path.display().to_string(),
        config_path: cli.config_path.as_ref().map(|p| p.display().to_string()),
        dataset: resolved.dataset_path().display().to_string(),
        contract_hash: resolved.contract_hash().to_string(),
        ingest_format: resolved.ingest_format().to_string(),
        session_limit: resolved.session_limit(),
        item_id: entry.question_id.clone(),
        session: cli.session,
        session_id: entry.haystack_session_ids.get(index).cloned(),
        session_date: date_str,
        timestamp: timestamp.to_rfc3339(),
        question: entry.question.clone(),
        answer: answer_to_string(&entry.answer),
        question_type: entry.reporting_category().to_string(),
        question_date: entry.question_date.clone(),
        chunk_config,
        total_units: units.len(),
        displayed_units: units.len(),
        units,
    })
}

fn html_escape(input: &str) -> String {
    let mut escaped = String::with_capacity(input.len());
    for ch in input.chars() {
        match ch {
            '&' => escaped.push_str("&amp;"),
            '<' => escaped.push_str("&lt;"),
            '>' => escaped.push_str("&gt;"),
            '"' => escaped.push_str("&quot;"),
            '\'' => escaped.push_str("&#39;"),
            _ => escaped.push(ch),
        }
    }
    escaped
}

fn render_annotated_input(unit: &InspectionUnit) -> String {
    #[derive(Clone, Copy)]
    enum MarkerKind {
        Start,
        End,
    }

    #[derive(Clone, Copy)]
    struct Marker {
        offset: usize,
        chunk_index: usize,
        kind: MarkerKind,
    }

    let mut markers = Vec::new();
    for chunk in &unit.chunks {
        markers.push(Marker {
            offset: chunk.start_byte,
            chunk_index: chunk.index,
            kind: MarkerKind::Start,
        });
        markers.push(Marker {
            offset: chunk.end_byte,
            chunk_index: chunk.index,
            kind: MarkerKind::End,
        });
    }

    markers.sort_by_key(|marker| {
        let order = match marker.kind {
            MarkerKind::End => 0usize,
            MarkerKind::Start => 1usize,
        };
        (marker.offset, order, marker.chunk_index)
    });

    let mut out = String::new();
    let mut cursor = 0usize;
    for marker in markers {
        if marker.offset > cursor {
            out.push_str(&html_escape(&unit.retain_input[cursor..marker.offset]));
            cursor = marker.offset;
        }
        match marker.kind {
            MarkerKind::Start => {
                let _ = write!(
                    out,
                    "<span class=\"marker start\">&#9654; chunk {}</span>",
                    marker.chunk_index
                );
            }
            MarkerKind::End => {
                let _ = write!(
                    out,
                    "<span class=\"marker end\">&#9664; chunk {}</span>",
                    marker.chunk_index
                );
            }
        }
    }
    if cursor < unit.retain_input.len() {
        out.push_str(&html_escape(&unit.retain_input[cursor..]));
    }
    out
}

fn unit_title(unit: &InspectionUnit) -> String {
    match unit.round {
        Some(round) => format!("Round {} / {}", round, unit.round_count.unwrap_or(round)),
        None => "Retain Unit".to_string(),
    }
}

fn render_text(doc: &InspectionBundle) -> String {
    let mut out = String::new();
    let _ = writeln!(out, "profile: {}", doc.profile);
    let _ = writeln!(out, "profile_path: {}", doc.profile_path);
    if let Some(config_path) = &doc.config_path {
        let _ = writeln!(out, "config_path: {}", config_path);
    }
    let _ = writeln!(out, "dataset: {}", doc.dataset);
    let _ = writeln!(out, "contract_hash: {}", doc.contract_hash);
    let _ = writeln!(out, "ingest_format: {}", doc.ingest_format);
    let _ = writeln!(out, "session_limit: {:?}", doc.session_limit);
    let _ = writeln!(out, "item_id: {}", doc.item_id);
    let _ = writeln!(out, "session: {}", doc.session);
    if let Some(session_id) = &doc.session_id {
        let _ = writeln!(out, "session_id: {}", session_id);
    }
    let _ = writeln!(out, "session_date: {}", doc.session_date);
    let _ = writeln!(out, "timestamp: {}", doc.timestamp);
    let _ = writeln!(out, "question_type: {}", doc.question_type);
    let _ = writeln!(out, "question_date: {}", doc.question_date);
    let _ = writeln!(out, "question: {}", doc.question);
    let _ = writeln!(out, "answer: {}", doc.answer);
    let _ = writeln!(
        out,
        "chunk_config: max_tokens={} overlap_tokens={} preserve_turns={}",
        doc.chunk_config.max_tokens,
        doc.chunk_config.overlap_tokens,
        doc.chunk_config.preserve_turns
    );
    let _ = writeln!(out, "displayed_units: {}", doc.displayed_units);
    let _ = writeln!(out, "total_units: {}", doc.total_units);

    for unit in &doc.units {
        let _ = writeln!(out);
        let _ = writeln!(out, "=== {} ===", unit_title(unit));
        let _ = writeln!(out, "mode: {}", unit.mode);
        let _ = writeln!(out, "turn_count: {}", unit.turn_count);
        let _ = writeln!(out, "retain_input_chars: {}", unit.retain_input_chars);
        let _ = writeln!(
            out,
            "retain_input_est_tokens: {}",
            unit.retain_input_est_tokens
        );
        let _ = writeln!(out, "extractor_chunks: {}", unit.total_chunk_count);
        let _ = writeln!(out, "displayed_chunks: {}", unit.displayed_chunk_count);
        let _ = writeln!(out, "--- retain input ---");
        let _ = writeln!(out, "{}", unit.retain_input);
        let _ = writeln!(out, "--- end retain input ---");

        for chunk in &unit.chunks {
            let _ = writeln!(out);
            let _ = writeln!(out, "=== Extractor Chunk {} ===", chunk.index);
            let _ = writeln!(out, "byte_range: {}..{}", chunk.start_byte, chunk.end_byte);
            let _ = writeln!(out, "content_chars: {}", chunk.content_chars);
            let _ = writeln!(out, "content_est_tokens: {}", chunk.content_est_tokens);
            let _ = writeln!(
                out,
                "speaker: {}",
                chunk.speaker.as_deref().unwrap_or("<none>")
            );
            let _ = writeln!(out, "context_chars: {}", chunk.context_chars);
            let _ = writeln!(out, "context_est_tokens: {}", chunk.context_est_tokens);
            let _ = writeln!(out, "--- content ---");
            let _ = writeln!(out, "{}", chunk.content);
            let _ = writeln!(out, "--- end content ---");
            if let Some(context) = &chunk.context {
                let _ = writeln!(out, "--- context ---");
                let _ = writeln!(out, "{}", context);
                let _ = writeln!(out, "--- end context ---");
            }
            let _ = writeln!(out, "--- extractor system prompt ---");
            let _ = writeln!(out, "{}", chunk.extractor_system_prompt);
            let _ = writeln!(out, "--- end extractor system prompt ---");
            let _ = writeln!(out, "--- extractor user message ---");
            let _ = writeln!(out, "{}", chunk.extractor_user_message);
            let _ = writeln!(out, "--- end extractor user message ---");
        }
    }

    out
}

fn render_html(doc: &InspectionBundle) -> String {
    let mut out = String::new();
    let title = format!(
        "{} retain inspection: {} session {}",
        doc.profile, doc.item_id, doc.session
    );

    out.push_str("<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">");
    let _ = write!(out, "<title>{}</title>", html_escape(&title));
    out.push_str(
        "<style>
        :root {
            --bg: #f3efe3;
            --panel: #fffaf2;
            --ink: #1f1b16;
            --muted: #6f6559;
            --line: #d9cdbf;
            --accent: #9d3f27;
            --accent-soft: #f6d7c8;
            --chip: #efe2cf;
            --code: #f8f2e8;
        }
        * { box-sizing: border-box; }
        body {
            margin: 0;
            padding: 24px;
            background: linear-gradient(180deg, #f5f0e7 0%, #efe6d7 100%);
            color: var(--ink);
            font: 14px/1.45 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
        }
        .layout {
            display: grid;
            grid-template-columns: 380px minmax(0, 1fr);
            gap: 20px;
            align-items: start;
        }
        .panel {
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 16px;
            padding: 18px;
            box-shadow: 0 16px 40px rgba(31, 27, 22, 0.06);
        }
        .sticky { position: sticky; top: 24px; }
        h1, h2, h3 { margin: 0 0 12px; line-height: 1.15; }
        h1 { font-size: 20px; }
        h2 { font-size: 16px; }
        h3 { font-size: 14px; }
        .summary-row {
            display: grid;
            grid-template-columns: 120px minmax(0, 1fr);
            gap: 12px;
            padding: 6px 0;
            border-bottom: 1px solid rgba(217, 205, 191, 0.55);
        }
        .summary-row:last-child { border-bottom: 0; }
        .label { color: var(--muted); }
        .value { white-space: pre-wrap; word-break: break-word; }
        pre {
            margin: 0;
            white-space: pre-wrap;
            word-break: break-word;
            background: var(--code);
            border: 1px solid var(--line);
            border-radius: 12px;
            padding: 14px;
            overflow-wrap: anywhere;
        }
        .input-view { max-height: 60vh; overflow: auto; }
        .marker {
            display: inline-block;
            margin: 0 2px;
            padding: 0 6px;
            border-radius: 999px;
            font-size: 11px;
            line-height: 1.8;
            vertical-align: middle;
        }
        .marker.start { background: var(--accent); color: white; }
        .marker.end { background: var(--accent-soft); color: var(--accent); }
        .chunk-card { margin-top: 16px; }
        .chunk-meta {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin: 10px 0 12px;
        }
        .chip {
            background: var(--chip);
            border: 1px solid var(--line);
            border-radius: 999px;
            padding: 4px 8px;
            color: var(--ink);
        }
        details { margin-top: 10px; }
        details > summary { cursor: pointer; color: var(--accent); }
        @media (max-width: 1080px) {
            .layout { grid-template-columns: 1fr; }
            .sticky { position: static; }
        }
        </style></head><body>",
    );
    out.push_str("<div class=\"layout\">");
    out.push_str("<aside class=\"panel sticky\">");
    let _ = write!(out, "<h1>{}</h1>", html_escape(&title));
    for (label, value) in [
        ("Profile", doc.profile.as_str()),
        ("Profile Path", doc.profile_path.as_str()),
        ("Dataset", doc.dataset.as_str()),
        ("Contract", doc.contract_hash.as_str()),
        ("Ingest", doc.ingest_format.as_str()),
        ("Item", doc.item_id.as_str()),
        ("Session Date", doc.session_date.as_str()),
        ("Timestamp", doc.timestamp.as_str()),
        ("Question Type", doc.question_type.as_str()),
        ("Question Date", doc.question_date.as_str()),
        ("Question", doc.question.as_str()),
        ("Answer", doc.answer.as_str()),
    ] {
        let _ = write!(
            out,
            "<div class=\"summary-row\"><span class=\"label\">{}</span><span class=\"value\">{}</span></div>",
            html_escape(label),
            html_escape(value)
        );
    }
    if let Some(config_path) = &doc.config_path {
        let _ = write!(
            out,
            "<div class=\"summary-row\"><span class=\"label\">Config</span><span class=\"value\">{}</span></div>",
            html_escape(config_path)
        );
    }
    if let Some(session_id) = &doc.session_id {
        let _ = write!(
            out,
            "<div class=\"summary-row\"><span class=\"label\">Session ID</span><span class=\"value\">{}</span></div>",
            html_escape(session_id)
        );
    }
    let _ = write!(
        out,
        "<div class=\"summary-row\"><span class=\"label\">Session</span><span class=\"value\">{}</span></div>",
        doc.session
    );
    let _ = write!(
        out,
        "<div class=\"summary-row\"><span class=\"label\">Chunking</span><span class=\"value\">{} max / {} overlap</span></div>",
        doc.chunk_config.max_tokens, doc.chunk_config.overlap_tokens
    );
    let _ = write!(
        out,
        "<div class=\"summary-row\"><span class=\"label\">Units</span><span class=\"value\">showing {} of {}</span></div>",
        doc.displayed_units, doc.total_units
    );
    out.push_str("</aside><main>");

    for unit in &doc.units {
        out.push_str("<section class=\"panel\">");
        let title = unit_title(unit);
        let _ = write!(out, "<h2>{}</h2>", html_escape(&title));
        out.push_str("<div class=\"chunk-meta\">");
        let _ = write!(
            out,
            "<span class=\"chip\">mode {}</span>",
            html_escape(&unit.mode)
        );
        let _ = write!(out, "<span class=\"chip\">turns {}</span>", unit.turn_count);
        let _ = write!(
            out,
            "<span class=\"chip\">input {} chars / {} est tokens</span>",
            unit.retain_input_chars, unit.retain_input_est_tokens
        );
        let _ = write!(
            out,
            "<span class=\"chip\">chunks showing {} of {}</span>",
            unit.displayed_chunk_count, unit.total_chunk_count
        );
        out.push_str("</div>");
        out.push_str("<div class=\"input-view\"><pre>");
        out.push_str(&render_annotated_input(unit));
        out.push_str("</pre></div></section>");

        for chunk in &unit.chunks {
            out.push_str("<section class=\"panel chunk-card\">");
            let _ = write!(
                out,
                "<h2>{}: Chunk {}</h2>",
                html_escape(&title),
                chunk.index
            );
            out.push_str("<div class=\"chunk-meta\">");
            let _ = write!(
                out,
                "<span class=\"chip\">bytes {}..{}</span>",
                chunk.start_byte, chunk.end_byte
            );
            let _ = write!(
                out,
                "<span class=\"chip\">content {} chars / {} est tokens</span>",
                chunk.content_chars, chunk.content_est_tokens
            );
            let _ = write!(
                out,
                "<span class=\"chip\">context {} chars / {} est tokens</span>",
                chunk.context_chars, chunk.context_est_tokens
            );
            if let Some(speaker) = &chunk.speaker {
                let _ = write!(
                    out,
                    "<span class=\"chip\">speaker {}</span>",
                    html_escape(speaker)
                );
            }
            out.push_str("</div>");

            out.push_str("<h3>Chunk Content</h3><pre>");
            out.push_str(&html_escape(&chunk.content));
            out.push_str("</pre>");
            if let Some(context) = &chunk.context {
                out.push_str("<h3>Chunk Context</h3><pre>");
                out.push_str(&html_escape(context));
                out.push_str("</pre>");
            }
            out.push_str("<details><summary>Extractor System Prompt</summary><pre>");
            out.push_str(&html_escape(&chunk.extractor_system_prompt));
            out.push_str("</pre></details>");
            out.push_str("<details open><summary>Extractor User Message</summary><pre>");
            out.push_str(&html_escape(&chunk.extractor_user_message));
            out.push_str("</pre></details>");
            out.push_str("</section>");
        }
    }

    out.push_str("</main></div></body></html>");
    out
}

fn emit_output(cli: &Cli, rendered: &str) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(path) = &cli.out {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(path, rendered)?;
        eprintln!(
            "wrote {} output to {}",
            cli.output_format.as_str(),
            path.display()
        );
    } else {
        print!("{rendered}");
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = parse_args().map_err(std::io::Error::other)?;
    let doc = inspect_longmemeval(&cli)?;
    let rendered = match cli.output_format {
        OutputFormat::Text => render_text(&doc),
        OutputFormat::Json => serde_json::to_string_pretty(&doc)?,
        OutputFormat::Html => render_html(&doc),
    };
    emit_output(&cli, &rendered)
}
