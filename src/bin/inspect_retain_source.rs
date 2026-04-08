use std::collections::HashMap;
use std::env;
use std::fmt::Write as _;
use std::fs;
use std::path::PathBuf;

use chrono::{DateTime, NaiveDate, NaiveDateTime, Utc};
use elephant::retain::chunker::{Chunker, SimpleChunker};
use elephant::retain::extractor::LlmFactExtractor;
use elephant::types::{BankId, ChunkConfig, ExtractionInput};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BenchmarkKind {
    Locomo,
    LongMemEval,
}

impl BenchmarkKind {
    fn as_str(self) -> &'static str {
        match self {
            Self::Locomo => "locomo",
            Self::LongMemEval => "longmemeval",
        }
    }

    fn default_dataset(self) -> PathBuf {
        match self {
            Self::Locomo => PathBuf::from("data/locomo10.json"),
            Self::LongMemEval => PathBuf::from("data/longmemeval_s_cleaned.json"),
        }
    }
}

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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum InspectMode {
    Session,
    Turn,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LongMemIngestFormat {
    Text,
    Json,
}

impl LongMemIngestFormat {
    fn as_str(self) -> &'static str {
        match self {
            Self::Text => "text",
            Self::Json => "json",
        }
    }
}

struct Cli {
    benchmark: BenchmarkKind,
    dataset: PathBuf,
    sample_id: Option<String>,
    question_id: Option<String>,
    session: usize,
    mode: InspectMode,
    turn: Option<usize>,
    raw_json: bool,
    longmem_format: LongMemIngestFormat,
    chunk: Option<usize>,
    output_format: OutputFormat,
    out: Option<PathBuf>,
}

#[derive(Debug, Deserialize)]
struct LocomoEntry {
    sample_id: String,
    conversation: LocomoConversation,
}

#[derive(Debug, Deserialize)]
struct LocomoConversation {
    #[serde(flatten)]
    sessions: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Deserialize)]
struct LocomoTurn {
    speaker: String,
    text: String,
    #[serde(default)]
    blip_caption: Option<String>,
}

#[derive(Debug, Deserialize)]
struct LongMemEvalInstance {
    question_id: String,
    #[serde(rename = "question_type")]
    question_type: String,
    question: String,
    answer: serde_json::Value,
    question_date: String,
    haystack_dates: Vec<String>,
    haystack_session_ids: Vec<String>,
    haystack_sessions: Vec<Vec<LongMemTurn>>,
}

#[derive(Debug, Deserialize)]
struct LongMemTurn {
    role: String,
    content: String,
}

#[derive(Debug, Serialize)]
struct InspectionDocument {
    benchmark: String,
    dataset: String,
    item_id: String,
    session: usize,
    session_id: Option<String>,
    mode: String,
    session_date: String,
    timestamp: String,
    item_label: Option<String>,
    question: Option<String>,
    answer: Option<String>,
    question_type: Option<String>,
    question_date: Option<String>,
    chunk_config: ChunkConfig,
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
    eprintln!("  cargo run --bin inspect_retain_source -- --sample <ID> --session <N> [options]");
    eprintln!(
        "  cargo run --bin inspect_retain_source -- --benchmark longmemeval --question <ID> --session <N> [options]"
    );
    eprintln!();
    eprintln!("Note: for bench-profile-aware LongMemEval inspection, use:");
    eprintln!(
        "  cargo run -p elephant-bench --bin inspect_retain_source -- --profile <name> --question <ID> --session <N> [options]"
    );
    eprintln!();
    eprintln!("General options:");
    eprintln!("  --benchmark <NAME>   locomo | longmemeval [default: locomo]");
    eprintln!("  --dataset <PATH>     Dataset path [default depends on --benchmark]");
    eprintln!("  --session <N>        Session number to inspect");
    eprintln!("  --chunk <N>          Show only one extractor chunk");
    eprintln!("  --output <FORMAT>    text | json | html [default: text]");
    eprintln!("  --out <PATH>         Write output to a file instead of stdout");
    eprintln!("  --help               Show this help");
    eprintln!();
    eprintln!("LoCoMo options:");
    eprintln!("  --sample <ID>        Conversation/sample id, e.g. conv-43");
    eprintln!("  --mode <MODE>        session | turn [default: session]");
    eprintln!("  --turn <N>           Turn number within the session (required for --mode turn)");
    eprintln!("  --raw-json           Use the raw-json session formatter");
    eprintln!();
    eprintln!("LongMemEval options:");
    eprintln!("  --question <ID>      Question id, e.g. e47becba");
    eprintln!("  --ingest-format <F>  text | json [default: text]");
}

fn parse_args() -> Result<Cli, String> {
    let mut benchmark = BenchmarkKind::Locomo;
    let mut dataset = None;
    let mut sample_id = None;
    let mut question_id = None;
    let mut session = None;
    let mut mode = InspectMode::Session;
    let mut turn = None;
    let mut raw_json = false;
    let mut longmem_format = LongMemIngestFormat::Text;
    let mut chunk = None;
    let mut output_format = OutputFormat::Text;
    let mut out = None;

    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--benchmark" => {
                benchmark = match args
                    .next()
                    .ok_or_else(|| "--benchmark requires a value".to_string())?
                    .as_str()
                {
                    "locomo" => BenchmarkKind::Locomo,
                    "longmemeval" => BenchmarkKind::LongMemEval,
                    other => {
                        return Err(format!(
                            "invalid --benchmark value: {other} (expected locomo or longmemeval)"
                        ));
                    }
                };
            }
            "--dataset" => {
                dataset = Some(PathBuf::from(
                    args.next()
                        .ok_or_else(|| "--dataset requires a path".to_string())?,
                ));
            }
            "--sample" => {
                sample_id = Some(
                    args.next()
                        .ok_or_else(|| "--sample requires a value".to_string())?,
                );
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
            "--mode" => {
                mode = match args
                    .next()
                    .ok_or_else(|| "--mode requires a value".to_string())?
                    .as_str()
                {
                    "session" => InspectMode::Session,
                    "turn" => InspectMode::Turn,
                    other => {
                        return Err(format!(
                            "invalid --mode value: {other} (expected session or turn)"
                        ));
                    }
                };
            }
            "--turn" => {
                turn = Some(
                    args.next()
                        .ok_or_else(|| "--turn requires a value".to_string())?
                        .parse()
                        .map_err(|_| "--turn must be an integer".to_string())?,
                );
            }
            "--raw-json" => raw_json = true,
            "--ingest-format" => {
                longmem_format = match args
                    .next()
                    .ok_or_else(|| "--ingest-format requires a value".to_string())?
                    .as_str()
                {
                    "text" => LongMemIngestFormat::Text,
                    "json" => LongMemIngestFormat::Json,
                    other => {
                        return Err(format!(
                            "invalid --ingest-format value: {other} (expected text or json)"
                        ));
                    }
                };
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

    let session = session.ok_or_else(|| "--session is required".to_string())?;
    let dataset = dataset.unwrap_or_else(|| benchmark.default_dataset());

    match benchmark {
        BenchmarkKind::Locomo => {
            if sample_id.is_none() {
                return Err("--sample is required for LoCoMo inspection".to_string());
            }
            if mode == InspectMode::Turn && turn.is_none() {
                return Err("--turn is required when --mode turn".to_string());
            }
        }
        BenchmarkKind::LongMemEval => {
            if question_id.is_none() {
                return Err("--question is required for LongMemEval inspection".to_string());
            }
            if mode != InspectMode::Session {
                return Err("LongMemEval inspector currently supports only --mode session".into());
            }
            if raw_json {
                return Err("--raw-json is only valid for LoCoMo inspection".into());
            }
            if turn.is_some() {
                return Err("--turn is only valid for LoCoMo turn inspection".into());
            }
        }
    }

    Ok(Cli {
        benchmark,
        dataset,
        sample_id,
        question_id,
        session,
        mode,
        turn,
        raw_json,
        longmem_format,
        chunk,
        output_format,
        out,
    })
}

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

fn parse_longmem_date(date_str: &str) -> DateTime<Utc> {
    let trimmed = date_str.trim();
    let before_paren = trimmed.split('(').next().unwrap_or(trimmed).trim();
    let after_paren = trimmed.split(')').nth(1).unwrap_or("00:00").trim();
    let cleaned = format!("{before_paren} {after_paren}");

    if let Ok(ndt) = NaiveDateTime::parse_from_str(&cleaned, "%Y/%m/%d %H:%M") {
        return ndt.and_utc();
    }

    let date_only = trimmed.split(" (").next().unwrap_or(trimmed);
    if let Ok(nd) = NaiveDate::parse_from_str(date_only, "%Y/%m/%d") {
        return nd.and_hms_opt(0, 0, 0).unwrap().and_utc();
    }

    Utc::now()
}

fn parse_longmem_date_prefix(date_str: &str) -> String {
    let date_part = date_str.split(" (").next().unwrap_or(date_str);
    let iso_date = date_part.replace('/', "-");
    format!("[Date: {iso_date}]")
}

fn get_locomo_session_turns(conv: &LocomoConversation, idx: usize) -> Vec<LocomoTurn> {
    let plain_key = format!("session_{idx}");
    if let Some(v) = conv.sessions.get(&plain_key)
        && let Ok(turns) = serde_json::from_value::<Vec<LocomoTurn>>(v.clone())
    {
        return turns;
    }

    for suffix in ["dialogue", "dialog"] {
        let key = format!("session_{idx}_{suffix}");
        if let Some(v) = conv.sessions.get(&key)
            && let Ok(turns) = serde_json::from_value::<Vec<LocomoTurn>>(v.clone())
        {
            return turns;
        }
    }

    Vec::new()
}

fn get_locomo_session_date(conv: &LocomoConversation, idx: usize) -> String {
    let key = format!("session_{idx}_date_time");
    conv.sessions
        .get(&key)
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string()
}

fn format_locomo_turn(turn: &LocomoTurn) -> String {
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

fn format_locomo_session(turns: &[LocomoTurn], date_str: &str) -> String {
    let dialogue = turns
        .iter()
        .map(format_locomo_turn)
        .collect::<Vec<_>>()
        .join("\n");
    format!("Date: {date_str}\n\n{dialogue}")
}

fn format_locomo_session_raw(conv: &LocomoConversation, idx: usize) -> String {
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

fn build_locomo_turn_input(
    turns: &[LocomoTurn],
    date_str: &str,
    turn_number: usize,
) -> Result<ExtractionInput, String> {
    if turn_number == 0 || turn_number > turns.len() {
        return Err(format!(
            "--turn {} is out of range for session with {} turns",
            turn_number,
            turns.len()
        ));
    }

    let mut prior_turns = Vec::new();
    for turn in turns.iter().take(turn_number - 1) {
        prior_turns.push(format_locomo_turn(turn));
    }

    let turn = &turns[turn_number - 1];
    Ok(ExtractionInput {
        content: format_locomo_turn(turn),
        bank_id: BankId::new(),
        context: if prior_turns.is_empty() {
            None
        } else {
            Some(prior_turns.join("\n"))
        },
        timestamp: parse_session_date(date_str),
        turn_id: None,
        custom_instructions: None,
        speaker: Some(turn.speaker.clone()),
    })
}

fn format_longmem_session_text(turns: &[LongMemTurn], date_str: &str) -> String {
    let date_prefix = parse_longmem_date_prefix(date_str);
    let dialogue = turns
        .iter()
        .map(|t| format!("{}: {}", t.role, t.content))
        .collect::<Vec<_>>()
        .join("\n");
    format!("{date_prefix}\n\n{dialogue}")
}

fn format_longmem_session_json(turns: &[LongMemTurn], date_str: &str) -> String {
    let cleaned: Vec<serde_json::Value> = turns
        .iter()
        .map(|t| serde_json::json!({"role": t.role, "content": t.content}))
        .collect();
    format!(
        "{}\n\n{}",
        parse_longmem_date_prefix(date_str),
        serde_json::to_string(&cleaned).unwrap_or_default()
    )
}

fn answer_to_string(val: &serde_json::Value) -> String {
    match val {
        serde_json::Value::String(s) => s.clone(),
        serde_json::Value::Number(n) => n.to_string(),
        other => other.to_string(),
    }
}

fn estimate_tokens(text: &str) -> usize {
    text.len().div_ceil(4)
}

fn build_chunk_config() -> ChunkConfig {
    ChunkConfig {
        max_tokens: 512,
        overlap_tokens: 64,
        preserve_turns: true,
    }
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

fn build_document(
    mut doc: InspectionDocument,
    base_input: &ExtractionInput,
    selected_chunk: Option<usize>,
) -> Result<InspectionDocument, String> {
    let chunk_config = doc.chunk_config.clone();
    let spans = reconstruct_chunk_spans(&base_input.content, &chunk_config);
    let inputs = extractor_inputs(base_input, &chunk_config);

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

    doc.total_chunk_count = inputs.len();
    doc.displayed_chunk_count = chunks.len();
    doc.chunks = chunks;
    Ok(doc)
}

fn inspect_locomo(cli: &Cli) -> Result<InspectionDocument, Box<dyn std::error::Error>> {
    let raw = fs::read_to_string(&cli.dataset)?;
    let entries: Vec<LocomoEntry> = serde_json::from_str(&raw)?;
    let sample_id = cli.sample_id.as_ref().expect("validated");
    let entry = entries
        .iter()
        .find(|entry| entry.sample_id == *sample_id)
        .ok_or_else(|| std::io::Error::other(format!("sample {} not found", sample_id)))?;

    let turns = get_locomo_session_turns(&entry.conversation, cli.session);
    if turns.is_empty() {
        return Err(std::io::Error::other(format!(
            "session {} not found for sample {}",
            cli.session, sample_id
        ))
        .into());
    }

    let date_str = get_locomo_session_date(&entry.conversation, cli.session);
    let chunk_config = build_chunk_config();
    let base_input = match cli.mode {
        InspectMode::Session => ExtractionInput {
            content: if cli.raw_json {
                format_locomo_session_raw(&entry.conversation, cli.session)
            } else {
                format_locomo_session(&turns, &date_str)
            },
            bank_id: BankId::new(),
            context: None,
            timestamp: parse_session_date(&date_str),
            turn_id: None,
            custom_instructions: None,
            speaker: None,
        },
        InspectMode::Turn => {
            build_locomo_turn_input(&turns, &date_str, cli.turn.expect("validated"))
                .map_err(std::io::Error::other)?
        }
    };

    let doc = InspectionDocument {
        benchmark: BenchmarkKind::Locomo.as_str().to_string(),
        dataset: cli.dataset.display().to_string(),
        item_id: sample_id.clone(),
        session: cli.session,
        session_id: None,
        mode: match cli.mode {
            InspectMode::Session => {
                if cli.raw_json {
                    "session-raw-json".to_string()
                } else {
                    "session".to_string()
                }
            }
            InspectMode::Turn => "turn".to_string(),
        },
        session_date: date_str,
        timestamp: base_input.timestamp.to_rfc3339(),
        item_label: None,
        question: None,
        answer: None,
        question_type: None,
        question_date: None,
        chunk_config,
        retain_input_chars: base_input.content.len(),
        retain_input_est_tokens: estimate_tokens(&base_input.content),
        retain_input: base_input.content.clone(),
        total_chunk_count: 0,
        displayed_chunk_count: 0,
        chunks: Vec::new(),
    };

    build_document(doc, &base_input, cli.chunk)
        .map_err(|e| -> Box<dyn std::error::Error> { Box::new(std::io::Error::other(e)) })
}

fn inspect_longmemeval(cli: &Cli) -> Result<InspectionDocument, Box<dyn std::error::Error>> {
    let raw = fs::read_to_string(&cli.dataset)?;
    let entries: Vec<LongMemEvalInstance> = serde_json::from_str(&raw)?;
    let question_id = cli.question_id.as_ref().expect("validated");
    let entry = entries
        .iter()
        .find(|entry| entry.question_id == *question_id)
        .ok_or_else(|| std::io::Error::other(format!("question {} not found", question_id)))?;

    if cli.session == 0 || cli.session > entry.haystack_sessions.len() {
        return Err(std::io::Error::other(format!(
            "session {} is out of range for question {} ({} sessions)",
            cli.session,
            question_id,
            entry.haystack_sessions.len()
        ))
        .into());
    }

    let index = cli.session - 1;
    let turns = &entry.haystack_sessions[index];
    let date_str = entry.haystack_dates[index].clone();
    let content = match cli.longmem_format {
        LongMemIngestFormat::Text => format_longmem_session_text(turns, &date_str),
        LongMemIngestFormat::Json => format_longmem_session_json(turns, &date_str),
    };
    let timestamp = parse_longmem_date(&date_str);
    let base_input = ExtractionInput {
        content,
        bank_id: BankId::new(),
        context: None,
        timestamp,
        turn_id: None,
        custom_instructions: None,
        speaker: None,
    };

    let doc = InspectionDocument {
        benchmark: BenchmarkKind::LongMemEval.as_str().to_string(),
        dataset: cli.dataset.display().to_string(),
        item_id: question_id.clone(),
        session: cli.session,
        session_id: entry.haystack_session_ids.get(index).cloned(),
        mode: format!("session-{}", cli.longmem_format.as_str()),
        session_date: date_str,
        timestamp: base_input.timestamp.to_rfc3339(),
        item_label: Some(format!("LongMemEval {}", entry.question_id)),
        question: Some(entry.question.clone()),
        answer: Some(answer_to_string(&entry.answer)),
        question_type: Some(entry.question_type.clone()),
        question_date: Some(entry.question_date.clone()),
        chunk_config: build_chunk_config(),
        retain_input_chars: base_input.content.len(),
        retain_input_est_tokens: estimate_tokens(&base_input.content),
        retain_input: base_input.content.clone(),
        total_chunk_count: 0,
        displayed_chunk_count: 0,
        chunks: Vec::new(),
    };

    build_document(doc, &base_input, cli.chunk)
        .map_err(|e| -> Box<dyn std::error::Error> { Box::new(std::io::Error::other(e)) })
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

fn render_annotated_input(doc: &InspectionDocument) -> String {
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
    for chunk in &doc.chunks {
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
            out.push_str(&html_escape(&doc.retain_input[cursor..marker.offset]));
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
    if cursor < doc.retain_input.len() {
        out.push_str(&html_escape(&doc.retain_input[cursor..]));
    }
    out
}

fn render_text(doc: &InspectionDocument) -> String {
    let mut out = String::new();
    let _ = writeln!(out, "benchmark: {}", doc.benchmark);
    let _ = writeln!(out, "dataset: {}", doc.dataset);
    let _ = writeln!(out, "item_id: {}", doc.item_id);
    if let Some(label) = &doc.item_label {
        let _ = writeln!(out, "item_label: {}", label);
    }
    let _ = writeln!(out, "session: {}", doc.session);
    if let Some(session_id) = &doc.session_id {
        let _ = writeln!(out, "session_id: {}", session_id);
    }
    let _ = writeln!(out, "mode: {}", doc.mode);
    let _ = writeln!(out, "session_date: {}", doc.session_date);
    let _ = writeln!(out, "timestamp: {}", doc.timestamp);
    if let Some(question_type) = &doc.question_type {
        let _ = writeln!(out, "question_type: {}", question_type);
    }
    if let Some(question_date) = &doc.question_date {
        let _ = writeln!(out, "question_date: {}", question_date);
    }
    if let Some(question) = &doc.question {
        let _ = writeln!(out, "question: {}", question);
    }
    if let Some(answer) = &doc.answer {
        let _ = writeln!(out, "answer: {}", answer);
    }
    let _ = writeln!(
        out,
        "chunk_config: max_tokens={} overlap_tokens={} preserve_turns={}",
        doc.chunk_config.max_tokens,
        doc.chunk_config.overlap_tokens,
        doc.chunk_config.preserve_turns
    );
    let _ = writeln!(out, "retain_input_chars: {}", doc.retain_input_chars);
    let _ = writeln!(
        out,
        "retain_input_est_tokens: {}",
        doc.retain_input_est_tokens
    );
    let _ = writeln!(out, "extractor_chunks: {}", doc.total_chunk_count);
    let _ = writeln!(out, "displayed_chunks: {}", doc.displayed_chunk_count);
    let _ = writeln!(out, "--- retain input ---");
    let _ = writeln!(out, "{}", doc.retain_input);
    let _ = writeln!(out, "--- end retain input ---");

    for chunk in &doc.chunks {
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

    out
}

fn render_html(doc: &InspectionDocument) -> String {
    let mut out = String::new();
    let title = format!(
        "{} retain inspection: {} session {}",
        doc.benchmark, doc.item_id, doc.session
    );
    let question_block = doc.question.as_ref().map(|question| {
        format!(
            "<div class=\"summary-row\"><span class=\"label\">Question</span><span class=\"value\">{}</span></div>",
            html_escape(question)
        )
    });
    let answer_block = doc.answer.as_ref().map(|answer| {
        format!(
            "<div class=\"summary-row\"><span class=\"label\">Answer</span><span class=\"value\">{}</span></div>",
            html_escape(answer)
        )
    });

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
            grid-template-columns: 360px minmax(0, 1fr);
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
    let _ = write!(
        out,
        "<div class=\"summary-row\"><span class=\"label\">Benchmark</span><span class=\"value\">{}</span></div>",
        html_escape(&doc.benchmark)
    );
    let _ = write!(
        out,
        "<div class=\"summary-row\"><span class=\"label\">Dataset</span><span class=\"value\">{}</span></div>",
        html_escape(&doc.dataset)
    );
    let _ = write!(
        out,
        "<div class=\"summary-row\"><span class=\"label\">Item</span><span class=\"value\">{}</span></div>",
        html_escape(&doc.item_id)
    );
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
        "<div class=\"summary-row\"><span class=\"label\">Mode</span><span class=\"value\">{}</span></div>",
        html_escape(&doc.mode)
    );
    let _ = write!(
        out,
        "<div class=\"summary-row\"><span class=\"label\">Session Date</span><span class=\"value\">{}</span></div>",
        html_escape(&doc.session_date)
    );
    let _ = write!(
        out,
        "<div class=\"summary-row\"><span class=\"label\">Timestamp</span><span class=\"value\">{}</span></div>",
        html_escape(&doc.timestamp)
    );
    let _ = write!(
        out,
        "<div class=\"summary-row\"><span class=\"label\">Retain Input</span><span class=\"value\">{} chars / {} est tokens</span></div>",
        doc.retain_input_chars, doc.retain_input_est_tokens
    );
    let _ = write!(
        out,
        "<div class=\"summary-row\"><span class=\"label\">Chunks</span><span class=\"value\">showing {} of {}</span></div>",
        doc.displayed_chunk_count, doc.total_chunk_count
    );
    if let Some(question_type) = &doc.question_type {
        let _ = write!(
            out,
            "<div class=\"summary-row\"><span class=\"label\">Question Type</span><span class=\"value\">{}</span></div>",
            html_escape(question_type)
        );
    }
    if let Some(question_date) = &doc.question_date {
        let _ = write!(
            out,
            "<div class=\"summary-row\"><span class=\"label\">Question Date</span><span class=\"value\">{}</span></div>",
            html_escape(question_date)
        );
    }
    if let Some(block) = question_block {
        out.push_str(&block);
    }
    if let Some(block) = answer_block {
        out.push_str(&block);
    }
    out.push_str("</aside>");

    out.push_str("<main>");
    out.push_str("<section class=\"panel\">");
    out.push_str("<h2>Retain Input With Chunk Boundaries</h2>");
    out.push_str("<div class=\"input-view\"><pre>");
    out.push_str(&render_annotated_input(doc));
    out.push_str("</pre></div></section>");

    for chunk in &doc.chunks {
        out.push_str("<section class=\"panel chunk-card\">");
        let _ = write!(out, "<h2>Chunk {}</h2>", chunk.index);
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
    let doc = match cli.benchmark {
        BenchmarkKind::Locomo => inspect_locomo(&cli)?,
        BenchmarkKind::LongMemEval => inspect_longmemeval(&cli)?,
    };

    let rendered = match cli.output_format {
        OutputFormat::Text => render_text(&doc),
        OutputFormat::Json => serde_json::to_string_pretty(&doc)?,
        OutputFormat::Html => render_html(&doc),
    };

    emit_output(&cli, &rendered)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reconstruct_chunk_spans_match_chunker_output() {
        let input = "Date: 2023-05-30\n\nuser: hello there this is a longer message about work and school.\nassistant: sounds good.\nuser: I graduated with a degree in Business Administration and I am tracking reimbursements.\nassistant: great.".repeat(6);
        let config = build_chunk_config();
        let base_input = ExtractionInput {
            content: input.clone(),
            bank_id: BankId::new(),
            context: None,
            timestamp: Utc::now(),
            turn_id: None,
            custom_instructions: None,
            speaker: None,
        };
        let spans = reconstruct_chunk_spans(&input, &config);
        let inputs = extractor_inputs(&base_input, &config);
        assert_eq!(spans.len(), inputs.len());
        for (span, chunk) in spans.iter().zip(inputs.iter()) {
            assert_eq!(&input[span.start..span.end], chunk.content.as_str());
        }
    }

    #[test]
    fn render_html_includes_markers_and_chunk_content() {
        let doc = InspectionDocument {
            benchmark: "locomo".into(),
            dataset: "data/locomo10.json".into(),
            item_id: "conv-26".into(),
            session: 1,
            session_id: None,
            mode: "session".into(),
            session_date: "8:00 AM on 1 January, 2024".into(),
            timestamp: "2024-01-01T08:00:00+00:00".into(),
            item_label: None,
            question: None,
            answer: None,
            question_type: None,
            question_date: None,
            chunk_config: build_chunk_config(),
            retain_input: "alpha bravo charlie delta".into(),
            retain_input_chars: 25,
            retain_input_est_tokens: 7,
            total_chunk_count: 1,
            displayed_chunk_count: 1,
            chunks: vec![InspectionChunk {
                index: 0,
                start_byte: 0,
                end_byte: 25,
                content_chars: 25,
                content_est_tokens: 7,
                context_chars: 0,
                context_est_tokens: 0,
                speaker: None,
                content: "alpha bravo charlie delta".into(),
                context: None,
                extractor_system_prompt: "system".into(),
                extractor_user_message: "user".into(),
            }],
        };

        let html = render_html(&doc);
        assert!(html.contains("chunk 0"));
        assert!(html.contains("alpha bravo charlie delta"));
        assert!(html.contains("Extractor User Message"));
    }
}
