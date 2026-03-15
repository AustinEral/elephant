use std::collections::HashMap;
use std::env;
use std::fs;
use std::path::PathBuf;

use chrono::{DateTime, NaiveDateTime, Utc};
use elephant::retain::chunker::{Chunker, SimpleChunker};
use elephant::retain::extractor::LlmFactExtractor;
use elephant::types::{ChunkConfig, ExtractionInput};
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct LocomoEntry {
    sample_id: String,
    conversation: Conversation,
}

#[derive(Debug, Deserialize)]
struct Conversation {
    #[serde(flatten)]
    sessions: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Deserialize)]
struct Turn {
    speaker: String,
    text: String,
    #[serde(default)]
    blip_caption: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum InspectMode {
    Session,
    Turn,
}

struct Cli {
    dataset: PathBuf,
    sample_id: String,
    session: usize,
    mode: InspectMode,
    turn: Option<usize>,
    raw_json: bool,
    chunk: Option<usize>,
}

fn print_help() {
    eprintln!("Usage:");
    eprintln!("  cargo run --bin inspect_retain_source -- --sample <ID> --session <N> [options]");
    eprintln!();
    eprintln!("Options:");
    eprintln!("  --dataset <PATH>     Dataset path [default: data/locomo10.json]");
    eprintln!("  --sample <ID>        Conversation/sample id, e.g. conv-43");
    eprintln!("  --session <N>        Session number to inspect");
    eprintln!("  --mode <MODE>        session | turn [default: session]");
    eprintln!("  --turn <N>           Turn number within the session (required for --mode turn)");
    eprintln!("  --raw-json           Use the raw-json session formatter");
    eprintln!("  --chunk <N>          Show only one extractor chunk");
    eprintln!("  --help               Show this help");
}

fn parse_args() -> Result<Cli, String> {
    let mut dataset = PathBuf::from("data/locomo10.json");
    let mut sample_id = None;
    let mut session = None;
    let mut mode = InspectMode::Session;
    let mut turn = None;
    let mut raw_json = false;
    let mut chunk = None;

    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--dataset" => {
                dataset = PathBuf::from(
                    args.next()
                        .ok_or_else(|| "--dataset requires a path".to_string())?,
                );
            }
            "--sample" => {
                sample_id = Some(
                    args.next()
                        .ok_or_else(|| "--sample requires a value".to_string())?,
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
            "--chunk" => {
                chunk = Some(
                    args.next()
                        .ok_or_else(|| "--chunk requires a value".to_string())?
                        .parse()
                        .map_err(|_| "--chunk must be an integer".to_string())?,
                );
            }
            "--help" | "-h" => {
                print_help();
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument: {other}")),
        }
    }

    let sample_id = sample_id.ok_or_else(|| "--sample is required".to_string())?;
    let session = session.ok_or_else(|| "--session is required".to_string())?;
    if mode == InspectMode::Turn && turn.is_none() {
        return Err("--turn is required when --mode turn".to_string());
    }

    Ok(Cli {
        dataset,
        sample_id,
        session,
        mode,
        turn,
        raw_json,
        chunk,
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

fn get_session_turns(conv: &Conversation, idx: usize) -> Vec<Turn> {
    let plain_key = format!("session_{idx}");
    if let Some(v) = conv.sessions.get(&plain_key)
        && let Ok(turns) = serde_json::from_value::<Vec<Turn>>(v.clone())
    {
        return turns;
    }

    for suffix in ["dialogue", "dialog"] {
        let key = format!("session_{idx}_{suffix}");
        if let Some(v) = conv.sessions.get(&key)
            && let Ok(turns) = serde_json::from_value::<Vec<Turn>>(v.clone())
        {
            return turns;
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

fn build_turn_retain_input(
    turns: &[Turn],
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
        prior_turns.push(format_turn(turn));
    }

    let turn = &turns[turn_number - 1];
    Ok(ExtractionInput {
        content: format_turn(turn),
        bank_id: elephant::types::BankId::new(),
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

fn build_session_content(
    conv: &Conversation,
    turns: &[Turn],
    idx: usize,
    raw_json: bool,
) -> String {
    if raw_json {
        format_session_raw(conv, idx)
    } else {
        format_session(turns, &get_session_date(conv, idx))
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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = parse_args().map_err(std::io::Error::other)?;
    let raw = fs::read_to_string(&cli.dataset)?;
    let entries: Vec<LocomoEntry> = serde_json::from_str(&raw)?;
    let entry = entries
        .iter()
        .find(|entry| entry.sample_id == cli.sample_id)
        .ok_or_else(|| std::io::Error::other(format!("sample {} not found", cli.sample_id)))?;

    let turns = get_session_turns(&entry.conversation, cli.session);
    if turns.is_empty() {
        return Err(std::io::Error::other(format!(
            "session {} not found for sample {}",
            cli.session, cli.sample_id
        ))
        .into());
    }

    let date_str = get_session_date(&entry.conversation, cli.session);
    let chunk_config = ChunkConfig {
        max_tokens: 512,
        overlap_tokens: 64,
        preserve_turns: true,
    };

    let base_input = match cli.mode {
        InspectMode::Session => ExtractionInput {
            content: build_session_content(&entry.conversation, &turns, cli.session, cli.raw_json),
            bank_id: elephant::types::BankId::new(),
            context: None,
            timestamp: parse_session_date(&date_str),
            turn_id: None,
            custom_instructions: None,
            speaker: None,
        },
        InspectMode::Turn => build_turn_retain_input(
            &turns,
            &date_str,
            cli.turn.expect("turn required for turn mode"),
        )
        .map_err(std::io::Error::other)?,
    };

    let inputs = extractor_inputs(&base_input, &chunk_config);

    println!("sample_id: {}", cli.sample_id);
    println!("session: {}", cli.session);
    println!(
        "mode: {}",
        match cli.mode {
            InspectMode::Session => "session",
            InspectMode::Turn => "turn",
        }
    );
    if let Some(turn) = cli.turn {
        println!("turn: {turn}");
    }
    println!("session_date: {}", date_str);
    println!("timestamp: {}", base_input.timestamp);
    println!(
        "chunk_config: max_tokens={} overlap_tokens={} preserve_turns={}",
        chunk_config.max_tokens, chunk_config.overlap_tokens, chunk_config.preserve_turns
    );
    println!("retain_input_chars: {}", base_input.content.len());
    println!("extractor_chunks: {}", inputs.len());

    for (idx, input) in inputs.iter().enumerate() {
        if cli.chunk.is_some_and(|selected| selected != idx) {
            continue;
        }

        println!();
        println!("=== Extractor Chunk {} ===", idx);
        println!("content_chars: {}", input.content.len());
        println!("speaker: {}", input.speaker.as_deref().unwrap_or("<none>"));
        println!(
            "context_chars: {}",
            input.context.as_ref().map(|ctx| ctx.len()).unwrap_or(0)
        );
        println!("--- content ---");
        println!("{}", input.content);
        println!("--- end content ---");
        if let Some(context) = &input.context {
            println!("--- context ---");
            println!("{context}");
            println!("--- end context ---");
        }
        let (system_prompt, user_message) = LlmFactExtractor::render_prompt(input);
        println!("--- extractor system prompt ---");
        println!("{system_prompt}");
        println!("--- end extractor system prompt ---");
        println!("--- extractor user message ---");
        println!("{user_message}");
        println!("--- end extractor user message ---");
    }

    Ok(())
}
