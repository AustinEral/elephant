use std::env;
use std::fs;
use std::path::Path;
use std::sync::Arc;

use chrono::{DateTime, Utc};
use datatest_stable::harness;
use dotenvy::dotenv;
use elephant::llm::retry::{RetryPolicy, RetryingLlmClient};
use elephant::llm::{self, LlmClient, Provider, ProviderConfig};
use elephant::retain::chunker::{Chunker, SimpleChunker};
use elephant::retain::extractor::{FactExtractor, LlmFactExtractor};
use elephant::types::{BankId, ChunkConfig, ExtractionInput};
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct ExtractCase {
    id: String,
    status: Status,
    input: Input,
    assertions: Vec<Assertion>,
}

#[derive(Debug, Deserialize, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum Status {
    Guard,
    Tracking,
    Limitation,
}

#[derive(Debug, Deserialize, Clone, Copy)]
#[serde(rename_all = "snake_case")]
enum ExtractMode {
    Session,
    Turn,
}

#[derive(Debug, Deserialize)]
struct Input {
    mode: ExtractMode,
    transcript: Vec<Message>,
}

#[derive(Debug, Deserialize)]
struct Message {
    speaker: String,
    timestamp: String,
    text: String,
}

#[derive(Debug, Deserialize)]
struct Assertion {
    kind: String,
    value: String,
}

fn build_llm() -> Result<Arc<dyn LlmClient>, String> {
    dotenv().ok();

    let provider_str =
        env::var("LLM_PROVIDER").map_err(|e| format!("LLM_PROVIDER must be set: {e}"))?;
    let api_key = env::var("LLM_API_KEY").map_err(|e| format!("LLM_API_KEY must be set: {e}"))?;
    let model = env::var("RETAIN_LLM_MODEL")
        .or_else(|_| env::var("LLM_MODEL"))
        .map_err(|e| format!("RETAIN_LLM_MODEL or LLM_MODEL must be set: {e}"))?;
    let base_url = env::var("LLM_BASE_URL").ok();

    let provider = match provider_str.as_str() {
        "openai" => Provider::OpenAi,
        _ => Provider::Anthropic,
    };

    let config = ProviderConfig {
        provider,
        api_key,
        model,
        base_url,
    };
    let base: Arc<dyn LlmClient> = Arc::from(
        llm::build_client(&config).map_err(|e| format!("failed to build LLM client: {e}"))?,
    );
    Ok(Arc::new(RetryingLlmClient::new(
        base,
        RetryPolicy::default(),
    )))
}

fn transcript_to_session_text(messages: &[Message]) -> String {
    let mut out = String::new();
    if let Some(first) = messages.first() {
        out.push_str("Date: ");
        out.push_str(&first.timestamp);
        out.push_str("\n\n");
    }
    for (idx, message) in messages.iter().enumerate() {
        if idx > 0 {
            out.push('\n');
        }
        out.push_str(&message.speaker);
        out.push_str(": ");
        out.push_str(&message.text);
    }
    out
}

async fn extract_facts(
    case: &ExtractCase,
    extractor: &LlmFactExtractor,
    chunker: &SimpleChunker,
    chunk_config: &ChunkConfig,
) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    let bank_id = BankId::new();
    let timestamp = case
        .input
        .transcript
        .first()
        .map(|msg| DateTime::parse_from_rfc3339(&msg.timestamp).map(|dt| dt.with_timezone(&Utc)))
        .transpose()?
        .unwrap_or_else(Utc::now);

    let mut out = Vec::new();
    match case.input.mode {
        ExtractMode::Session => {
            let content = transcript_to_session_text(&case.input.transcript);
            let chunks = chunker.chunk(&content, chunk_config);
            for chunk in chunks {
                let input = ExtractionInput {
                    content: chunk.content,
                    bank_id,
                    context: chunk.context,
                    timestamp,
                    turn_id: None,
                    custom_instructions: None,
                    speaker: None,
                };
                let facts = extractor.extract(&input).await?;
                out.extend(facts.into_iter().map(|fact| fact.content));
            }
        }
        ExtractMode::Turn => {
            for message in &case.input.transcript {
                let input = ExtractionInput {
                    content: format!("{}: {}", message.speaker, message.text),
                    bank_id,
                    context: None,
                    timestamp: DateTime::parse_from_rfc3339(&message.timestamp)?
                        .with_timezone(&Utc),
                    turn_id: None,
                    custom_instructions: None,
                    speaker: Some(message.speaker.clone()),
                };
                let facts = extractor.extract(&input).await?;
                out.extend(facts.into_iter().map(|fact| fact.content));
            }
        }
    }
    Ok(out)
}

fn contains_match(facts: &[String], needle: &str) -> bool {
    let needle = needle.to_lowercase();
    facts.iter()
        .any(|fact| fact.to_lowercase().contains(&needle))
}

fn extract_case_live(path: &Path) -> datatest_stable::Result<()> {
    dotenv().ok();
    let raw = fs::read_to_string(path)?;
    let case: ExtractCase = serde_json::from_str(&raw)?;

    let rt = tokio::runtime::Runtime::new()?;
    let llm = build_llm().map_err(std::io::Error::other)?;
    let extractor = LlmFactExtractor::new(llm);
    let chunker = SimpleChunker;
    let chunk_config = ChunkConfig {
        max_tokens: 512,
        overlap_tokens: 64,
        preserve_turns: true,
    };

    let facts = rt.block_on(extract_facts(&case, &extractor, &chunker, &chunk_config))?;

    let mut failed = false;
    for assertion in &case.assertions {
        let passed = match assertion.kind.as_str() {
            "fact_contains" => contains_match(&facts, &assertion.value),
            "fact_not_contains" => !contains_match(&facts, &assertion.value),
            other => {
                return Err(std::io::Error::other(format!(
                    "{}: unsupported assertion kind {other}",
                    case.id
                ))
                .into())
            }
        };

        match case.status {
            Status::Guard => {
                if !passed {
                    failed = true;
                }
            }
            Status::Tracking | Status::Limitation => {
                if passed {
                    return Err(std::io::Error::other(format!(
                        "{}: {:?} case now passes and should be reviewed: {}",
                        case.id, case.status, assertion.value
                    ))
                    .into());
                }
            }
        }
    }

    if failed {
        return Err(std::io::Error::other(format!(
            "{}: guard case failed; extracted facts: {:?}",
            case.id, facts
        ))
        .into());
    }

    Ok(())
}

harness! {
    { test = extract_case_live, root = "tests/evals/extract", pattern = r".*\.json$" },
}
