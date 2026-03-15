use std::collections::BTreeMap;
use std::str::FromStr;
use std::time::Instant;

use chrono::{DateTime, NaiveDate, NaiveDateTime, Utc};
use serde::{Deserialize, Serialize};
use tracing::{debug, info, warn};

use elephant::error::Result;
use elephant::metrics::{LlmStage, StageUsage};
use elephant::runtime::ElephantRuntime;
use elephant::types::id::BankId;
use elephant::types::{Disposition, MemoryBank, RetainInput};

use super::dataset::{LongMemEvalInstance, Turn};

/// Session formatting mode for ingestion.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum IngestFormat {
    Text,
    Json,
}

impl Default for IngestFormat {
    fn default() -> Self {
        Self::Text
    }
}

impl IngestFormat {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Text => "text",
            Self::Json => "json",
        }
    }
}

impl FromStr for IngestFormat {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s {
            "text" => Ok(Self::Text),
            "json" => Ok(Self::Json),
            other => Err(format!(
                "invalid --ingest-format value: {other} (expected one of: text, json)"
            )),
        }
    }
}

/// Consolidation strategy after ingestion.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ConsolidationMode {
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
    pub fn as_str(self) -> &'static str {
        match self {
            Self::End => "end",
            Self::PerSession => "per-session",
            Self::Off => "off",
        }
    }
}

impl FromStr for ConsolidationMode {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
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

impl ConsolidationMode {
    /// Whether consolidation is enabled (true for End and PerSession).
    pub fn enabled(self) -> bool {
        !matches!(self, Self::Off)
    }

    /// Whether consolidation runs after each session.
    pub fn per_session(self) -> bool {
        matches!(self, Self::PerSession)
    }
}

/// Configuration for LongMemEval ingestion.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IngestConfig {
    /// Session formatting: text (default) or json.
    pub format: IngestFormat,
    /// Consolidation strategy: end (default), per-session, or off.
    pub consolidation: ConsolidationMode,
}

impl Default for IngestConfig {
    fn default() -> Self {
        Self {
            format: IngestFormat::default(),
            consolidation: ConsolidationMode::default(),
        }
    }
}

/// Result of ingesting one LongMemEval instance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IngestResult {
    /// Which question this result is for.
    pub question_id: String,
    /// The bank created for this instance.
    pub bank_id: BankId,
    /// Per-stage LLM usage metrics.
    pub stage_metrics: BTreeMap<LlmStage, StageUsage>,
    /// Ingestion statistics.
    pub stats: IngestStats,
    /// Wall-clock timing.
    pub timing: IngestTiming,
}

/// Ingestion statistics for one instance.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct IngestStats {
    pub sessions_ingested: usize,
    pub facts_stored: usize,
    pub entities_resolved: usize,
    pub links_created: usize,
    pub opinions_reinforced: usize,
    pub opinions_weakened: usize,
    pub session_failures: usize,
    pub observations_created: usize,
    pub observations_updated: usize,
}

/// Wall-clock timing for one instance ingestion.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct IngestTiming {
    pub ingest_time_s: f64,
    pub consolidation_time_s: f64,
    pub total_time_s: f64,
}

/// Parse a LongMemEval date string to a `[Date: YYYY-MM-DD]` prefix.
///
/// Input: `"2023/05/20 (Sat) 02:21"`
/// Output: `"[Date: 2023-05-20]"`
pub fn parse_date_prefix(date_str: &str) -> String {
    let date_part = date_str.split(" (").next().unwrap_or(date_str);
    let iso_date = date_part.replace('/', "-");
    format!("[Date: {iso_date}]")
}

/// Parse a LongMemEval date string to `DateTime<Utc>`.
///
/// Handles: `"2023/05/20 (Sat) 02:21"` and date-only fallback `"2023/05/20"`.
pub fn parse_haystack_date(date_str: &str) -> DateTime<Utc> {
    let trimmed = date_str.trim();

    // Try full format: "2023/05/20 (Sat) 02:21"
    // Strip the day-of-week portion: split on '(' to get "2023/05/20 ",
    // split on ')' to get " 02:21", combine as "2023/05/20 02:21"
    let before_paren = trimmed.split('(').next().unwrap_or(trimmed).trim();
    let after_paren = trimmed.split(')').nth(1).unwrap_or("00:00").trim();
    let cleaned = format!("{before_paren} {after_paren}");

    if let Ok(ndt) = NaiveDateTime::parse_from_str(&cleaned, "%Y/%m/%d %H:%M") {
        return ndt.and_utc();
    }

    // Fallback: try date-only
    let date_only = trimmed.split(" (").next().unwrap_or(trimmed);
    if let Ok(nd) = NaiveDate::parse_from_str(date_only, "%Y/%m/%d") {
        return nd.and_hms_opt(0, 0, 0).unwrap().and_utc();
    }

    // Last resort
    Utc::now()
}

/// Format a session's turns as date-prefixed plain text.
///
/// Output: `[Date: 2023-05-20]\n\nuser: Hello\nassistant: Hi there`
pub fn format_session_text(turns: &[Turn], date_str: &str) -> String {
    let date_prefix = parse_date_prefix(date_str);
    let dialogue = turns
        .iter()
        .map(|t| format!("{}: {}", t.role, t.content))
        .collect::<Vec<_>>()
        .join("\n");
    format!("{date_prefix}\n\n{dialogue}")
}

/// Format a session's turns as a JSON array of `{role, content}` objects.
///
/// No `has_answer` leakage -- Turn only has role and content.
pub fn format_session_json(turns: &[Turn]) -> String {
    let cleaned: Vec<serde_json::Value> = turns
        .iter()
        .map(|t| serde_json::json!({"role": t.role, "content": t.content}))
        .collect();
    serde_json::to_string(&cleaned).unwrap_or_default()
}

/// Ingest a single LongMemEval instance into its own isolated bank.
///
/// Creates a new bank, ingests all haystack sessions sequentially, then
/// optionally runs consolidation based on the configured mode.
///
/// Note: INGEST-05 (pool sizing for concurrent ops) is deferred to Phase 5.
/// This function assumes sequential invocation.
pub async fn ingest_instance(
    instance: &LongMemEvalInstance,
    runtime: &ElephantRuntime,
    config: &IngestConfig,
) -> Result<IngestResult> {
    let total_start = Instant::now();
    let mut stats = IngestStats::default();

    // 1. Create bank
    let bank = MemoryBank {
        id: BankId::new(),
        name: format!("longmemeval-{}", instance.question_id),
        mission: "Long-term conversational memory benchmark".into(),
        directives: vec![],
        disposition: Disposition::default(),
        embedding_model: runtime.embeddings.model_name().to_string(),
        embedding_dimensions: runtime.embeddings.dimensions() as u16,
    };
    runtime.store.create_bank(&bank).await?;

    let total_sessions = instance.haystack_sessions.len();
    info!(
        question_id = %instance.question_id,
        sessions = total_sessions,
        format = ?config.format,
        consolidation = ?config.consolidation,
        "starting ingestion"
    );

    // 2. Ingest sessions sequentially
    let ingest_start = Instant::now();

    for (idx, (session, date_str)) in instance
        .haystack_sessions
        .iter()
        .zip(instance.haystack_dates.iter())
        .enumerate()
    {
        debug!(
            question_id = %instance.question_id,
            session = idx + 1,
            total = total_sessions,
            "ingesting session"
        );

        let content = match config.format {
            IngestFormat::Text => format_session_text(session, date_str),
            IngestFormat::Json => {
                format!("{}\n\n{}", parse_date_prefix(date_str), format_session_json(session))
            }
        };
        let timestamp = parse_haystack_date(date_str);

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
                stats.sessions_ingested += 1;
                stats.facts_stored += resp.facts_stored;
                stats.entities_resolved += resp.entities_resolved;
                stats.links_created += resp.links_created;
                stats.opinions_reinforced += resp.opinions_reinforced;
                stats.opinions_weakened += resp.opinions_weakened;
            }
            Err(e) => {
                warn!(
                    question_id = %instance.question_id,
                    session = idx + 1,
                    total = total_sessions,
                    error = %e,
                    "session ingestion failed"
                );
                stats.session_failures += 1;
            }
        }

        // Per-session consolidation
        if config.consolidation.per_session() {
            match runtime.consolidator.consolidate(bank.id).await {
                Ok(cr) => {
                    stats.observations_created += cr.observations_created;
                    stats.observations_updated += cr.observations_updated;
                }
                Err(e) => {
                    warn!(
                        question_id = %instance.question_id,
                        session = idx + 1,
                        error = %e,
                        "per-session consolidation failed"
                    );
                }
            }
        }
    }

    let ingest_time_s = ingest_start.elapsed().as_secs_f64();

    // 3. End-of-ingestion consolidation
    let mut consolidation_time_s = 0.0;
    if config.consolidation.enabled() && !config.consolidation.per_session() {
        let t0 = Instant::now();
        match runtime.consolidator.consolidate(bank.id).await {
            Ok(cr) => {
                stats.observations_created += cr.observations_created;
                stats.observations_updated += cr.observations_updated;
            }
            Err(e) => {
                warn!(
                    question_id = %instance.question_id,
                    error = %e,
                    "end-of-ingestion consolidation failed"
                );
            }
        }
        consolidation_time_s = t0.elapsed().as_secs_f64();
    }

    let total_time_s = total_start.elapsed().as_secs_f64();

    info!(
        question_id = %instance.question_id,
        bank_id = %bank.id,
        sessions = stats.sessions_ingested,
        facts = stats.facts_stored,
        failures = stats.session_failures,
        duration_s = format!("{total_time_s:.1}"),
        "ingestion complete"
    );

    Ok(IngestResult {
        question_id: instance.question_id.clone(),
        bank_id: bank.id,
        stage_metrics: BTreeMap::new(),
        stats,
        timing: IngestTiming {
            ingest_time_s,
            consolidation_time_s,
            total_time_s,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_turns() -> Vec<Turn> {
        vec![
            Turn {
                role: "user".into(),
                content: "Hello there".into(),
            },
            Turn {
                role: "assistant".into(),
                content: "Hi! How can I help?".into(),
            },
        ]
    }

    // --- parse_date_prefix ---

    #[test]
    fn parse_date_prefix_full_format() {
        assert_eq!(
            parse_date_prefix("2023/05/20 (Sat) 02:21"),
            "[Date: 2023-05-20]"
        );
    }

    #[test]
    fn parse_date_prefix_new_year() {
        assert_eq!(
            parse_date_prefix("2024/01/01 (Mon) 00:00"),
            "[Date: 2024-01-01]"
        );
    }

    // --- parse_haystack_date ---

    #[test]
    fn parse_haystack_date_full_format() {
        let dt = parse_haystack_date("2023/05/20 (Sat) 02:21");
        assert_eq!(dt.format("%Y-%m-%dT%H:%M").to_string(), "2023-05-20T02:21");
    }

    #[test]
    fn parse_haystack_date_new_year() {
        let dt = parse_haystack_date("2024/01/01 (Mon) 00:00");
        assert_eq!(dt.format("%Y-%m-%dT%H:%M").to_string(), "2024-01-01T00:00");
    }

    #[test]
    fn parse_haystack_date_date_only_fallback() {
        let dt = parse_haystack_date("2023/05/20");
        assert_eq!(dt.format("%Y-%m-%d").to_string(), "2023-05-20");
    }

    // --- format_session_text ---

    #[test]
    fn format_session_text_produces_date_prefix() {
        let turns = make_turns();
        let result = format_session_text(&turns, "2023/05/20 (Sat) 02:21");
        assert!(result.starts_with("[Date: 2023-05-20]\n\n"));
        assert!(result.contains("user: Hello there"));
        assert!(result.contains("assistant: Hi! How can I help?"));
    }

    #[test]
    fn format_session_text_multiple_turns_joined() {
        let turns = make_turns();
        let result = format_session_text(&turns, "2023/05/20 (Sat) 02:21");
        let lines: Vec<&str> = result.lines().collect();
        // [Date: ...], empty line, user line, assistant line
        assert_eq!(lines.len(), 4);
        assert_eq!(lines[0], "[Date: 2023-05-20]");
        assert_eq!(lines[1], "");
        assert_eq!(lines[2], "user: Hello there");
        assert_eq!(lines[3], "assistant: Hi! How can I help?");
    }

    // --- format_session_json ---

    #[test]
    fn format_session_json_produces_array() {
        let turns = make_turns();
        let result = format_session_json(&turns);
        let parsed: Vec<serde_json::Value> = serde_json::from_str(&result).unwrap();
        assert_eq!(parsed.len(), 2);
        assert_eq!(parsed[0]["role"], "user");
        assert_eq!(parsed[0]["content"], "Hello there");
        assert_eq!(parsed[1]["role"], "assistant");
        assert_eq!(parsed[1]["content"], "Hi! How can I help?");
    }

    #[test]
    fn format_session_json_no_has_answer() {
        let turns = make_turns();
        let result = format_session_json(&turns);
        assert!(!result.contains("has_answer"));
    }

    // --- ConsolidationMode ---

    #[test]
    fn consolidation_mode_end_enabled() {
        assert!(ConsolidationMode::End.enabled());
    }

    #[test]
    fn consolidation_mode_end_not_per_session() {
        assert!(!ConsolidationMode::End.per_session());
    }

    #[test]
    fn consolidation_mode_per_session_enabled() {
        assert!(ConsolidationMode::PerSession.enabled());
    }

    #[test]
    fn consolidation_mode_per_session_is_per_session() {
        assert!(ConsolidationMode::PerSession.per_session());
    }

    #[test]
    fn consolidation_mode_off_not_enabled() {
        assert!(!ConsolidationMode::Off.enabled());
    }

    // --- IngestConfig default ---

    #[test]
    fn ingest_config_default() {
        let config = IngestConfig::default();
        assert_eq!(config.format, IngestFormat::Text);
        assert_eq!(config.consolidation, ConsolidationMode::End);
    }

    // --- IngestResult / IngestStats / IngestTiming serde roundtrip ---

    #[test]
    fn ingest_result_roundtrip() {
        let result = IngestResult {
            question_id: "q1".into(),
            bank_id: BankId::new(),
            stage_metrics: BTreeMap::new(),
            stats: IngestStats {
                sessions_ingested: 5,
                facts_stored: 20,
                entities_resolved: 3,
                links_created: 7,
                opinions_reinforced: 1,
                opinions_weakened: 0,
                session_failures: 0,
                observations_created: 4,
                observations_updated: 2,
            },
            timing: IngestTiming {
                ingest_time_s: 10.5,
                consolidation_time_s: 3.2,
                total_time_s: 13.7,
            },
        };
        let json = serde_json::to_string(&result).unwrap();
        let back: IngestResult = serde_json::from_str(&json).unwrap();
        assert_eq!(back.question_id, "q1");
        assert_eq!(back.stats.sessions_ingested, 5);
        assert_eq!(back.stats.facts_stored, 20);
        assert!((back.timing.total_time_s - 13.7).abs() < f64::EPSILON);
    }

    #[test]
    fn ingest_stats_default() {
        let stats = IngestStats::default();
        assert_eq!(stats.sessions_ingested, 0);
        assert_eq!(stats.facts_stored, 0);
        assert_eq!(stats.session_failures, 0);
    }

    #[test]
    fn ingest_timing_default() {
        let timing = IngestTiming::default();
        assert!((timing.total_time_s - 0.0).abs() < f64::EPSILON);
    }
}
