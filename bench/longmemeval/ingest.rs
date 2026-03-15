use std::collections::BTreeMap;

use chrono::{DateTime, NaiveDate, NaiveDateTime, Utc};
use serde::{Deserialize, Serialize};

use elephant::metrics::{LlmStage, StageUsage};
use elephant::types::id::BankId;

use super::dataset::Turn;

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
