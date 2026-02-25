//! Temporal retriever — parses time references from queries and filters by time range.

use std::sync::Arc;

use async_trait::async_trait;
use chrono::{DateTime, Datelike, Duration, NaiveDate, TimeZone, Utc};
use regex::Regex;

use crate::error::Result;
use crate::storage::MemoryStore;
use crate::types::{FactFilter, RecallQuery, RetrievalSource, ScoredFact, TemporalRange};

use super::Retriever;

/// Retrieves facts that fall within a parsed temporal range.
pub struct TemporalRetriever {
    store: Arc<dyn MemoryStore>,
}

impl TemporalRetriever {
    /// Create a new temporal retriever.
    pub fn new(store: Arc<dyn MemoryStore>) -> Self {
        Self { store }
    }
}

#[async_trait]
impl Retriever for TemporalRetriever {
    async fn retrieve(&self, query: &RecallQuery) -> Result<Vec<ScoredFact>> {
        // Use explicit temporal anchor if set, otherwise parse from query text
        let range = query
            .temporal_anchor
            .clone()
            .or_else(|| parse_temporal_reference(&query.query, Utc::now()));

        let Some(range) = range else {
            return Ok(Vec::new());
        };

        let filter = FactFilter {
            network: query.network_filter.clone(),
            temporal_range: Some(range.clone()),
            ..Default::default()
        };

        let facts = self
            .store
            .get_facts_by_bank(query.bank_id, filter)
            .await?;

        let now = Utc::now();
        let mut scored: Vec<ScoredFact> = facts
            .into_iter()
            .map(|f| {
                let score = temporal_proximity_score(&f.temporal_range, &range, now);
                ScoredFact {
                    fact: f,
                    score,
                    sources: vec![RetrievalSource::Temporal],
                }
            })
            .collect();

        scored.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        Ok(scored)
    }
}

/// Score facts by temporal proximity to the query range: `1.0 / (1.0 + days_distance * 0.01)`.
///
/// Measures distance between the fact's temporal midpoint and the query range's midpoint.
fn temporal_proximity_score(
    fact_range: &Option<TemporalRange>,
    query_range: &TemporalRange,
    now: DateTime<Utc>,
) -> f32 {
    let fact_mid = fact_range
        .as_ref()
        .and_then(|r| match (r.start, r.end) {
            (Some(s), Some(e)) => Some(s + (e - s) / 2),
            (Some(s), None) => Some(s),
            (None, Some(e)) => Some(e),
            (None, None) => None,
        })
        .unwrap_or(now);

    let query_mid = match (query_range.start, query_range.end) {
        (Some(s), Some(e)) => s + (e - s) / 2,
        (Some(s), None) => s,
        (None, Some(e)) => e,
        (None, None) => now,
    };

    let days = (query_mid - fact_mid).num_days().unsigned_abs() as f32;
    1.0 / (1.0 + days * 0.01)
}

/// Parse temporal references from natural language query text.
///
/// Supports: ISO dates, "last week/month/year", "yesterday/today",
/// named months (e.g. "in January"), and "recently" (7 days).
pub fn parse_temporal_reference(query: &str, now: DateTime<Utc>) -> Option<TemporalRange> {
    let lower = query.to_lowercase();

    // "recently" — last 7 days
    if lower.contains("recently") || lower.contains("recent") {
        return Some(TemporalRange {
            start: Some(now - Duration::days(7)),
            end: Some(now),
        });
    }

    // "today"
    if lower.contains("today") {
        let start = start_of_day(now);
        return Some(TemporalRange {
            start: Some(start),
            end: Some(now),
        });
    }

    // "yesterday"
    if lower.contains("yesterday") {
        let yesterday = now - Duration::days(1);
        let start = start_of_day(yesterday);
        let end = start + Duration::days(1) - Duration::seconds(1);
        return Some(TemporalRange {
            start: Some(start),
            end: Some(end),
        });
    }

    // "last week"
    if lower.contains("last week") {
        return Some(TemporalRange {
            start: Some(now - Duration::weeks(1)),
            end: Some(now),
        });
    }

    // "last month"
    if lower.contains("last month") {
        return Some(TemporalRange {
            start: Some(now - Duration::days(30)),
            end: Some(now),
        });
    }

    // "last year"
    if lower.contains("last year") {
        return Some(TemporalRange {
            start: Some(now - Duration::days(365)),
            end: Some(now),
        });
    }

    // Named months: "in January", "in February", etc.
    let month_re = Regex::new(
        r"(?i)\b(?:in\s+)?(january|february|march|april|may|june|july|august|september|october|november|december)\b",
    )
    .unwrap();
    if let Some(caps) = month_re.captures(&lower) {
        let month_name = &caps[1];
        let month_num = match month_name {
            "january" => 1,
            "february" => 2,
            "march" => 3,
            "april" => 4,
            "may" => 5,
            "june" => 6,
            "july" => 7,
            "august" => 8,
            "september" => 9,
            "october" => 10,
            "november" => 11,
            "december" => 12,
            _ => return None,
        };
        let year = if month_num <= now.month() {
            now.year()
        } else {
            now.year() - 1
        };
        let start = Utc.with_ymd_and_hms(year, month_num, 1, 0, 0, 0).single()?;
        let end_date = if month_num == 12 {
            NaiveDate::from_ymd_opt(year + 1, 1, 1)?
        } else {
            NaiveDate::from_ymd_opt(year, month_num + 1, 1)?
        };
        let end = Utc
            .with_ymd_and_hms(end_date.year(), end_date.month(), 1, 0, 0, 0)
            .single()?
            - Duration::seconds(1);
        return Some(TemporalRange {
            start: Some(start),
            end: Some(end),
        });
    }

    // ISO date: YYYY-MM-DD
    let iso_re = Regex::new(r"\b(\d{4}-\d{2}-\d{2})\b").unwrap();
    if let Some(caps) = iso_re.captures(query) {
        let date_str = &caps[1];
        if let Ok(date) = NaiveDate::parse_from_str(date_str, "%Y-%m-%d") {
            let start = date.and_hms_opt(0, 0, 0)?;
            let end = date.and_hms_opt(23, 59, 59)?;
            return Some(TemporalRange {
                start: Some(DateTime::from_naive_utc_and_offset(start, Utc)),
                end: Some(DateTime::from_naive_utc_and_offset(end, Utc)),
            });
        }
    }

    // ISO date range: YYYY-MM-DD to YYYY-MM-DD
    let range_re = Regex::new(r"\b(\d{4}-\d{2}-\d{2})\s+to\s+(\d{4}-\d{2}-\d{2})\b").unwrap();
    if let Some(caps) = range_re.captures(query) {
        let start_str = &caps[1];
        let end_str = &caps[2];
        if let (Ok(start_date), Ok(end_date)) = (
            NaiveDate::parse_from_str(start_str, "%Y-%m-%d"),
            NaiveDate::parse_from_str(end_str, "%Y-%m-%d"),
        ) {
            let start = start_date.and_hms_opt(0, 0, 0)?;
            let end = end_date.and_hms_opt(23, 59, 59)?;
            return Some(TemporalRange {
                start: Some(DateTime::from_naive_utc_and_offset(start, Utc)),
                end: Some(DateTime::from_naive_utc_and_offset(end, Utc)),
            });
        }
    }

    None
}

fn start_of_day(dt: DateTime<Utc>) -> DateTime<Utc> {
    dt.date_naive()
        .and_hms_opt(0, 0, 0)
        .map(|ndt| DateTime::from_naive_utc_and_offset(ndt, Utc))
        .unwrap_or(dt)
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    fn fixed_now() -> DateTime<Utc> {
        Utc.with_ymd_and_hms(2026, 2, 25, 12, 0, 0).unwrap()
    }

    #[test]
    fn parse_recently() {
        let result = parse_temporal_reference("what happened recently?", fixed_now()).unwrap();
        assert!(result.start.is_some());
        let days = (fixed_now() - result.start.unwrap()).num_days();
        assert_eq!(days, 7);
    }

    #[test]
    fn parse_today() {
        let result = parse_temporal_reference("what happened today?", fixed_now()).unwrap();
        assert!(result.start.is_some());
        assert_eq!(result.start.unwrap().date_naive(), fixed_now().date_naive());
    }

    #[test]
    fn parse_yesterday() {
        let result = parse_temporal_reference("what about yesterday?", fixed_now()).unwrap();
        let expected = (fixed_now() - Duration::days(1)).date_naive();
        assert_eq!(result.start.unwrap().date_naive(), expected);
    }

    #[test]
    fn parse_last_week() {
        let result = parse_temporal_reference("last week events", fixed_now()).unwrap();
        let days = (fixed_now() - result.start.unwrap()).num_days();
        assert_eq!(days, 7);
    }

    #[test]
    fn parse_last_month() {
        let result = parse_temporal_reference("last month summary", fixed_now()).unwrap();
        let days = (fixed_now() - result.start.unwrap()).num_days();
        assert_eq!(days, 30);
    }

    #[test]
    fn parse_last_year() {
        let result = parse_temporal_reference("last year review", fixed_now()).unwrap();
        let days = (fixed_now() - result.start.unwrap()).num_days();
        assert_eq!(days, 365);
    }

    #[test]
    fn parse_named_month() {
        let result = parse_temporal_reference("what happened in January?", fixed_now()).unwrap();
        // January 2026 (current year since Jan < Feb)
        assert_eq!(result.start.unwrap().month(), 1);
        assert_eq!(result.start.unwrap().year(), 2026);
    }

    #[test]
    fn parse_future_month_goes_to_previous_year() {
        let result = parse_temporal_reference("in December", fixed_now()).unwrap();
        // December 2025 (previous year since Dec > Feb)
        assert_eq!(result.start.unwrap().month(), 12);
        assert_eq!(result.start.unwrap().year(), 2025);
    }

    #[test]
    fn parse_iso_date() {
        let result = parse_temporal_reference("events on 2026-01-15", fixed_now()).unwrap();
        assert_eq!(result.start.unwrap().day(), 15);
        assert_eq!(result.start.unwrap().month(), 1);
    }

    #[test]
    fn parse_no_temporal_ref() {
        let result = parse_temporal_reference("tell me about Rust", fixed_now());
        assert!(result.is_none());
    }

    #[test]
    fn temporal_proximity_closer_to_query_scores_higher() {
        let now = fixed_now();
        // Query asks about "last week"
        let query_range = TemporalRange {
            start: Some(now - Duration::days(7)),
            end: Some(now),
        };
        // Fact from 3 days ago (close to query midpoint)
        let close = Some(TemporalRange {
            start: Some(now - Duration::days(3)),
            end: Some(now - Duration::days(2)),
        });
        // Fact from 100 days ago (far from query midpoint)
        let far = Some(TemporalRange {
            start: Some(now - Duration::days(100)),
            end: Some(now - Duration::days(90)),
        });

        let score_close = temporal_proximity_score(&close, &query_range, now);
        let score_far = temporal_proximity_score(&far, &query_range, now);
        assert!(score_close > score_far);
    }

    #[test]
    fn temporal_proximity_exact_match_scores_highest() {
        let now = fixed_now();
        let query_range = TemporalRange {
            start: Some(now - Duration::days(30)),
            end: Some(now),
        };
        // Fact centered exactly at query midpoint
        let mid = now - Duration::days(15);
        let exact = Some(TemporalRange {
            start: Some(mid - Duration::days(1)),
            end: Some(mid + Duration::days(1)),
        });
        let score = temporal_proximity_score(&exact, &query_range, now);
        // Should be close to 1.0 (midpoints nearly identical)
        assert!(score > 0.9);
    }
}
