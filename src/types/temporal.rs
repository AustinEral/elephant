//! Temporal range for time-bounded facts.

use chrono::{DateTime, NaiveDate, Utc};
use serde::{Deserialize, Deserializer, Serialize};

/// A time range with optional start and end bounds.
///
/// Both bounds are inclusive. If `start` is `None`, the range extends
/// infinitely into the past. If `end` is `None`, it extends into the future.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TemporalRange {
    /// Start of the range (inclusive). `None` means unbounded past.
    #[serde(default, deserialize_with = "deserialize_flexible_datetime")]
    pub start: Option<DateTime<Utc>>,
    /// End of the range (inclusive). `None` means unbounded future.
    #[serde(default, deserialize_with = "deserialize_flexible_datetime")]
    pub end: Option<DateTime<Utc>>,
}

/// Deserialize a DateTime<Utc> accepting both full ISO8601 ("2015-01-01T00:00:00Z")
/// and date-only ("2015-01-01") formats. LLMs commonly produce date-only strings.
fn deserialize_flexible_datetime<'de, D>(
    deserializer: D,
) -> std::result::Result<Option<DateTime<Utc>>, D::Error>
where
    D: Deserializer<'de>,
{
    let opt: Option<String> = Option::deserialize(deserializer)?;
    match opt {
        None => Ok(None),
        Some(s) => {
            // Try full datetime first
            if let Ok(dt) = s.parse::<DateTime<Utc>>() {
                return Ok(Some(dt));
            }
            // Try date-only (e.g. "2015-01-01")
            if let Ok(date) = NaiveDate::parse_from_str(&s, "%Y-%m-%d") {
                return Ok(Some(date.and_hms_opt(0, 0, 0).unwrap().and_utc()));
            }
            Err(serde::de::Error::custom(format!(
                "cannot parse '{s}' as datetime or date"
            )))
        }
    }
}

impl TemporalRange {
    /// Check if this range contains a specific point in time.
    pub fn contains(&self, instant: DateTime<Utc>) -> bool {
        let after_start = self.start.is_none_or(|s| instant >= s);
        let before_end = self.end.is_none_or(|e| instant <= e);
        after_start && before_end
    }

    /// Check if this range overlaps with another range.
    pub fn overlaps(&self, other: &TemporalRange) -> bool {
        let self_before_other = match (self.end, other.start) {
            (Some(se), Some(os)) => se < os,
            _ => false,
        };
        let other_before_self = match (other.end, self.start) {
            (Some(oe), Some(ss)) => oe < ss,
            _ => false,
        };
        !self_before_other && !other_before_self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    fn dt(year: i32, month: u32, day: u32) -> DateTime<Utc> {
        Utc.with_ymd_and_hms(year, month, day, 0, 0, 0).unwrap()
    }

    #[test]
    fn contains_bounded() {
        let range = TemporalRange {
            start: Some(dt(2024, 1, 1)),
            end: Some(dt(2024, 12, 31)),
        };
        assert!(range.contains(dt(2024, 6, 15)));
        assert!(range.contains(dt(2024, 1, 1))); // inclusive start
        assert!(range.contains(dt(2024, 12, 31))); // inclusive end
        assert!(!range.contains(dt(2023, 12, 31)));
        assert!(!range.contains(dt(2025, 1, 1)));
    }

    #[test]
    fn contains_open_start() {
        let range = TemporalRange {
            start: None,
            end: Some(dt(2024, 6, 1)),
        };
        assert!(range.contains(dt(2000, 1, 1)));
        assert!(!range.contains(dt(2025, 1, 1)));
    }

    #[test]
    fn contains_open_end() {
        let range = TemporalRange {
            start: Some(dt(2024, 1, 1)),
            end: None,
        };
        assert!(!range.contains(dt(2023, 1, 1)));
        assert!(range.contains(dt(2099, 1, 1)));
    }

    #[test]
    fn contains_fully_open() {
        let range = TemporalRange {
            start: None,
            end: None,
        };
        assert!(range.contains(dt(2024, 6, 15)));
    }

    #[test]
    fn overlaps_basic() {
        let a = TemporalRange {
            start: Some(dt(2024, 1, 1)),
            end: Some(dt(2024, 6, 30)),
        };
        let b = TemporalRange {
            start: Some(dt(2024, 3, 1)),
            end: Some(dt(2024, 12, 31)),
        };
        assert!(a.overlaps(&b));
        assert!(b.overlaps(&a));
    }

    #[test]
    fn overlaps_no_overlap() {
        let a = TemporalRange {
            start: Some(dt(2024, 1, 1)),
            end: Some(dt(2024, 3, 31)),
        };
        let b = TemporalRange {
            start: Some(dt(2024, 6, 1)),
            end: Some(dt(2024, 12, 31)),
        };
        assert!(!a.overlaps(&b));
        assert!(!b.overlaps(&a));
    }

    #[test]
    fn overlaps_open_ended() {
        let a = TemporalRange {
            start: Some(dt(2024, 1, 1)),
            end: None,
        };
        let b = TemporalRange {
            start: Some(dt(2025, 1, 1)),
            end: Some(dt(2025, 6, 30)),
        };
        assert!(a.overlaps(&b));
    }

    #[test]
    fn overlaps_both_open() {
        let a = TemporalRange {
            start: None,
            end: None,
        };
        let b = TemporalRange {
            start: Some(dt(2024, 1, 1)),
            end: Some(dt(2024, 12, 31)),
        };
        assert!(a.overlaps(&b));
    }

    #[test]
    fn serde_roundtrip() {
        let range = TemporalRange {
            start: Some(dt(2024, 1, 1)),
            end: None,
        };
        let json = serde_json::to_string(&range).unwrap();
        let back: TemporalRange = serde_json::from_str(&json).unwrap();
        assert_eq!(range, back);
    }
}
