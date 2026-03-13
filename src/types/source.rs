//! Provenance source types for tracing facts back to their origin.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use super::id::{BankId, FactId, SourceId};

/// A source unit that produced one or more facts during retain.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Source {
    /// Unique identifier.
    pub id: SourceId,
    /// Which memory bank this source belongs to.
    pub bank_id: BankId,
    /// Primary content passed to the extractor for this source.
    pub content: String,
    /// Optional preceding context shown to the extractor.
    pub context: Option<String>,
    /// Optional speaker label shown to the extractor.
    pub speaker: Option<String>,
    /// Exact extractor user message shown to the LLM, if stored.
    pub rendered_input: Option<String>,
    /// When this source content was produced.
    pub timestamp: DateTime<Utc>,
    /// When this source record was created.
    pub created_at: DateTime<Utc>,
}

/// A deduplicated link from a fact to a source that supports it.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct FactSource {
    /// Fact supported by the source.
    pub fact_id: FactId,
    /// Supporting source id.
    pub source_id: SourceId,
}

/// Sources linked to a fact, returned by provenance lookup.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct FactSourceLookup {
    /// Fact whose provenance was requested.
    pub fact_id: FactId,
    /// Matching sources for that fact, ordered chronologically.
    pub sources: Vec<Source>,
}
