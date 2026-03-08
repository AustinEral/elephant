//! Pipeline input/output types for retain, recall, and reflect operations.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use super::entity::EntityType;
use super::fact::{Fact, FactType, ScoredFact};
use super::id::{BankId, EntityId, FactId, TurnId};
use super::network::NetworkType;
use super::temporal::TemporalRange;

// --- Retain pipeline ---

/// Input to the retain pipeline.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RetainInput {
    /// The memory bank to store into.
    pub bank_id: BankId,
    /// Raw text content to be processed and stored.
    pub content: String,
    /// When this content was produced.
    pub timestamp: DateTime<Utc>,
    /// Optional conversation turn ID for provenance.
    pub turn_id: Option<TurnId>,
    /// Summary of preceding context for cross-chunk coreference resolution.
    pub context: Option<String>,
    /// Domain-specific extraction guidelines injected into the extraction prompt.
    pub custom_instructions: Option<String>,
    /// Name of the speaker/author for resolving first-person references ("I", "me", "my").
    #[serde(default)]
    pub speaker: Option<String>,
}

/// Output from the retain pipeline.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RetainOutput {
    /// IDs of facts that were created.
    pub fact_ids: Vec<FactId>,
    /// Number of facts stored.
    pub facts_stored: usize,
    /// IDs of newly created entities.
    pub new_entities: Vec<EntityId>,
    /// Total entities resolved (new + existing matches).
    pub entities_resolved: usize,
    /// Number of graph links created.
    pub links_created: usize,
    /// Number of existing opinions reinforced by new evidence.
    pub opinions_reinforced: usize,
    /// Number of existing opinions weakened by contradicting evidence.
    pub opinions_weakened: usize,
}

// --- Recall pipeline ---

/// Query for the recall (retrieval) pipeline.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RecallQuery {
    /// The memory bank to search.
    pub bank_id: BankId,
    /// Natural language query.
    pub query: String,
    /// Maximum token budget for returned facts.
    pub budget_tokens: usize,
    /// Optional network type filter.
    pub network_filter: Option<Vec<NetworkType>>,
    /// Optional temporal anchor for temporal retrieval.
    pub temporal_anchor: Option<TemporalRange>,
}

// --- Reflect pipeline ---

/// Query for the reflect (reasoning) pipeline.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ReflectQuery {
    /// The memory bank to reason over.
    pub bank_id: BankId,
    /// The question or topic to reflect on.
    pub question: String,
    /// Maximum token budget for retrieved context.
    pub budget_tokens: usize,
}

// --- Extraction ---

/// Input for fact extraction from a single chunk.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ExtractionInput {
    /// The chunk text to extract facts from.
    pub content: String,
    /// Which bank this extraction is for.
    pub bank_id: BankId,
    /// Summary of preceding chunks for coreference resolution.
    pub context: Option<String>,
    /// When this content was produced (for resolving relative temporal references).
    pub timestamp: DateTime<Utc>,
    /// Optional conversation turn ID for provenance.
    pub turn_id: Option<TurnId>,
    /// Domain-specific extraction guidelines.
    pub custom_instructions: Option<String>,
    /// Name of the speaker/author for resolving first-person references.
    pub speaker: Option<String>,
}

/// A fact extracted by the LLM before storage.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ExtractedFact {
    /// The natural language content.
    pub content: String,
    /// Extracted fact type.
    pub fact_type: FactType,
    /// Which network this should be stored in.
    pub network: NetworkType,
    /// Entity names mentioned (not yet resolved to IDs).
    pub entity_mentions: Vec<String>,
    /// Optional temporal range.
    pub temporal_range: Option<TemporalRange>,
    /// Confidence score if applicable.
    pub confidence: Option<f32>,
}

// --- Chunking ---

/// Configuration for text chunking.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ChunkConfig {
    /// Maximum number of tokens per chunk.
    pub max_tokens: usize,
    /// Number of overlapping tokens between chunks.
    pub overlap_tokens: usize,
    /// Don't split in the middle of a conversation turn.
    pub preserve_turns: bool,
}

/// A chunk of text produced by the chunker.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Chunk {
    /// The chunk text.
    pub content: String,
    /// Position of this chunk in the original document (0-indexed).
    pub index: usize,
    /// Summary of preceding chunks for coreference resolution.
    pub context: Option<String>,
}

// --- Filtering ---

/// Filter criteria for fact queries.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct FactFilter {
    /// Filter by network type.
    pub network: Option<Vec<NetworkType>>,
    /// Filter by fact type.
    pub fact_type: Option<FactType>,
    /// Filter by temporal range.
    pub temporal_range: Option<TemporalRange>,
    /// Filter by entity IDs (facts must reference at least one).
    pub entity_ids: Option<Vec<EntityId>>,
    /// Only return facts created at or after this timestamp.
    #[serde(default)]
    pub created_since: Option<DateTime<Utc>>,
    /// Only return facts that have not been consolidated yet.
    #[serde(default)]
    pub unconsolidated_only: bool,
}

// --- Entity resolution ---

/// An entity resolved from a mention string.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ResolvedEntity {
    /// The original mention text.
    pub mention: String,
    /// The resolved entity ID.
    pub entity_id: EntityId,
    /// The canonical name of the resolved entity.
    pub canonical_name: String,
    /// The entity type.
    pub entity_type: EntityType,
    /// Whether this entity was newly created.
    pub is_new: bool,
    /// How confident the resolution is (1.0 = exact match, lower = fuzzy/LLM).
    pub confidence: f32,
}

// --- Recall/Reflect results ---

/// Result of a recall operation.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RecallResult {
    /// Retrieved facts with scores.
    pub facts: Vec<ScoredFact>,
    /// Total tokens used by the returned facts.
    pub total_tokens: usize,
}

/// A single retrieved fact with its relevance score.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RetrievedFact {
    /// Fact ID.
    pub id: FactId,
    /// Fact text content.
    pub content: String,
    /// Relevance score from the recall pipeline.
    pub score: f32,
    /// Which memory network this fact belongs to.
    pub network: NetworkType,
}

/// Result of a reflect operation.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ReflectResult {
    /// The generated reflection response.
    pub response: String,
    /// Fact IDs that grounded the response.
    pub sources: Vec<FactId>,
    /// New opinions formed during reflection.
    pub new_opinions: Vec<Fact>,
    /// Confidence in the reflection.
    pub confidence: f32,
    /// All facts retrieved during the reflect agent loop, in ranked order.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub retrieved_context: Vec<RetrievedFact>,
}

// --- Reflect pipeline types ---

/// Assembled memory context for the reflect pipeline (Phase 5A output).
///
/// Contains memory organized by network type with budget-allocated content.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AssembledContext {
    /// Entity-level summaries.
    pub observations: Vec<Fact>,
    /// Individual world/experience facts.
    pub raw_facts: Vec<Fact>,
    /// Relevant beliefs with confidence scores.
    pub opinions: Vec<Fact>,
    /// Total tokens across all sections.
    pub total_tokens: usize,
    /// Pre-formatted string ready to inject into the LLM prompt.
    pub formatted: String,
}

/// Verbalized bank profile for the reflect pipeline (Phase 5B output).
///
/// Contains prompt components derived from the bank's disposition, directives, and mission.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct BankPromptContext {
    /// Soft personality/reasoning style instructions derived from disposition.
    pub disposition_prompt: String,
    /// Hard compliance rules derived from directives (never violate).
    pub directives_prompt: String,
    /// Bank mission statement.
    pub mission_prompt: String,
}

// --- Consolidation reports ---

/// Report from observation consolidation.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct ConsolidationReport {
    /// Number of new observations created.
    pub observations_created: usize,
    /// Number of existing observations updated with new evidence.
    pub observations_updated: usize,
}

/// Report from opinion merging.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct OpinionMergeReport {
    /// Number of opinion clusters that were merged into a single opinion.
    pub opinions_merged: usize,
    /// Number of opinions superseded (older opinion weakened).
    pub opinions_superseded: usize,
    /// Number of opinions found to be conflicting (both weakened).
    pub opinions_conflicting: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn retain_input_roundtrip() {
        let input = RetainInput {
            bank_id: BankId::new(),
            content: "hello world".into(),
            timestamp: chrono::Utc::now(),
            turn_id: None,
            context: Some("previous context".into()),
            custom_instructions: None,
            speaker: None,
        };
        let json = serde_json::to_string(&input).unwrap();
        let back: RetainInput = serde_json::from_str(&json).unwrap();
        assert_eq!(input, back);
    }

    #[test]
    fn retain_output_roundtrip() {
        let output = RetainOutput {
            fact_ids: vec![FactId::new()],
            facts_stored: 1,
            new_entities: vec![EntityId::new()],
            entities_resolved: 3,
            links_created: 5,
            opinions_reinforced: 1,
            opinions_weakened: 0,
        };
        let json = serde_json::to_string(&output).unwrap();
        let back: RetainOutput = serde_json::from_str(&json).unwrap();
        assert_eq!(output, back);
    }

    #[test]
    fn recall_query_roundtrip() {
        let query = RecallQuery {
            bank_id: BankId::new(),
            query: "what happened?".into(),
            budget_tokens: 2048,
            network_filter: Some(vec![NetworkType::World, NetworkType::Experience]),
            temporal_anchor: None,
        };
        let json = serde_json::to_string(&query).unwrap();
        let back: RecallQuery = serde_json::from_str(&json).unwrap();
        assert_eq!(query, back);
    }

    #[test]
    fn reflect_query_roundtrip() {
        let query = ReflectQuery {
            bank_id: BankId::new(),
            question: "why did we choose Postgres?".into(),
            budget_tokens: 4096,
        };
        let json = serde_json::to_string(&query).unwrap();
        let back: ReflectQuery = serde_json::from_str(&json).unwrap();
        assert_eq!(query, back);
    }

    #[test]
    fn extracted_fact_roundtrip() {
        let ef = ExtractedFact {
            content: "Rust uses ownership for memory safety".into(),
            fact_type: FactType::World,
            network: NetworkType::World,
            entity_mentions: vec!["Rust".into()],
            temporal_range: None,
            confidence: None,
        };
        let json = serde_json::to_string(&ef).unwrap();
        let back: ExtractedFact = serde_json::from_str(&json).unwrap();
        assert_eq!(ef, back);
    }

    #[test]
    fn chunk_config_roundtrip() {
        let config = ChunkConfig {
            max_tokens: 512,
            overlap_tokens: 64,
            preserve_turns: true,
        };
        let json = serde_json::to_string(&config).unwrap();
        let back: ChunkConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(config, back);
    }

    #[test]
    fn chunk_roundtrip() {
        let chunk = Chunk {
            content: "some text".into(),
            index: 0,
            context: Some("preceding summary".into()),
        };
        let json = serde_json::to_string(&chunk).unwrap();
        let back: Chunk = serde_json::from_str(&json).unwrap();
        assert_eq!(chunk, back);
    }

    #[test]
    fn fact_filter_default_is_empty() {
        let f = FactFilter::default();
        assert!(f.network.is_none());
        assert!(f.fact_type.is_none());
        assert!(f.temporal_range.is_none());
        assert!(f.entity_ids.is_none());
    }

    #[test]
    fn resolved_entity_roundtrip() {
        let re = ResolvedEntity {
            mention: "pg".into(),
            entity_id: EntityId::new(),
            canonical_name: "PostgreSQL".into(),
            entity_type: EntityType::Concept,
            is_new: false,
            confidence: 0.95,
        };
        let json = serde_json::to_string(&re).unwrap();
        let back: ResolvedEntity = serde_json::from_str(&json).unwrap();
        assert_eq!(re, back);
    }

    #[test]
    fn assembled_context_roundtrip() {
        let ctx = AssembledContext {
            observations: vec![],
            raw_facts: vec![],
            opinions: vec![],
            total_tokens: 0,
            formatted: String::new(),
        };
        let json = serde_json::to_string(&ctx).unwrap();
        let back: AssembledContext = serde_json::from_str(&json).unwrap();
        assert_eq!(ctx, back);
    }

    #[test]
    fn bank_prompt_context_roundtrip() {
        let bpc = BankPromptContext {
            disposition_prompt: "You are moderately skeptical.".into(),
            directives_prompt: "Never recommend competitors.".into(),
            mission_prompt: "Remember developer context.".into(),
        };
        let json = serde_json::to_string(&bpc).unwrap();
        let back: BankPromptContext = serde_json::from_str(&json).unwrap();
        assert_eq!(bpc, back);
    }
}
