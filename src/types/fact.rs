//! The core Fact type — the atomic unit of memory.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use super::id::{BankId, EntityId, FactId, TurnId};
use super::network::NetworkType;
use super::temporal::TemporalRange;

/// The atomic unit of memory in the engine.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Fact {
    /// Unique identifier.
    pub id: FactId,
    /// Which memory bank this fact belongs to.
    pub bank_id: BankId,
    /// The natural language content of the fact.
    pub content: String,
    /// Source classification (World or Experience).
    pub fact_type: FactType,
    /// Which memory network this fact lives in.
    pub network: NetworkType,
    /// Resolved entity references mentioned in this fact.
    pub entity_ids: Vec<EntityId>,
    /// When this fact was true.
    pub temporal_range: Option<TemporalRange>,
    /// Embedding vector for semantic search.
    pub embedding: Option<Vec<f32>>,
    /// Confidence score, primarily for opinions.
    pub confidence: Option<f32>,
    /// Facts that support this one (for observations, opinions, mental models).
    pub evidence_ids: Vec<FactId>,
    /// The conversation turn that produced this fact.
    pub source_turn_id: Option<TurnId>,
    /// When this fact was created.
    pub created_at: DateTime<Utc>,
    /// When this fact was last updated.
    pub updated_at: DateTime<Utc>,
}

/// Source classification for a fact.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FactType {
    /// Factual knowledge about the world.
    World,
    /// User experiences and interactions.
    Experience,
}

/// A fact with its retrieval score and provenance.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ScoredFact {
    /// The retrieved fact.
    pub fact: Fact,
    /// Combined retrieval score.
    pub score: f32,
    /// Which retrieval strategies found this fact.
    pub sources: Vec<RetrievalSource>,
}

/// Which retrieval strategy found a fact.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RetrievalSource {
    /// Found via embedding similarity search.
    Semantic,
    /// Found via keyword/text search.
    Keyword,
    /// Found via graph traversal.
    Graph,
    /// Found via temporal filtering.
    Temporal,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_fact() -> Fact {
        Fact {
            id: FactId::new(),
            bank_id: BankId::new(),
            content: "Rust is a systems programming language".into(),
            fact_type: FactType::World,
            network: NetworkType::World,
            entity_ids: vec![EntityId::new()],
            temporal_range: None,
            embedding: Some(vec![0.1, 0.2, 0.3]),
            confidence: None,
            evidence_ids: vec![],
            source_turn_id: Some(TurnId::new()),
            created_at: Utc::now(),
            updated_at: Utc::now(),
        }
    }

    #[test]
    fn serde_roundtrip() {
        let fact = sample_fact();
        let json = serde_json::to_string(&fact).unwrap();
        let back: Fact = serde_json::from_str(&json).unwrap();
        assert_eq!(fact, back);
    }

    #[test]
    fn scored_fact_roundtrip() {
        let sf = ScoredFact {
            fact: sample_fact(),
            score: 0.95,
            sources: vec![RetrievalSource::Semantic, RetrievalSource::Graph],
        };
        let json = serde_json::to_string(&sf).unwrap();
        let back: ScoredFact = serde_json::from_str(&json).unwrap();
        assert_eq!(sf, back);
    }

    #[test]
    fn fact_type_roundtrip() {
        for variant in [FactType::World, FactType::Experience] {
            let json = serde_json::to_string(&variant).unwrap();
            let back: FactType = serde_json::from_str(&json).unwrap();
            assert_eq!(variant, back);
        }
    }

    #[test]
    fn retrieval_source_roundtrip() {
        for variant in [
            RetrievalSource::Semantic,
            RetrievalSource::Keyword,
            RetrievalSource::Graph,
            RetrievalSource::Temporal,
        ] {
            let json = serde_json::to_string(&variant).unwrap();
            let back: RetrievalSource = serde_json::from_str(&json).unwrap();
            assert_eq!(variant, back);
        }
    }

    mod proptest_facts {
        use super::*;
        use proptest::prelude::*;

        fn arb_network_type() -> impl Strategy<Value = NetworkType> {
            prop_oneof![
                Just(NetworkType::World),
                Just(NetworkType::Experience),
                Just(NetworkType::Observation),
                Just(NetworkType::Opinion),
                Just(NetworkType::MentalModel),
            ]
        }

        fn arb_fact_type() -> impl Strategy<Value = FactType> {
            prop_oneof![Just(FactType::World), Just(FactType::Experience),]
        }

        fn arb_fact() -> impl Strategy<Value = Fact> {
            (
                "[a-zA-Z0-9 ]+",
                arb_fact_type(),
                arb_network_type(),
                proptest::option::of(0.0f32..=1.0),
                proptest::collection::vec(-1.0f32..1.0, 0..4),
            )
                .prop_map(|(content, fact_type, network, confidence, embedding)| Fact {
                    id: FactId::new(),
                    bank_id: BankId::new(),
                    content,
                    fact_type,
                    network,
                    entity_ids: vec![],
                    temporal_range: None,
                    embedding: if embedding.is_empty() {
                        None
                    } else {
                        Some(embedding)
                    },
                    confidence,
                    evidence_ids: vec![],
                    source_turn_id: None,
                    created_at: Utc::now(),
                    updated_at: Utc::now(),
                })
        }

        proptest! {
            #[test]
            fn fact_serde_roundtrip(fact in arb_fact()) {
                let json = serde_json::to_string(&fact).unwrap();
                let back: Fact = serde_json::from_str(&json).unwrap();
                prop_assert_eq!(fact, back);
            }
        }
    }
}
