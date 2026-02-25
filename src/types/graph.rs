//! Graph link types connecting facts in the knowledge graph.

use serde::{Deserialize, Serialize};

use super::id::FactId;

/// A directed edge in the fact knowledge graph.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GraphLink {
    /// Source fact ID.
    pub source_id: FactId,
    /// Target fact ID.
    pub target_id: FactId,
    /// Type of relationship.
    pub link_type: LinkType,
    /// Edge weight (0.0 to 1.0).
    pub weight: f32,
}

/// Classification of a graph edge.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LinkType {
    /// Facts are temporally related.
    Temporal,
    /// Facts are semantically similar.
    Semantic,
    /// Facts share an entity reference.
    Entity,
    /// One fact causally leads to another.
    Causal,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serde_roundtrip() {
        let link = GraphLink {
            source_id: FactId::new(),
            target_id: FactId::new(),
            link_type: LinkType::Causal,
            weight: 0.8,
        };
        let json = serde_json::to_string(&link).unwrap();
        let back: GraphLink = serde_json::from_str(&json).unwrap();
        assert_eq!(link, back);
    }

    #[test]
    fn link_type_variants() {
        for variant in [
            LinkType::Temporal,
            LinkType::Semantic,
            LinkType::Entity,
            LinkType::Causal,
        ] {
            let json = serde_json::to_string(&variant).unwrap();
            let back: LinkType = serde_json::from_str(&json).unwrap();
            assert_eq!(variant, back);
        }
    }
}
