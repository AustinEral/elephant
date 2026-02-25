//! Entity types for named entity resolution.

use serde::{Deserialize, Serialize};

use super::id::{BankId, EntityId};

/// A named entity referenced by facts.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Entity {
    /// Unique identifier.
    pub id: EntityId,
    /// The canonical name for this entity.
    pub canonical_name: String,
    /// Alternative names that map to this entity.
    pub aliases: Vec<String>,
    /// Classification of the entity.
    pub entity_type: EntityType,
    /// Which memory bank this entity belongs to.
    pub bank_id: BankId,
}

/// Classification of a named entity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EntityType {
    /// A person or user.
    Person,
    /// An organization or company.
    Organization,
    /// A physical location.
    Location,
    /// A product or service.
    Product,
    /// An abstract concept or technology.
    Concept,
    /// A specific event.
    Event,
    /// Anything that doesn't fit the other categories.
    Other,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serde_roundtrip() {
        let entity = Entity {
            id: EntityId::new(),
            canonical_name: "Rust".into(),
            aliases: vec!["rust-lang".into(), "Rust language".into()],
            entity_type: EntityType::Concept,
            bank_id: BankId::new(),
        };
        let json = serde_json::to_string(&entity).unwrap();
        let back: Entity = serde_json::from_str(&json).unwrap();
        assert_eq!(entity, back);
    }

    #[test]
    fn entity_type_variants() {
        for variant in [
            EntityType::Person,
            EntityType::Organization,
            EntityType::Location,
            EntityType::Product,
            EntityType::Concept,
            EntityType::Event,
            EntityType::Other,
        ] {
            let json = serde_json::to_string(&variant).unwrap();
            let back: EntityType = serde_json::from_str(&json).unwrap();
            assert_eq!(variant, back);
        }
    }
}
