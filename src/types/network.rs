//! Memory network classification.

use serde::{Deserialize, Serialize};

/// Which memory network a fact belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NetworkType {
    /// Factual knowledge about the world.
    World,
    /// User experiences and interactions.
    Experience,
    /// Consolidated observations derived from experiences.
    Observation,
    /// Opinions formed through reflection.
    Opinion,
    /// High-level mental models synthesized from other networks.
    MentalModel,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serde_roundtrip() {
        for variant in [
            NetworkType::World,
            NetworkType::Experience,
            NetworkType::Observation,
            NetworkType::Opinion,
            NetworkType::MentalModel,
        ] {
            let json = serde_json::to_string(&variant).unwrap();
            let back: NetworkType = serde_json::from_str(&json).unwrap();
            assert_eq!(variant, back);
        }
    }

    #[test]
    fn json_format() {
        assert_eq!(
            serde_json::to_string(&NetworkType::MentalModel).unwrap(),
            "\"mental_model\""
        );
    }
}
