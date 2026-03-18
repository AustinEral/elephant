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
}

/// Extraction-facing network labels.
///
/// LLMs sometimes emit nearby semantic categories from the extraction prompt
/// rather than the canonical storage enum. We accept those variants here and
/// normalize them into [`NetworkType`] before storing facts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExtractedNetworkType {
    /// Factual knowledge about the world.
    World,
    /// User experiences and compatible semantic categories like plans/intentions.
    #[serde(
        alias = "experiences",
        alias = "plan",
        alias = "plans",
        alias = "intention",
        alias = "intentions",
        alias = "plans and intentions"
    )]
    Experience,
    /// Consolidated observations derived from experiences.
    #[serde(alias = "observations")]
    Observation,
    /// Opinions formed through reflection.
    #[serde(alias = "opinions")]
    Opinion,
}

impl ExtractedNetworkType {
    /// Map extraction-time labels to the canonical storage enum.
    pub fn normalize(self) -> NetworkType {
        match self {
            Self::World => NetworkType::World,
            Self::Experience => NetworkType::Experience,
            Self::Observation => NetworkType::Observation,
            Self::Opinion => NetworkType::Opinion,
        }
    }
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
        ] {
            let json = serde_json::to_string(&variant).unwrap();
            let back: NetworkType = serde_json::from_str(&json).unwrap();
            assert_eq!(variant, back);
        }
    }

    #[test]
    fn json_format() {
        assert_eq!(
            serde_json::to_string(&NetworkType::Opinion).unwrap(),
            "\"opinion\""
        );
    }

    #[test]
    fn normalizes_plan_like_labels_to_experience() {
        for label in [
            "plan",
            "plans",
            "intention",
            "intentions",
            "plans and intentions",
        ] {
            let parsed: ExtractedNetworkType =
                serde_json::from_str(&format!("\"{label}\"")).unwrap();
            assert_eq!(parsed.normalize(), NetworkType::Experience);
        }
    }
}
