//! Memory bank configuration and disposition.

use serde::{Deserialize, Serialize};

use super::id::BankId;
use crate::error::{Error, Result};

/// A memory bank — an isolated memory store with its own configuration.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MemoryBank {
    /// Unique identifier.
    pub id: BankId,
    /// Human-readable name.
    pub name: String,
    /// What this bank should focus on retaining.
    pub mission: String,
    /// Guardrails and compliance rules.
    pub directives: Vec<String>,
    /// Personality parameters that influence recall and reflection.
    pub disposition: Disposition,
}

/// Raw disposition data used for deserialization before validation.
#[derive(Deserialize)]
struct DispositionRaw {
    skepticism: u8,
    literalism: u8,
    empathy: u8,
    bias_strength: f32,
}

/// Personality parameters for a memory bank.
///
/// Controls how the bank interprets, weights, and responds to information.
/// Constructed via [`Disposition::new()`] which validates all fields.
#[derive(Debug, Clone, Serialize, PartialEq)]
#[serde(into = "DispositionSer")]
pub struct Disposition {
    /// How much evidence is required before accepting claims (1-5).
    skepticism: u8,
    /// How literally to interpret statements (1-5).
    literalism: u8,
    /// Weight given to emotional/human factors (1-5).
    empathy: u8,
    /// How strongly the disposition influences responses (0.0-1.0).
    bias_strength: f32,
}

/// Serialization helper to avoid custom Serialize impl.
#[derive(Serialize)]
struct DispositionSer {
    skepticism: u8,
    literalism: u8,
    empathy: u8,
    bias_strength: f32,
}

impl From<Disposition> for DispositionSer {
    fn from(d: Disposition) -> Self {
        DispositionSer {
            skepticism: d.skepticism,
            literalism: d.literalism,
            empathy: d.empathy,
            bias_strength: d.bias_strength,
        }
    }
}

impl Disposition {
    /// Create a new disposition with validated parameters.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidDisposition`] if any value is out of range:
    /// - `skepticism`, `literalism`, `empathy`: must be 1-5
    /// - `bias_strength`: must be 0.0-1.0
    pub fn new(skepticism: u8, literalism: u8, empathy: u8, bias_strength: f32) -> Result<Self> {
        if !(1..=5).contains(&skepticism) {
            return Err(Error::InvalidDisposition(format!(
                "skepticism must be 1-5, got {skepticism}"
            )));
        }
        if !(1..=5).contains(&literalism) {
            return Err(Error::InvalidDisposition(format!(
                "literalism must be 1-5, got {literalism}"
            )));
        }
        if !(1..=5).contains(&empathy) {
            return Err(Error::InvalidDisposition(format!(
                "empathy must be 1-5, got {empathy}"
            )));
        }
        if !(0.0..=1.0).contains(&bias_strength) {
            return Err(Error::InvalidDisposition(format!(
                "bias_strength must be 0.0-1.0, got {bias_strength}"
            )));
        }
        Ok(Self {
            skepticism,
            literalism,
            empathy,
            bias_strength,
        })
    }

    /// Skepticism level (1-5).
    pub fn skepticism(&self) -> u8 {
        self.skepticism
    }

    /// Literalism level (1-5).
    pub fn literalism(&self) -> u8 {
        self.literalism
    }

    /// Empathy level (1-5).
    pub fn empathy(&self) -> u8 {
        self.empathy
    }

    /// Bias strength (0.0-1.0).
    pub fn bias_strength(&self) -> f32 {
        self.bias_strength
    }
}

impl Default for Disposition {
    fn default() -> Self {
        // Safety: these values are known-valid.
        Self {
            skepticism: 3,
            literalism: 3,
            empathy: 3,
            bias_strength: 0.5,
        }
    }
}

impl TryFrom<DispositionRaw> for Disposition {
    type Error = Error;

    fn try_from(raw: DispositionRaw) -> Result<Self> {
        Disposition::new(raw.skepticism, raw.literalism, raw.empathy, raw.bias_strength)
    }
}

impl<'de> Deserialize<'de> for Disposition {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let raw = DispositionRaw::deserialize(deserializer)?;
        Disposition::try_from(raw).map_err(serde::de::Error::custom)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn valid_disposition() {
        let d = Disposition::new(1, 5, 3, 0.5).unwrap();
        assert_eq!(d.skepticism(), 1);
        assert_eq!(d.literalism(), 5);
        assert_eq!(d.empathy(), 3);
        assert!((d.bias_strength() - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn invalid_skepticism() {
        assert!(Disposition::new(0, 3, 3, 0.5).is_err());
        assert!(Disposition::new(6, 3, 3, 0.5).is_err());
    }

    #[test]
    fn invalid_literalism() {
        assert!(Disposition::new(3, 0, 3, 0.5).is_err());
        assert!(Disposition::new(3, 6, 3, 0.5).is_err());
    }

    #[test]
    fn invalid_empathy() {
        assert!(Disposition::new(3, 3, 0, 0.5).is_err());
        assert!(Disposition::new(3, 3, 6, 0.5).is_err());
    }

    #[test]
    fn invalid_bias_strength() {
        assert!(Disposition::new(3, 3, 3, -0.1).is_err());
        assert!(Disposition::new(3, 3, 3, 1.1).is_err());
    }

    #[test]
    fn default_disposition() {
        let d = Disposition::default();
        assert_eq!(d.skepticism(), 3);
        assert_eq!(d.literalism(), 3);
        assert_eq!(d.empathy(), 3);
        assert!((d.bias_strength() - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn serde_roundtrip() {
        let d = Disposition::new(2, 4, 1, 0.8).unwrap();
        let json = serde_json::to_string(&d).unwrap();
        let back: Disposition = serde_json::from_str(&json).unwrap();
        assert_eq!(d, back);
    }

    #[test]
    fn deserialize_invalid_json() {
        let json = r#"{"skepticism": 0, "literalism": 3, "empathy": 3, "bias_strength": 0.5}"#;
        let result: std::result::Result<Disposition, _> = serde_json::from_str(json);
        assert!(result.is_err());
    }

    #[test]
    fn memory_bank_roundtrip() {
        let bank = MemoryBank {
            id: BankId::new(),
            name: "test bank".into(),
            mission: "remember everything".into(),
            directives: vec!["be helpful".into()],
            disposition: Disposition::default(),
        };
        let json = serde_json::to_string(&bank).unwrap();
        let back: MemoryBank = serde_json::from_str(&json).unwrap();
        assert_eq!(bank, back);
    }
}
