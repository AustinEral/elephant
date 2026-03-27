//! Provider/model capability policy for LLM request features.

use super::config::{ClientConfig, Provider};
use super::types::ReasoningEffort;
use serde::{Deserialize, Serialize};

/// Benchmark-facing determinism profile for a concrete stage configuration.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum DeterminismSupport {
    /// Determinism was not assessed for this artifact/config.
    #[default]
    Unknown,
    /// The provider/model offers a strong determinism contract for this setup.
    Strong,
    /// The setup is only suitable for best-effort low-variance benchmarking.
    BestEffort,
    /// The setup is not suitable for deterministic/low-variance benchmarking.
    Unsupported,
}

/// Per-stage determinism assessment for benchmark provenance and preflight checks.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct DeterminismAssessment {
    /// Coarse support tier.
    pub support: DeterminismSupport,
    /// Human-readable explanation of the classification.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
}

impl DeterminismAssessment {
    fn best_effort(reason: impl Into<String>) -> Self {
        Self {
            support: DeterminismSupport::BestEffort,
            reason: Some(reason.into()),
        }
    }

    fn unsupported(reason: impl Into<String>) -> Self {
        Self {
            support: DeterminismSupport::Unsupported,
            reason: Some(reason.into()),
        }
    }

    /// Return whether this assessment satisfies the requested requirement.
    pub fn satisfies(&self, requirement: DeterminismRequirement) -> bool {
        match requirement {
            DeterminismRequirement::BestEffort => matches!(
                self.support,
                DeterminismSupport::Strong | DeterminismSupport::BestEffort
            ),
            DeterminismRequirement::Strong => self.support == DeterminismSupport::Strong,
        }
    }
}

/// Minimum determinism guarantee required by a benchmark run.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum DeterminismRequirement {
    /// Allow best-effort low-variance setups.
    BestEffort,
    /// Require a strong determinism contract.
    Strong,
}

impl DeterminismRequirement {
    /// Return the stable config label.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::BestEffort => "best_effort",
            Self::Strong => "strong",
        }
    }
}

/// Provider/model-specific resolution of a requested sampling temperature.
#[derive(Debug, Clone, PartialEq)]
pub struct TemperatureResolution {
    requested: Option<f32>,
    effective: Option<f32>,
    unsupported_reason: Option<String>,
}

impl TemperatureResolution {
    fn supported(requested: Option<f32>, effective: Option<f32>) -> Self {
        Self {
            requested,
            effective,
            unsupported_reason: None,
        }
    }

    fn unsupported(
        requested: Option<f32>,
        effective: Option<f32>,
        reason: impl Into<String>,
    ) -> Self {
        Self {
            requested,
            effective,
            unsupported_reason: Some(reason.into()),
        }
    }

    /// Return the originally requested temperature.
    pub fn requested(&self) -> Option<f32> {
        self.requested
    }

    /// Return the temperature that will actually be forwarded to the provider.
    pub fn effective(&self) -> Option<f32> {
        self.effective
    }

    /// Return whether the requested temperature is supported as-is.
    pub fn is_supported(&self) -> bool {
        self.unsupported_reason.is_none()
    }

    /// Return the reason the requested temperature is unsupported, if any.
    pub fn unsupported_reason(&self) -> Option<&str> {
        self.unsupported_reason.as_deref()
    }
}

fn resolve_supported_temperature(
    provider_label: &str,
    requested: f32,
    min: f32,
    max: f32,
) -> TemperatureResolution {
    if !(min..=max).contains(&requested) {
        return TemperatureResolution::unsupported(
            Some(requested),
            None,
            format!("{provider_label} supports temperature in the range {min}..={max}"),
        );
    }

    TemperatureResolution::supported(Some(requested), Some(requested))
}

/// Resolve the effective temperature for a provider/model request.
pub fn resolve_temperature_for_target(
    provider: Provider,
    model: &str,
    requested: Option<f32>,
    reasoning_effort: Option<ReasoningEffort>,
) -> TemperatureResolution {
    let Some(requested) = requested else {
        return TemperatureResolution::supported(None, None);
    };

    match provider {
        Provider::OpenAi => {
            if model.starts_with("gpt-5.1") || model.starts_with("gpt-5.2") {
                if matches!(reasoning_effort, Some(ReasoningEffort::None)) {
                    return resolve_supported_temperature(
                        &format!("openai/{model}"),
                        requested,
                        0.0,
                        2.0,
                    );
                }

                return TemperatureResolution::unsupported(
                    Some(requested),
                    None,
                    format!(
                        "openai/{model} only supports custom temperature when reasoning_effort=none"
                    ),
                );
            }

            if model.starts_with("gpt-5") {
                return TemperatureResolution::unsupported(
                    Some(requested),
                    None,
                    format!("openai/{model} does not support custom temperature"),
                );
            }

            resolve_supported_temperature(&format!("openai/{model}"), requested, 0.0, 2.0)
        }
        Provider::Anthropic => {
            resolve_supported_temperature(&format!("anthropic/{model}"), requested, 0.0, 1.0)
        }
        Provider::Gemini => {
            resolve_supported_temperature(&format!("gemini/{model}"), requested, 0.0, 2.0)
        }
        Provider::Vertex => {
            resolve_supported_temperature(&format!("vertex/{model}"), requested, 0.0, 2.0)
        }
    }
}

/// Resolve the effective temperature for a validated client configuration.
pub fn resolve_temperature(
    config: &ClientConfig,
    requested: Option<f32>,
    reasoning_effort: Option<ReasoningEffort>,
) -> TemperatureResolution {
    resolve_temperature_for_target(
        config.provider(),
        config.model(),
        requested,
        reasoning_effort,
    )
}

/// Assess whether a stage configuration is suitable for low-variance benchmarking.
pub fn assess_determinism_for_target(
    provider: Provider,
    model: &str,
    requested_temperature: Option<f32>,
    reasoning_effort: Option<ReasoningEffort>,
) -> DeterminismAssessment {
    let resolution =
        resolve_temperature_for_target(provider, model, requested_temperature, reasoning_effort);

    if resolution.effective() != Some(0.0) {
        return DeterminismAssessment::unsupported(format!(
            "{provider_label}/{model} does not have effective temperature 0.0",
            provider_label = provider.as_str()
        ));
    }

    match provider {
        Provider::OpenAi => {
            if model.starts_with("gpt-5.1") || model.starts_with("gpt-5.2") {
                return DeterminismAssessment::best_effort(format!(
                    "openai/{model} accepts temperature=0 only with reasoning_effort=none, but OpenAI does not guarantee deterministic outputs"
                ));
            }
            DeterminismAssessment::best_effort(format!(
                "openai/{model} accepts temperature=0, but OpenAI does not guarantee deterministic outputs"
            ))
        }
        Provider::Anthropic => DeterminismAssessment::best_effort(format!(
            "anthropic/{model} supports temperature=0, but Anthropic docs note outputs are not fully deterministic"
        )),
        Provider::Gemini => DeterminismAssessment::best_effort(format!(
            "gemini/{model} supports temperature=0, but Google docs describe outputs as mostly deterministic and warn of degraded behavior"
        )),
        Provider::Vertex => DeterminismAssessment::best_effort(format!(
            "vertex/{model} supports temperature=0, but Google docs describe outputs as mostly deterministic and warn of degraded behavior"
        )),
    }
}

/// Assess determinism for a validated client configuration.
pub fn assess_determinism(
    config: &ClientConfig,
    requested_temperature: Option<f32>,
    reasoning_effort: Option<ReasoningEffort>,
) -> DeterminismAssessment {
    assess_determinism_for_target(
        config.provider(),
        config.model(),
        requested_temperature,
        reasoning_effort,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn temperature_resolution_omits_gpt54_temperature() {
        let resolution = resolve_temperature_for_target(
            Provider::OpenAi,
            "gpt-5.4-mini",
            Some(0.0),
            Some(ReasoningEffort::None),
        );
        assert_eq!(resolution.requested(), Some(0.0));
        assert_eq!(resolution.effective(), None);
        assert!(!resolution.is_supported());
        assert!(
            resolution
                .unsupported_reason()
                .unwrap()
                .contains("does not support custom temperature")
        );
    }

    #[test]
    fn temperature_resolution_allows_gpt51_with_reasoning_none() {
        let resolution = resolve_temperature_for_target(
            Provider::OpenAi,
            "gpt-5.1",
            Some(0.0),
            Some(ReasoningEffort::None),
        );
        assert_eq!(resolution.effective(), Some(0.0));
        assert!(resolution.is_supported());
        assert_eq!(resolution.unsupported_reason(), None);
    }

    #[test]
    fn temperature_resolution_rejects_gpt51_without_reasoning_none() {
        let resolution = resolve_temperature_for_target(
            Provider::OpenAi,
            "gpt-5.1",
            Some(0.0),
            Some(ReasoningEffort::Low),
        );
        assert_eq!(resolution.effective(), None);
        assert!(!resolution.is_supported());
        assert!(
            resolution
                .unsupported_reason()
                .unwrap()
                .contains("reasoning_effort=none")
        );
    }

    #[test]
    fn temperature_resolution_rejects_anthropic_out_of_range_temperature() {
        let resolution = resolve_temperature_for_target(
            Provider::Anthropic,
            "claude-sonnet-4-5",
            Some(1.1),
            None,
        );
        assert_eq!(resolution.effective(), None);
        assert!(!resolution.is_supported());
        assert!(resolution.unsupported_reason().unwrap().contains("0..=1"));
    }

    #[test]
    fn temperature_resolution_allows_gemini_zero_temperature() {
        let resolution =
            resolve_temperature_for_target(Provider::Gemini, "gemini-2.5-flash", Some(0.0), None);
        assert_eq!(resolution.effective(), Some(0.0));
        assert!(resolution.is_supported());
    }

    #[test]
    fn temperature_resolution_allows_vertex_upper_bound() {
        let resolution =
            resolve_temperature_for_target(Provider::Vertex, "gemini-2.0-flash", Some(2.0), None);
        assert_eq!(resolution.effective(), Some(2.0));
        assert!(resolution.is_supported());
    }

    #[test]
    fn determinism_assessment_marks_gpt54_unsupported() {
        let assessment = assess_determinism_for_target(
            Provider::OpenAi,
            "gpt-5.4-mini",
            Some(0.0),
            Some(ReasoningEffort::None),
        );
        assert_eq!(assessment.support, DeterminismSupport::Unsupported);
    }

    #[test]
    fn determinism_assessment_marks_anthropic_best_effort() {
        let assessment = assess_determinism_for_target(
            Provider::Anthropic,
            "claude-sonnet-4-5",
            Some(0.0),
            None,
        );
        assert_eq!(assessment.support, DeterminismSupport::BestEffort);
    }

    #[test]
    fn determinism_assessment_requires_zero_temperature() {
        let assessment =
            assess_determinism_for_target(Provider::Gemini, "gemini-2.5-flash", Some(0.2), None);
        assert_eq!(assessment.support, DeterminismSupport::Unsupported);
        assert!(
            assessment
                .reason
                .as_deref()
                .unwrap()
                .contains("effective temperature 0.0")
        );
    }
}
