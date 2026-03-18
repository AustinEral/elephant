use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::common;

/// The 6 question type categories in LongMemEval.
///
/// Abstention is NOT a separate question type -- it is identified by the `_abs`
/// suffix on `question_id`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum QuestionType {
    SingleSessionUser,
    SingleSessionAssistant,
    SingleSessionPreference,
    MultiSession,
    TemporalReasoning,
    KnowledgeUpdate,
}

/// A single conversation turn (user or assistant message).
#[derive(Debug, Clone, Deserialize)]
pub struct Turn {
    pub role: String,
    pub content: String,
}

/// A single LongMemEval benchmark instance.
#[derive(Debug, Clone, Deserialize)]
pub struct LongMemEvalInstance {
    pub question_id: String,
    pub question_type: QuestionType,
    pub question: String,
    pub answer: serde_json::Value,
    pub question_date: String,
    pub haystack_dates: Vec<String>,
    pub haystack_session_ids: Vec<String>,
    pub haystack_sessions: Vec<Vec<Turn>>,
    pub answer_session_ids: Vec<String>,
}

impl LongMemEvalInstance {
    /// Returns true if this is an abstention (false-premise) question.
    ///
    /// Matches upstream Python: `'_abs' in entry['question_id']`.
    pub fn is_abstention(&self) -> bool {
        self.question_id.contains("_abs")
    }

    /// Returns the reporting category: one of the 6 question type strings, or
    /// `"abstention"` for false-premise questions.
    pub fn reporting_category(&self) -> &str {
        if self.is_abstention() {
            "abstention"
        } else {
            match self.question_type {
                QuestionType::SingleSessionUser => "single-session-user",
                QuestionType::SingleSessionAssistant => "single-session-assistant",
                QuestionType::SingleSessionPreference => "single-session-preference",
                QuestionType::MultiSession => "multi-session",
                QuestionType::TemporalReasoning => "temporal-reasoning",
                QuestionType::KnowledgeUpdate => "knowledge-update",
            }
        }
    }

    /// Coerce the mixed-type `answer` field to a string.
    pub fn answer_string(&self) -> String {
        answer_to_string(&self.answer)
    }
}

/// Coerce a `serde_json::Value` to a string representation.
///
/// - String -> clone the inner string
/// - Number -> decimal string
/// - Other -> JSON serialization fallback
pub fn answer_to_string(val: &serde_json::Value) -> String {
    match val {
        serde_json::Value::String(s) => s.clone(),
        serde_json::Value::Number(n) => n.to_string(),
        other => other.to_string(),
    }
}

/// A semantic validation error for one dataset instance.
#[derive(Debug)]
pub struct ValidationError {
    pub instance_index: usize,
    pub question_id: String,
    pub errors: Vec<String>,
}

/// Validate semantic invariants across all instances, collecting ALL errors
/// before returning (not fail-fast).
pub fn validate_dataset(instances: &[LongMemEvalInstance]) -> Result<(), Vec<ValidationError>> {
    let mut all_errors = Vec::new();

    for (i, inst) in instances.iter().enumerate() {
        let mut errors = Vec::new();

        // DATA-07: haystack_sessions and haystack_dates must have equal length
        if inst.haystack_sessions.len() != inst.haystack_dates.len() {
            errors.push(format!(
                "haystack_sessions ({}) and haystack_dates ({}) length mismatch",
                inst.haystack_sessions.len(),
                inst.haystack_dates.len()
            ));
        }

        // haystack_sessions and haystack_session_ids must have equal length
        if inst.haystack_sessions.len() != inst.haystack_session_ids.len() {
            errors.push(format!(
                "haystack_sessions ({}) and haystack_session_ids ({}) length mismatch",
                inst.haystack_sessions.len(),
                inst.haystack_session_ids.len()
            ));
        }

        // answer must be a string or number (not null, object, or array)
        if inst.answer.is_null() || inst.answer.is_object() || inst.answer.is_array() {
            errors.push(format!(
                "answer is not a string or number: {:?}",
                inst.answer
            ));
        }

        // question_date must not be empty
        if inst.question_date.is_empty() {
            errors.push("question_date is empty".into());
        }

        if !errors.is_empty() {
            all_errors.push(ValidationError {
                instance_index: i,
                question_id: inst.question_id.clone(),
                errors,
            });
        }
    }

    if all_errors.is_empty() {
        Ok(())
    } else {
        Err(all_errors)
    }
}

/// Load the LongMemEval dataset from a JSON file.
///
/// Returns `(instances, fingerprint)` on success.
/// The fingerprint is a deterministic FNV1a-64 hex hash of the raw file bytes.
///
/// When the file is not found, returns a helpful error message with download
/// instructions for the HuggingFace dataset.
pub fn load_dataset(path: &Path) -> Result<(Vec<LongMemEvalInstance>, String), String> {
    if !path.exists() {
        return Err(format!(
            "Dataset file not found: {}\n\n\
             Download from HuggingFace:\n\
             \n\
             # Install git-lfs first, then:\n\
             git clone https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned\n\
             cp longmemeval-cleaned/longmemeval_s_cleaned.json data/\n\
             cp longmemeval-cleaned/longmemeval_m_cleaned.json data/",
            path.display()
        ));
    }

    let raw_bytes =
        fs::read(path).map_err(|e| format!("failed to read {}: {e}", path.display()))?;

    let fingerprint = format!("{:016x}", common::fnv1a64(&raw_bytes));

    let instances: Vec<LongMemEvalInstance> = serde_json::from_slice(&raw_bytes)
        .map_err(|e| format!("failed to parse {}: {e}", path.display()))?;

    if let Err(validation_errors) = validate_dataset(&instances) {
        let mut msg = format!(
            "Dataset validation failed for {} ({} error(s)):\n",
            path.display(),
            validation_errors.len()
        );
        for ve in &validation_errors {
            msg.push_str(&format!(
                "\n  [{}] {}: {}",
                ve.instance_index,
                ve.question_id,
                ve.errors.join("; ")
            ));
        }
        return Err(msg);
    }

    Ok((instances, fingerprint))
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- QuestionType deserialization ---

    #[test]
    fn deserialize_single_session_user() {
        let qt: QuestionType = serde_json::from_str("\"single-session-user\"").unwrap();
        assert_eq!(qt, QuestionType::SingleSessionUser);
    }

    #[test]
    fn deserialize_single_session_assistant() {
        let qt: QuestionType = serde_json::from_str("\"single-session-assistant\"").unwrap();
        assert_eq!(qt, QuestionType::SingleSessionAssistant);
    }

    #[test]
    fn deserialize_single_session_preference() {
        let qt: QuestionType = serde_json::from_str("\"single-session-preference\"").unwrap();
        assert_eq!(qt, QuestionType::SingleSessionPreference);
    }

    #[test]
    fn deserialize_multi_session() {
        let qt: QuestionType = serde_json::from_str("\"multi-session\"").unwrap();
        assert_eq!(qt, QuestionType::MultiSession);
    }

    #[test]
    fn deserialize_temporal_reasoning() {
        let qt: QuestionType = serde_json::from_str("\"temporal-reasoning\"").unwrap();
        assert_eq!(qt, QuestionType::TemporalReasoning);
    }

    #[test]
    fn deserialize_knowledge_update() {
        let qt: QuestionType = serde_json::from_str("\"knowledge-update\"").unwrap();
        assert_eq!(qt, QuestionType::KnowledgeUpdate);
    }

    // --- answer_to_string ---

    #[test]
    fn answer_string_from_string() {
        let val = serde_json::Value::String("hello".into());
        assert_eq!(answer_to_string(&val), "hello");
    }

    #[test]
    fn answer_number_from_int() {
        let val = serde_json::json!(42);
        assert_eq!(answer_to_string(&val), "42");
    }

    #[test]
    fn answer_fallback_from_bool() {
        let val = serde_json::json!(true);
        assert_eq!(answer_to_string(&val), "true");
    }

    // --- is_abstention ---

    #[test]
    fn is_abstention_with_abs_suffix() {
        let inst = make_instance("single_hop_1_abs", QuestionType::SingleSessionUser);
        assert!(inst.is_abstention());
    }

    #[test]
    fn is_abstention_without_abs() {
        let inst = make_instance("single_hop_1", QuestionType::SingleSessionUser);
        assert!(!inst.is_abstention());
    }

    // --- reporting_category ---

    #[test]
    fn reporting_category_abstention() {
        let inst = make_instance("multi_session_3_abs", QuestionType::MultiSession);
        assert_eq!(inst.reporting_category(), "abstention");
    }

    #[test]
    fn reporting_category_normal() {
        let inst = make_instance("temporal_1", QuestionType::TemporalReasoning);
        assert_eq!(inst.reporting_category(), "temporal-reasoning");
    }

    // --- validate_dataset ---

    #[test]
    fn validate_catches_session_date_length_mismatch() {
        let mut inst = make_instance("q1", QuestionType::SingleSessionUser);
        inst.haystack_sessions = vec![vec![]];
        inst.haystack_dates = vec!["2023/01/01".into(), "2023/01/02".into()];
        inst.haystack_session_ids = vec!["s1".into()];

        let err = validate_dataset(&[inst]).unwrap_err();
        assert_eq!(err.len(), 1);
        assert!(err[0].errors[0].contains("haystack_dates"));
    }

    #[test]
    fn validate_catches_session_id_length_mismatch() {
        let mut inst = make_instance("q1", QuestionType::SingleSessionUser);
        inst.haystack_sessions = vec![vec![]];
        inst.haystack_dates = vec!["2023/01/01".into()];
        inst.haystack_session_ids = vec!["s1".into(), "s2".into()];

        let err = validate_dataset(&[inst]).unwrap_err();
        assert_eq!(err.len(), 1);
        assert!(err[0].errors[0].contains("haystack_session_ids"));
    }

    #[test]
    fn validate_catches_null_answer() {
        let mut inst = make_instance("q1", QuestionType::SingleSessionUser);
        inst.answer = serde_json::Value::Null;

        let err = validate_dataset(&[inst]).unwrap_err();
        assert_eq!(err.len(), 1);
        assert!(err[0].errors[0].contains("answer"));
    }

    #[test]
    fn validate_collects_all_errors() {
        let mut inst1 = make_instance("q1", QuestionType::SingleSessionUser);
        inst1.answer = serde_json::Value::Null;

        let mut inst2 = make_instance("q2", QuestionType::MultiSession);
        inst2.haystack_sessions = vec![vec![]];
        inst2.haystack_dates = vec!["d1".into(), "d2".into()];
        inst2.haystack_session_ids = vec!["s1".into()];

        let err = validate_dataset(&[inst1, inst2]).unwrap_err();
        assert_eq!(err.len(), 2);
        assert_eq!(err[0].question_id, "q1");
        assert_eq!(err[1].question_id, "q2");
    }

    #[test]
    fn validate_passes_for_valid_instance() {
        let inst = make_instance("q1", QuestionType::SingleSessionUser);
        assert!(validate_dataset(&[inst]).is_ok());
    }

    // --- load_dataset ---

    #[test]
    fn load_dataset_missing_file_gives_helpful_error() {
        let result = load_dataset(Path::new("/nonexistent/longmemeval_s_cleaned.json"));
        let err = result.unwrap_err();
        assert!(err.contains("Dataset file not found"));
        assert!(err.contains("huggingface"));
    }

    // --- Helper ---

    fn make_instance(question_id: &str, question_type: QuestionType) -> LongMemEvalInstance {
        LongMemEvalInstance {
            question_id: question_id.into(),
            question_type,
            question: "What is the answer?".into(),
            answer: serde_json::json!("yes"),
            question_date: "2023/05/25 (Thu) 14:30".into(),
            haystack_dates: vec!["2023/05/20 (Mon) 10:15".into()],
            haystack_session_ids: vec!["session_1".into()],
            haystack_sessions: vec![vec![Turn {
                role: "user".into(),
                content: "Hello".into(),
            }]],
            answer_session_ids: vec!["session_1".into()],
        }
    }
}
