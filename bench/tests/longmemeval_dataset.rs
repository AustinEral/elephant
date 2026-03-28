//! Integration tests for LongMemEval dataset loading.
//!
//! Tests marked `#[ignore]` require dataset files in `data/`:
//!   data/longmemeval_s_cleaned.json
//!   data/longmemeval_m_cleaned.json
//!
//! Download from HuggingFace:
//!   git clone https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned
//!   cp longmemeval-cleaned/longmemeval_s_cleaned.json data/
//!   cp longmemeval-cleaned/longmemeval_m_cleaned.json data/

#[path = "../common/mod.rs"]
mod common;
#[path = "../longmemeval/dataset.rs"]
mod dataset;

use std::collections::HashMap;
use std::path::Path;

use dataset::{QuestionType, load_dataset};

#[test]
#[ignore]
fn test_load_s() {
    let path = Path::new("data/longmemeval_s_cleaned.json");
    let (instances, fingerprint) = load_dataset(path).expect("failed to load S dataset");

    assert_eq!(instances.len(), 500, "S dataset should have 500 instances");
    assert_eq!(fingerprint.len(), 16, "fingerprint should be 16 chars");
    assert!(
        fingerprint.chars().all(|c| c.is_ascii_hexdigit()),
        "fingerprint should be hex: {fingerprint}"
    );

    let mut type_counts: HashMap<QuestionType, usize> = HashMap::new();
    for inst in &instances {
        *type_counts.entry(inst.question_type).or_insert(0) += 1;
    }
    assert!(type_counts.contains_key(&QuestionType::SingleSessionUser));
    assert!(type_counts.contains_key(&QuestionType::SingleSessionAssistant));
    assert!(type_counts.contains_key(&QuestionType::SingleSessionPreference));
    assert!(type_counts.contains_key(&QuestionType::MultiSession));
    assert!(type_counts.contains_key(&QuestionType::TemporalReasoning));
    assert!(type_counts.contains_key(&QuestionType::KnowledgeUpdate));

    let abs_count = instances.iter().filter(|i| i.is_abstention()).count();
    assert!(
        abs_count > 0,
        "should have at least one abstention instance"
    );

    for inst in &instances {
        assert!(
            !inst.question_date.is_empty(),
            "question_date should not be empty for {}",
            inst.question_id
        );
    }

    eprintln!("=== LongMemEval S Dataset ===");
    eprintln!("Instances: {}", instances.len());
    eprintln!("Fingerprint: {fingerprint}");
    eprintln!("Abstention: {abs_count}");
    for (qt, count) in &type_counts {
        eprintln!("  {qt:?}: {count}");
    }
}

#[test]
#[ignore]
fn test_load_m() {
    let path_m = Path::new("data/longmemeval_m_cleaned.json");
    let (instances, fingerprint_m) = load_dataset(path_m).expect("failed to load M dataset");

    assert!(!instances.is_empty(), "M dataset should have instances");
    assert_eq!(fingerprint_m.len(), 16, "fingerprint should be 16 chars");
    assert!(
        fingerprint_m.chars().all(|c| c.is_ascii_hexdigit()),
        "fingerprint should be hex: {fingerprint_m}"
    );

    let path_s = Path::new("data/longmemeval_s_cleaned.json");
    let (_, fingerprint_s) = load_dataset(path_s).expect("failed to load S dataset for comparison");
    assert_ne!(
        fingerprint_m, fingerprint_s,
        "M and S fingerprints should differ"
    );

    let mut type_counts: HashMap<QuestionType, usize> = HashMap::new();
    for inst in &instances {
        *type_counts.entry(inst.question_type).or_insert(0) += 1;
    }
    assert!(type_counts.contains_key(&QuestionType::SingleSessionUser));
    assert!(type_counts.contains_key(&QuestionType::SingleSessionAssistant));
    assert!(type_counts.contains_key(&QuestionType::SingleSessionPreference));
    assert!(type_counts.contains_key(&QuestionType::MultiSession));
    assert!(type_counts.contains_key(&QuestionType::TemporalReasoning));
    assert!(type_counts.contains_key(&QuestionType::KnowledgeUpdate));

    eprintln!("=== LongMemEval M Dataset ===");
    eprintln!("Instances: {}", instances.len());
    eprintln!("Fingerprint: {fingerprint_m}");
    for (qt, count) in &type_counts {
        eprintln!("  {qt:?}: {count}");
    }
}

#[test]
#[ignore]
fn test_fingerprint_determinism() {
    let path = Path::new("data/longmemeval_s_cleaned.json");
    let (_, fp1) = load_dataset(path).expect("first load failed");
    let (_, fp2) = load_dataset(path).expect("second load failed");
    assert_eq!(fp1, fp2, "fingerprint should be deterministic across loads");
}

#[test]
#[ignore]
fn test_question_type_distribution() {
    let path = Path::new("data/longmemeval_s_cleaned.json");
    let (instances, _) = load_dataset(path).expect("failed to load S dataset");

    let mut category_counts: HashMap<&str, usize> = HashMap::new();
    for inst in &instances {
        *category_counts
            .entry(inst.reporting_category())
            .or_insert(0) += 1;
    }

    let expected_categories = [
        "single-session-user",
        "single-session-assistant",
        "single-session-preference",
        "multi-session",
        "temporal-reasoning",
        "knowledge-update",
        "abstention",
    ];

    for cat in &expected_categories {
        assert!(
            category_counts.get(cat).copied().unwrap_or(0) > 0,
            "category '{cat}' should have at least 1 instance"
        );
    }

    eprintln!("=== Question Type Distribution (S) ===");
    for cat in &expected_categories {
        let count = category_counts.get(cat).copied().unwrap_or(0);
        eprintln!("  {cat}: {count}");
    }
}

#[test]
fn test_missing_file_error() {
    let result = load_dataset(Path::new("data/nonexistent.json"));
    let err = result.unwrap_err();
    assert!(
        err.contains("Dataset file not found"),
        "error should mention 'Dataset file not found': {err}"
    );
    assert!(
        err.contains("xiaowu0162/longmemeval-cleaned"),
        "error should contain HuggingFace URL: {err}"
    );
}
