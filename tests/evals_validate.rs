use std::fs;
use std::path::Path;

use datatest_stable::harness;
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct ExtractCase {
    id: String,
    description: Option<String>,
    status: Status,
    input: Input,
    assertions: Vec<Assertion>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
enum Status {
    Guard,
    Tracking,
    Limitation,
}

#[derive(Debug, Deserialize)]
struct Input {
    mode: String,
    transcript: Vec<Message>,
}

#[derive(Debug, Deserialize)]
struct Message {
    speaker: String,
    timestamp: String,
    text: String,
}

#[derive(Debug, Deserialize)]
struct Assertion {
    kind: String,
    value: String,
}

fn validate_extract_case(path: &Path) -> datatest_stable::Result<()> {
    let raw = fs::read_to_string(path)?;
    let case: ExtractCase = serde_json::from_str(&raw)?;

    assert!(
        !case.id.trim().is_empty(),
        "{}: id must not be empty",
        path.display()
    );
    if let Some(description) = &case.description {
        assert!(
            !description.trim().is_empty(),
            "{}: description must not be empty when present",
            path.display()
        );
    }
    match case.status {
        Status::Guard | Status::Tracking | Status::Limitation => {}
    }
    assert!(
        matches!(case.input.mode.as_str(), "turn" | "session"),
        "{}: input.mode must be turn or session",
        path.display()
    );
    assert!(
        !case.input.transcript.is_empty(),
        "{}: transcript must not be empty",
        path.display()
    );
    for (idx, message) in case.input.transcript.iter().enumerate() {
        assert!(
            !message.speaker.trim().is_empty(),
            "{}: transcript[{idx}].speaker must not be empty",
            path.display()
        );
        assert!(
            !message.timestamp.trim().is_empty(),
            "{}: transcript[{idx}].timestamp must not be empty",
            path.display()
        );
        assert!(
            !message.text.trim().is_empty(),
            "{}: transcript[{idx}].text must not be empty",
            path.display()
        );
    }
    assert!(
        !case.assertions.is_empty(),
        "{}: assertions must not be empty",
        path.display()
    );
    for (idx, assertion) in case.assertions.iter().enumerate() {
        assert!(
            matches!(assertion.kind.as_str(), "fact_contains" | "fact_not_contains"),
            "{}: assertions[{idx}].kind must be fact_contains or fact_not_contains",
            path.display()
        );
        assert!(
            !assertion.value.trim().is_empty(),
            "{}: assertions[{idx}].value must not be empty",
            path.display()
        );
    }

    Ok(())
}

harness! {
    { test = validate_extract_case, root = "tests/evals/extract", pattern = r".*\.json$" },
}
