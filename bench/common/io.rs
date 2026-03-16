use std::fs;
use std::path::{Path, PathBuf};

use serde::Serialize;

/// Build a sidecar file path by inserting `suffix` before the `.jsonl` extension.
///
/// Example: `sidecar_path("results/foo.json", "questions")` -> `"results/foo.questions.jsonl"`.
pub fn sidecar_path(output_path: &Path, suffix: &str) -> PathBuf {
    let parent = output_path
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_default();
    let stem = output_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("results");
    parent.join(format!("{stem}.{suffix}.jsonl"))
}

/// Atomically write a JSON value to a file (write-to-tmp then rename).
pub fn atomic_write_json<T: Serialize>(path: &Path, value: &T) {
    let json = match serde_json::to_string_pretty(value) {
        Ok(j) => j,
        Err(e) => {
            eprintln!("atomic_write_json: serialize failed: {e}");
            return;
        }
    };
    let tmp = path.with_extension("tmp");
    if fs::write(&tmp, &json).is_ok() {
        if fs::rename(&tmp, path).is_err() {
            // rename failed (cross-device?), fall back to direct write
            let _ = fs::write(path, &json);
            let _ = fs::remove_file(&tmp);
        }
    } else {
        // tmp write failed, fall back to direct write
        let _ = fs::write(path, &json);
    }
}

/// Append a single JSON-serialized line to a JSONL file, creating the file if needed.
pub fn append_jsonl<T: Serialize>(path: &Path, value: &T) {
    if let Ok(line) = serde_json::to_string(value)
        && let Ok(mut file) = fs::OpenOptions::new().create(true).append(true).open(path)
    {
        use std::io::Write;
        let _ = writeln!(file, "{line}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sidecar_path_format() {
        let p = sidecar_path(Path::new("results/foo.json"), "questions");
        assert_eq!(p, PathBuf::from("results/foo.questions.jsonl"));
    }

    #[test]
    fn sidecar_path_no_parent() {
        let p = sidecar_path(Path::new("foo.json"), "debug");
        assert_eq!(p, PathBuf::from("foo.debug.jsonl"));
    }

    #[test]
    fn append_jsonl_creates_and_appends() {
        let dir = std::env::temp_dir().join("elephant_test_io");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("test_append.jsonl");
        let _ = std::fs::remove_file(&path);

        append_jsonl(&path, &serde_json::json!({"a": 1}));
        append_jsonl(&path, &serde_json::json!({"b": 2}));

        let contents = std::fs::read_to_string(&path).unwrap();
        let lines: Vec<&str> = contents.lines().collect();
        assert_eq!(lines.len(), 2);
        assert!(lines[0].contains("\"a\":1") || lines[0].contains("\"a\": 1"));

        let _ = std::fs::remove_file(&path);
    }
}
