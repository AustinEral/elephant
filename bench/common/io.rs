use std::fs;
use std::path::{Path, PathBuf};

use serde::Serialize;

/// Resolve logical repo-relative benchmark paths against the workspace root.
pub fn resolve_workspace_path(path: &Path) -> PathBuf {
    if path.is_absolute() {
        return path.to_path_buf();
    }

    let Some(first) = path.components().next() else {
        return path.to_path_buf();
    };
    let anchor_to_workspace = matches!(
        first.as_os_str().to_str(),
        Some("bench" | "data" | "models" | "lib")
    );
    if !anchor_to_workspace {
        return path.to_path_buf();
    }

    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("bench crate must live under the workspace root")
        .join(path)
}

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
#[allow(dead_code)]
pub fn atomic_write_json<T: Serialize>(path: &Path, value: &T) {
    let path = resolve_workspace_path(path);
    let json = match serde_json::to_string_pretty(value) {
        Ok(j) => j,
        Err(e) => {
            eprintln!("atomic_write_json: serialize failed: {e}");
            return;
        }
    };
    let tmp = path.with_extension("tmp");
    if fs::write(&tmp, &json).is_ok() {
        if fs::rename(&tmp, &path).is_err() {
            // rename failed (cross-device?), fall back to direct write
            let _ = fs::write(&path, &json);
            let _ = fs::remove_file(&tmp);
        }
    } else {
        // tmp write failed, fall back to direct write
        let _ = fs::write(&path, &json);
    }
}

/// Append a single JSON-serialized line to a JSONL file, creating the file if needed.
pub fn append_jsonl<T: Serialize>(path: &Path, value: &T) {
    let path = resolve_workspace_path(path);
    if let Ok(line) = serde_json::to_string(value)
        && let Ok(mut file) = fs::OpenOptions::new().create(true).append(true).open(&path)
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
    fn resolve_workspace_path_rewrites_logical_bench_paths() {
        let resolved = resolve_workspace_path(Path::new("bench/locomo/profiles/smoke.toml"));
        let expected = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .join("bench/locomo/profiles/smoke.toml");
        assert_eq!(resolved, expected);
    }

    #[test]
    fn resolve_workspace_path_rewrites_repo_relative_data_and_model_paths() {
        let workspace_root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
        assert_eq!(
            resolve_workspace_path(Path::new("data/longmemeval_s_cleaned.json")),
            workspace_root.join("data/longmemeval_s_cleaned.json")
        );
        assert_eq!(
            resolve_workspace_path(Path::new("models/bge-small-en-v1.5")),
            workspace_root.join("models/bge-small-en-v1.5")
        );
        assert_eq!(
            resolve_workspace_path(Path::new(
                "lib/onnxruntime-linux-x64-1.23.0/lib/libonnxruntime.so.1.23.0"
            )),
            workspace_root.join("lib/onnxruntime-linux-x64-1.23.0/lib/libonnxruntime.so.1.23.0")
        );
    }

    #[test]
    fn resolve_workspace_path_leaves_non_bench_paths_unchanged() {
        let absolute = std::env::temp_dir().join("elephant-absolute-test.json");
        assert_eq!(resolve_workspace_path(&absolute), absolute);

        let other_relative = Path::new("tmp/example.json");
        assert_eq!(resolve_workspace_path(other_relative), other_relative);
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
