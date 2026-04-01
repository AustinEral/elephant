use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use elephant::llm::DEFAULT_TIMEOUT_SECS;

fn default_dataset_path() -> PathBuf {
    PathBuf::from("data/locomo10.json")
}

fn default_output_dir() -> PathBuf {
    PathBuf::from("bench/locomo/results/local")
}

fn default_bench_database_url() -> String {
    "postgres://postgres:postgres@localhost:5433/elephant_bench".into()
}

fn default_embedding_model_path() -> PathBuf {
    PathBuf::from("models/bge-small-en-v1.5")
}

fn default_reranker_model_path() -> PathBuf {
    PathBuf::from("models/ms-marco-MiniLM-L-6-v2")
}

fn default_longmemeval_dataset_path() -> PathBuf {
    PathBuf::from("data/longmemeval_s_cleaned.json")
}

fn default_longmemeval_output_dir() -> PathBuf {
    PathBuf::from("bench/longmemeval/results/local")
}

fn default_jobs() -> usize {
    1
}

fn default_timeout_secs() -> u64 {
    DEFAULT_TIMEOUT_SECS
}

fn default_embedding_max_seq_len() -> usize {
    512
}

fn default_reranker_max_seq_len() -> usize {
    512
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct BenchExecutionOverlayFile {
    #[serde(default)]
    pub(crate) database_url: Option<String>,
    #[serde(default)]
    pub(crate) dataset_path: Option<PathBuf>,
    #[serde(default)]
    pub(crate) output_dir: Option<PathBuf>,
    #[serde(default)]
    pub(crate) tag: Option<String>,
    #[serde(default)]
    pub(crate) conversation_jobs: Option<usize>,
    #[serde(default)]
    pub(crate) question_jobs: Option<usize>,
    #[serde(default)]
    pub(crate) llm_base_url: Option<String>,
    #[serde(default)]
    pub(crate) llm_timeout_secs: Option<u64>,
    #[serde(default)]
    pub(crate) llm_vertex_project: Option<String>,
    #[serde(default)]
    pub(crate) llm_vertex_location: Option<String>,
    #[serde(default)]
    pub(crate) embedding_model_path: Option<PathBuf>,
    #[serde(default)]
    pub(crate) embedding_max_seq_len: Option<usize>,
    #[serde(default)]
    pub(crate) reranker_model_path: Option<PathBuf>,
    #[serde(default)]
    pub(crate) reranker_api_url: Option<String>,
    #[serde(default)]
    pub(crate) reranker_max_seq_len: Option<usize>,
    #[serde(default)]
    pub(crate) judge_base_url: Option<String>,
    #[serde(default)]
    pub(crate) judge_timeout_secs: Option<u64>,
    #[serde(default)]
    pub(crate) judge_vertex_project: Option<String>,
    #[serde(default)]
    pub(crate) judge_vertex_location: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct BenchExecution {
    pub(crate) dataset_path: PathBuf,
    pub(crate) output_dir: PathBuf,
    pub(crate) tag: Option<String>,
    pub(crate) conversation_jobs: usize,
    pub(crate) question_jobs: usize,
    pub(crate) database_url: String,
    pub(crate) llm_base_url: Option<String>,
    pub(crate) llm_timeout_secs: u64,
    pub(crate) llm_vertex_project: Option<String>,
    pub(crate) llm_vertex_location: Option<String>,
    pub(crate) embedding_model_path: Option<PathBuf>,
    pub(crate) embedding_max_seq_len: usize,
    pub(crate) reranker_model_path: Option<PathBuf>,
    pub(crate) reranker_api_url: Option<String>,
    pub(crate) reranker_max_seq_len: usize,
    pub(crate) judge_base_url: Option<String>,
    pub(crate) judge_timeout_secs: u64,
    pub(crate) judge_vertex_project: Option<String>,
    pub(crate) judge_vertex_location: Option<String>,
}

impl BenchExecution {
    pub(crate) fn from_overlay(overlay: Option<BenchExecutionOverlayFile>) -> Self {
        let overlay = overlay.unwrap_or(BenchExecutionOverlayFile {
            database_url: None,
            dataset_path: None,
            output_dir: None,
            tag: None,
            conversation_jobs: None,
            question_jobs: None,
            llm_base_url: None,
            llm_timeout_secs: None,
            llm_vertex_project: None,
            llm_vertex_location: None,
            embedding_model_path: None,
            embedding_max_seq_len: None,
            reranker_model_path: None,
            reranker_api_url: None,
            reranker_max_seq_len: None,
            judge_base_url: None,
            judge_timeout_secs: None,
            judge_vertex_project: None,
            judge_vertex_location: None,
        });

        Self {
            dataset_path: overlay.dataset_path.unwrap_or_else(default_dataset_path),
            output_dir: overlay.output_dir.unwrap_or_else(default_output_dir),
            tag: overlay.tag,
            conversation_jobs: overlay.conversation_jobs.unwrap_or_else(default_jobs),
            question_jobs: overlay.question_jobs.unwrap_or_else(default_jobs),
            database_url: overlay
                .database_url
                .unwrap_or_else(default_bench_database_url),
            llm_base_url: overlay.llm_base_url,
            llm_timeout_secs: overlay
                .llm_timeout_secs
                .unwrap_or_else(default_timeout_secs),
            llm_vertex_project: overlay.llm_vertex_project,
            llm_vertex_location: overlay.llm_vertex_location,
            embedding_model_path: Some(
                overlay
                    .embedding_model_path
                    .unwrap_or_else(default_embedding_model_path),
            ),
            embedding_max_seq_len: overlay
                .embedding_max_seq_len
                .unwrap_or_else(default_embedding_max_seq_len),
            reranker_model_path: Some(
                overlay
                    .reranker_model_path
                    .unwrap_or_else(default_reranker_model_path),
            ),
            reranker_api_url: overlay.reranker_api_url,
            reranker_max_seq_len: overlay
                .reranker_max_seq_len
                .unwrap_or_else(default_reranker_max_seq_len),
            judge_base_url: overlay.judge_base_url,
            judge_timeout_secs: overlay
                .judge_timeout_secs
                .unwrap_or_else(default_timeout_secs),
            judge_vertex_project: overlay.judge_vertex_project,
            judge_vertex_location: overlay.judge_vertex_location,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct LongMemEvalExecutionOverlayFile {
    #[serde(default)]
    pub(crate) database_url: Option<String>,
    #[serde(default)]
    pub(crate) dataset_path: Option<PathBuf>,
    #[serde(default)]
    pub(crate) output_dir: Option<PathBuf>,
    #[serde(default)]
    pub(crate) tag: Option<String>,
    #[serde(default)]
    pub(crate) instance_jobs: Option<usize>,
    #[serde(default)]
    pub(crate) llm_base_url: Option<String>,
    #[serde(default)]
    pub(crate) llm_timeout_secs: Option<u64>,
    #[serde(default)]
    pub(crate) llm_vertex_project: Option<String>,
    #[serde(default)]
    pub(crate) llm_vertex_location: Option<String>,
    #[serde(default)]
    pub(crate) embedding_model_path: Option<PathBuf>,
    #[serde(default)]
    pub(crate) embedding_max_seq_len: Option<usize>,
    #[serde(default)]
    pub(crate) reranker_model_path: Option<PathBuf>,
    #[serde(default)]
    pub(crate) reranker_api_url: Option<String>,
    #[serde(default)]
    pub(crate) reranker_max_seq_len: Option<usize>,
    #[serde(default)]
    pub(crate) judge_base_url: Option<String>,
    #[serde(default)]
    pub(crate) judge_timeout_secs: Option<u64>,
    #[serde(default)]
    pub(crate) judge_vertex_project: Option<String>,
    #[serde(default)]
    pub(crate) judge_vertex_location: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct LongMemEvalExecution {
    pub(crate) dataset_path: PathBuf,
    pub(crate) output_dir: PathBuf,
    pub(crate) tag: Option<String>,
    pub(crate) instance_jobs: usize,
    pub(crate) database_url: String,
    pub(crate) llm_base_url: Option<String>,
    pub(crate) llm_timeout_secs: u64,
    pub(crate) llm_vertex_project: Option<String>,
    pub(crate) llm_vertex_location: Option<String>,
    pub(crate) embedding_model_path: Option<PathBuf>,
    pub(crate) embedding_max_seq_len: usize,
    pub(crate) reranker_model_path: Option<PathBuf>,
    pub(crate) reranker_api_url: Option<String>,
    pub(crate) reranker_max_seq_len: usize,
    pub(crate) judge_base_url: Option<String>,
    pub(crate) judge_timeout_secs: u64,
    pub(crate) judge_vertex_project: Option<String>,
    pub(crate) judge_vertex_location: Option<String>,
}

impl LongMemEvalExecution {
    pub(crate) fn from_overlay(overlay: Option<LongMemEvalExecutionOverlayFile>) -> Self {
        let overlay = overlay.unwrap_or(LongMemEvalExecutionOverlayFile {
            database_url: None,
            dataset_path: None,
            output_dir: None,
            tag: None,
            instance_jobs: None,
            llm_base_url: None,
            llm_timeout_secs: None,
            llm_vertex_project: None,
            llm_vertex_location: None,
            embedding_model_path: None,
            embedding_max_seq_len: None,
            reranker_model_path: None,
            reranker_api_url: None,
            reranker_max_seq_len: None,
            judge_base_url: None,
            judge_timeout_secs: None,
            judge_vertex_project: None,
            judge_vertex_location: None,
        });

        Self {
            dataset_path: overlay
                .dataset_path
                .unwrap_or_else(default_longmemeval_dataset_path),
            output_dir: overlay
                .output_dir
                .unwrap_or_else(default_longmemeval_output_dir),
            tag: overlay.tag,
            instance_jobs: overlay.instance_jobs.unwrap_or_else(default_jobs),
            database_url: overlay
                .database_url
                .unwrap_or_else(default_bench_database_url),
            llm_base_url: overlay.llm_base_url,
            llm_timeout_secs: overlay
                .llm_timeout_secs
                .unwrap_or_else(default_timeout_secs),
            llm_vertex_project: overlay.llm_vertex_project,
            llm_vertex_location: overlay.llm_vertex_location,
            embedding_model_path: Some(
                overlay
                    .embedding_model_path
                    .unwrap_or_else(default_embedding_model_path),
            ),
            embedding_max_seq_len: overlay
                .embedding_max_seq_len
                .unwrap_or_else(default_embedding_max_seq_len),
            reranker_model_path: Some(
                overlay
                    .reranker_model_path
                    .unwrap_or_else(default_reranker_model_path),
            ),
            reranker_api_url: overlay.reranker_api_url,
            reranker_max_seq_len: overlay
                .reranker_max_seq_len
                .unwrap_or_else(default_reranker_max_seq_len),
            judge_base_url: overlay.judge_base_url,
            judge_timeout_secs: overlay
                .judge_timeout_secs
                .unwrap_or_else(default_timeout_secs),
            judge_vertex_project: overlay.judge_vertex_project,
            judge_vertex_location: overlay.judge_vertex_location,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn locomo_execution_defaults_support_standard_local_benchmark_setup() {
        let execution = BenchExecution::from_overlay(None);

        assert_eq!(
            execution.database_url,
            "postgres://postgres:postgres@localhost:5433/elephant_bench"
        );
        assert_eq!(
            execution.embedding_model_path,
            Some(PathBuf::from("models/bge-small-en-v1.5"))
        );
        assert_eq!(
            execution.reranker_model_path,
            Some(PathBuf::from("models/ms-marco-MiniLM-L-6-v2"))
        );
    }

    #[test]
    fn longmemeval_execution_defaults_support_standard_local_benchmark_setup() {
        let execution = LongMemEvalExecution::from_overlay(None);

        assert_eq!(
            execution.database_url,
            "postgres://postgres:postgres@localhost:5433/elephant_bench"
        );
        assert_eq!(
            execution.embedding_model_path,
            Some(PathBuf::from("models/bge-small-en-v1.5"))
        );
        assert_eq!(
            execution.reranker_model_path,
            Some(PathBuf::from("models/ms-marco-MiniLM-L-6-v2"))
        );
    }
}
