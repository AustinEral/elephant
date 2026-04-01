use std::path::PathBuf;

use serde::{Deserialize, Serialize};

fn default_output_dir() -> PathBuf {
    PathBuf::from("bench/locomo/results/local")
}

fn default_bench_database_url() -> String {
    "postgres://postgres:postgres@localhost:5433/elephant_bench".into()
}

fn default_longmemeval_output_dir() -> PathBuf {
    PathBuf::from("bench/longmemeval/results/local")
}

fn default_jobs() -> usize {
    1
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct LocomoShardOverlayFile {
    #[serde(default)]
    pub(crate) conversations: Vec<String>,
}

#[derive(Debug, Clone, Default, Serialize)]
pub(crate) struct LocomoShardExecution {
    pub(crate) conversations: Vec<String>,
}

impl LocomoShardExecution {
    pub(crate) fn from_overlay(overlay: Option<LocomoShardOverlayFile>) -> Self {
        let overlay = overlay.unwrap_or_default();
        Self {
            conversations: overlay.conversations,
        }
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.conversations.is_empty()
    }
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct LongMemEvalShardOverlayFile {
    #[serde(default)]
    pub(crate) instances: Vec<String>,
    #[serde(default)]
    pub(crate) instance_limit: Option<usize>,
    #[serde(default)]
    pub(crate) instance_offset: usize,
}

#[derive(Debug, Clone, Default, Serialize)]
pub(crate) struct LongMemEvalShardExecution {
    pub(crate) instances: Vec<String>,
    pub(crate) instance_limit: Option<usize>,
    pub(crate) instance_offset: usize,
}

impl LongMemEvalShardExecution {
    pub(crate) fn from_overlay(overlay: Option<LongMemEvalShardOverlayFile>) -> Self {
        let overlay = overlay.unwrap_or_default();
        Self {
            instances: overlay.instances,
            instance_limit: overlay.instance_limit,
            instance_offset: overlay.instance_offset,
        }
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.instances.is_empty() && self.instance_limit.is_none() && self.instance_offset == 0
    }
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ClientTargetOverlayFile {
    #[serde(default)]
    pub(crate) base_url: Option<String>,
    #[serde(default)]
    pub(crate) vertex_project: Option<String>,
    #[serde(default)]
    pub(crate) vertex_location: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize)]
pub(crate) struct ClientTargetExecution {
    pub(crate) base_url: Option<String>,
    pub(crate) vertex_project: Option<String>,
    pub(crate) vertex_location: Option<String>,
}

impl ClientTargetExecution {
    pub(crate) fn from_overlay(overlay: Option<ClientTargetOverlayFile>) -> Self {
        let overlay = overlay.unwrap_or_default();
        Self {
            base_url: overlay.base_url,
            vertex_project: overlay.vertex_project,
            vertex_location: overlay.vertex_location,
        }
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.base_url.is_none() && self.vertex_project.is_none() && self.vertex_location.is_none()
    }
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
    pub(crate) shard: Option<LocomoShardOverlayFile>,
    #[serde(default)]
    pub(crate) runtime_target: Option<ClientTargetOverlayFile>,
    #[serde(default)]
    pub(crate) judge_target: Option<ClientTargetOverlayFile>,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct BenchExecution {
    pub(crate) dataset_path: PathBuf,
    pub(crate) output_dir: PathBuf,
    pub(crate) tag: Option<String>,
    pub(crate) conversation_jobs: usize,
    pub(crate) question_jobs: usize,
    pub(crate) database_url: String,
    pub(crate) shard: LocomoShardExecution,
    pub(crate) runtime_target: ClientTargetExecution,
    pub(crate) judge_target: ClientTargetExecution,
}

impl BenchExecution {
    pub(crate) fn from_overlay(
        overlay: Option<BenchExecutionOverlayFile>,
        default_dataset_path: PathBuf,
    ) -> Self {
        let overlay = overlay.unwrap_or(BenchExecutionOverlayFile {
            database_url: None,
            dataset_path: None,
            output_dir: None,
            tag: None,
            conversation_jobs: None,
            question_jobs: None,
            shard: None,
            runtime_target: None,
            judge_target: None,
        });

        Self {
            dataset_path: overlay.dataset_path.unwrap_or(default_dataset_path),
            output_dir: overlay.output_dir.unwrap_or_else(default_output_dir),
            tag: overlay.tag,
            conversation_jobs: overlay.conversation_jobs.unwrap_or_else(default_jobs),
            question_jobs: overlay.question_jobs.unwrap_or_else(default_jobs),
            database_url: overlay
                .database_url
                .unwrap_or_else(default_bench_database_url),
            shard: LocomoShardExecution::from_overlay(overlay.shard),
            runtime_target: ClientTargetExecution::from_overlay(overlay.runtime_target),
            judge_target: ClientTargetExecution::from_overlay(overlay.judge_target),
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
    pub(crate) shard: Option<LongMemEvalShardOverlayFile>,
    #[serde(default)]
    pub(crate) runtime_target: Option<ClientTargetOverlayFile>,
    #[serde(default)]
    pub(crate) judge_target: Option<ClientTargetOverlayFile>,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct LongMemEvalExecution {
    pub(crate) dataset_path: PathBuf,
    pub(crate) output_dir: PathBuf,
    pub(crate) tag: Option<String>,
    pub(crate) instance_jobs: usize,
    pub(crate) database_url: String,
    pub(crate) shard: LongMemEvalShardExecution,
    pub(crate) runtime_target: ClientTargetExecution,
    pub(crate) judge_target: ClientTargetExecution,
}

impl LongMemEvalExecution {
    pub(crate) fn from_overlay(
        overlay: Option<LongMemEvalExecutionOverlayFile>,
        default_dataset_path: PathBuf,
    ) -> Self {
        let overlay = overlay.unwrap_or(LongMemEvalExecutionOverlayFile {
            database_url: None,
            dataset_path: None,
            output_dir: None,
            tag: None,
            instance_jobs: None,
            shard: None,
            runtime_target: None,
            judge_target: None,
        });

        Self {
            dataset_path: overlay.dataset_path.unwrap_or(default_dataset_path),
            output_dir: overlay
                .output_dir
                .unwrap_or_else(default_longmemeval_output_dir),
            tag: overlay.tag,
            instance_jobs: overlay.instance_jobs.unwrap_or_else(default_jobs),
            database_url: overlay
                .database_url
                .unwrap_or_else(default_bench_database_url),
            shard: LongMemEvalShardExecution::from_overlay(overlay.shard),
            runtime_target: ClientTargetExecution::from_overlay(overlay.runtime_target),
            judge_target: ClientTargetExecution::from_overlay(overlay.judge_target),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn locomo_execution_defaults_support_standard_local_benchmark_setup() {
        let execution = BenchExecution::from_overlay(None, PathBuf::from("data/locomo10.json"));

        assert_eq!(
            execution.database_url,
            "postgres://postgres:postgres@localhost:5433/elephant_bench"
        );
        assert_eq!(execution.dataset_path, PathBuf::from("data/locomo10.json"));
    }

    #[test]
    fn longmemeval_execution_defaults_support_standard_local_benchmark_setup() {
        let execution = LongMemEvalExecution::from_overlay(
            None,
            PathBuf::from("data/longmemeval_m_cleaned.json"),
        );

        assert_eq!(
            execution.database_url,
            "postgres://postgres:postgres@localhost:5433/elephant_bench"
        );
        assert_eq!(
            execution.dataset_path,
            PathBuf::from("data/longmemeval_m_cleaned.json")
        );
    }
}
