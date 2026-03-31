use serde::{Deserialize, Serialize};

use elephant::llm::{DeterminismRequirement, ReasoningEffort};

fn default_schema_version() -> u32 {
    1
}

fn default_protocol_version() -> String {
    "2026-03-31-locomo-contract-v1".into()
}

fn default_dataset_identifier() -> String {
    "locomo10".into()
}

fn default_category_filter() -> Vec<u8> {
    vec![1, 2, 3, 4]
}

fn default_reflect_budget_tokens() -> usize {
    4096
}

fn default_source_limit() -> usize {
    elephant::reflect::DEFAULT_SOURCE_LOOKUP_LIMIT
}

fn default_enable_source_lookup() -> bool {
    elephant::reflect::DEFAULT_ENABLE_SOURCE_LOOKUP
}

fn default_semantic_threshold() -> f32 {
    0.7
}

fn default_temporal_max_days() -> i64 {
    30
}

fn default_true() -> bool {
    true
}

fn default_consolidation_batch_size() -> usize {
    8
}

fn default_consolidation_max_tokens() -> usize {
    4096
}

fn default_consolidation_recall_budget() -> usize {
    512
}

fn default_structured_output_max_attempts() -> usize {
    3
}

fn default_retriever_limit() -> usize {
    40
}

fn default_max_facts() -> usize {
    50
}

fn default_reflect_max_iterations() -> usize {
    8
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub(crate) enum BenchmarkKind {
    #[default]
    Locomo,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ProviderKind {
    Anthropic,
    #[serde(rename = "openai")]
    OpenAi,
    Gemini,
    Vertex,
}

impl ProviderKind {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Anthropic => "anthropic",
            Self::OpenAi => "openai",
            Self::Gemini => "gemini",
            Self::Vertex => "vertex",
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "kebab-case")]
pub(crate) enum LocomoIngestMode {
    #[default]
    Turn,
    Session,
    RawJson,
}

impl LocomoIngestMode {
    pub(crate) fn image_policy(self) -> &'static str {
        match self {
            Self::RawJson => "raw_json_session_payload",
            Self::Turn | Self::Session => "blip_caption_inline",
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "kebab-case")]
pub(crate) enum LocomoConsolidationMode {
    #[default]
    End,
    PerSession,
    Off,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum EmbeddingProviderKind {
    Local,
    #[serde(rename = "openai")]
    OpenAi,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum RerankerProviderKind {
    Local,
    Api,
    None,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct LocomoContractFile {
    #[serde(default = "default_schema_version")]
    pub(crate) schema_version: u32,
    #[serde(default)]
    pub(crate) benchmark: BenchmarkKind,
    #[serde(default = "default_protocol_version")]
    pub(crate) protocol_version: String,
    #[serde(default)]
    pub(crate) dataset: DatasetContract,
    #[serde(default)]
    pub(crate) slice: LocomoSliceContract,
    #[serde(default)]
    pub(crate) ingest: LocomoIngestMode,
    #[serde(default)]
    pub(crate) consolidation: LocomoConsolidationMode,
    #[serde(default)]
    pub(crate) determinism_requirement: Option<DeterminismRequirement>,
    pub(crate) runtime: RuntimeContract,
    pub(crate) judge: JudgeContract,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct DatasetContract {
    #[serde(default = "default_dataset_identifier")]
    pub(crate) identifier: String,
    #[serde(default)]
    pub(crate) expected_fingerprint: Option<String>,
}

impl Default for DatasetContract {
    fn default() -> Self {
        Self {
            identifier: default_dataset_identifier(),
            expected_fingerprint: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct LocomoSliceContract {
    #[serde(default = "default_category_filter")]
    pub(crate) category_filter: Vec<u8>,
    #[serde(default)]
    pub(crate) conversations: Vec<String>,
    #[serde(default)]
    pub(crate) session_limit: Option<usize>,
    #[serde(default)]
    pub(crate) question_limit: Option<usize>,
}

impl Default for LocomoSliceContract {
    fn default() -> Self {
        Self {
            category_filter: default_category_filter(),
            conversations: Vec::new(),
            session_limit: None,
            question_limit: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct RuntimeContract {
    pub(crate) llm: RuntimeLlmContract,
    pub(crate) embedding: EmbeddingContract,
    pub(crate) reranker: RerankerContract,
    #[serde(default)]
    pub(crate) tuning: RuntimeTuningContract,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct RuntimeLlmContract {
    pub(crate) provider: ProviderKind,
    pub(crate) retain_model: String,
    pub(crate) reflect_model: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct EmbeddingContract {
    pub(crate) provider: EmbeddingProviderKind,
    pub(crate) model: String,
    #[serde(default)]
    pub(crate) dimensions: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct RerankerContract {
    pub(crate) provider: RerankerProviderKind,
    #[serde(default)]
    pub(crate) model: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct JudgeContract {
    pub(crate) provider: ProviderKind,
    pub(crate) model: String,
    #[serde(default)]
    pub(crate) temperature: Option<f32>,
    #[serde(default = "default_judge_max_tokens")]
    pub(crate) max_tokens: usize,
    #[serde(default = "default_judge_max_attempts")]
    pub(crate) max_attempts: usize,
}

fn default_judge_max_tokens() -> usize {
    200
}

fn default_judge_max_attempts() -> usize {
    3
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct RuntimeTuningContract {
    #[serde(default = "default_reflect_budget_tokens")]
    pub(crate) reflect_budget_tokens: usize,
    #[serde(default = "default_dedup_threshold")]
    pub(crate) dedup_threshold: Option<f32>,
    #[serde(default)]
    pub(crate) reasoning_effort: ReasoningEffortContract,
    #[serde(default)]
    pub(crate) extraction: ExtractionTuningContract,
    #[serde(default)]
    pub(crate) resolve_temperature: Option<f32>,
    #[serde(default)]
    pub(crate) graph: GraphTuningContract,
    #[serde(default)]
    pub(crate) consolidation: ConsolidationTuningContract,
    #[serde(default)]
    pub(crate) opinion_merge: OpinionMergeTuningContract,
    #[serde(default)]
    pub(crate) reflect: ReflectTuningContract,
    #[serde(default)]
    pub(crate) reflect_temperature: Option<f32>,
    #[serde(default)]
    pub(crate) retrieval: RetrievalTuningContract,
}

fn default_dedup_threshold() -> Option<f32> {
    Some(0.95)
}

impl Default for RuntimeTuningContract {
    fn default() -> Self {
        Self {
            reflect_budget_tokens: default_reflect_budget_tokens(),
            dedup_threshold: default_dedup_threshold(),
            reasoning_effort: ReasoningEffortContract::default(),
            extraction: ExtractionTuningContract::default(),
            resolve_temperature: None,
            graph: GraphTuningContract::default(),
            consolidation: ConsolidationTuningContract::default(),
            opinion_merge: OpinionMergeTuningContract::default(),
            reflect: ReflectTuningContract::default(),
            reflect_temperature: None,
            retrieval: RetrievalTuningContract::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub(crate) struct ReasoningEffortContract {
    #[serde(default)]
    pub(crate) retain_extract: Option<ReasoningEffort>,
    #[serde(default)]
    pub(crate) retain_resolve: Option<ReasoningEffort>,
    #[serde(default)]
    pub(crate) retain_graph: Option<ReasoningEffort>,
    #[serde(default)]
    pub(crate) reflect: Option<ReasoningEffort>,
    #[serde(default)]
    pub(crate) consolidate: Option<ReasoningEffort>,
    #[serde(default)]
    pub(crate) opinion_merge: Option<ReasoningEffort>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ExtractionTuningContract {
    #[serde(default = "default_structured_output_max_attempts")]
    pub(crate) structured_output_max_attempts: usize,
    #[serde(default)]
    pub(crate) temperature: Option<f32>,
}

impl Default for ExtractionTuningContract {
    fn default() -> Self {
        Self {
            structured_output_max_attempts: default_structured_output_max_attempts(),
            temperature: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct GraphTuningContract {
    #[serde(default = "default_semantic_threshold")]
    pub(crate) semantic_threshold: f32,
    #[serde(default = "default_temporal_max_days")]
    pub(crate) temporal_max_days: i64,
    #[serde(default = "default_true")]
    pub(crate) enable_causal: bool,
    #[serde(default)]
    pub(crate) causal_temperature: Option<f32>,
}

impl Default for GraphTuningContract {
    fn default() -> Self {
        Self {
            semantic_threshold: default_semantic_threshold(),
            temporal_max_days: default_temporal_max_days(),
            enable_causal: default_true(),
            causal_temperature: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ConsolidationTuningContract {
    #[serde(default = "default_consolidation_batch_size")]
    pub(crate) batch_size: usize,
    #[serde(default = "default_consolidation_max_tokens")]
    pub(crate) max_tokens: usize,
    #[serde(default = "default_consolidation_recall_budget")]
    pub(crate) recall_budget: usize,
    #[serde(default = "default_structured_output_max_attempts")]
    pub(crate) structured_output_max_attempts: usize,
    #[serde(default)]
    pub(crate) temperature: Option<f32>,
}

impl Default for ConsolidationTuningContract {
    fn default() -> Self {
        Self {
            batch_size: default_consolidation_batch_size(),
            max_tokens: default_consolidation_max_tokens(),
            recall_budget: default_consolidation_recall_budget(),
            structured_output_max_attempts: default_structured_output_max_attempts(),
            temperature: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub(crate) struct OpinionMergeTuningContract {
    #[serde(default)]
    pub(crate) temperature: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ReflectTuningContract {
    #[serde(default = "default_reflect_max_iterations")]
    pub(crate) max_iterations: usize,
    #[serde(default)]
    pub(crate) max_tokens: Option<usize>,
    #[serde(default = "default_source_limit")]
    pub(crate) source_limit: usize,
    #[serde(default)]
    pub(crate) source_max_chars: Option<usize>,
    #[serde(default = "default_enable_source_lookup")]
    pub(crate) enable_source_lookup: bool,
}

impl Default for ReflectTuningContract {
    fn default() -> Self {
        Self {
            max_iterations: default_reflect_max_iterations(),
            max_tokens: None,
            source_limit: default_source_limit(),
            source_max_chars: None,
            enable_source_lookup: default_enable_source_lookup(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct RetrievalTuningContract {
    #[serde(default = "default_retriever_limit")]
    pub(crate) retriever_limit: usize,
    #[serde(default = "default_max_facts")]
    pub(crate) max_facts: usize,
}

impl Default for RetrievalTuningContract {
    fn default() -> Self {
        Self {
            retriever_limit: default_retriever_limit(),
            max_facts: default_max_facts(),
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct ResolvedLocomoContract {
    pub(crate) benchmark: &'static str,
    pub(crate) schema_version: u32,
    pub(crate) protocol_version: String,
    pub(crate) dataset_identifier: String,
    pub(crate) dataset_fingerprint: String,
    pub(crate) expected_dataset_fingerprint: Option<String>,
    pub(crate) category_filter: Vec<u8>,
    pub(crate) conversations: Vec<String>,
    pub(crate) session_limit: Option<usize>,
    pub(crate) question_limit: Option<usize>,
    pub(crate) ingest: LocomoIngestMode,
    pub(crate) image_policy: &'static str,
    pub(crate) consolidation: LocomoConsolidationMode,
    pub(crate) determinism_requirement: Option<DeterminismRequirement>,
    pub(crate) runtime: RuntimeContract,
    pub(crate) judge: JudgeContract,
    pub(crate) judge_prompt_hash: String,
}
