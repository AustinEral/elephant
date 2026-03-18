//! Core types for the memory engine.

pub mod bank;
pub mod entity;
pub mod fact;
pub mod graph;
pub mod id;
pub mod llm;
pub mod network;
pub mod pipeline;
pub mod source;
pub mod temporal;

pub use bank::{Disposition, MemoryBank};
pub use entity::{Entity, EntityType};
pub use fact::{Fact, FactType, RetrievalSource, ScoredFact};
pub use graph::{GraphLink, LinkType};
pub use id::{BankId, EntityId, FactId, SourceId, TurnId};
pub use llm::{
    AnthropicPromptCacheConfig, AnthropicPromptCacheTtl, CompletionRequest, CompletionResponse,
    Message, OpenAiPromptCacheConfig, OpenAiPromptCacheRetention, PromptCacheConfig,
    PromptCacheUsage,
};
pub use network::NetworkType;
pub use pipeline::{
    AssembledContext, BankPromptContext, Chunk, ChunkConfig, ConsolidationReport, ExtractedFact,
    ExtractionInput, FactFilter, OpinionMergeReport, RecallQuery, RecallResult, ReflectDoneTrace,
    ReflectQuery, ReflectResult, ReflectTraceStep, ResolvedEntity, RetainInput, RetainOutput,
    RetrievedFact, RetrievedSource,
};
pub use source::{FactSource, FactSourceLookup, Source};
pub use temporal::TemporalRange;
