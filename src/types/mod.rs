//! Core types for the memory engine.

pub mod bank;
pub mod entity;
pub mod fact;
pub mod graph;
pub mod id;
pub mod llm;
pub mod network;
pub mod pipeline;
pub mod temporal;

pub use bank::{Disposition, MemoryBank};
pub use entity::{Entity, EntityType};
pub use fact::{Fact, FactType, RetrievalSource, ScoredFact};
pub use graph::{GraphLink, LinkType};
pub use id::{BankId, EntityId, FactId, TurnId};
pub use llm::{CompletionRequest, CompletionResponse, Message};
pub use network::NetworkType;
pub use pipeline::{
    AssembledContext, BankPromptContext, Chunk, ChunkConfig, ConsolidationReport, ExtractedFact,
    ExtractionInput, FactFilter, OpinionMergeReport, RecallQuery, RecallResult, ReflectQuery,
    ReflectResult, ReflectTraceStep, ResolvedEntity, RetainInput, RetainOutput, RetrievedFact,
};
pub use temporal::TemporalRange;
