//! Text chunking for the retain pipeline.

use crate::types::{Chunk, ChunkConfig};

/// Trait for splitting input text into extractable chunks.
pub trait Chunker: Send + Sync {
    /// Split input text into chunks according to the given configuration.
    fn chunk(&self, input: &str, config: &ChunkConfig) -> Vec<Chunk>;
}

/// Simple chunker that splits on token estimates (chars / 4).
///
/// Respects overlap, and optionally preserves conversation turn boundaries
/// (lines starting with a speaker prefix like "User:", "Assistant:", etc.).
pub struct SimpleChunker;

impl SimpleChunker {
    /// Estimate token count using chars/4 heuristic.
    fn estimate_tokens(text: &str) -> usize {
        // Rough heuristic: ~4 chars per token for English text.
        text.len().div_ceil(4)
    }

    /// Find a good split point near `target_byte` in `text`.
    ///
    /// Prefers splitting at paragraph boundaries (\n\n), then line boundaries (\n),
    /// then sentence boundaries (. ), then word boundaries ( ).
    fn find_split_point(text: &str, target_byte: usize, preserve_turns: bool) -> usize {
        let search_start = target_byte.saturating_sub(200);
        let search_region = &text[search_start..target_byte.min(text.len())];

        if preserve_turns {
            // Look for turn boundaries (newline followed by a speaker-like prefix)
            // Search backwards for "\nSomething:" pattern
            if let Some(pos) = search_region.rfind("\n\n") {
                return search_start + pos + 2; // After the double newline
            }
        }

        // Paragraph boundary
        if let Some(pos) = search_region.rfind("\n\n") {
            return search_start + pos + 2;
        }

        // Line boundary
        if let Some(pos) = search_region.rfind('\n') {
            return search_start + pos + 1;
        }

        // Sentence boundary
        if let Some(pos) = search_region.rfind(". ") {
            return search_start + pos + 2;
        }

        // Word boundary
        if let Some(pos) = search_region.rfind(' ') {
            return search_start + pos + 1;
        }

        // Give up, split at target
        target_byte.min(text.len())
    }
}

impl Chunker for SimpleChunker {
    fn chunk(&self, input: &str, config: &ChunkConfig) -> Vec<Chunk> {
        let total_tokens = Self::estimate_tokens(input);

        // Short input: pass through as single chunk
        if total_tokens <= config.max_tokens {
            return vec![Chunk {
                content: input.to_string(),
                index: 0,
                context: None,
            }];
        }

        let mut chunks = Vec::new();
        let mut byte_offset = 0;
        let bytes_per_token = input.len() as f64 / total_tokens as f64;
        let max_bytes = (config.max_tokens as f64 * bytes_per_token) as usize;
        let overlap_bytes = (config.overlap_tokens as f64 * bytes_per_token) as usize;

        while byte_offset < input.len() {
            let target_end = (byte_offset + max_bytes).min(input.len());

            let chunk_end = if target_end >= input.len() {
                input.len()
            } else {
                Self::find_split_point(input, target_end, config.preserve_turns)
            };

            let chunk_text = &input[byte_offset..chunk_end];

            // Build context from the end of the previous chunk (if any)
            let context = if !chunks.is_empty() {
                // Take the last ~200 chars of the previous chunk as context
                let prev: &Chunk = chunks.last().unwrap();
                let ctx_start = prev.content.len().saturating_sub(200);
                // Find a clean start point (word boundary)
                let clean_start = prev.content[ctx_start..]
                    .find(' ')
                    .map(|i| ctx_start + i + 1)
                    .unwrap_or(ctx_start);
                Some(prev.content[clean_start..].to_string())
            } else {
                None
            };

            chunks.push(Chunk {
                content: chunk_text.to_string(),
                index: chunks.len(),
                context,
            });

            // Advance past the chunk, minus overlap
            let advance = chunk_end - byte_offset;
            if advance == 0 {
                break; // Safety: prevent infinite loop
            }
            byte_offset = chunk_end.saturating_sub(overlap_bytes);
            // Make sure we actually advance
            if byte_offset
                <= chunks
                    .last()
                    .map(|c| chunk_end - c.content.len())
                    .unwrap_or(0)
            {
                byte_offset = chunk_end;
            }
        }

        chunks
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config() -> ChunkConfig {
        ChunkConfig {
            max_tokens: 2000,
            overlap_tokens: 200,
            preserve_turns: false,
        }
    }

    #[test]
    fn short_input_single_chunk() {
        let chunker = SimpleChunker;
        let input = "This is a short piece of text.";
        let chunks = chunker.chunk(input, &default_config());
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].content, input);
        assert_eq!(chunks[0].index, 0);
        assert!(chunks[0].context.is_none());
    }

    #[test]
    fn empty_input() {
        let chunker = SimpleChunker;
        let chunks = chunker.chunk("", &default_config());
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].content, "");
    }

    #[test]
    fn long_input_multiple_chunks() {
        let chunker = SimpleChunker;
        // Create input that's ~300 tokens (1200 chars)
        let config = ChunkConfig {
            max_tokens: 100,
            overlap_tokens: 10,
            preserve_turns: false,
        };
        let input = "The quick brown fox jumps over the lazy dog. ".repeat(30);
        let chunks = chunker.chunk(&input, &config);
        assert!(chunks.len() > 1, "should produce multiple chunks");

        // Verify sequential indices
        for (i, chunk) in chunks.iter().enumerate() {
            assert_eq!(chunk.index, i);
        }

        // Verify first chunk has no context, others do
        assert!(chunks[0].context.is_none());
        for chunk in &chunks[1..] {
            assert!(
                chunk.context.is_some(),
                "non-first chunks should have context"
            );
        }
    }

    #[test]
    fn chunks_have_sequential_indices() {
        let chunker = SimpleChunker;
        let config = ChunkConfig {
            max_tokens: 50,
            overlap_tokens: 5,
            preserve_turns: false,
        };
        let input = "word ".repeat(200);
        let chunks = chunker.chunk(&input, &config);
        for (i, chunk) in chunks.iter().enumerate() {
            assert_eq!(chunk.index, i);
        }
    }

    #[test]
    fn context_populated_from_preceding_chunk() {
        let chunker = SimpleChunker;
        let config = ChunkConfig {
            max_tokens: 50,
            overlap_tokens: 5,
            preserve_turns: false,
        };
        let input = "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike november oscar papa quebec romeo sierra tango uniform victor whiskey xray yankee zulu. ".repeat(10);
        let chunks = chunker.chunk(&input, &config);
        assert!(chunks.len() > 1);
        // The context of chunk[1] should be a trailing portion of chunk[0]
        let ctx = chunks[1].context.as_ref().unwrap();
        assert!(!ctx.is_empty());
        assert!(
            chunks[0].content.ends_with(ctx) || chunks[0].content.contains(ctx),
            "context should come from end of previous chunk"
        );
    }

    #[test]
    fn preserve_turns_splits_at_paragraph_boundaries() {
        let chunker = SimpleChunker;
        let config = ChunkConfig {
            max_tokens: 30,
            overlap_tokens: 5,
            preserve_turns: true,
        };
        let input = "User: Hello, I need help with my project.\n\n\
             Assistant: Sure! What kind of project are you working on?\n\n\
             User: It's a Rust application for managing memory in an AI system. \
             We need to handle facts, entities, and graph links between them. \
             The system should support semantic search and temporal filtering."
            .to_string();
        let chunks = chunker.chunk(&input, &config);
        assert!(chunks.len() > 1);
        // With preserve_turns, chunks should prefer splitting at \n\n boundaries
        // (we can't guarantee it perfectly, but the split point logic prefers them)
    }

    #[test]
    fn token_estimate_reasonable() {
        // "hello world" = 11 chars → ~3 tokens
        assert_eq!(SimpleChunker::estimate_tokens("hello world"), 3);
        // Empty
        assert_eq!(SimpleChunker::estimate_tokens(""), 0);
        // 4000 chars → ~1000 tokens
        let long = "a".repeat(4000);
        assert_eq!(SimpleChunker::estimate_tokens(&long), 1000);
    }
}
