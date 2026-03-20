# Memory Engine - Rust Implementation

## Project

Rust reimplementation of the Hindsight memory architecture (arXiv:2512.12818). Four-network memory (world, experience, observation, opinion), three operations (retain, recall, reflect), TEMPR retrieval with RRF fusion, CARA preference-conditioned reasoning.

Architecture reference: see `docs/architecture.md`

Agent reading list: see `docs/agent-reading.md`

Before starting any work, agents should read the full Rust API Guidelines:
https://rust-lang.github.io/api-guidelines/

## Hindsight Memory (MCP)

A Hindsight MCP server is available for project memory. Bank ID: `memory-engine`.

Use the MCP tools directly — no curl commands needed.

### When to RETAIN (save to memory)

Retain after any of these events using `mcp__hindsight__retain` with a clear, standalone summary:

- Completing a significant component or milestone
- Making a design decision or changing an interface
- Discovering something that didn't work and why
- Changing a dependency or tool choice
- Finishing a significant debugging session
- Any moment where future-you would want context on what happened

Call: `mcp__hindsight__retain(content="CLEAR STANDALONE SUMMARY", bank_id="memory-engine")`

The summary should be self-contained. Bad: "Fixed the bug." Good: "Fixed entity resolver duplicate creation bug. When resolving multiple mentions of the same entity in a single batch, the resolver was creating a new entity for each mention instead of deduplicating within the batch. Added a local cache keyed by canonical name that persists for the duration of a single resolve() call. Test added: resolve(['Postgres', 'Postgres', 'pg']) now correctly returns one entity."

### When to RECALL (retrieve from memory)

Recall at the start of any session or when starting work on a component:

Call: `mcp__hindsight__recall(query="RELEVANT QUERY", bank_id="memory-engine")`

### When to REFLECT (ask a question about history)

Use reflect for "why" questions about past decisions:

Call: `mcp__hindsight__reflect(query="QUESTION", bank_id="memory-engine")`

### Failure handling

If a memory MCP call fails, tell the user briefly so they know memory isn't accumulating. Don't retry, don't debug it — just flag it and keep working.

## Code Conventions

- Rust 2021 edition
- Use `thiserror` for error types, `anyhow` in tests
- `#[async_trait]` for async trait definitions
- All public interfaces defined as traits for testability
- Tests use `testcontainers` for Postgres, mock LLM/embedding clients for unit tests
- `#[ignore]` on tests that call real LLM APIs
- Prompts live in `prompts/` as plain text files, not in Rust source
- One crate, modules matching the current architecture boundaries
