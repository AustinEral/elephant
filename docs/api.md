# API

Elephant exposes two interfaces:

- a REST API under `/v1`
- an MCP server under `/mcp`

This document describes the current public surface in the repo today.

## Base URLs

- REST: `http://localhost:3001/v1`
- MCP: `http://localhost:3001/mcp`

## Common Notes

- All REST requests and responses use JSON.
- IDs are opaque string identifiers. Treat them as stable IDs, not user-facing names.
- Most write/query endpoints are bank-scoped under `/v1/banks/{id}/...`.
- For current bank-scoped POST endpoints, include `bank_id` in the JSON body as well as in the path. The handler overwrites it from the path, but the request still must deserialize successfully.
- Timestamps should be RFC 3339 / ISO 8601 UTC strings like `2024-03-01T00:00:00Z`.

## Error Model

Errors are returned as:

```json
{ "error": "..." }
```

Common status classes:

- `400` for invalid IDs, invalid disposition values, and malformed request bodies
- `404` for missing banks or entities
- `409` for embedding dimension mismatches
- `429` for upstream rate limiting
- `502` for LLM / embedding / reranker provider failures
- `500` for storage or internal errors

## REST Endpoints

### `GET /v1/info`

Returns the active model configuration for the running server.

Example response:

```json
{
  "retain_model": "openai/gpt-5.4-mini",
  "reflect_model": "openai/gpt-5.4-mini",
  "embedding_model": "local/bge-small-en-v1.5",
  "reranker_model": "local/ms-marco-MiniLM-L-6-v2"
}
```

### `GET /v1/banks`

Lists all memory banks.

Returns an array of bank objects.

### `POST /v1/banks`

Creates a new memory bank.

Required fields:

- `name`
- `mission`

Optional fields:

- `directives`: array of strings
- `disposition`: object with `skepticism`, `literalism`, `empathy`, `bias_strength`

Example request:

```json
{
  "name": "demo",
  "mission": "Remember user facts, events, and preferences",
  "directives": ["Never fabricate facts"],
  "disposition": {
    "skepticism": 3,
    "literalism": 3,
    "empathy": 3,
    "bias_strength": 0.5
  }
}
```

Example response:

```json
{
  "id": "01KMCM5R8X4T25KEZWYH0D3NMR",
  "name": "demo",
  "mission": "Remember user facts, events, and preferences",
  "directives": ["Never fabricate facts"],
  "disposition": {
    "skepticism": 3,
    "literalism": 3,
    "empathy": 3,
    "bias_strength": 0.5
  },
  "embedding_model": "bge-small-en-v1.5",
  "embedding_dimensions": 384
}
```

### `GET /v1/banks/{id}`

Returns the full bank configuration for one bank.

### `POST /v1/banks/{id}/retain`

Stores raw text in memory by extracting facts, resolving entities, and linking evidence.

Required fields:

- `bank_id`
- `content`
- `timestamp`

Optional fields:

- `turn_id`
- `context`
- `custom_instructions`
- `speaker`

Example request:

```json
{
  "bank_id": "01KMCM5R8X4T25KEZWYH0D3NMR",
  "content": "Alice joined Acme Corp in March 2024. She prefers Rust over Go.",
  "timestamp": "2024-03-01T00:00:00Z",
  "turn_id": null,
  "context": null,
  "custom_instructions": null,
  "speaker": "Alice"
}
```

Example response:

```json
{
  "fact_ids": ["01...", "01..."],
  "facts_stored": 2,
  "new_entities": ["01...", "01..."],
  "entities_resolved": 5,
  "links_created": 1,
  "opinions_reinforced": 0,
  "opinions_weakened": 0
}
```

### `POST /v1/banks/{id}/recall`

Retrieves relevant facts for a query.

Required fields:

- `bank_id`
- `query`
- `budget_tokens`

Optional fields:

- `network_filter`: array of network names: `world`, `experience`, `observation`, `opinion`
- `temporal_anchor`

Example request:

```json
{
  "bank_id": "01KMCM5R8X4T25KEZWYH0D3NMR",
  "query": "Alice employment start date",
  "budget_tokens": 2048,
  "network_filter": null,
  "temporal_anchor": null
}
```

Returns:

- `facts`: scored fact objects
- `total_tokens`: estimated token cost of the returned fact set

### `POST /v1/banks/{id}/reflect`

Answers a question by retrieving and synthesizing stored memory.

Required fields:

- `bank_id`
- `question`
- `budget_tokens`

Optional fields:

- `temporal_context`

Example request:

```json
{
  "bank_id": "01KMCM5R8X4T25KEZWYH0D3NMR",
  "question": "When did Alice join her company?",
  "budget_tokens": 2048,
  "temporal_context": null
}
```

Returns:

- `response`: final answer text
- `sources`: fact IDs cited by the answer
- `new_opinions`: any opinions formed during reflection
- `confidence`: model confidence score
- `retrieved_context`: ranked facts used during reflection
- `retrieved_sources`: source snippets surfaced through source lookup
- `trace`: tool/query trace for the reflect loop
- `final_done`: normalized final completion payload for debugging

This is the richest endpoint in the API and is the best place to inspect how Elephant arrived at an answer.

### `GET /v1/banks/{id}/entities`

Lists resolved entities in a bank.

Each entity includes:

- `id`
- `canonical_name`
- `aliases`
- `entity_type`
- `bank_id`

### `GET /v1/banks/{id}/entities/{eid}/facts`

Returns facts associated with one entity.

Each fact includes its content, type, network, timestamps, evidence IDs, and optional source turn.

### `POST /v1/banks/{id}/consolidate`

Runs observation consolidation for the bank.

Returns:

```json
{
  "observations_created": 0,
  "observations_updated": 0
}
```

### `POST /v1/banks/{id}/merge-opinions`

Runs opinion clustering / merge logic for the bank.

Returns:

```json
{
  "opinions_merged": 0,
  "opinions_superseded": 0,
  "opinions_conflicting": 0
}
```

## MCP

Elephant also exposes an MCP server at `/mcp`.

Current tools:

- `create_bank`
- `list_banks`
- `retain`
- `recall`
- `reflect`

The MCP surface is intentionally narrower than the REST API. It is designed for agent clients that want the core memory workflow without raw maintenance endpoints.

### MCP tool notes

- `create_bank`
  - accepts `bank_id`, optional `name`, optional `mission`
  - if `bank_id` parses as an existing bank ID, the current bank is returned instead of creating a new one
- `retain`
  - accepts `bank_id`, `content`, optional `context`, optional `timestamp`
- `recall`
  - accepts `bank_id`, `query`, optional `max_tokens`
- `reflect`
  - accepts `bank_id`, `query`, optional `context`, `budget` (`low`, `mid`, `high`)

## Related Docs

- [getting-started.md](getting-started.md)
- [architecture.md](architecture.md)
- [../bench/locomo/result-card.md](../bench/locomo/result-card.md)
