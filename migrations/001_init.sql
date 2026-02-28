CREATE EXTENSION IF NOT EXISTS vector;

-- Memory banks: isolated memory stores with personality configuration.
CREATE TABLE IF NOT EXISTS memory_banks (
    id            UUID PRIMARY KEY,
    name          TEXT NOT NULL,
    mission       TEXT NOT NULL DEFAULT '',
    directives    JSONB NOT NULL DEFAULT '[]',
    skepticism    SMALLINT NOT NULL DEFAULT 3,
    literalism    SMALLINT NOT NULL DEFAULT 3,
    empathy       SMALLINT NOT NULL DEFAULT 3,
    bias_strength REAL NOT NULL DEFAULT 0.5,
    embedding_model TEXT NOT NULL DEFAULT '',
    embedding_dims SMALLINT NOT NULL DEFAULT 0,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Facts: the atomic unit of memory.
CREATE TABLE IF NOT EXISTS facts (
    id              UUID PRIMARY KEY,
    bank_id         UUID NOT NULL REFERENCES memory_banks(id),
    content         TEXT NOT NULL,
    fact_type       TEXT NOT NULL,
    network         TEXT NOT NULL,
    entity_ids      JSONB NOT NULL DEFAULT '[]',
    temporal_start  TIMESTAMPTZ,
    temporal_end    TIMESTAMPTZ,
    embedding       vector,
    confidence      REAL,
    evidence_ids    JSONB NOT NULL DEFAULT '[]',
    source_turn_id  UUID,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_facts_bank_id ON facts(bank_id);
CREATE INDEX IF NOT EXISTS idx_facts_bank_network ON facts(bank_id, network);
CREATE INDEX IF NOT EXISTS idx_facts_temporal ON facts(temporal_start, temporal_end);
CREATE INDEX IF NOT EXISTS idx_facts_entity_ids ON facts USING GIN(entity_ids);
CREATE INDEX IF NOT EXISTS idx_facts_fts ON facts USING GIN(to_tsvector('english', content));

-- Entities: named entities referenced by facts.
CREATE TABLE IF NOT EXISTS entities (
    id              UUID PRIMARY KEY,
    bank_id         UUID NOT NULL REFERENCES memory_banks(id),
    canonical_name  TEXT NOT NULL,
    aliases         JSONB NOT NULL DEFAULT '[]',
    entity_type     TEXT NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE(bank_id, canonical_name)
);

CREATE INDEX IF NOT EXISTS idx_entities_aliases ON entities USING GIN(aliases);

-- Graph links: directed edges in the fact knowledge graph.
CREATE TABLE IF NOT EXISTS graph_links (
    source_id  UUID NOT NULL REFERENCES facts(id),
    target_id  UUID NOT NULL REFERENCES facts(id),
    link_type  TEXT NOT NULL,
    weight     REAL NOT NULL DEFAULT 1.0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (source_id, target_id, link_type)
);

CREATE INDEX IF NOT EXISTS idx_graph_links_source ON graph_links(source_id);
CREATE INDEX IF NOT EXISTS idx_graph_links_target ON graph_links(target_id);
