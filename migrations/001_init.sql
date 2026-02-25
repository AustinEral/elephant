CREATE EXTENSION IF NOT EXISTS vector;

-- Memory banks: isolated memory stores with personality configuration.
CREATE TABLE memory_banks (
    id            UUID PRIMARY KEY,
    name          TEXT NOT NULL,
    mission       TEXT NOT NULL DEFAULT '',
    directives    JSONB NOT NULL DEFAULT '[]',
    skepticism    SMALLINT NOT NULL DEFAULT 3,
    literalism    SMALLINT NOT NULL DEFAULT 3,
    empathy       SMALLINT NOT NULL DEFAULT 3,
    bias_strength REAL NOT NULL DEFAULT 0.5,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Facts: the atomic unit of memory.
CREATE TABLE facts (
    id              UUID PRIMARY KEY,
    bank_id         UUID NOT NULL REFERENCES memory_banks(id),
    content         TEXT NOT NULL,
    fact_type       TEXT NOT NULL,
    network         TEXT NOT NULL,
    entity_ids      JSONB NOT NULL DEFAULT '[]',
    temporal_start  TIMESTAMPTZ,
    temporal_end    TIMESTAMPTZ,
    embedding       vector(384),
    confidence      REAL,
    evidence_ids    JSONB NOT NULL DEFAULT '[]',
    source_turn_id  UUID,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_facts_bank_id ON facts(bank_id);
CREATE INDEX idx_facts_bank_network ON facts(bank_id, network);
CREATE INDEX idx_facts_temporal ON facts(temporal_start, temporal_end);
CREATE INDEX idx_facts_entity_ids ON facts USING GIN(entity_ids);

-- Entities: named entities referenced by facts.
CREATE TABLE entities (
    id              UUID PRIMARY KEY,
    bank_id         UUID NOT NULL REFERENCES memory_banks(id),
    canonical_name  TEXT NOT NULL,
    aliases         JSONB NOT NULL DEFAULT '[]',
    entity_type     TEXT NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE(bank_id, canonical_name)
);

CREATE INDEX idx_entities_aliases ON entities USING GIN(aliases);

-- Graph links: directed edges in the fact knowledge graph.
CREATE TABLE graph_links (
    source_id  UUID NOT NULL REFERENCES facts(id),
    target_id  UUID NOT NULL REFERENCES facts(id),
    link_type  TEXT NOT NULL,
    weight     REAL NOT NULL DEFAULT 1.0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (source_id, target_id, link_type)
);

CREATE INDEX idx_graph_links_source ON graph_links(source_id);
CREATE INDEX idx_graph_links_target ON graph_links(target_id);
