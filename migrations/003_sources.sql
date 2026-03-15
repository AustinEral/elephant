-- Provenance sources: raw source units that facts can be traced back to.
CREATE TABLE IF NOT EXISTS sources (
    id          UUID PRIMARY KEY,
    bank_id     UUID NOT NULL REFERENCES memory_banks(id),
    content     TEXT NOT NULL,
    context     TEXT,
    speaker     TEXT,
    rendered_input TEXT,
    timestamp   TIMESTAMPTZ NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

ALTER TABLE sources ADD COLUMN IF NOT EXISTS context TEXT;
ALTER TABLE sources ADD COLUMN IF NOT EXISTS speaker TEXT;
ALTER TABLE sources ADD COLUMN IF NOT EXISTS rendered_input TEXT;

CREATE INDEX IF NOT EXISTS idx_sources_bank_timestamp ON sources(bank_id, timestamp);

-- Deduplicated fact-to-source provenance links.
CREATE TABLE IF NOT EXISTS fact_sources (
    fact_id     UUID NOT NULL REFERENCES facts(id) ON DELETE CASCADE,
    source_id   UUID NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (fact_id, source_id)
);

CREATE INDEX IF NOT EXISTS idx_fact_sources_source_id ON fact_sources(source_id);
