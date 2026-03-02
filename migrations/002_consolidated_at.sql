ALTER TABLE facts ADD COLUMN IF NOT EXISTS consolidated_at TIMESTAMPTZ;
CREATE INDEX IF NOT EXISTS idx_facts_unconsolidated
    ON facts(bank_id) WHERE consolidated_at IS NULL;
