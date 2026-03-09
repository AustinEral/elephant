//! Reset consolidation state for banks in a benchmark results file.
//!
//! Deletes all Observation facts and sets consolidated_at = NULL on remaining facts,
//! so the next bench run with --reuse re-consolidates from scratch.
//!
//! Usage:
//!     cargo run --bin reset-consolidation -- <results.json>

use std::collections::HashMap;
use std::env;
use std::fs;

use serde::Deserialize;

#[derive(Deserialize)]
struct BenchmarkOutput {
    #[serde(rename = "bank_ids", alias = "banks", default)]
    banks: HashMap<String, String>,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let _ = dotenvy::dotenv();

    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: reset-consolidation <results.json>");
        std::process::exit(1);
    }

    let path = &args[1];
    let raw = fs::read_to_string(path)?;
    let output: BenchmarkOutput = serde_json::from_str(&raw)?;

    if output.banks.is_empty() {
        eprintln!("No banks found in {path}");
        std::process::exit(1);
    }

    let database_url = env::var("DATABASE_URL")?;
    let pool = sqlx::PgPool::connect(&database_url).await?;

    let bank_ids: Vec<uuid::Uuid> = output
        .banks
        .values()
        .filter_map(|id| id.parse::<ulid::Ulid>().ok())
        .map(|u| uuid::Uuid::from(u))
        .collect();

    println!("Resetting consolidation for {} banks...", bank_ids.len());

    // Delete all observation facts
    let deleted =
        sqlx::query("DELETE FROM facts WHERE network = 'observation' AND bank_id = ANY($1)")
            .bind(&bank_ids)
            .execute(&pool)
            .await?;
    println!("  Deleted {} observations", deleted.rows_affected());

    // Reset consolidated_at on remaining facts
    let reset = sqlx::query("UPDATE facts SET consolidated_at = NULL WHERE bank_id = ANY($1) AND consolidated_at IS NOT NULL")
        .bind(&bank_ids)
        .execute(&pool)
        .await?;
    println!("  Reset consolidated_at on {} facts", reset.rows_affected());

    // Delete orphaned graph links (referencing deleted observation facts)
    let links = sqlx::query("DELETE FROM graph_links WHERE source_id NOT IN (SELECT id FROM facts) OR target_id NOT IN (SELECT id FROM facts)")
        .execute(&pool)
        .await?;
    if links.rows_affected() > 0 {
        println!("  Cleaned {} orphaned links", links.rows_affected());
    }

    println!("Done. Run bench with --reuse to re-consolidate.");
    Ok(())
}
