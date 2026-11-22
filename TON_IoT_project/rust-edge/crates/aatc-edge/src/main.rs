//! AATC-BAD Edge Runtime — single binary entry point
//!
//! Modes:
//!   replay  — Run the full evaluation pipeline on a CSV telemetry file
//!             (reproduces the paper's Table 2 / Table 4 results)
//!   bench   — Throughput benchmark: how many frames/sec can the engine handle?
//!
//! Usage:
//!   aatc-edge replay --csv data/DNN-EdgeIIoT-dataset.csv --model model.onnx
//!   aatc-edge bench  --model model.onnx --frames 100000

use std::path::PathBuf;

use anyhow::Result;
use clap::{Parser, Subcommand};
use tracing::info;
use tracing_subscriber::EnvFilter;

mod inference;
mod window;
mod xai_inference;
mod replay;
mod bench;

#[derive(Parser, Debug)]
#[command(
    name    = "aatc-edge",
    version = "0.1.0",
    about   = "AATC-BAD edge runtime: ONNX inference + trust engine + policy enforcement"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    /// Path to engine config JSON (uses frozen paper defaults if omitted)
    #[arg(long, global = true)]
    config: Option<PathBuf>,

    /// Log level: error | warn | info | debug | trace
    #[arg(long, global = true, default_value = "info")]
    log_level: String,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Replay a CSV telemetry file and emit per-frame decisions
    Replay {
        /// Path to Edge-IIoTset CSV
        #[arg(long)]
        csv: PathBuf,
        /// Path to exported Isolation Forest ONNX model
        #[arg(long)]
        model: PathBuf,
        /// Output JSONL file for per-frame decisions (stdout if omitted)
        #[arg(long)]
        output: Option<PathBuf>,
        /// Maximum rows to process (paper evaluation uses 17459; omit for full CSV)
        #[arg(long)]
        max_rows: Option<u64>,
        /// Optional path to metadata JSON
        #[arg(long)]
        meta: Option<PathBuf>,
        /// Enable Single-Pass Autoencoder XAI mode (outputs Phi_t in decision records)
        #[arg(long)]
        xai: bool,
    },
    /// Throughput benchmark
    Bench {
        /// Path to ONNX model
        #[arg(long)]
        model: PathBuf,
        /// Number of synthetic frames to process
        #[arg(long, default_value = "100000")]
        frames: u64,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    // Initialise tracing
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::new(&cli.log_level))
        .init();

    // Load engine config
    let engine_config = match &cli.config {
        Some(path) => {
            let raw = std::fs::read_to_string(path)?;
            serde_json::from_str(&raw)?
        }
        None => {
            info!("No --config supplied; using frozen paper defaults");
            aatc_engine::EngineConfig::default()
        }
    };

    match cli.command {
        Commands::Replay { csv, model, output, max_rows, meta, xai } => {
            replay::run(engine_config, csv, model, output, max_rows, meta, xai).await?;
        }
        Commands::Bench { model, frames } => {
            bench::run(engine_config, model, frames).await?;
        }
    }

    Ok(())
}
