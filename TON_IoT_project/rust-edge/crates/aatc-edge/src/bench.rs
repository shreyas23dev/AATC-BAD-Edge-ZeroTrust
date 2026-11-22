//! Throughput benchmark mode.
//!
//! Generates synthetic windowed feature vectors (88-dim = [mean_44 | std_44])
//! and measures raw engine throughput (frames/sec) with ONNX inference.

use std::time::Instant;

use anyhow::Result;
use tracing::info;

use aatc_engine::{AatcEngine, EngineConfig, TelemetryFrame};

use crate::inference::{OnnxModel, FEATURE_DIM};
use crate::window::{DeviceWindow, MODEL_INPUT_DIM};

pub async fn run(config: EngineConfig, model_path: std::path::PathBuf, frames: u64) -> Result<()> {
    info!("Loading ONNX model from {}", model_path.display());
    let mut model = OnnxModel::load(&model_path)?;
    let mut engine = AatcEngine::new(config);

    // Synthetic raw frames — 44-dim; we push through DeviceWindow to get 88-dim
    let raw_benign:    [f32; FEATURE_DIM] = [0.1f32; FEATURE_DIM];
    let raw_anomalous: [f32; FEATURE_DIM] = [0.8f32; FEATURE_DIM];

    // Pre-compute stable windowed feature vectors for benchmark
    let mut win = DeviceWindow::new();
    for _ in 0..16 { win.push(raw_benign); }
    let benign_win: [f32; MODEL_INPUT_DIM] = win.aggregate();

    let mut win2 = DeviceWindow::new();
    for _ in 0..16 { win2.push(raw_anomalous); }
    let anomalous_win: [f32; MODEL_INPUT_DIM] = win2.aggregate();

    info!("Warming up (10,000 frames)...");
    for i in 0..10_000u64 {
        let feats = if i % 11 == 0 { &anomalous_win } else { &benign_win };
        let score = model.score(feats)?;
        engine.process(&TelemetryFrame {
            device_id:    format!("dev-{}", i % 500),
            anomaly_score: score,
            timestamp_ms:  i * 100,
        });
    }
    engine = AatcEngine::new(EngineConfig::default()); // reset after warmup

    info!("Benchmarking {frames} frames...");
    let start = Instant::now();

    for i in 0..frames {
        let feats = if i % 11 == 0 { &anomalous_win } else { &benign_win };
        let score = model.score(feats)?;
        engine.process(&TelemetryFrame {
            device_id:    format!("dev-{}", i % 500),
            anomaly_score: score,
            timestamp_ms:  i * 100,
        });
    }

    let elapsed = start.elapsed();
    let fps     = frames as f64 / elapsed.as_secs_f64();
    let ns_per  = elapsed.as_nanos() as f64 / frames as f64;

    info!("── Benchmark Results ─────────────────────────────────────────────");
    info!("  Frames      : {frames}");
    info!("  Elapsed     : {:.3?}", elapsed);
    info!("  Throughput  : {fps:.0} frames/sec");
    info!("  Latency     : {ns_per:.1} ns/frame  (88-dim ONNX + trust update)");

    Ok(())
}
