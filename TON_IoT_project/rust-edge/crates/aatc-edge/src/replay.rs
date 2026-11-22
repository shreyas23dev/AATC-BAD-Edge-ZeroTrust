//! CSV replay mode — AATC-BAD v2 with per-device flow-level windowing and Single-Pass XAI.
//!
//! Modes:
//!   - Standard: Isolation Forest (model_pipeline.onnx)
//!   - XAI: Single-Pass INT8/f32 Autoencoder (autoencoder_int8.onnx) emitting Phi_t
//!
//! Each row's raw 44 features are pushed into a per-device DeviceWindow.
//! The window aggregates to [mean_44 | std_44] = 88-dim input for ONNX.

use std::collections::HashMap;
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use serde::Serialize;
use tracing::{info, warn};

use aatc_engine::{AatcEngine, EngineConfig, TelemetryFrame};

use crate::inference::{OnnxModel, FEATURE_DIM};
use crate::window::DeviceWindow;
use crate::xai_inference::AeXaiModel;

/// One output record per CSV row (Audit Tuple R_t).
#[derive(Serialize)]
struct DecisionRecord {
    row:           u64,
    device_id:     String,
    anomaly_score: f64,
    trust:         f64,
    debt:          f64,
    ceiling:       f64,
    decision:      String,
    timestamp_ms:  u64,
    window_len:    usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    phi:           Option<Vec<f32>>,
}

/// Score normalisation constants + feature list from model_meta.json or xai_meta.json.
struct ModelMeta {
    feature_indices: Vec<usize>,
    score_min:       f32,
    score_max:       f32,
    a_thr:           f64,
    r_thr:           f64,
    window_size:     usize,
}

fn load_meta(model_path: &PathBuf, headers: &csv::StringRecord, meta_override: Option<&PathBuf>, is_xai: bool) -> ModelMeta {
    let col_idx: HashMap<&str, usize> = headers.iter()
        .enumerate()
        .map(|(i, h)| (h.trim(), i))
        .collect();

    let meta_file = if is_xai { "xai_meta.json" } else { "model_meta.json" };
    let meta_path = if let Some(p) = meta_override {
        Some(p.clone())
    } else {
        model_path.parent()
            .and_then(|p| {
                let direct = p.join(meta_file);
                if direct.exists() { return Some(direct); }
                p.parent().map(|pp| pp.join(meta_file)).filter(|pp| pp.exists())
            })
    };

    let Some(mp) = meta_path else {
        warn!("{} not found — using first {} columns and default config", meta_file, FEATURE_DIM);
        return ModelMeta {
            feature_indices: (0..FEATURE_DIM).collect(),
            score_min: -0.659_714,
            score_max: -0.404_530,
            a_thr: if is_xai { 0.0017 } else { 0.7042 },
            r_thr: 0.0040,
            window_size: 16,
        };
    };

    let text = std::fs::read_to_string(&mp).expect("read metadata json");
    let v: serde_json::Value = serde_json::from_str(&text).expect("parse metadata json");

    let feature_cols: Vec<String> = v["feature_cols"]
        .as_array().unwrap_or(&vec![])
        .iter()
        .filter_map(|x| x.as_str().map(str::to_owned))
        .collect();

    let feature_indices: Vec<usize> = feature_cols.iter().map(|c| {
        col_idx.get(c.trim()).copied().unwrap_or_else(|| {
            warn!("Feature '{}' not in CSV headers — using 0", c);
            0
        })
    }).collect();

    let a_thr = if is_xai {
        v["a_thr_ae"].as_f64().unwrap_or(0.0017)
    } else {
        v["score_norm"]["a_thr_recalibrated"].as_f64().unwrap_or(0.7042)
    };

    let score_min = v["score_norm"]["score_min"].as_f64().unwrap_or(-0.6597) as f32;
    let score_max = v["score_norm"]["score_max"].as_f64().unwrap_or(-0.4045) as f32;
    let window_size = v["window_size"].as_u64().unwrap_or(16) as usize;
    let r_thr = v["r_thr"].as_f64()
        .or_else(|| v["quantization"]["calibrated_mse_threshold"].as_f64())
        .unwrap_or(0.0040);

    info!("Metadata loaded: {} features, window={}, a_thr={:.6}, r_thr={:.6} (XAI={})",
        feature_indices.len(), window_size, a_thr, r_thr, is_xai);

    ModelMeta { feature_indices, score_min, score_max, a_thr, r_thr, window_size }
}

pub async fn run(
    mut config:    EngineConfig,
    csv_path:      PathBuf,
    model_path:    PathBuf,
    output:        Option<PathBuf>,
    max_rows:      Option<u64>,
    meta_override: Option<PathBuf>,
    xai:           bool,
) -> Result<()> {
    info!("Starting replay mode (Single-Pass XAI: {})", xai);
    info!("Loading model from: {}", model_path.display());

    let mut reader = csv::ReaderBuilder::new()
        .has_headers(true)
        .from_path(&csv_path)
        .with_context(|| format!("Cannot open CSV: {}", csv_path.display()))?;
    let headers = reader.headers()?.clone();

    let meta = load_meta(&model_path, &headers, meta_override.as_ref(), xai);

    // Override A_thr threshold for the chosen model (use frozen paper A_thr for XAI)
    let chosen_a_thr = if xai { 0.4554 } else { meta.a_thr };
    config.a_thr = chosen_a_thr;
    info!("AATC-BAD Engine A_thr set to: {:.4}", config.a_thr);

    let device_col = headers.iter()
        .position(|h| h.trim() == "ip.src_host" || h.trim() == "src_ip")
        .unwrap_or(0);

    info!("CSV: {} | max_rows: {:?}", csv_path.display(), max_rows);

    let mut engine = AatcEngine::new(config);
    let mut windows: HashMap<String, DeviceWindow> = HashMap::new();

    // Model variant initialization
    let mut if_model = if !xai {
        Some(OnnxModel::load_with_norm(&model_path, meta.score_min, meta.score_max)?)
    } else {
        None
    };

    let mut ae_xai_model = if xai {
        Some(AeXaiModel::load(&model_path, chosen_a_thr, meta.r_thr)?)
    } else {
        None
    };

    let output_sink: Box<dyn Write> = match output {
        Some(ref path) => Box::new(BufWriter::new(
            std::fs::File::create(path)
                .with_context(|| format!("Cannot create output file: {}", path.display()))?,
        )),
        None => Box::new(BufWriter::new(std::io::stdout())),
    };
    let mut sink = output_sink;

    let start = Instant::now();
    let mut row_idx: u64 = 0;
    let mut allow  = 0u64;
    let mut verify = 0u64;
    let mut block  = 0u64;

    for record in reader.records() {
        let record = match record {
            Ok(r)  => r,
            Err(e) => { warn!("Skipping malformed row {}: {}", row_idx, e); continue; }
        };

        row_idx += 1;
        if let Some(limit) = max_rows {
            if row_idx > limit { break; }
        }

        // Feature extraction
        let mut raw_frame = [0f32; FEATURE_DIM];
        for (slot, &csv_idx) in meta.feature_indices.iter().enumerate() {
            if slot >= FEATURE_DIM { break; }
            raw_frame[slot] = record.get(csv_idx)
                .and_then(|v| v.trim().parse::<f32>().ok())
                .unwrap_or(0.0);
        }

        let device_id = record.get(device_col)
            .unwrap_or("unknown")
            .trim()
            .to_string();

        // Sliding window aggregation
        let win = windows.entry(device_id.clone()).or_default();
        win.push(raw_frame);
        let win_len = win.len();
        let agg = win.aggregate();   // [mean_44 | std_44] = 88-dim

        // Model inference & attribution
        let (anomaly_score, phi) = if let Some(ref mut ae) = ae_xai_model {
            let res = ae.score_and_explain(&agg).unwrap_or_else(|_| crate::xai_inference::XaiResult {
                anomaly_score: 0.5,
                phi: [0.0; crate::window::MODEL_INPUT_DIM],
            });
            (res.anomaly_score, Some(res.phi.to_vec()))
        } else if let Some(ref mut if_m) = if_model {
            let score = if_m.score(&agg).unwrap_or(0.5);
            (score, None)
        } else {
            (0.5, None)
        };

        // AATC-BAD Trust Engine
        let frame = TelemetryFrame {
            device_id:    device_id.clone(),
            anomaly_score,
            timestamp_ms: row_idx * 100,
        };
        let decision = engine.process(&frame);

        let state = engine.device_state(&device_id).unwrap();
        let rec = DecisionRecord {
            row: row_idx,
            device_id,
            anomaly_score,
            trust:    state.alpha,
            debt:     state.debt,
            ceiling:  state.ceiling,
            decision: decision.to_string(),
            timestamp_ms: frame.timestamp_ms,
            window_len: win_len,
            phi,
        };

        writeln!(sink, "{}", serde_json::to_string(&rec)?)?;

        match decision {
            aatc_engine::PolicyDecision::Allow  => allow  += 1,
            aatc_engine::PolicyDecision::Verify => verify += 1,
            aatc_engine::PolicyDecision::Block  => block  += 1,
        }
    }

    let elapsed = start.elapsed();
    let fps = row_idx as f64 / elapsed.as_secs_f64();

    info!("── Replay complete (Single-Pass XAI: {}) ──────────────────────────", xai);
    info!("  Frames processed : {row_idx}");
    info!("  ALLOW            : {allow}");
    info!("  VERIFY           : {verify}");
    info!("  BLOCK            : {block}");
    info!("  Devices tracked  : {}", engine.device_count());
    info!("  Window size      : {} frames", meta.window_size);
    info!("  Elapsed          : {:.3?}", elapsed);
    info!("  Throughput       : {fps:.0} frames/sec");

    Ok(())
}
