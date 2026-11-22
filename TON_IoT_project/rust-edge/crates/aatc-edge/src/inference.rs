//! ONNX Isolation Forest inference using `ort` 2.0.0-rc.15.
//!
//! Input: 88-dimensional windowed features [mean_44 | std_44] computed by
//! `window::DeviceWindow::aggregate()` — do NOT pass raw 44-dim features.
//!
//! Score normalisation:
//!   anomaly = (score_max - raw) / (score_max - score_min)
//!   Constants are loaded from model_meta.json → score_norm at startup.

use anyhow::{Context, Result};
use ort::{environment::Environment, session::Session, value::TensorRef};
use std::path::Path;
use std::sync::Arc;

use crate::window::MODEL_INPUT_DIM;

/// Raw per-frame feature count (used by DeviceWindow buffer).
pub const FEATURE_DIM: usize = 44;

/// Loaded ONNX Runtime session ready for inference.
pub struct OnnxModel {
    session:   Session,
    _env:      Arc<Environment>,
    score_min: f32,
    score_max: f32,
}

impl OnnxModel {
    pub fn load(path: &Path) -> Result<Self> {
        Self::load_with_norm(path, -0.659_714, -0.404_530)
    }

    /// Load with explicit normalisation constants (from model_meta.json).
    pub fn load_with_norm(path: &Path, score_min: f32, score_max: f32) -> Result<Self> {
        let env = Arc::new(
            ort::init()
                .with_name("aatc-bad")
                .build()
                .context("Failed to build ORT environment")?,
        );
        let session = Session::builder(&env)
            .context("Failed to create ORT session builder")?
            .commit_from_file(path)
            .context("Failed to load ONNX model")?;
        Ok(Self { session, _env: env, score_min, score_max })
    }

    /// Run inference on 88-dim windowed features.
    /// Returns normalised anomaly score in [0, 1] (higher = more anomalous).
    pub fn score(&mut self, features: &[f32; MODEL_INPUT_DIM]) -> Result<f64> {
        let shape = vec![1i64, MODEL_INPUT_DIM as i64];
        let input = TensorRef::<f32>::from_array_view((shape, features.as_slice()))
            .context("Failed to create input TensorRef")?;

        let outputs = self
            .session
            .run(ort::inputs!["float_input" => input])
            .context("ORT inference failed")?;

        // Output [2] = score_samples (f32, shape [1,1])
        // More negative = more anomalous (shorter isolation path).
        let (_, scores) = outputs[2]
            .try_extract_tensor::<f32>()
            .context("Failed to extract score_samples")?;

        let raw   = scores.iter().next().copied().unwrap_or(self.score_max);
        let range = (self.score_max - self.score_min) as f64;
        let norm  = ((self.score_max as f64 - raw as f64) / range).clamp(0.0, 1.0);
        Ok(norm)
    }
}
