//! Single-Pass XAI Autoencoder inference using `ort` 2.0.0-rc.15.
//!
//! Input: 88-dimensional windowed features [mean_44 | std_44] computed by
//! `window::DeviceWindow::aggregate()`.
//!
//! Outputs (simultaneously in 1 forward pass):
//!   [0] anomaly_score: scalar reconstruction MSE across all 88 features [1, 1]
//!   [1] phi          : per-feature squared reconstruction errors [1, 88]
//!
//! Zero-Sampling XAI:
//!   No sampling, no perturbations, and no tree traversals are performed.
//!   Marginal computational cost of attribution vector Phi_t is ZERO.

use anyhow::{Context, Result};
use ort::{environment::Environment, session::Session, value::TensorRef};
use std::path::Path;
use std::sync::Arc;

use crate::window::MODEL_INPUT_DIM;

/// Simultaneous outputs from single-pass XAI model.
pub struct XaiResult {
    /// Normalised anomaly score in [0, 1] (higher = more anomalous).
    pub anomaly_score: f64,
    /// Per-feature attribution vector Phi_t (squared reconstruction error per feature).
    pub phi: [f32; MODEL_INPUT_DIM],
}

/// Loaded ONNX Runtime session for Single-Pass Autoencoder XAI.
pub struct AeXaiModel {
    session:     Session,
    _env:        Arc<Environment>,
    r_thr:       f64,
    paper_a_thr: f64,
}

impl AeXaiModel {
    pub fn load(path: &Path, a_thr: f64, r_thr: f64) -> Result<Self> {
        let env = Arc::new(
            ort::init()
                .with_name("aatc-bad-xai")
                .build()
                .context("Failed to build ORT environment for XAI")?,
        );
        let session = Session::builder(&env)
            .context("Failed to create ORT session builder for XAI")?
            .commit_from_file(path)
            .context("Failed to load Autoencoder XAI ONNX model")?;
        Ok(Self {
            session,
            _env: env,
            r_thr: if r_thr > 0.0 { r_thr } else { 0.0040 },
            paper_a_thr: a_thr,
        })
    }

    /// Single-pass forward inference:
    /// Returns both A_t and Phi_t simultaneously.
    pub fn score_and_explain(&mut self, features: &[f32; MODEL_INPUT_DIM]) -> Result<XaiResult> {
        let shape = vec![1i64, MODEL_INPUT_DIM as i64];
        let input = TensorRef::<f32>::from_array_view((shape, features.as_slice()))
            .context("Failed to create input TensorRef for XAI")?;

        let outputs = self
            .session
            .run(ort::inputs!["float_input" => input])
            .context("XAI ONNX inference failed")?;

        // Output [0] = anomaly_score (f32, shape [1, 1])
        let (_, raw_scores) = outputs[0]
            .try_extract_tensor::<f32>()
            .context("Failed to extract anomaly_score from XAI model")?;
        let raw_score = raw_scores.iter().next().copied().unwrap_or(0.0_f32);

        // Output [1] = phi (f32, shape [1, 88])
        let (_, raw_phi) = outputs[1]
            .try_extract_tensor::<f32>()
            .context("Failed to extract phi attribution from XAI model")?;

        let mut phi = [0f32; MODEL_INPUT_DIM];
        for (i, val) in raw_phi.iter().take(MODEL_INPUT_DIM).enumerate() {
            phi[i] = *val;
        }

        // Dynamically scale calibrated reconstruction error MSE to match frozen paper A_thr:
        let norm_score = ((raw_score as f64 / self.r_thr) * self.paper_a_thr).clamp(0.0, 1.0);

        Ok(XaiResult {
            anomaly_score: norm_score,
            phi,
        })
    }
}
