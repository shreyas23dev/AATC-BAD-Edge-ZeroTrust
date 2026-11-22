//! Frozen parameter configuration — matches Table 1 of the paper exactly.

use serde::{Deserialize, Serialize};

/// All model parameters. Load from `config.json` at runtime or use defaults.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineConfig {
    // ── Baseline B0 parameters ──────────────────────────────────────────────
    /// Anomaly threshold from Isolation Forest calibration on Edge-IIoTset
    pub a_thr: f64,
    /// Trust decay rate (exponential smoothing on trust)
    pub beta: f64,
    /// Trust recovery rate (benign evidence)
    pub gamma: f64,
    /// Penalty scaling factor
    pub theta: f64,
    /// Smoothing coefficient for anomaly score EMA
    pub lambda: f64,
    /// Baseline trust floor delta
    pub delta: f64,
    /// Penalty magnitude per ALLOW under anomaly
    pub p_pen: f64,
    /// Initial trust value for new devices
    pub alpha_0: f64,
    /// Trust threshold for ALLOW decision
    pub t_allow: f64,
    /// Trust threshold for QUARANTINE decision
    pub t_quar: f64,

    // ── AATC-BAD extension parameters ──────────────────────────────────────
    /// Borderline-low threshold for debt accumulation
    pub q_low: f64,
    /// Debt increment per borderline ALLOW event
    pub delta_d: f64,
    /// Maximum cumulative debt before ceiling is permanently suppressed
    pub k_allow: f64,
    /// Ceiling decay rate per ALLOW event under debt
    pub eta_c: f64,
    /// Hard floor for dynamic recovery ceiling
    pub c_floor: f64,
}

impl Default for EngineConfig {
    /// Frozen paper values — do not change without re-running experiments.
    fn default() -> Self {
        Self {
            // Recalibrated from 0.4554 (paper) to 0.7042 to match the
            // range-normalised score_samples from the ONNX pipeline.
            // See model_meta.json → score_norm for derivation.
            a_thr:   0.7042,
            beta:    0.40,
            gamma:   0.335,
            theta:   1.0,
            lambda:  0.55,
            delta:   0.05,
            p_pen:   0.25,
            alpha_0: 0.2057,
            t_allow: 0.80,
            t_quar:  0.30,
            q_low:   0.35,
            delta_d: 0.10,
            k_allow: 1.70,
            eta_c:   0.30,
            c_floor: 0.70,
        }
    }
}
