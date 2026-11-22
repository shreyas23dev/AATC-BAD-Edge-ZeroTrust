//! Per-device sliding window buffer for flow-level feature aggregation.
//!
//! For each device (keyed by IP string), we maintain a ring buffer of the
//! last WINDOW_SIZE raw feature frames. Before calling ONNX inference, we
//! aggregate the window into [mean_44, std_44] = 88-dimensional input.
//!
//! This matches the Python `compute_window_features()` in `retrain_and_export.py`
//! exactly (population std, ddof=0).

use std::collections::VecDeque;
use crate::inference::FEATURE_DIM;

/// Number of frames per device in the sliding window.
/// Must match WINDOW_SIZE in retrain_and_export.py.
pub const WINDOW_SIZE: usize = 16;

/// Windowed model input dimension: mean (44) + std (44).
pub const MODEL_INPUT_DIM: usize = FEATURE_DIM * 2;

/// Ring buffer storing the last WINDOW_SIZE raw frames for one device.
pub struct DeviceWindow {
    frames: VecDeque<[f32; FEATURE_DIM]>,
}

impl DeviceWindow {
    pub fn new() -> Self {
        Self {
            frames: VecDeque::with_capacity(WINDOW_SIZE + 1),
        }
    }

    /// Push a new raw frame, evicting the oldest if the buffer is full.
    pub fn push(&mut self, frame: [f32; FEATURE_DIM]) {
        self.frames.push_back(frame);
        if self.frames.len() > WINDOW_SIZE {
            self.frames.pop_front();
        }
    }

    /// Compute [mean_44 | std_44] from the current window contents.
    ///
    /// Single-frame window → std = 0 (cold-start behaviour matches Python).
    pub fn aggregate(&self) -> [f32; MODEL_INPUT_DIM] {
        let n = self.frames.len() as f32;
        let mut means = [0f32; FEATURE_DIM];
        let mut stds  = [0f32; FEATURE_DIM];

        // ── Pass 1: means ────────────────────────────────────────────────────
        for frame in &self.frames {
            for (i, &v) in frame.iter().enumerate() {
                means[i] += v / n;
            }
        }

        // ── Pass 2: population std (ddof=0) ──────────────────────────────────
        if self.frames.len() > 1 {
            let mut variance = [0f32; FEATURE_DIM];
            for frame in &self.frames {
                for (i, &v) in frame.iter().enumerate() {
                    let diff = v - means[i];
                    variance[i] += diff * diff / n;
                }
            }
            for (s, &var) in stds.iter_mut().zip(variance.iter()) {
                *s = var.sqrt();
            }
        }

        // ── Concatenate [mean | std] ──────────────────────────────────────────
        let mut result = [0f32; MODEL_INPUT_DIM];
        result[..FEATURE_DIM].copy_from_slice(&means);
        result[FEATURE_DIM..].copy_from_slice(&stds);
        result
    }

    pub fn len(&self) -> usize {
        self.frames.len()
    }
}

impl Default for DeviceWindow {
    fn default() -> Self { Self::new() }
}
