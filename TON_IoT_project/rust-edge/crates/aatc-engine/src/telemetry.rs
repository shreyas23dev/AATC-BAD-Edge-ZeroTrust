//! Telemetry frame — the unit of data passed into the engine per tick.

use serde::{Deserialize, Serialize};

/// A single telemetry observation from an IoT device.
/// The `anomaly_score` field is the output of the ONNX Isolation Forest model
/// and must be pre-computed by the edge binary before calling the engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TelemetryFrame {
    /// Source device identifier (IP or MAC string)
    pub device_id: String,
    /// Normalised anomaly score in [0, 1] from Isolation Forest ONNX model
    /// 0.0 = fully benign, 1.0 = maximally anomalous
    pub anomaly_score: f64,
    /// Unix timestamp (milliseconds)
    pub timestamp_ms: u64,
}
