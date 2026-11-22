//! AATC-BAD Trust Engine — pure Rust implementation
//!
//! Implements the two-state-variable countermeasure from:
//!   "When Trust Recovery Creates Moral Hazard: Strategic Evasion and
//!    Policy-Layer Defense in Zero-Trust IoT"
//!
//! Frozen parameters (Table 1 of the paper):
//!   A_thr=0.4554, beta=0.40, gamma=0.335, theta=1.0, lambda=0.55,
//!   delta=0.05,   P_pen=0.25, alpha_0=0.2057, T_allow=0.80, T_quar=0.30
//!   AATC-BAD: q_low=0.35, delta_D=0.10, K_allow=1.70, eta_C=0.30, C_floor=0.70

pub mod config;
pub mod engine;
pub mod policy;
pub mod telemetry;

pub use config::EngineConfig;
pub use engine::{AatcEngine, DeviceState};
pub use policy::PolicyDecision;
pub use telemetry::TelemetryFrame;
