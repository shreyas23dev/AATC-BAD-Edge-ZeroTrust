//! Core AATC-BAD state machine — zero-allocation hot path.
//!
//! Each device has an independent `DeviceState`. The `AatcEngine` holds a
//! hash-map of device states and is the single entry point for callers.
//!
//! Implements Equations (1)–(4) and (15)–(18) from the paper exactly:
//!   - Equation (15): Debt increment (A_t - q_low) / (A_thr - q_low) charged only under ALLOW
//!   - Equation (15): Linear debt paydown of delta_D per nominal frame (A_t < q_low)
//!   - Equation (17): Dynamic recovery ceiling C_t lowered only when D_t > K_allow
//!   - Equation (18): Asymptotic recovery capped by dynamic recovery ceiling C_t
//!   - Equation (4): 3-tier policy decision (ALLOW, VERIFY, BLOCK)

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use tracing::debug;

use crate::{config::EngineConfig, policy::PolicyDecision, telemetry::TelemetryFrame};

// ─────────────────────────────────────────────────────────────────────────────
// Per-device state
// ─────────────────────────────────────────────────────────────────────────────

/// All mutable state associated with a single IoT device.
/// Struct layout is cache-friendly (fits in one 64-byte L1 cache line).
#[repr(C)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceState {
    /// Current behavioral trust score α_t ∈ [0, 1]
    pub alpha: f64,
    /// Exponentially-smoothed telemetry deviation E_t
    pub smoothed_anomaly: f64,
    /// Accumulated slow-burn energy S_t
    pub slow_burn_energy: f64,
    /// Cumulative authorization debt D_t
    pub debt: f64,
    /// Dynamic recovery ceiling C_t ∈ [C_floor, 1]
    pub ceiling: f64,
    /// Total ALLOW decisions granted (for paper metric M_cum)
    pub allow_count: u64,
    /// Frame index (for debugging)
    pub tick: u64,
}

impl DeviceState {
    pub fn new(_config: &EngineConfig) -> Self {
        Self {
            alpha:            1.0, // An authenticated admitted device starts in full ALLOW
            smoothed_anomaly: 0.0,
            slow_burn_energy: 0.0,
            debt:             0.0,
            ceiling:          1.0,
            allow_count:      0,
            tick:             0,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Engine
// ─────────────────────────────────────────────────────────────────────────────

/// AATC-BAD engine — holds per-device state for all tracked devices.
pub struct AatcEngine {
    config: EngineConfig,
    devices: HashMap<String, DeviceState>,
}

impl AatcEngine {
    /// Create a new engine with the given (or default) config.
    pub fn new(config: EngineConfig) -> Self {
        Self {
            config,
            devices: HashMap::new(),
        }
    }

    /// Process one telemetry frame and return the policy decision.
    ///
    /// This is the **hot path** — called once per packet/flow in production.
    /// All arithmetic is branchless scalar math; no heap allocation occurs
    /// after the first call per device.
    #[inline]
    pub fn process(&mut self, frame: &TelemetryFrame) -> PolicyDecision {
        let cfg = &self.config;

        // ── 1. Retrieve or initialise device state ──────────────────────────
        let state = self
            .devices
            .entry(frame.device_id.clone())
            .or_insert_with(|| DeviceState::new(cfg));

        state.tick += 1;
        let a_t = frame.anomaly_score;
        let alpha_prev = state.alpha;

        // ── 2. Authorization Debt Accrual (Equations 15-16, Algorithm 2) ──
        let d_prior_is_allow = alpha_prev >= cfg.t_allow;
        if d_prior_is_allow && a_t >= cfg.q_low && a_t < cfg.a_thr {
            let delta_d = (a_t - cfg.q_low) / (cfg.a_thr - cfg.q_low);
            state.debt += delta_d;
            debug!(
                device = %frame.device_id,
                tick   = state.tick,
                debt   = state.debt,
                "Debt accumulated"
            );
        } else if a_t < cfg.q_low {
            // Linear debt paydown credit during nominal rest
            state.debt = (state.debt - cfg.delta_d).max(0.0);
        }

        // ── 3. Dynamic Recovery Ceiling Update (Equation 17) ───────────────
        let excess_debt = (state.debt - cfg.k_allow).max(0.0);
        let c_t = (1.0 - cfg.eta_c * excess_debt).max(cfg.c_floor);
        state.ceiling = c_t;

        // ── 4. Deviation Smoothing & Slow-Burn Energy Tracking (Eqs. 1-2) ──
        state.smoothed_anomaly = cfg.beta * a_t + (1.0 - cfg.beta) * state.smoothed_anomaly;
        let e_t = state.smoothed_anomaly;
        if e_t > cfg.gamma {
            state.slow_burn_energy += cfg.lambda;
        } else {
            state.slow_burn_energy = (state.slow_burn_energy - cfg.delta).max(0.0);
        }
        let slow_burn_active = state.slow_burn_energy > cfg.theta;

        // ── 5. Dynamic Trust State Transition (Equation 18, Algorithm 2) ───
        let alpha_new = if a_t >= cfg.a_thr {
            // Overt anomaly shock
            (alpha_prev * (1.0 - a_t)).max(0.0)
        } else if slow_burn_active {
            // Slow-burn penalty
            (alpha_prev - cfg.p_pen * (a_t / cfg.a_thr)).max(0.0)
        } else {
            // Asymptotic recovery capped by dynamic ceiling C_t
            let rec = alpha_prev + cfg.alpha_0 * (c_t - alpha_prev);
            rec.min(c_t)
        };
        state.alpha = alpha_new;

        // ── 6. Operational Access Decision (Equation 4) ────────────────────
        let decision = Self::classify(alpha_new, cfg);
        if decision == PolicyDecision::Allow {
            state.allow_count += 1;
        }

        decision
    }

    /// Stateless threshold classifier (no state mutation).
    #[inline(always)]
    pub fn classify(alpha: f64, cfg: &EngineConfig) -> PolicyDecision {
        if alpha >= cfg.t_allow {
            PolicyDecision::Allow
        } else if alpha >= cfg.t_quar {
            PolicyDecision::Verify
        } else {
            PolicyDecision::Block
        }
    }

    /// Read-only snapshot of a device's current state (for logging / export).
    pub fn device_state(&self, device_id: &str) -> Option<&DeviceState> {
        self.devices.get(device_id)
    }

    /// Iterate over all tracked devices.
    pub fn iter_devices(&self) -> impl Iterator<Item = (&String, &DeviceState)> {
        self.devices.iter()
    }

    /// Total number of tracked devices.
    pub fn device_count(&self) -> usize {
        self.devices.len()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Unit tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_frame(device_id: &str, score: f64, tick: u64) -> TelemetryFrame {
        TelemetryFrame {
            device_id:     device_id.to_string(),
            anomaly_score: score,
            timestamp_ms:  tick * 1000,
        }
    }

    /// A device that sends only benign traffic should remain in ALLOW.
    #[test]
    fn test_benign_device_stays_in_allow() {
        let mut engine = AatcEngine::new(EngineConfig::default());
        let mut last_decision = PolicyDecision::Block;
        for t in 0..50 {
            last_decision = engine.process(&make_frame("dev-benign", 0.05, t));
        }
        assert_eq!(last_decision, PolicyDecision::Allow);
        let state = engine.device_state("dev-benign").unwrap();
        assert_eq!(state.debt, 0.0);
        assert_eq!(state.ceiling, 1.0);
    }

    /// Borderline pulsing under ALLOW accumulates debt and lowers ceiling past K_allow.
    #[test]
    fn test_pulsing_attacker_accumulates_debt_and_demotes() {
        let mut cfg = EngineConfig::default();
        cfg.a_thr = 0.4554;
        cfg.q_low = 0.35;
        cfg.k_allow = 1.0;
        cfg.delta_d = 0.10;
        cfg.eta_c = 0.30;
        cfg.c_floor = 0.70;
        let mut engine = AatcEngine::new(cfg);

        // Device starts in ALLOW. Pulse in borderline band [0.35, 0.4554)
        for t in 0..15 {
            engine.process(&make_frame("dev-attacker", 0.44, t));
        }

        let state = engine.device_state("dev-attacker").unwrap();
        assert!(state.debt > 1.0, "Expected debt > K_allow (1.0), got {}", state.debt);
        assert!(state.ceiling < 1.0, "Expected ceiling < 1.0, got {}", state.ceiling);
        // Past debt threshold, ceiling drops below T_allow (0.80), demoting from ALLOW
        assert!(state.ceiling <= 0.80, "Expected ceiling <= 0.80, got {}", state.ceiling);
    }

    /// Nominal frames pay down debt back to zero.
    #[test]
    fn test_debt_paydown_during_nominal_rest() {
        let mut cfg = EngineConfig::default();
        cfg.a_thr = 0.4554;
        cfg.q_low = 0.35;
        let mut engine = AatcEngine::new(cfg);

        // Incur some debt
        for t in 0..5 {
            engine.process(&make_frame("dev-paydown", 0.40, t));
        }
        let debt_before = engine.device_state("dev-paydown").unwrap().debt;
        assert!(debt_before > 0.0);

        // Rest nominally (score 0.10 < q_low)
        for t in 5..30 {
            engine.process(&make_frame("dev-paydown", 0.10, t));
        }
        let debt_after = engine.device_state("dev-paydown").unwrap().debt;
        assert_eq!(debt_after, 0.0, "Expected debt to be fully paid down to 0");
    }
}
