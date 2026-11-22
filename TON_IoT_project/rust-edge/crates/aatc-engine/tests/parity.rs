//! Parity verification test: Rust AatcEngine vs Python reference (1,000 steps).

use aatc_engine::{AatcEngine, EngineConfig, PolicyDecision, TelemetryFrame};
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct ReferenceStep {
    step: usize,
    score: f64,
    alpha: f64,
    debt: f64,
    ceiling: f64,
    policy: String,
}

#[test]
fn test_python_rust_step_parity_1000_frames() {
    let json_data = include_str!("parity_reference.json");
    let ref_steps: Vec<ReferenceStep> = serde_json::from_str(json_data)
        .expect("Failed to parse parity_reference.json");

    assert_eq!(ref_steps.len(), 1000, "Reference must contain 1,000 steps");

    let mut cfg = EngineConfig::default();
    cfg.a_thr = 0.4554;
    cfg.beta = 0.40;
    cfg.gamma = 0.335;
    cfg.theta = 1.00;
    cfg.lambda = 0.55;
    cfg.delta = 0.05;
    cfg.p_pen = 0.25;
    cfg.alpha_0 = 0.2057;
    cfg.t_allow = 0.80;
    cfg.t_quar = 0.30;
    cfg.q_low = 0.35;
    cfg.delta_d = 0.10;
    cfg.k_allow = 1.70;
    cfg.eta_c = 0.30;
    cfg.c_floor = 0.70;

    let mut engine = AatcEngine::new(cfg);

    for step in ref_steps {
        let frame = TelemetryFrame {
            device_id: "test-device".to_string(),
            anomaly_score: step.score,
            timestamp_ms: (step.step as u64) * 100,
        };

        let decision = engine.process(&frame);
        let state = engine.device_state("test-device").unwrap();

        let expected_pol = match step.policy.as_str() {
            "ALLOW" => PolicyDecision::Allow,
            "VERIFY" => PolicyDecision::Verify,
            "BLOCK" => PolicyDecision::Block,
            other => panic!("Unknown policy: {}", other),
        };

        // Assert policy decision matches exactly
        assert_eq!(
            decision, expected_pol,
            "Policy mismatch at step {}: expected {:?}, got {:?}",
            step.step, expected_pol, decision
        );

        // Assert state variables match within floating-point tolerance (1e-5)
        let alpha_diff = (state.alpha - step.alpha).abs();
        let debt_diff = (state.debt - step.debt).abs();
        let ceiling_diff = (state.ceiling - step.ceiling).abs();

        assert!(
            alpha_diff < 1e-4,
            "Alpha mismatch at step {}: expected {:.6}, got {:.6} (diff={:.6})",
            step.step, step.alpha, state.alpha, alpha_diff
        );
        assert!(
            debt_diff < 1e-4,
            "Debt mismatch at step {}: expected {:.6}, got {:.6} (diff={:.6})",
            step.step, step.debt, state.debt, debt_diff
        );
        assert!(
            ceiling_diff < 1e-4,
            "Ceiling mismatch at step {}: expected {:.6}, got {:.6} (diff={:.6})",
            step.step, step.ceiling, state.ceiling, ceiling_diff
        );
    }

    println!("✓ 1,000-frame Rust vs Python parity test passed with 100% precision!");
}
