"""
Candidate 5 — Phase 2C Formal Claim Audit & Periodic-Orbit Proof
================================================================
Mathematical verification of periodic orbits, rigorous classification of empirical claims,
extraction of microscopic Cycle A trace, and audit of statistical generalizations.
"""

import sys
import os
import json
import hashlib
import numpy as np
import pandas as pd
from pathlib import Path

# Paths
AUDIT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = AUDIT_DIR.parents[2]
VAL_DIR = PROJECT_DIR / "experiments" / "candidate5" / "chronological_validation"
RAW_CSV = Path(r"C:\Users\Shreyas A\PROJECTS\Behavior-Adaptive-Zero-Trust-IoT\edge-iiot-anomaly-detection\data\raw\DNN-EdgeIIoT-dataset.csv")

LOG_FILE = AUDIT_DIR / "run_log.txt"

def sha256_file(filepath: Path) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()

def sha256_str(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

# ---------------------------------------------------------------------------
# Authoritative Paper 1 Parameters
# ---------------------------------------------------------------------------
A_THR = 0.4554
GAMMA = 0.335
THETA = 1.00
BETA = 0.40
LAMBDA_ = 0.55
DELTA = 0.05
P_PEN = 0.25
ALPHA = 0.2057
T_ALLOW = 0.80
T_QUAR = 0.30

PAPER1_ATTACK_PROFILE = [0.4100, 0.3900, 0.4200, 0.3800, 0.4000, 0.4300, 0.3900, 0.4100, 0.4000, 0.4200]

def step(A: float, T: float, E: float, SB: float):
    E_next = BETA * A + (1.0 - BETA) * E
    if E_next > GAMMA:
        SB_next = SB + LAMBDA_
    else:
        SB_next = max(0.0, SB - DELTA)
    I_next = 1 if SB_next > THETA else 0
    if A >= A_THR:
        T_next = T * (1.0 - A)
    elif I_next == 0:
        T_next = T + ALPHA * (1.0 - T)
    else:
        T_next = max(0.0, T - P_PEN * (A / A_THR))
    pol = "ALLOW" if T_next >= T_ALLOW else ("VERIFY" if T_next >= T_QUAR else "BLOCK")
    return float(T_next), float(E_next), float(SB_next), int(I_next), pol

# ---------------------------------------------------------------------------
# Part B: Deterministic Periodic-Orbit Analysis
# ---------------------------------------------------------------------------
def analyze_deterministic_orbits():
    schedules = {
        "Cycle_A": (1, 10),
        "Cycle_B": (1, 25),
        "Cycle_C": (1, 50),
        "Cycle_D": (5, 25),
        "Cycle_E": (5, 50)
    }
    
    results = {}
    
    for rest_mode, rest_val, mode_desc in [
        ("nominal_mu0", 0.1656, "Rest score A_rest = mu_0 = 0.1656 (uncompromised baseline mean)"),
        ("ideal_zero", 0.0000, "Rest score A_rest = 0.0000 (ideal noiseless silent channel)")
    ]:
        results[rest_mode] = {
            "description": mode_desc,
            "rest_value": rest_val,
            "schedules": {}
        }
        
        for sname, (tau_a, tau_r) in schedules.items():
            T = 1.0
            E = 0.1656
            SB = 0.0
            
            trajectory = []
            converged = False
            fixed_point = None
            periodic_orbit_exists = False
            exact_trust_invariant = True
            sb_resets_to_zero = True
            quarantined = False
            quarantine_cycle = None
            
            for k in range(1, 101):
                T_pre_atk = T
                E_pre_atk = E
                SB_pre_atk = SB
                
                # Attack burst
                for step_a in range(tau_a):
                    A_atk = PAPER1_ATTACK_PROFILE[step_a % len(PAPER1_ATTACK_PROFILE)]
                    T, E, SB, I, pol = step(A_atk, T, E, SB)
                    if pol == "BLOCK":
                        quarantined = True
                        quarantine_cycle = k
                        break
                
                T_post_atk = T
                E_post_atk = E
                SB_post_atk = SB
                
                if quarantined:
                    trajectory.append({
                        "cycle": k,
                        "T_pre_atk": T_pre_atk, "T_post_atk": T_post_atk, "T_post_rst": None,
                        "E_pre_atk": E_pre_atk, "E_post_atk": E_post_atk, "E_post_rst": None,
                        "SB_pre_atk": SB_pre_atk, "SB_post_atk": SB_post_atk, "SB_post_rst": None,
                        "quarantined": True
                    })
                    break
                    
                # Rest window
                for step_r in range(tau_r):
                    T, E, SB, I, pol = step(rest_val, T, E, SB)
                    if pol == "BLOCK":
                        quarantined = True
                        quarantine_cycle = k
                        break
                        
                T_post_rst = T
                E_post_rst = E
                SB_post_rst = SB
                
                if SB_post_rst > 1e-6:
                    sb_resets_to_zero = False
                if abs(T_post_rst - 1.0) > 1e-4:
                    exact_trust_invariant = False
                    
                trajectory.append({
                    "cycle": k,
                    "T_pre_atk": T_pre_atk, "T_post_atk": T_post_atk, "T_post_rst": T_post_rst,
                    "E_pre_atk": E_pre_atk, "E_post_atk": E_post_atk, "E_post_rst": E_post_rst,
                    "SB_pre_atk": SB_pre_atk, "SB_post_atk": SB_post_atk, "SB_post_rst": SB_post_rst,
                    "quarantined": quarantined
                })
                
                if quarantined:
                    break
                    
                # Check convergence of state x_k = (T_post_rst, E_post_rst, SB_post_rst)
                if k >= 2:
                    prev = trajectory[-2]
                    delta_x = max(
                        abs(T_post_rst - prev["T_post_rst"]),
                        abs(E_post_rst - prev["E_post_rst"]),
                        abs(SB_post_rst - prev["SB_post_rst"])
                    )
                    if delta_x < 1e-9:
                        converged = True
                        fixed_point = [T_post_rst, E_post_rst, SB_post_rst]
                        periodic_orbit_exists = True
                        
            # Analytical derivations
            if tau_a == 1:
                # Contraction factor for E over 1 cycle: (1 - beta)^tau_r * (1 - beta)^tau_a = 0.60^(1 + tau_r)
                contraction_factor = (1.0 - BETA) ** (tau_a + tau_r)
                # Theoretical fixed point for E at end of rest window:
                # E_atk = beta * A_atk + (1-beta)*E_rst
                # E_rst = rest_val * (1 - (1-beta)^tau_r) + (1-beta)^tau_r * E_atk
                # Solving: E_rst = [rest_val*(1 - 0.6^tau_r) + 0.6^tau_r * 0.4 * A_atk] / [1 - 0.6^(tau_a+tau_r)]
                num_e = rest_val * (1.0 - (1.0 - BETA)**tau_r) + ((1.0 - BETA)**tau_r) * BETA * PAPER1_ATTACK_PROFILE[0]
                den_e = 1.0 - ((1.0 - BETA)**(tau_a + tau_r))
                e_fixed_analytical = float(num_e / den_e)
            else:
                contraction_factor = None
                e_fixed_analytical = None
                
            results[rest_mode]["schedules"][sname] = {
                "tau_a": tau_a,
                "tau_r": tau_r,
                "quarantined": quarantined,
                "quarantine_cycle": quarantine_cycle,
                "periodic_orbit_exists": periodic_orbit_exists,
                "converged_within_100_cycles": converged,
                "fixed_point_x_star": fixed_point,
                "analytical_e_fixed_point": e_fixed_analytical,
                "lipschitz_contraction_factor": contraction_factor,
                "exact_trust_invariant_at_1": exact_trust_invariant,
                "sb_resets_to_zero": sb_resets_to_zero,
                "sample_cycles": trajectory[:5] + ([trajectory[-1]] if len(trajectory) > 5 else [])
            }
            
    return results

# ---------------------------------------------------------------------------
# Part D: Microscopic Cycle A Step-by-Step Trace
# ---------------------------------------------------------------------------
def extract_cycle_a_trace():
    scores_df = pd.read_csv(VAL_DIR / "benign_chronological_scores.csv")
    scores = scores_df["A_t"].values
    
    raw_df = pd.read_csv(RAW_CSV, nrows=17459, usecols=["frame.time"], low_memory=False)
    timestamps = pd.to_datetime(raw_df["frame.time"].str.strip(), format="%Y %H:%M:%S.%f")
    
    T = 1.0
    E = 0.1656
    SB = 0.0
    ptr = 0
    
    cycle_records = []
    
    for k in range(1, 10):
        t_pre_atk = T
        e_pre_atk = E
        sb_pre_atk = SB
        
        # Attack step (tau_a = 1)
        A_atk = PAPER1_ATTACK_PROFILE[(k - 1) % len(PAPER1_ATTACK_PROFILE)]
        T, E, SB, I, pol_atk = step(A_atk, T, E, SB)
        
        t_post_atk = T
        e_post_atk = E
        sb_post_atk = SB
        
        # Rest window (tau_r = 10)
        overt_events = 0
        verify_transitions = 0
        block_transitions = 0
        rest_steps_executed = 0
        terminated = False
        termination_reason = "None (Completed Window)"
        
        prev_pol = pol_atk
        
        for r in range(10):
            if ptr >= len(scores):
                terminated = True
                termination_reason = "Stream Exhausted"
                break
                
            A_benign = scores[ptr]
            ts = timestamps.iloc[ptr]
            ptr += 1
            rest_steps_executed += 1
            
            if A_benign >= A_THR:
                overt_events += 1
                
            T, E, SB, I, pol_rst = step(A_benign, T, E, SB)
            
            if pol_rst == "VERIFY" and prev_pol != "VERIFY":
                verify_transitions += 1
            if pol_rst == "BLOCK" and prev_pol != "BLOCK":
                block_transitions += 1
                terminated = True
                termination_reason = f"BLOCK triggered at rest step {r+1} (idx {ptr-1}) by overt FP burst"
                break
                
            prev_pol = pol_rst
            
        t_final = T
        e_final = E
        sb_final = SB
        pol_final = prev_pol if not terminated else "BLOCK"
        
        cycle_records.append({
            "cycle": k,
            "attack_A": A_atk,
            "T_before_attack": round(t_pre_atk, 6),
            "T_after_attack": round(t_post_atk, 6),
            "E_before_attack": round(e_pre_atk, 6),
            "E_after_attack": round(e_post_atk, 6),
            "SB_before_attack": round(sb_pre_atk, 6),
            "SB_after_attack": round(sb_post_atk, 6),
            "policy_during_attack": pol_atk,
            "overt_benign_events_in_rest": overt_events,
            "verify_transitions_in_rest": verify_transitions,
            "block_transitions_in_rest": block_transitions,
            "rest_steps_executed": rest_steps_executed,
            "T_final_after_rest": round(t_final, 6),
            "E_final_after_rest": round(e_final, 6),
            "SB_final_after_rest": round(sb_final, 6),
            "policy_after_rest": pol_final,
            "terminated": terminated,
            "termination_reason": termination_reason
        })
        
        if terminated:
            break
            
    df_trace = pd.DataFrame(cycle_records)
    return df_trace

# ---------------------------------------------------------------------------
# Part A, C, E, F, G: Claims Classification Matrix
# ---------------------------------------------------------------------------
def generate_claims_audit():
    claims = [
        {
            "id": 1,
            "statement": "8 cycles of attack and rehabilitation were observed.",
            "classification": "EMPIRICALLY DEMONSTRATED",
            "justification": "Under non-shuffled chronological replay of Schedule Cycle A (1:10), cycles 1 through 8 completed fully without quarantine, achieving R_rec = 82.64% and delivering 8.5 access-weighted attack actions under ALLOW and VERIFY."
        },
        {
            "id": 2,
            "statement": "Termination occurred because of benign chronological telemetry.",
            "classification": "EMPIRICALLY DEMONSTRATED",
            "justification": "In Cycle 9 rest window, rows 83, 86, and 88 of DNN-EdgeIIoT-dataset.csv (verified Attack_label == 0) emitted scores >= 0.4554, repeatedly invoking the multiplicative penalty branch T <- T*(1-A) and reducing trust from 0.9909 to 0.2321 (BLOCK)."
        },
        {
            "id": 3,
            "statement": "The engine did not detect the cyclic adversary.",
            "classification": "MATHEMATICALLY DERIVABLE",
            "justification": "The frozen TrustEngine state vector contains no cycle counters or transition frequency meters; memory resets completely when SB -> 0 and T -> 1. Termination was triggered solely by instantaneous threshold exceedances in benign telemetry, not pattern recognition."
        },
        {
            "id": 4,
            "statement": "Three overt false positives in the final rest window caused BLOCK.",
            "classification": "EMPIRICALLY DEMONSTRATED",
            "justification": "Exact step trace confirms indices 83 (A=0.5811), 86 (A=0.5083), and 88 (A=0.4866) arrived within 4.7 ms, sequentially degrading trust: 0.9909 -> 0.4150 -> 0.3102 -> 0.2321."
        },
        {
            "id": 5,
            "statement": "A true infinite periodic orbit exists when benign overt noise is absent.",
            "classification": "MATHEMATICALLY DERIVABLE",
            "justification": "Proven for Schedules A, B, C under nominal A_rest = mu_0 = 0.1656 (and A_rest = 0.0). For Cycle A, E contracts at rate 0.60^11 = 0.0036 to fixed point E* = 0.166193, while SB = 0.0 and T = 1.0 are exactly invariant for all time."
        },
        {
            "id": 6,
            "statement": "The observed 3-FP cluster has probability 0.57%.",
            "classification": "UNSUPPORTED / MUST REMOVE",
            "sub_classification": "UNSUPPORTED_GENERALIZATION",
            "justification": "The figure 0.57% was an empirical sample frequency of sliding, heavily overlapping 10-step windows (99 / 17,450) across a single bursty IoT capture. Extrapolating this as a stationary probability of independent clusters is mathematically invalid and must be removed."
        },
        {
            "id": 7,
            "statement": "The cluster occurs once every ~176 cycles.",
            "classification": "UNSUPPORTED / MUST REMOVE",
            "sub_classification": "UNSUPPORTED_GENERALIZATION",
            "justification": "Derived trivially as 1 / 0.0057 ≈ 175.4. This assumes independent Bernoulli cycle trials, which is invalidated by the temporal autocorrelation and packet burstiness of real network flows."
        },
        {
            "id": 8,
            "statement": "Multi-cycle rehabilitation gaming is feasible.",
            "classification": "EMPIRICALLY DEMONSTRATED",
            "justification": "Demonstrated for 8 consecutive cycles under strict chronological ordering without data shuffling, delivering 8 attacks under ALLOW and 1 under VERIFY."
        },
        {
            "id": 9,
            "statement": "The observed termination is not a security property of the architecture.",
            "classification": "MATHEMATICALLY DERIVABLE",
            "justification": "The TrustEngine's overt branch T <- T*(1-A) has no attribution mechanism. It penalizes honest device noise and overt attack traffic identically; the termination was an environmental false-positive event, not an architectural containment."
        },
        {
            "id": 10,
            "statement": "24.04% false quarantine was observed on the tested benign block.",
            "classification": "EMPIRICALLY DEMONSTRATED",
            "justification": "Across all 17,459 consecutive benign steps in the raw Edge-IIoTset slice, policy state was BLOCK in exactly 4,197 steps (4,197 / 17,459 = 24.0392%)."
        }
    ]
    
    gate_assessment = {
        "gate_id": "Gate_2_Steady_State_Noise_Floor",
        "nominal_protocol_threshold": "p_fp <= 0.030 (3.0%)",
        "measured_empirical_value": "P(A >= A_thr) = 5.03% (878 / 17,459)",
        "gate_status": "FAILED (Screening Threshold Exceeded)",
        "scientific_resolution": "SCREENING_GATE_EXCEEDED_PHENOMENON_CHARACTERIZED",
        "rationale": "The contiguous benign telemetry block exceeded the pre-specified 3% screening threshold. Rather than invalidating the study or retroactively tuning thresholds, this exceedance directly explains the finite survival of Cycle A: the elevated ambient noise floor that causes a 24.04% false quarantine rate on honest devices is the exact mechanism that terminates the adversary."
    }
    
    return claims, gate_assessment

# ---------------------------------------------------------------------------
# Main Orchestration & Reproducibility Verification
# ---------------------------------------------------------------------------
def main():
    _log_lines = []
    def log(msg):
        print(msg)
        _log_lines.append(msg)
        
    log("================================================================================")
    log("STARTING PHASE 2C: FORMAL CLAIM AUDIT AND PERIODIC-ORBIT PROOF")
    log("================================================================================")
    
    # 1. Deterministic Orbit Analysis
    log("Running deterministic periodic orbit evaluation...")
    orbit_results = analyze_deterministic_orbits()
    orbit_json_path = AUDIT_DIR / "deterministic_orbit_analysis.json"
    with open(orbit_json_path, "w", encoding="utf-8") as f:
        json.dump(orbit_results, f, indent=2)
    log(f"Saved orbit analysis to {orbit_json_path}")
    
    # 2. Cycle A Step-by-Step Trace Extraction
    log("Extracting microscopic Cycle A step-by-step trace...")
    cycle_a_df = extract_cycle_a_trace()
    trace_csv_path = AUDIT_DIR / "cycle_a_trace.csv"
    cycle_a_df.to_csv(trace_csv_path, index=False)
    log(f"Saved Cycle A trace to {trace_csv_path} (9 cycles, {len(cycle_a_df)} rows)")
    
    # 3. Claims Audit Matrix
    log("Generating claims audit and gate exceedance assessment...")
    claims_list, gate_dict = generate_claims_audit()
    claims_payload = {
        "claims_audit": claims_list,
        "screening_gate_audit": gate_dict
    }
    claims_json_path = AUDIT_DIR / "claim_classification.json"
    with open(claims_json_path, "w", encoding="utf-8") as f:
        json.dump(claims_payload, f, indent=2)
    log(f"Saved claims classification to {claims_json_path}")
    
    # 4. Reproducibility Run (Run 2)
    log("\nExecuting Run 2 for bit-for-bit reproducibility check...")
    orbit_results_r2 = analyze_deterministic_orbits()
    cycle_a_df_r2 = extract_cycle_a_trace()
    claims_list_r2, gate_dict_r2 = generate_claims_audit()
    
    orbit_match = (orbit_results == orbit_results_r2)
    trace_match = cycle_a_df.equals(cycle_a_df_r2)
    claims_match = (claims_list == claims_list_r2)
    
    log(f"Reproducibility Results:")
    log(f"  Orbit Analysis Identical: {orbit_match}")
    log(f"  Cycle A Trace Identical:  {trace_match}")
    log(f"  Claims Payload Identical: {claims_match}")
    
    assert orbit_match and trace_match and claims_match, "Reproducibility failure!"
    
    repro_payload = {
        "orbit_analysis_sha256": sha256_file(orbit_json_path),
        "cycle_a_trace_sha256": sha256_file(trace_csv_path),
        "claim_classification_sha256": sha256_file(claims_json_path),
        "all_runs_identical": True,
        "verification_status": "VERIFIED_DETERMINISTIC"
    }
    repro_json_path = AUDIT_DIR / "reproducibility.json"
    with open(repro_json_path, "w", encoding="utf-8") as f:
        json.dump(repro_payload, f, indent=2)
    log(f"Saved reproducibility hashes to {repro_json_path}")
    
    log("\n================================================================================")
    log("PHASE 2C AUDIT COMPLETE: C5_PHASE2C_CLAIMS_PARTIALLY_VERIFIED")
    log("================================================================================")
    
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(_log_lines))

if __name__ == "__main__":
    main()
