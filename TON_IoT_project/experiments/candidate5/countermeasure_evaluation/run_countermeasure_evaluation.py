"""
Candidate 5 — Phase 4 Empirical Countermeasure Evaluation
=========================================================
Empirical evaluation of Baseline B0, Mechanism M1, Mechanism M2, and Mechanism M3
under non-shuffled chronological replay of Paper 1 sub-threshold attack profiles
into the contiguous Edge-IIoTset benign telemetry stream (N=17,459).
"""

import sys
import os
import json
import hashlib
import numpy as np
import pandas as pd
from pathlib import Path

# Paths
EVAL_DIR = Path(__file__).resolve().parent
PROJECT_DIR = EVAL_DIR.parents[2]
SCORES_CSV = PROJECT_DIR / "experiments" / "candidate5" / "chronological_validation" / "benign_chronological_scores.csv"
LOG_FILE = EVAL_DIR / "run_log.txt"

# ---------------------------------------------------------------------------
# Authoritative Baseline Parameters (Frozen B0)
# ---------------------------------------------------------------------------
A_THR = 0.4554
GAMMA = 0.335
THETA = 1.00
BETA = 0.40
LAMBDA_ = 0.55
DELTA = 0.05
P_PEN = 0.25
ALPHA_0 = 0.2057
T_ALLOW = 0.80
T_QUAR = 0.30

# Attack profile (Paper 1 sub-threshold)
ATK_PROFILE = [0.4100, 0.3900, 0.4200, 0.3800, 0.4000, 0.4300, 0.3900, 0.4100, 0.4000, 0.4200]

# Primary Evaluation Schedules
SCHEDULES = {
    "Cycle_A": (1, 10),
    "Cycle_B": (1, 25),
    "Cycle_C": (1, 50),
    "Cycle_D": (5, 25),
    "Cycle_E": (5, 50)
}

# Mechanism Parameters (from Phase 3 Definitions)
PARAMS_M1 = {"kappa": 0.50, "h_max": 0.80, "delta_h": 0.001}
PARAMS_M2 = {"eta": 0.50, "k_max": 10, "n_clean_req": 100}
PARAMS_M3 = {"H_req": 15}

def sha256_file(filepath: Path) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()

def sha256_obj(obj) -> str:
    s = json.dumps(obj, sort_keys=True)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

# ---------------------------------------------------------------------------
# Step Functions
# ---------------------------------------------------------------------------

def step_b0(A: float, state: dict) -> dict:
    T, E, SB = state["T"], state["E"], state["SB"]
    E_next = BETA * A + (1.0 - BETA) * E
    if E_next > GAMMA:
        SB_next = SB + LAMBDA_
    else:
        SB_next = max(0.0, SB - DELTA)
    I_next = 1 if SB_next > THETA else 0
    if A >= A_THR:
        T_next = T * (1.0 - A)
        branch = "OVERT"
    elif I_next == 0:
        T_next = T + ALPHA_0 * (1.0 - T)
        branch = "BENIGN_RECOVERY"
    else:
        T_next = max(0.0, T - P_PEN * (A / A_THR))
        branch = "SLOWBURN_PENALTY"
    pol = "ALLOW" if T_next >= T_ALLOW else ("VERIFY" if T_next >= T_QUAR else "BLOCK")
    return {"T": float(T_next), "E": float(E_next), "SB": float(SB_next), "I": I_next, "policy": pol, "branch": branch}

def step_m1(A: float, state: dict) -> dict:
    T, E, SB, h = state["T"], state["E"], state["SB"], state["h"]
    E_next = BETA * A + (1.0 - BETA) * E
    if E_next > GAMMA:
        SB_accum = SB + LAMBDA_
        h_next = min(PARAMS_M1["h_max"], max(h, PARAMS_M1["kappa"] * SB_accum))
        SB_next = SB_accum
    else:
        h_next = max(0.0, h - PARAMS_M1["delta_h"])
        SB_next = max(h_next, SB - DELTA)
    I_next = 1 if SB_next > THETA else 0
    if A >= A_THR:
        T_next = T * (1.0 - A)
        branch = "OVERT"
    elif I_next == 0:
        T_next = T + ALPHA_0 * (1.0 - T)
        branch = "BENIGN_RECOVERY"
    else:
        T_next = max(0.0, T - P_PEN * (A / A_THR))
        branch = "SLOWBURN_PENALTY"
    pol = "ALLOW" if T_next >= T_ALLOW else ("VERIFY" if T_next >= T_QUAR else "BLOCK")
    return {"T": float(T_next), "E": float(E_next), "SB": float(SB_next), "h": float(h_next), "I": I_next, "policy": pol, "branch": branch}

def step_m2(A: float, state: dict) -> dict:
    T, E, SB, k_inc, clean_c = state["T"], state["E"], state["SB"], state["k_inc"], state["clean_counter"]
    E_next = BETA * A + (1.0 - BETA) * E
    if E_next > GAMMA:
        SB_next = SB + LAMBDA_
    else:
        SB_next = max(0.0, SB - DELTA)
    I_next = 1 if SB_next > THETA else 0
    alpha_eff = ALPHA_0 / (1.0 + PARAMS_M2["eta"] * k_inc)
    if A >= A_THR:
        T_next = T * (1.0 - A)
        branch = "OVERT"
    elif I_next == 0:
        T_next = T + alpha_eff * (1.0 - T)
        branch = "BENIGN_RECOVERY"
    else:
        T_next = max(0.0, T - P_PEN * (A / A_THR))
        branch = "SLOWBURN_PENALTY"
    pol = "ALLOW" if T_next >= T_ALLOW else ("VERIFY" if T_next >= T_QUAR else "BLOCK")
    prev_pol = state.get("policy", "ALLOW")
    if prev_pol == "ALLOW" and pol != "ALLOW":
        k_inc_next = min(PARAMS_M2["k_max"], k_inc + 1)
        clean_c_next = 0
    elif pol == "ALLOW":
        clean_c_next = clean_c + 1
        if clean_c_next >= PARAMS_M2["n_clean_req"]:
            k_inc_next = max(0, k_inc - 1)
            clean_c_next = 0
        else:
            k_inc_next = k_inc
    else:
        k_inc_next = k_inc
        clean_c_next = 0
    return {"T": float(T_next), "E": float(E_next), "SB": float(SB_next), "k_inc": int(k_inc_next), "clean_counter": int(clean_c_next), "alpha_eff": float(alpha_eff), "I": I_next, "policy": pol, "branch": branch}

def step_m3(A: float, state: dict) -> dict:
    T, E, SB, prob, streak = state["T"], state["E"], state["SB"], state["probation"], state["streak"]
    E_next = BETA * A + (1.0 - BETA) * E
    if E_next > GAMMA:
        SB_next = SB + LAMBDA_
    else:
        SB_next = max(0.0, SB - DELTA)
    I_next = 1 if SB_next > THETA else 0
    if A >= A_THR:
        T_next = T * (1.0 - A)
        branch = "OVERT"
    elif I_next == 0:
        T_next = T + ALPHA_0 * (1.0 - T)
        branch = "BENIGN_RECOVERY"
    else:
        T_next = max(0.0, T - P_PEN * (A / A_THR))
        branch = "SLOWBURN_PENALTY"

    if T_next < T_ALLOW:
        prob_next = 1
        streak_next = 0
    else:
        if prob == 1:
            if A < A_THR and E_next <= GAMMA:
                streak_next = streak + 1
            else:
                streak_next = 0
            if streak_next >= PARAMS_M3["H_req"]:
                prob_next = 0
            else:
                prob_next = 1
        else:
            prob_next = 0
            streak_next = streak + 1

    if T_next < T_QUAR:
        pol = "BLOCK"
    elif T_next >= T_ALLOW and prob_next == 0:
        pol = "ALLOW"
    else:
        pol = "VERIFY"

    return {"T": float(T_next), "E": float(E_next), "SB": float(SB_next), "probation": int(prob_next), "streak": int(streak_next), "I": I_next, "policy": pol, "branch": branch}

# ---------------------------------------------------------------------------
# Part 1: Attack Replay Simulation
# ---------------------------------------------------------------------------

def run_attack_replay(mechanism: str, schedule_name: str, scores: np.ndarray, max_cycles: int = 20) -> dict:
    tau_a, tau_r = SCHEDULES[schedule_name]
    step_fn = {"B0": step_b0, "M1": step_m1, "M2": step_m2, "M3": step_m3}[mechanism]

    if mechanism == "B0":
        state = {"T": 1.0, "E": 0.1656, "SB": 0.0, "policy": "ALLOW"}
    elif mechanism == "M1":
        state = {"T": 1.0, "E": 0.1656, "SB": 0.0, "h": 0.0, "policy": "ALLOW"}
    elif mechanism == "M2":
        state = {"T": 1.0, "E": 0.1656, "SB": 0.0, "k_inc": 0, "clean_counter": 0, "policy": "ALLOW"}
    elif mechanism == "M3":
        state = {"T": 1.0, "E": 0.1656, "SB": 0.0, "probation": 0, "streak": 0, "policy": "ALLOW"}

    ptr = 0
    quarantined = False
    quarantine_cycle = None
    quarantine_phase = None
    quarantine_step = None
    survived_cycles = 0

    allow_atk = 0
    verify_atk = 0
    block_atk = 0
    total_atk_delivered = 0

    cycle_records = []

    for k in range(1, max_cycles + 1):
        t_pre_atk = state["T"]
        sb_pre_atk = state["SB"]

        # Attack burst
        for step_a in range(tau_a):
            a_val = ATK_PROFILE[step_a % len(ATK_PROFILE)]
            state = step_fn(a_val, state)
            total_atk_delivered += 1
            if state["policy"] == "ALLOW": allow_atk += 1
            elif state["policy"] == "VERIFY": verify_atk += 1
            else: block_atk += 1

            if state["policy"] == "BLOCK" and not quarantined:
                quarantined = True
                quarantine_cycle = k
                quarantine_phase = "ATTACK"
                quarantine_step = step_a + 1
                break

        if quarantined:
            cycle_records.append({
                "cycle": k,
                "T_post_atk": round(state["T"], 6),
                "SB_post_atk": round(state["SB"], 6),
                "T_post_rst": None,
                "SB_post_rst": None,
                "quarantined": True,
                "quarantine_phase": "ATTACK",
                "quarantine_step": quarantine_step
            })
            break

        t_post_atk = state["T"]
        sb_post_atk = state["SB"]

        # Rest window
        for step_r in range(tau_r):
            if ptr >= len(scores):
                break
            a_benign = scores[ptr]
            ptr += 1
            state = step_fn(a_benign, state)

            if state["policy"] == "BLOCK" and not quarantined:
                quarantined = True
                quarantine_cycle = k
                quarantine_phase = "REST"
                quarantine_step = step_r + 1
                break

        cycle_records.append({
            "cycle": k,
            "T_post_atk": round(t_post_atk, 6),
            "SB_post_atk": round(sb_post_atk, 6),
            "T_post_rst": round(state["T"], 6),
            "SB_post_rst": round(state["SB"], 6),
            "quarantined": quarantined,
            "quarantine_phase": "REST" if quarantined else "NONE",
            "quarantine_step": quarantine_step if quarantined else None,
            "end_policy": state["policy"],
            "extra_state": {k_key: round(state[k_key], 6) if isinstance(state[k_key], float) else state[k_key] for k_key in state if k_key not in ["T", "E", "SB", "I", "policy", "branch"]}
        })

        if quarantined:
            break
        survived_cycles += 1

    malicious_yield = 1.0 * allow_atk + 0.5 * verify_atk

    return {
        "mechanism": mechanism,
        "schedule": schedule_name,
        "tau_a": tau_a,
        "tau_r": tau_r,
        "survived_cycles_S": survived_cycles,
        "attack_steps_delivered": total_atk_delivered,
        "attack_steps_allow": allow_atk,
        "attack_steps_verify": verify_atk,
        "attack_steps_block": block_atk,
        "malicious_access_yield_M_cum": round(malicious_yield, 2),
        "quarantined": quarantined,
        "time_to_first_block": {
            "cycle": quarantine_cycle,
            "phase": quarantine_phase,
            "step": quarantine_step
        } if quarantined else None,
        "absorbing_block": quarantined,
        "rehabilitation_cycles_completed": survived_cycles,
        "residual_evidence_last_cycle": round(cycle_records[-1]["SB_post_rst"], 6) if (cycle_records and cycle_records[-1]["SB_post_rst"] is not None) else round(cycle_records[-1]["SB_post_atk"], 6),
        "cycle_records": cycle_records
    }

# ---------------------------------------------------------------------------
# Part 2: Pure Benign Telemetry Availability Simulation
# ---------------------------------------------------------------------------

def run_benign_availability(mechanism: str, scores: np.ndarray) -> dict:
    step_fn = {"B0": step_b0, "M1": step_m1, "M2": step_m2, "M3": step_m3}[mechanism]

    if mechanism == "B0":
        state = {"T": 1.0, "E": 0.1656, "SB": 0.0, "policy": "ALLOW"}
    elif mechanism == "M1":
        state = {"T": 1.0, "E": 0.1656, "SB": 0.0, "h": 0.0, "policy": "ALLOW"}
    elif mechanism == "M2":
        state = {"T": 1.0, "E": 0.1656, "SB": 0.0, "k_inc": 0, "clean_counter": 0, "policy": "ALLOW"}
    elif mechanism == "M3":
        state = {"T": 1.0, "E": 0.1656, "SB": 0.0, "probation": 0, "streak": 0, "policy": "ALLOW"}

    allow_count = 0
    verify_count = 0
    block_count = 0
    block_transitions = 0
    longest_block = 0
    current_block = 0

    recovery_times = []
    under_degradation = False
    degradation_start = 0

    h_trajectory = []
    k_trajectory = []
    streak_trajectory = []

    prev_pol = "ALLOW"

    for idx, a in enumerate(scores):
        state = step_fn(a, state)
        pol = state["policy"]

        if pol == "ALLOW": allow_count += 1
        elif pol == "VERIFY": verify_count += 1
        else: block_count += 1

        # Block transitions & duration
        if pol == "BLOCK":
            current_block += 1
            if prev_pol != "BLOCK":
                block_transitions += 1
        else:
            if current_block > longest_block:
                longest_block = current_block
            current_block = 0

        # Degradation recovery time tracking
        if prev_pol == "ALLOW" and pol != "ALLOW":
            under_degradation = True
            degradation_start = idx
        elif under_degradation and pol == "ALLOW":
            recovery_times.append(idx - degradation_start)
            under_degradation = False

        if mechanism == "M1" and idx % 500 == 0:
            h_trajectory.append({"step": idx, "h": round(state["h"], 6), "SB": round(state["SB"], 6)})
        elif mechanism == "M2" and idx % 500 == 0:
            k_trajectory.append({"step": idx, "k_inc": state["k_inc"], "alpha_eff": round(state["alpha_eff"], 6)})
        elif mechanism == "M3" and idx % 500 == 0:
            streak_trajectory.append({"step": idx, "probation": state["probation"], "streak": state["streak"]})

        prev_pol = pol

    if current_block > longest_block:
        longest_block = current_block

    n_total = len(scores)
    mean_recovery = float(np.mean(recovery_times)) if recovery_times else 0.0
    median_recovery = float(np.median(recovery_times)) if recovery_times else 0.0

    res = {
        "mechanism": mechanism,
        "total_benign_observations": n_total,
        "allow_fraction": round(allow_count / n_total, 4),
        "verify_fraction": round(verify_count / n_total, 4),
        "block_fraction": round(block_count / n_total, 4),
        "allow_count": allow_count,
        "false_verify_count": verify_count,
        "false_block_count": block_count,
        "block_transitions": block_transitions,
        "longest_continuous_block_duration": longest_block,
        "mean_recovery_time_steps": round(mean_recovery, 2),
        "median_recovery_time_steps": round(median_recovery, 2),
        "final_trust": round(state["T"], 6),
        "final_state": {k_key: round(state[k_key], 6) if isinstance(state[k_key], float) else state[k_key] for k_key in state}
    }

    if mechanism == "M1":
        res["h_trajectory_sample"] = h_trajectory
        res["min_residual_floor"] = min(p["h"] for p in h_trajectory) if h_trajectory else 0.0
        res["max_residual_floor"] = max(p["h"] for p in h_trajectory) if h_trajectory else 0.0
    elif mechanism == "M2":
        res["k_trajectory_sample"] = k_trajectory
        res["min_alpha_eff"] = min(p["alpha_eff"] for p in k_trajectory) if k_trajectory else ALPHA_0
        res["max_k_inc"] = max(p["k_inc"] for p in k_trajectory) if k_trajectory else 0
    elif mechanism == "M3":
        res["streak_trajectory_sample"] = streak_trajectory
        res["probation_fraction"] = round(sum(1 for p in streak_trajectory if p["probation"] == 1) / len(streak_trajectory), 4) if streak_trajectory else 0.0

    return res

# ---------------------------------------------------------------------------
# Main Orchestration & Reproducibility Verification
# ---------------------------------------------------------------------------

def run_full_evaluation():
    scores = pd.read_csv(SCORES_CSV)["A_t"].values

    config = {
        "evaluation_name": "Candidate 5 Phase 4 Empirical Countermeasure Evaluation",
        "dataset_source": "benign_chronological_scores.csv (Edge-IIoTset)",
        "dataset_rows": len(scores),
        "ambient_noise_rate_p_fp": float(np.mean(scores >= A_THR)),
        "attack_profile_sha256": hashlib.sha256(str(ATK_PROFILE).encode("utf-8")).hexdigest(),
        "baseline_parameters": {
            "A_thr": A_THR, "gamma": GAMMA, "theta": THETA, "beta": BETA,
            "lambda": LAMBDA_, "delta": DELTA, "P_pen": P_PEN, "alpha_0": ALPHA_0,
            "T_allow": T_ALLOW, "T_quar": T_QUAR
        },
        "mechanisms_evaluated": ["B0", "M1", "M2", "M3"],
        "schedules_evaluated": list(SCHEDULES.keys())
    }

    # Execute Replay for All Architectures and Schedules
    all_sec_results = {}
    for m in ["B0", "M1", "M2", "M3"]:
        all_sec_results[m] = {}
        for s in SCHEDULES:
            all_sec_results[m][s] = run_attack_replay(m, s, scores, max_cycles=20)

    # Execute Pure Benign Availability for All Architectures
    all_avail_results = {}
    for m in ["B0", "M1", "M2", "M3"]:
        all_avail_results[m] = run_benign_availability(m, scores)

    # Trade-Off Metrics & Negative Result Detection
    tradeoff_matrix = {}
    b0_avail = all_avail_results["B0"]
    b0_sec_a = all_sec_results["B0"]["Cycle_A"]

    for m in ["B0", "M1", "M2", "M3"]:
        m_avail = all_avail_results[m]
        m_sec_a = all_sec_results[m]["Cycle_A"]

        delta_sec_yield = round(m_sec_a["malicious_access_yield_M_cum"] - b0_sec_a["malicious_access_yield_M_cum"], 2)
        delta_sec_cycles = m_sec_a["survived_cycles_S"] - b0_sec_a["survived_cycles_S"]
        delta_avail_block = round(m_avail["block_fraction"] - b0_avail["block_fraction"], 4)
        delta_avail_verify = round(m_avail["verify_fraction"] - b0_avail["verify_fraction"], 4)
        delta_avail_allow = round(m_avail["allow_fraction"] - b0_avail["allow_fraction"], 4)

        # Classification of mechanism outcome
        if m == "B0":
            classif = "BASELINE"
            interpretation = "Memoryless baseline with 8-cycle survival and 24.04% legitimate false BLOCK."
        elif delta_sec_yield < 0 and (delta_avail_block > 0.05 or delta_avail_verify > 0.15):
            classif = "SECURITY_GAIN_WITH_AVAILABILITY_COST"
            interpretation = f"Yield reduced by {abs(delta_sec_yield)} units, but legitimate ALLOW dropped by {abs(delta_avail_allow)*100:.1f}%."
        elif delta_sec_yield == 0 and delta_avail_block > 0.05:
            classif = "NO_SECURITY_GAIN_WITH_AVAILABILITY_COST"
            interpretation = f"No security improvement on Cycle A, while false BLOCK increased by {delta_avail_block*100:.1f}%."
        elif delta_sec_yield < 0 and delta_avail_block <= 0.02 and delta_avail_verify <= 0.05:
            classif = "PARETO_SUPERIOR_COUNTERMEASURE"
            interpretation = "Achieved meaningful security gain with minimal legitimate degradation."
        else:
            classif = "NEUTRAL_TRADE_OFF"
            interpretation = "Mixed marginal trade-offs."

        tradeoff_matrix[m] = {
            "mechanism": m,
            "classification": classif,
            "interpretation": interpretation,
            "security_cycle_a": {
                "survived_cycles": m_sec_a["survived_cycles_S"],
                "malicious_yield_M_cum": m_sec_a["malicious_access_yield_M_cum"],
                "delta_yield_vs_B0": delta_sec_yield,
                "delta_cycles_vs_B0": delta_sec_cycles
            },
            "availability_benign": {
                "allow_fraction": m_avail["allow_fraction"],
                "verify_fraction": m_avail["verify_fraction"],
                "block_fraction": m_avail["block_fraction"],
                "delta_allow_vs_B0": delta_avail_allow,
                "delta_verify_vs_B0": delta_avail_verify,
                "delta_block_vs_B0": delta_avail_block,
                "block_transitions": m_avail["block_transitions"],
                "mean_recovery_steps": m_avail["mean_recovery_time_steps"]
            }
        }

    return config, all_sec_results, all_avail_results, tradeoff_matrix

def main():
    _log_lines = []
    def log(msg):
        print(msg)
        _log_lines.append(msg)

    log("================================================================================")
    log("STARTING PHASE 4: EMPIRICAL COUNTERMEASURE EVALUATION")
    log("================================================================================")

    # Run 1
    log("Executing Evaluation Run 1...")
    config1, sec1, avail1, trade1 = run_full_evaluation()

    # Save artifact files
    with open(EVAL_DIR / "config.json", "w", encoding="utf-8") as f:
        json.dump(config1, f, indent=2)
    with open(EVAL_DIR / "B0_results.json", "w", encoding="utf-8") as f:
        json.dump({"security": sec1["B0"], "availability": avail1["B0"]}, f, indent=2)
    with open(EVAL_DIR / "M1_results.json", "w", encoding="utf-8") as f:
        json.dump({"security": sec1["M1"], "availability": avail1["M1"]}, f, indent=2)
    with open(EVAL_DIR / "M2_results.json", "w", encoding="utf-8") as f:
        json.dump({"security": sec1["M2"], "availability": avail1["M2"]}, f, indent=2)
    with open(EVAL_DIR / "M3_results.json", "w", encoding="utf-8") as f:
        json.dump({"security": sec1["M3"], "availability": avail1["M3"]}, f, indent=2)
    with open(EVAL_DIR / "security_metrics.json", "w", encoding="utf-8") as f:
        json.dump(sec1, f, indent=2)
    with open(EVAL_DIR / "availability_metrics.json", "w", encoding="utf-8") as f:
        json.dump(avail1, f, indent=2)
    with open(EVAL_DIR / "tradeoff_metrics.json", "w", encoding="utf-8") as f:
        json.dump(trade1, f, indent=2)

    log("Saved all evaluation JSON artifacts.")

    # Run 2 for bit-for-bit reproducibility check
    log("Executing Evaluation Run 2 for reproducibility check...")
    config2, sec2, avail2, trade2 = run_full_evaluation()

    match_cfg = (config1 == config2)
    match_sec = (sec1 == sec2)
    match_avail = (avail1 == avail2)
    match_trade = (trade1 == trade2)

    log(f"Reproducibility verification:")
    log(f"  Config identical:       {match_cfg}")
    log(f"  Security identical:     {match_sec}")
    log(f"  Availability identical: {match_avail}")
    log(f"  Tradeoff identical:     {match_trade}")

    assert match_cfg and match_sec and match_avail and match_trade, "Phase 4 reproducibility failed!"

    file_hashes = {
        "config_json_sha256": sha256_file(EVAL_DIR / "config.json"),
        "B0_results_json_sha256": sha256_file(EVAL_DIR / "B0_results.json"),
        "M1_results_json_sha256": sha256_file(EVAL_DIR / "M1_results.json"),
        "M2_results_json_sha256": sha256_file(EVAL_DIR / "M2_results.json"),
        "M3_results_json_sha256": sha256_file(EVAL_DIR / "M3_results.json"),
        "security_metrics_json_sha256": sha256_file(EVAL_DIR / "security_metrics.json"),
        "availability_metrics_json_sha256": sha256_file(EVAL_DIR / "availability_metrics.json"),
        "tradeoff_metrics_json_sha256": sha256_file(EVAL_DIR / "tradeoff_metrics.json")
    }

    with open(EVAL_DIR / "trajectory_hashes.json", "w", encoding="utf-8") as f:
        json.dump(file_hashes, f, indent=2)

    repro_payload = {
        "hashes": file_hashes,
        "all_runs_identical": True,
        "verification_status": "VERIFIED_DETERMINISTIC"
    }
    with open(EVAL_DIR / "reproducibility.json", "w", encoding="utf-8") as f:
        json.dump(repro_payload, f, indent=2)

    log("Saved trajectory hashes and reproducibility ledger.")
    log("================================================================================")
    log("PHASE 4 COMPLETE: C5_PHASE4_EMPIRICAL_EVALUATION_COMPLETE")
    log("================================================================================")

    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(_log_lines))

if __name__ == "__main__":
    main()
