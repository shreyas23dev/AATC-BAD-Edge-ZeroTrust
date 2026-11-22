#!/usr/bin/env python3
"""
benchmark_monte_carlo_pulsing.py
Performs rigorous Monte Carlo evaluation (100 independent random seeds) of AATC-BAD
versus Authoritative Baseline B0 across multiple pacing schedules (tau_a, tau_r)
with stochastic jitter and randomized background telemetry offsets.
Evaluates on both Edge-IIoTset and CIC-IoT-2023 benign telemetry streams.
Addresses InCIS Reviewer 1 (Weakness 1 & Comment 2: lack of variance/repeated runs).
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path

# Paths
BASE_DIR = Path("/mnt/c/Users/Shreyas A/PROJECTS/BAZT-TON_IoT/TON_IoT_project")
EDGE_CSV = BASE_DIR / "experiments/candidate5/chronological_validation/benign_chronological_scores.csv"
CICIOT_JSONL = BASE_DIR / "rust-edge/decisions_ciciot_expanded.jsonl"
CICIOT_CSV = BASE_DIR / "data/ciciot2023/ciciot_eval_expanded_44.csv"
OUT_JSON = BASE_DIR / "rust-edge/xai/monte_carlo_pulsing_results.json"

# Frozen Parameters (Paper 1 & Phase 9 Specification)
A_THR   = 0.4554
BETA    = 0.40
GAMMA   = 0.335
THETA   = 1.00
LAMBDA_ = 0.55
DELTA   = 0.05
P_PEN   = 0.25
ALPHA_0 = 0.2057
T_ALLOW = 0.80
T_QUAR  = 0.30

# AATC-BAD Parameters
Q_LOW   = 0.35
DENOM   = A_THR - Q_LOW  # 0.1054
DELTA_D = 0.10
K_ALLOW = 1.70
ETA_C   = 0.30
C_FLOOR = 0.70

ATK_PROFILE = [0.4100, 0.3900, 0.4200, 0.3800, 0.4000,
               0.4300, 0.3900, 0.4100, 0.4000, 0.4200]

def step_b0(a: float, state: dict) -> dict:
    T, E, SB = state["T"], state["E"], state["SB"]
    E_next = BETA * a + (1.0 - BETA) * E
    if E_next > GAMMA:
        SB_next = SB + LAMBDA_
    else:
        SB_next = max(0.0, SB - DELTA)
    I_next = 1 if SB_next > THETA else 0
    if a >= A_THR:
        T_next = T * (1.0 - a)
    elif I_next == 0:
        T_next = T + ALPHA_0 * (1.0 - T)
    else:
        T_next = max(0.0, T - P_PEN * (a / A_THR))
    pol = "ALLOW" if T_next >= T_ALLOW else ("VERIFY" if T_next >= T_QUAR else "BLOCK")
    return {"T": float(T_next), "E": float(E_next), "SB": float(SB_next), "I": I_next, "policy": pol}

def step_aatc(a: float, state: dict) -> dict:
    T, E, SB, D, C = state["T"], state["E"], state["SB"], state["D"], state["C"]
    
    # Stage 1: Debt increment
    if (T >= T_ALLOW) and (Q_LOW <= a < A_THR):
        delta_D = (a - Q_LOW) / DENOM
    else:
        delta_D = 0.0
        
    # Stage 2: Debt update
    if a < Q_LOW:
        D_next = max(0.0, D - DELTA_D)
    else:
        D_next = D + delta_D
        
    # Stage 3: Ceiling update
    C_next = max(C_FLOOR, 1.0 - ETA_C * max(0.0, D_next - K_ALLOW))
    
    # Stage 4: Preserved B0 Energy and Slow-burn
    E_next = BETA * a + (1.0 - BETA) * E
    if E_next > GAMMA:
        SB_next = SB + LAMBDA_
    else:
        SB_next = max(0.0, SB - DELTA)
    I_next = 1 if SB_next > THETA else 0
    
    # Stage 5: Trust recovery capped by C_next
    if a >= A_THR:
        T_next = T * (1.0 - a)
    elif I_next == 0:
        T_next = min(C_next, T + ALPHA_0 * (C_next - T))
    else:
        T_next = max(0.0, T - P_PEN * (a / A_THR))
        
    pol = "ALLOW" if T_next >= T_ALLOW else ("VERIFY" if T_next >= T_QUAR else "BLOCK")
    return {"T": float(T_next), "E": float(E_next), "SB": float(SB_next),
            "D": float(D_next), "C": float(C_next), "I": I_next, "policy": pol}

def simulate_single_run(scores, start_idx, tau_a, tau_r, jitter_range, seed, max_cycles=25):
    rng = np.random.RandomState(seed)
    
    # 1. Simulate B0
    state_b0 = {"T": 1.0, "E": 0.1656, "SB": 0.0}
    ptr_b0 = start_idx
    b0_allow = 0
    b0_verify = 0
    b0_survived = 0
    
    # 2. Simulate AATC
    state_aatc = {"T": 1.0, "E": 0.1656, "SB": 0.0, "D": 0.0, "C": 1.0}
    ptr_aatc = start_idx
    aatc_allow = 0
    aatc_verify = 0
    aatc_survived = 0
    
    # Pre-generate jitter per cycle so both algorithms face identical telemetry
    jitters = [rng.randint(-jitter_range, jitter_range + 1) for _ in range(max_cycles)]
    noise_deltas = [rng.normal(0, 0.005, size=tau_a) for _ in range(max_cycles)]
    
    b0_quarantined = False
    for k in range(max_cycles):
        if b0_quarantined:
            break
            
        # Attack burst
        for step_a in range(tau_a):
            base_a = ATK_PROFILE[step_a % len(ATK_PROFILE)]
            a = float(np.clip(base_a + noise_deltas[k][step_a], Q_LOW + 0.01, A_THR - 0.005))
            state_b0 = step_b0(a, state_b0)
            if state_b0["policy"] == "ALLOW": b0_allow += 1
            elif state_b0["policy"] == "VERIFY": b0_verify += 1
            else:
                b0_quarantined = True
                break
                
        if b0_quarantined:
            break
            
        # Rest window
        cur_tau_r = max(2, tau_r + jitters[k])
        for _ in range(cur_tau_r):
            if ptr_b0 >= len(scores): break
            a = scores[ptr_b0]
            ptr_b0 += 1
            state_b0 = step_b0(a, state_b0)
            if state_b0["policy"] == "BLOCK":
                b0_quarantined = True
                break
                
        if not b0_quarantined:
            b0_survived += 1
            
    m_cum_b0 = b0_allow * 1.0 + b0_verify * 0.5
    
    # Now simulate AATC with identical stream & jitter
    aatc_quarantined = False
    for k in range(max_cycles):
        if aatc_quarantined:
            break
            
        # Attack burst
        for step_a in range(tau_a):
            base_a = ATK_PROFILE[step_a % len(ATK_PROFILE)]
            a = float(np.clip(base_a + noise_deltas[k][step_a], Q_LOW + 0.01, A_THR - 0.005))
            state_aatc = step_aatc(a, state_aatc)
            if state_aatc["policy"] == "ALLOW": aatc_allow += 1
            elif state_aatc["policy"] == "VERIFY": aatc_verify += 1
            else:
                aatc_quarantined = True
                break
                
        if aatc_quarantined:
            break
            
        # Rest window
        cur_tau_r = max(2, tau_r + jitters[k])
        for _ in range(cur_tau_r):
            if ptr_aatc >= len(scores): break
            a = scores[ptr_aatc]
            ptr_aatc += 1
            state_aatc = step_aatc(a, state_aatc)
            if state_aatc["policy"] == "BLOCK":
                aatc_quarantined = True
                break
                
        if not aatc_quarantined:
            aatc_survived += 1
            
    m_cum_aatc = aatc_allow * 1.0 + aatc_verify * 0.5
    
    return {
        "b0_mcum": m_cum_b0,
        "b0_allow": b0_allow,
        "b0_survived": b0_survived,
        "aatc_mcum": m_cum_aatc,
        "aatc_allow": aatc_allow,
        "aatc_survived": aatc_survived,
        "delta_mcum": m_cum_b0 - m_cum_aatc,
        "pct_mcum_reduction": ((m_cum_b0 - m_cum_aatc) / m_cum_b0 * 100.0) if m_cum_b0 > 0 else 0.0,
        "delta_allow": b0_allow - aatc_allow
    }

def run_monte_carlo(dataset_name, scores, n_seeds=100):
    print(f"\n================================================================================")
    print(f"Running Monte Carlo Evaluation on {dataset_name} ({n_seeds} Independent Runs)")
    print(f"Telemetry stream length: {len(scores)} observations")
    print(f"================================================================================")
    
    schedules = [
        {"name": "Pacing (1, 8)",  "tau_a": 1, "tau_r": 8,  "jitter": 1},
        {"name": "Pacing (1, 10)", "tau_a": 1, "tau_r": 10, "jitter": 2},
        {"name": "Pacing (1, 12)", "tau_a": 1, "tau_r": 12, "jitter": 2},
        {"name": "Pacing (1, 15)", "tau_a": 1, "tau_r": 15, "jitter": 2},
        {"name": "Pacing (2, 12)", "tau_a": 2, "tau_r": 12, "jitter": 2},
        {"name": "Pacing (2, 18)", "tau_a": 2, "tau_r": 18, "jitter": 3},
    ]
    
    master_rng = np.random.RandomState(42)
    max_start = max(100, len(scores) - 1500)
    
    summary_results = []
    
    print(f"{'Schedule':<16} | {'B0 M_cum (mean±std)':<20} | {'AATC M_cum (mean±std)':<22} | {'M_cum Reduction (%)':<20} | {'ALLOW Pulses Cut'}")
    print("-" * 98)
    
    for sched in schedules:
        tau_a = sched["tau_a"]
        tau_r = sched["tau_r"]
        jitter = sched["jitter"]
        name = sched["name"]
        
        b0_mcums = []
        aatc_mcums = []
        pct_reductions = []
        b0_allows = []
        aatc_allows = []
        delta_allows = []
        
        for run_id in range(n_seeds):
            seed = master_rng.randint(0, 1_000_000)
            start_idx = master_rng.randint(0, max_start)
            
            res = simulate_single_run(
                scores=scores,
                start_idx=start_idx,
                tau_a=tau_a,
                tau_r=tau_r,
                jitter_range=jitter,
                seed=seed,
                max_cycles=25
            )
            
            b0_mcums.append(res["b0_mcum"])
            aatc_mcums.append(res["aatc_mcum"])
            pct_reductions.append(res["pct_mcum_reduction"])
            b0_allows.append(res["b0_allow"])
            aatc_allows.append(res["aatc_allow"])
            delta_allows.append(res["delta_allow"])
            
        b0_m_mean = float(np.mean(b0_mcums))
        b0_m_std = float(np.std(b0_mcums))
        aatc_m_mean = float(np.mean(aatc_mcums))
        aatc_m_std = float(np.std(aatc_mcums))
        pct_red_mean = float(np.mean(pct_reductions))
        pct_red_std = float(np.std(pct_reductions))
        allow_cut_mean = float(np.mean(delta_allows))
        allow_cut_std = float(np.std(delta_allows))
        
        summary_results.append({
            "schedule": name,
            "tau_a": tau_a,
            "tau_r": tau_r,
            "b0_mcum_mean": round(b0_m_mean, 2),
            "b0_mcum_std": round(b0_m_std, 2),
            "aatc_mcum_mean": round(aatc_m_mean, 2),
            "aatc_mcum_std": round(aatc_m_std, 2),
            "pct_reduction_mean": round(pct_red_mean, 2),
            "pct_reduction_std": round(pct_red_std, 2),
            "allow_cut_mean": round(allow_cut_mean, 2),
            "allow_cut_std": round(allow_cut_std, 2)
        })
        
        print(f"{name:<16} | {b0_m_mean:>6.2f} ± {b0_m_std:<8.2f} | {aatc_m_mean:>6.2f} ± {aatc_m_std:<10.2f} | {pct_red_mean:>6.2f}% ± {pct_red_std:<7.2f}% | {allow_cut_mean:>5.2f} ± {allow_cut_std:<5.2f}")
        
    return summary_results

def main():
    # 1. Load Edge-IIoTset Evaluation Scores
    print("Loading Edge-IIoTset benign chronological scores...")
    df_edge = pd.read_csv(EDGE_CSV)
    # Use evaluation split (rows 10,000 to 17,458)
    edge_scores = df_edge["A_t"].values[10000:17459]
    
    # 2. Load CIC-IoT-2023 Benign Scores
    print("Loading CIC-IoT-2023 benign telemetry scores...")
    df_ciciot = pd.read_csv(CICIOT_CSV, usecols=["Attack_type"])
    ciciot_norm_indices = df_ciciot[df_ciciot["Attack_type"] == "Normal"].index.values
    
    # Extract raw reconstruction scores from decisions_ciciot_expanded.jsonl
    raw_scores = []
    with open(CICIOT_JSONL, "r") as f:
        for idx, line in enumerate(f):
            if idx in ciciot_norm_indices:
                rec = json.loads(line)
                if "phi" in rec and rec["phi"] is not None:
                    raw_scores.append(np.mean(rec["phi"]))
                else:
                    raw_scores.append(0.0)
                    
    # Scale to A_t matching AATC-BAD calibrated r_thr = 0.0360
    ciciot_a_t = np.clip((np.array(raw_scores) / 0.0360) * A_THR, 0.0, 1.0)
    
    results = {
        "metadata": {
            "n_monte_carlo_seeds": 100,
            "attack_profile": ATK_PROFILE,
            "r_thr_ciciot": 0.0360,
            "a_thr": A_THR
        },
        "edge_iiotset": run_monte_carlo("Edge-IIoTset (N=7,459)", edge_scores, n_seeds=100),
        "cic_iot_2023": run_monte_carlo("CIC-IoT-2023 (N=10,000)", ciciot_a_t, n_seeds=100)
    }
    
    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nMonte Carlo benchmark results saved to {OUT_JSON}")

if __name__ == "__main__":
    main()
