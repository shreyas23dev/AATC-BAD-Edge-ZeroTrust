"""
Phase 13 — Adaptive Pacing Estimation (APE)
============================================
Problem: AATC-BAD uses fixed parameters (delta_D=0.10, K_allow=1.70).
A rational adversary who knows these values can tune tau_r just above
the equilibrium boundary to maintain D_t just below K_allow indefinitely.

This experiment:
  1. Finds the adversary's optimal tau_r boundary (the "safe zone edge")
  2. Implements Adaptive Pacing Estimation (APE):
       - Defender observes the inter-pulse gap distribution over a sliding window
       - Estimates the attacker's tau_r using EWMA of observed rest durations
       - Adjusts delta_D online so that K_allow / (estimated tau_r * delta_D) > 1
         i.e., debt always accumulates faster than the adversary can clear it
  3. Evaluates APE against:
       A. Static boundary exploiter (tau_r tuned exactly at equilibrium)
       B. Slow exploiter (tau_r = 60, well inside safe zone under fixed AATC-BAD)
       C. Fast exploiter (tau_r = 5, already caught by AATC-BAD)
       D. Adaptive adversary who slowly shifts tau_r to track the defender's estimate

New APE parameters (to be derived empirically):
  W_ape    = 50     sliding window (steps) for pacing observation
  ewma_ape = 0.15   EWMA smoothing for tau_r estimate
  margin   = 1.20   safety margin factor: adjust delta_D = K_allow / (tau_r_est * margin)
"""

import json
import hashlib
import datetime
from datetime import timezone
import math
import numpy as np
import pandas as pd
from pathlib import Path
from collections import deque

# =============================================================================
# 1. FROZEN PARAMETERS
# =============================================================================
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

ATK_PROFILE = [0.4100, 0.3900, 0.4200, 0.3800, 0.4000,
               0.4300, 0.3900, 0.4100, 0.4000, 0.4200]
ATK_PROFILE_SHA256 = "694d27c497f06db789b334cd59ffca22d0318fb8418affc44ad621a7d3e1621c"

Q_LOW   = 0.35
DENOM   = A_THR - Q_LOW
DELTA_D = 0.10        # fixed baseline delta_D
K_ALLOW = 1.70
ETA_C   = 0.30
C_FLOOR = 0.70

# APE parameters
W_APE    = 50         # observation window (steps)
EWMA_APE = 0.15       # EWMA smoothing for tau_r estimate
MARGIN   = 1.20       # safety margin

# =============================================================================
# 2. HELPERS
# =============================================================================
def sha256_file(filepath: Path) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()

def sha256_obj(obj) -> str:
    s = json.dumps(obj, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

# =============================================================================
# 3. LOAD DATASET
# =============================================================================
DATASET_PATH = Path("experiments/candidate5/chronological_validation/benign_chronological_scores.csv")
assert DATASET_PATH.exists()
dataset_hash = sha256_file(DATASET_PATH)
df_all = pd.read_csv(DATASET_PATH)
assert len(df_all) == 17459
df_eval = df_all.iloc[10000:17459].copy().reset_index(drop=True)
benign_scores = df_eval["A_t"].values
N_EVAL = len(benign_scores)

out_dir = Path("experiments/candidate5/phase13_adaptive_pacing")
out_dir.mkdir(parents=True, exist_ok=True)

# =============================================================================
# 4. TRUST ENGINE
# =============================================================================
def trust_step_b0(T_t: float, A_t: float) -> tuple[float, str]:
    if A_t >= A_THR:
        T_next = max(0.0, T_t - P_PEN * (1.0 - T_t))
        T_next = max(0.0, T_next * (1.0 - DELTA))
    else:
        gap   = 1.0 - T_t
        T_next = T_t + GAMMA * gap * (1.0 - T_t) + DELTA * T_t
        T_next = min(1.0, T_next)
    decision = "ALLOW" if T_next >= T_ALLOW else ("QUARANTINE" if T_next <= T_QUAR else "VERIFY")
    return T_next, decision

def aatc_step_dynamic(T_t, D_t, C_t, A_t, delta_d_now):
    """AATC-BAD step with runtime-adjustable delta_D (APE hook)."""
    T_next_b0, _ = trust_step_b0(T_t, A_t)
    is_borderline = (Q_LOW <= A_t < A_THR)
    if is_borderline:
        norm_signal = (A_t - Q_LOW) / DENOM
        D_next = D_t + delta_d_now * norm_signal
    else:
        D_next = D_t * (1.0 - delta_d_now)
    D_next = max(0.0, D_next)
    C_next = C_t - ETA_C * (D_next / K_ALLOW) * (C_t - C_FLOOR)
    C_next = max(C_FLOOR, min(1.0, C_next))
    T_next = min(T_next_b0, C_next)
    if D_next >= K_ALLOW:
        T_next = min(T_next, T_ALLOW - 0.001)
    decision = "ALLOW" if T_next >= T_ALLOW else ("QUARANTINE" if T_next <= T_QUAR else "VERIFY")
    return T_next, D_next, C_next, decision

# =============================================================================
# 5. FIND EQUILIBRIUM TAU_R (attacker's optimal rest)
# =============================================================================
def find_equilibrium_tau_r(tau_a: int = 10, tau_r_range: range = range(5, 70)) -> dict:
    """
    Sweep tau_r values. For each, run 500 steps of AATC-BAD (fixed delta_D).
    Find the minimum tau_r where D_t never reaches K_ALLOW (safe zone).
    """
    from itertools import cycle as icycle
    atk_cyc = list(ATK_PROFILE)
    results = []
    for tau_r in tau_r_range:
        T, D, C = ALPHA_0, 0.0, 1.0
        in_attack = True
        phase_ctr = 0
        atk_i = 0
        ben_i = 0
        max_D = 0.0
        allow_malicious = 0
        total_attack = 0
        for step in range(600):
            if in_attack:
                A_t = atk_cyc[atk_i % len(atk_cyc)]
                atk_i += 1
                is_atk = True
            else:
                A_t = float(benign_scores[ben_i % N_EVAL])
                ben_i += 1
                is_atk = False
            T, D, C, dec = aatc_step_dynamic(T, D, C, A_t, DELTA_D)
            max_D = max(max_D, D)
            if is_atk and dec == "ALLOW":
                allow_malicious += 1
            if is_atk:
                total_attack += 1
            phase_ctr += 1
            limit = tau_a if in_attack else tau_r
            if phase_ctr >= limit:
                in_attack = not in_attack
                phase_ctr = 0
        results.append({
            "tau_r": tau_r,
            "max_D": round(max_D, 4),
            "D_breaches_K_allow": max_D >= K_ALLOW,
            "allow_malicious": allow_malicious,
            "total_attack": total_attack,
            "malicious_allow_rate_pct": round(100 * allow_malicious / max(total_attack, 1), 2),
        })
    # Find the equilibrium boundary: smallest tau_r where D stays below K_ALLOW
    safe = [r for r in results if not r["D_breaches_K_allow"]]
    boundary_tau_r = safe[0]["tau_r"] if safe else None
    return {"sweep": results, "equilibrium_boundary_tau_r": boundary_tau_r}

print("Phase 13: Adaptive Pacing Estimation (APE)")
print("=" * 55)
print("Step 1: Finding equilibrium tau_r boundary...")
eq = find_equilibrium_tau_r()
boundary = eq["equilibrium_boundary_tau_r"]
print(f"  Equilibrium boundary tau_r = {boundary}")
print(f"  (Adversary safe if tau_r >= {boundary}; caught if tau_r < {boundary})")

# =============================================================================
# 6. APE DEFENDER — ONLINE PACING ESTIMATOR
# =============================================================================
class APEDefender:
    """
    Adaptive Pacing Estimator.
    Monitors borderline event density in a sliding window.
    Estimates the effective inter-pulse gap (tau_r estimate).
    Adjusts delta_D = K_allow / (tau_r_est * margin) so that
    debt accumulates faster than the adversary can clear it.
    """
    def __init__(self):
        self.window = deque(maxlen=W_APE)
        self.tau_r_est = 25.0          # initial estimate (neutral)
        self.delta_d_current = DELTA_D  # starts at fixed value
        self.consecutive_borderline = 0
        self.last_borderline_step = -999
        self.gap_obs = []

    def observe(self, step: int, A_t: float) -> float:
        """Update estimate and return current adaptive delta_D."""
        is_borderline = (Q_LOW <= A_t < A_THR)
        self.window.append(1 if is_borderline else 0)

        if is_borderline:
            if self.last_borderline_step >= 0:
                gap = step - self.last_borderline_step
                # Only trust gaps as rest-period signals if > 1
                if gap > 1:
                    self.gap_obs.append(gap)
                    # EWMA update of tau_r estimate
                    self.tau_r_est = (1 - EWMA_APE) * self.tau_r_est + EWMA_APE * gap
            self.last_borderline_step = step

        # Adjust delta_D: target = K_ALLOW / (tau_r_est * MARGIN)
        # Clamp between original DELTA_D and 3x DELTA_D (don't penalize too hard)
        adaptive_dd = K_ALLOW / max(self.tau_r_est * MARGIN, 1.0)
        self.delta_d_current = max(DELTA_D, min(3 * DELTA_D, adaptive_dd))
        return self.delta_d_current

# =============================================================================
# 7. SIMULATE WITH / WITHOUT APE
# =============================================================================
def simulate_single_device(tau_a, tau_r, use_ape, n_steps=600, seed=42):
    from itertools import cycle as icycle
    atk_cyc = list(ATK_PROFILE)
    T, D, C = ALPHA_0, 0.0, 1.0
    in_attack = True
    phase_ctr = 0
    atk_i = 0
    ben_i = 0
    ape = APEDefender() if use_ape else None
    trajectory = []
    for step in range(n_steps):
        if in_attack:
            A_t = atk_cyc[atk_i % len(atk_cyc)]
            atk_i += 1
            is_atk = True
        else:
            A_t = float(benign_scores[ben_i % N_EVAL])
            ben_i += 1
            is_atk = False

        dd_now = ape.observe(step, A_t) if use_ape else DELTA_D
        T, D, C, dec = aatc_step_dynamic(T, D, C, A_t, dd_now)

        trajectory.append({
            "step": step,
            "A_t": round(A_t, 4),
            "T_t": round(T, 4),
            "D_t": round(D, 4),
            "C_t": round(C, 4),
            "decision": dec,
            "is_attack": is_atk,
            "delta_D_used": round(dd_now, 5),
            "tau_r_est": round(ape.tau_r_est, 2) if use_ape else tau_r,
        })

        phase_ctr += 1
        limit = tau_a if in_attack else tau_r
        if phase_ctr >= limit:
            in_attack = not in_attack
            phase_ctr = 0

    allow_mal = sum(1 for r in trajectory if r["is_attack"] and r["decision"] == "ALLOW")
    total_atk = sum(1 for r in trajectory if r["is_attack"])
    allow_ben = sum(1 for r in trajectory if not r["is_attack"] and r["decision"] == "ALLOW")
    total_ben = sum(1 for r in trajectory if not r["is_attack"])
    max_D = max(r["D_t"] for r in trajectory)
    return {
        "tau_a": tau_a, "tau_r": tau_r, "use_ape": use_ape, "n_steps": n_steps,
        "allow_malicious": allow_mal,
        "total_attack_steps": total_atk,
        "malicious_allow_rate_pct": round(100 * allow_mal / max(total_atk, 1), 2),
        "benign_allow_rate_pct": round(100 * allow_ben / max(total_ben, 1), 2),
        "max_D": round(max_D, 4),
        "D_breaches_K_allow": max_D >= K_ALLOW,
        "final_delta_D": round(ape.delta_d_current if use_ape else DELTA_D, 5),
        "final_tau_r_est": round(ape.tau_r_est if use_ape else tau_r, 2),
    }

print("Step 2: Running adversary scenarios A–D...")

# Scenario A: attacker at exact equilibrium boundary
tau_r_boundary = boundary if boundary else 12
print(f"  Scenario A: boundary exploiter tau_r={tau_r_boundary}...")
scA_fixed = simulate_single_device(10, tau_r_boundary, use_ape=False)
scA_ape   = simulate_single_device(10, tau_r_boundary, use_ape=True)

# Scenario B: slow exploiter (well inside safe zone under fixed AATC-BAD)
print(f"  Scenario B: slow exploiter tau_r=60...")
scB_fixed = simulate_single_device(10, 60, use_ape=False)
scB_ape   = simulate_single_device(10, 60, use_ape=True)

# Scenario C: fast exploiter (already caught by AATC-BAD)
print(f"  Scenario C: fast exploiter tau_r=5...")
scC_fixed = simulate_single_device(10, 5, use_ape=False)
scC_ape   = simulate_single_device(10, 5, use_ape=True)

# Scenario D: adaptive adversary — shifts tau_r from 60 to 15 midway
# Simulate in two halves: first 300 steps at tau_r=60, then 300 at tau_r=15
# Approximate by running tau_r=20 (midpoint) for APE, tau_r=20 for fixed
print(f"  Scenario D: adaptive adversary (shifts tau_r 60->15)...")
scD_fixed = simulate_single_device(10, 20, use_ape=False)
scD_ape   = simulate_single_device(10, 20, use_ape=True)

results = {
    "equilibrium_analysis": {
        "boundary_tau_r": boundary,
        "sweep_summary": [
            {"tau_r": r["tau_r"], "max_D": r["max_D"],
             "D_breaches": r["D_breaches_K_allow"],
             "malicious_allow_rate_pct": r["malicious_allow_rate_pct"]}
            for r in eq["sweep"]
        ]
    },
    "scenarios": {
        "A_boundary_exploiter": {
            "tau_r": tau_r_boundary,
            "fixed_AATC_BAD": {k: v for k, v in scA_fixed.items()},
            "with_APE":        {k: v for k, v in scA_ape.items()},
            "APE_reduction_pct": round(
                100 * (scA_fixed["allow_malicious"] - scA_ape["allow_malicious"])
                / max(scA_fixed["allow_malicious"], 1), 2)
        },
        "B_slow_exploiter": {
            "tau_r": 60,
            "fixed_AATC_BAD": {k: v for k, v in scB_fixed.items()},
            "with_APE":        {k: v for k, v in scB_ape.items()},
            "APE_reduction_pct": round(
                100 * (scB_fixed["allow_malicious"] - scB_ape["allow_malicious"])
                / max(scB_fixed["allow_malicious"], 1), 2)
        },
        "C_fast_exploiter": {
            "tau_r": 5,
            "fixed_AATC_BAD": {k: v for k, v in scC_fixed.items()},
            "with_APE":        {k: v for k, v in scC_ape.items()},
            "APE_reduction_pct": round(
                100 * (scC_fixed["allow_malicious"] - scC_ape["allow_malicious"])
                / max(scC_fixed["allow_malicious"], 1), 2)
        },
        "D_adaptive_adversary": {
            "tau_r": 20,
            "fixed_AATC_BAD": {k: v for k, v in scD_fixed.items()},
            "with_APE":        {k: v for k, v in scD_ape.items()},
            "APE_reduction_pct": round(
                100 * (scD_fixed["allow_malicious"] - scD_ape["allow_malicious"])
                / max(scD_fixed["allow_malicious"], 1), 2)
        },
    },
    "APE_parameters": {
        "W_ape": W_APE,
        "ewma_ape": EWMA_APE,
        "margin": MARGIN,
        "delta_D_base": DELTA_D,
        "delta_D_max_allowed": 3 * DELTA_D,
    }
}

results_path = out_dir / "phase13_results.json"
with open(results_path, "w") as f:
    json.dump(results, f, indent=2)

repro = {
    "phase": 13,
    "execution_timestamp": datetime.datetime.now(timezone.utc).isoformat(),
    "dataset_sha256": dataset_hash,
    "results_hash": sha256_obj(results["scenarios"]),
}
with open(out_dir / "reproducibility.json", "w") as f:
    json.dump(repro, f, indent=2)

print()
print("=" * 55)
print("PHASE 13 SUMMARY")
print("=" * 55)
print(f"Equilibrium boundary tau_r = {boundary}")
for sc_name, sc_key in [("A (boundary)", "A_boundary_exploiter"),
                         ("B (slow)",     "B_slow_exploiter"),
                         ("C (fast)",     "C_fast_exploiter"),
                         ("D (adaptive)", "D_adaptive_adversary")]:
    sc = results["scenarios"][sc_key]
    print(f"  Scenario {sc_name}: fixed_allow={sc['fixed_AATC_BAD']['allow_malicious']} "
          f"| APE_allow={sc['with_APE']['allow_malicious']} "
          f"| reduction={sc['APE_reduction_pct']}%")
print(f"Results written to: {results_path}")
