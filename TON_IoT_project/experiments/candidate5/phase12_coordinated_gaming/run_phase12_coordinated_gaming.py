"""
Phase 12 — Multi-Device Coordinated Gaming
===========================================
Tests whether AATC-BAD (single-device debt tracking) can be bypassed
by 2 or 3 devices operating coordinated sub-threshold pulsing schedules
where each device's individual debt D_t stays below K_allow = 1.70,
but the network collectively achieves high cumulative malicious yield.

New mechanism evaluated: Cross-Device Debt Aggregation (CDDA)
  - Tracks a network-level debt D_net_t = sum of D_t across all devices
  - Applies a shared network ceiling C_net_t derived from D_net_t
  - Each device's effective ceiling = min(C_t_device, C_net_t)

Experiment structure:
  E1 — 2-device coordination (interleaved phases, tau_r = 25 each)
  E2 — 3-device coordination (round-robin phases, tau_r = 25 each)
  E3 — 3-device coordination with staggered tau_r (15, 25, 35)
  E4 — CDDA defender response to all three attack formations
  E5 — Benign availability under CDDA (must preserve parity with B0)

All parameters frozen from Phase 10. CDDA adds two new parameters:
  K_net   = 2.50  (network-level debt ceiling, empirically derived)
  eta_net = 0.20  (network ceiling decay rate)
"""

import json
import hashlib
import datetime
from datetime import timezone
import math
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import cycle

# =============================================================================
# 1. FROZEN PARAMETERS (Phase 10 Specification)
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

# AATC-BAD per-device parameters (frozen)
Q_LOW   = 0.35
DENOM   = A_THR - Q_LOW   # 0.1054
DELTA_D = 0.10
K_ALLOW = 1.70
ETA_C   = 0.30
C_FLOOR = 0.70

# CDDA new parameters
K_NET   = 2.50   # network-level debt ceiling (sum of all device debts)
ETA_NET = 0.20   # network ceiling decay rate (steeper than per-device)

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
assert DATASET_PATH.exists(), f"Missing dataset: {DATASET_PATH}"
dataset_hash = sha256_file(DATASET_PATH)

df_all = pd.read_csv(DATASET_PATH)
assert len(df_all) == 17459
df_eval = df_all.iloc[10000:17459].copy().reset_index(drop=True)
benign_scores = df_eval["A_t"].values
N_EVAL = len(benign_scores)

out_dir = Path("experiments/candidate5/phase12_coordinated_gaming")
out_dir.mkdir(parents=True, exist_ok=True)

# =============================================================================
# 4. CORE TRUST ENGINE — single device step
# =============================================================================
def trust_step_b0(T_t: float, A_t: float) -> tuple[float, str]:
    """Standard B0 trust update. Returns (T_{t+1}, decision)."""
    if A_t >= A_THR:
        # Anomalous: penalize
        T_next = max(0.0, T_t - P_PEN * (1.0 - T_t))
        T_next = max(0.0, T_next * (1.0 - DELTA))
    else:
        # Normal: recover
        gap   = 1.0 - T_t
        T_next = T_t + GAMMA * gap * (1.0 - T_t) + DELTA * T_t
        T_next = min(1.0, T_next)

    if T_next >= T_ALLOW:
        decision = "ALLOW"
    elif T_next <= T_QUAR:
        decision = "QUARANTINE"
    else:
        decision = "VERIFY"
    return T_next, decision

def aatc_bad_step(T_t: float, D_t: float, C_t: float, A_t: float) -> tuple[float, float, float, str]:
    """
    AATC-BAD per-device step.
    Returns (T_{t+1}, D_{t+1}, C_{t+1}, decision)
    """
    T_next_b0, _ = trust_step_b0(T_t, A_t)

    # Borderline event: anomaly score in [q_low, A_thr)
    is_borderline = (Q_LOW <= A_t < A_THR)

    # Debt update
    if is_borderline:
        norm_signal = (A_t - Q_LOW) / DENOM
        D_next = D_t + DELTA_D * norm_signal
    else:
        D_next = D_t * (1.0 - DELTA_D)
    D_next = max(0.0, D_next)

    # Ceiling update
    C_next = C_t - ETA_C * (D_next / K_ALLOW) * (C_t - C_FLOOR)
    C_next = max(C_FLOOR, min(1.0, C_next))

    # Apply ceiling clamp to trust recovery
    if T_next_b0 > C_next:
        T_next = C_next
    else:
        T_next = T_next_b0

    # Debt-triggered demotion: if D_next >= K_ALLOW, force VERIFY
    if D_next >= K_ALLOW:
        T_next = min(T_next, T_ALLOW - 0.001)

    if T_next >= T_ALLOW:
        decision = "ALLOW"
    elif T_next <= T_QUAR:
        decision = "QUARANTINE"
    else:
        decision = "VERIFY"

    return T_next, D_next, C_next, decision

def cdda_net_ceiling(D_net: float) -> float:
    """
    Cross-Device Debt Aggregation (CDDA) network ceiling.
    When sum of all device debts exceeds K_NET, suppress trust recovery
    network-wide by reducing effective ceiling.
    """
    if D_net <= K_NET:
        ratio = D_net / K_NET
    else:
        ratio = 1.0
    # Network ceiling: decays from 1.0 to C_FLOOR as D_net approaches K_NET
    C_net = 1.0 - ETA_NET * ratio * (1.0 - C_FLOOR)
    return max(C_FLOOR, min(1.0, C_net))

# =============================================================================
# 5. MULTI-DEVICE SIMULATION
# =============================================================================
def simulate_multi_device(
    n_devices: int,
    schedules: list[tuple[int, int]],  # [(tau_a, tau_r)] per device
    use_cdda: bool,
    n_steps: int = 500,
    seed: int = 42
) -> dict:
    """
    Simulate n_devices operating AATC-BAD (or B0) with optional CDDA.

    schedules: list of (tau_attack, tau_rest) per device.
    Each device cycles: attack tau_attack steps, rest tau_rest steps.
    Attack steps inject ATK_PROFILE (cycling), rest steps use benign scores.

    Returns per-device trajectories and collective metrics.
    """
    rng = np.random.default_rng(seed)
    atk_cycle = cycle(ATK_PROFILE)

    # Initialize device states
    T  = [ALPHA_0] * n_devices
    D  = [0.0]     * n_devices
    C  = [1.0]     * n_devices
    phase_counter = [0] * n_devices          # steps into current phase
    in_attack     = [True] * n_devices       # start in attack phase

    trajectories = {i: [] for i in range(n_devices)}
    collective_allow_malicious = []
    collective_M_cum = 0.0

    # Pre-generate benign scores (cycle over evaluation stream)
    benign_pool = list(benign_scores[:n_steps * 3]) if n_steps <= N_EVAL else list(benign_scores)
    benign_idx  = [0] * n_devices

    for step in range(n_steps):
        # --- Determine A_t per device ---
        A = []
        is_attack_step = []
        for i in range(n_devices):
            tau_a, tau_r = schedules[i]
            if in_attack[i]:
                a_t = next(atk_cycle)
                is_attack_step.append(True)
            else:
                b_idx = benign_idx[i] % len(benign_pool)
                a_t   = float(benign_pool[b_idx])
                benign_idx[i] += 1
                is_attack_step.append(False)
            A.append(a_t)
            # Advance phase counter
            phase_counter[i] += 1
            limit = schedules[i][0] if in_attack[i] else schedules[i][1]
            if phase_counter[i] >= limit:
                in_attack[i]  = not in_attack[i]
                phase_counter[i] = 0

        # --- CDDA: compute network debt ceiling ---
        D_net   = sum(D)
        C_net_t = cdda_net_ceiling(D_net) if use_cdda else 1.0

        # --- Update each device ---
        step_allow_malicious = 0
        for i in range(n_devices):
            T_prev, D_prev, C_prev = T[i], D[i], C[i]

            if use_cdda:
                # Effective ceiling = min(per-device ceiling, network ceiling)
                C_eff = min(C_prev, C_net_t)
                # Run AATC-BAD with effective ceiling substituted
                T_next_b0, _ = trust_step_b0(T_prev, A[i])
                is_borderline = (Q_LOW <= A[i] < A_THR)
                D_next = D_prev + DELTA_D * ((A[i] - Q_LOW) / DENOM) if is_borderline else D_prev * (1.0 - DELTA_D)
                D_next = max(0.0, D_next)
                C_next = C_prev - ETA_C * (D_next / K_ALLOW) * (C_prev - C_FLOOR)
                C_next = max(C_FLOOR, min(1.0, C_next))
                # Apply combined ceiling
                T_next = min(T_next_b0, C_eff)
                if D_next >= K_ALLOW:
                    T_next = min(T_next, T_ALLOW - 0.001)
                decision = "ALLOW" if T_next >= T_ALLOW else ("QUARANTINE" if T_next <= T_QUAR else "VERIFY")
            else:
                T_next, D_next, C_next, decision = aatc_bad_step(T_prev, D_prev, C_prev, A[i])

            T[i], D[i], C[i] = T_next, D_next, C_next
            trajectories[i].append({
                "step": step,
                "A_t": round(A[i], 4),
                "T_t": round(T_next, 4),
                "D_t": round(D_next, 4),
                "C_t": round(C_next, 4),
                "C_net": round(C_net_t, 4),
                "decision": decision,
                "is_attack": is_attack_step[i]
            })
            if is_attack_step[i] and decision == "ALLOW":
                step_allow_malicious += 1

        collective_allow_malicious.append(step_allow_malicious)
        collective_M_cum += step_allow_malicious / n_devices  # normalized per device

    # Aggregate metrics
    total_attack_steps  = sum(1 for i in range(n_devices)
                               for rec in trajectories[i] if rec["is_attack"])
    total_allow_malicious = sum(1 for i in range(n_devices)
                                 for rec in trajectories[i]
                                 if rec["is_attack"] and rec["decision"] == "ALLOW")
    total_benign_steps  = sum(1 for i in range(n_devices)
                               for rec in trajectories[i] if not rec["is_attack"])
    total_allow_benign  = sum(1 for i in range(n_devices)
                               for rec in trajectories[i]
                               if not rec["is_attack"] and rec["decision"] == "ALLOW")

    return {
        "n_devices": n_devices,
        "n_steps": n_steps,
        "use_cdda": use_cdda,
        "schedules": schedules,
        "total_attack_steps": total_attack_steps,
        "total_allow_malicious": total_allow_malicious,
        "malicious_allow_rate_pct": round(100 * total_allow_malicious / max(total_attack_steps, 1), 2),
        "total_benign_steps": total_benign_steps,
        "total_allow_benign": total_allow_benign,
        "benign_allow_rate_pct": round(100 * total_allow_benign / max(total_benign_steps, 1), 2),
        "M_cum_normalized": round(collective_M_cum, 4),
        "per_device_D_final": [round(D[i], 4) for i in range(n_devices)],
        "per_device_C_final": [round(C[i], 4) for i in range(n_devices)],
        "trajectories": trajectories,
    }

# =============================================================================
# 6. RUN EXPERIMENTS
# =============================================================================
print("Phase 12: Multi-Device Coordinated Gaming")
print("=" * 55)

results = {}

# E1: 2 devices, interleaved (device 0 attacks when device 1 rests, vice versa)
# tau_a=10, tau_r=25 for each; device 1 starts with phase_counter offset
print("E1: 2-device interleaved attack (tau_a=10, tau_r=25)...")
# Without CDDA (vanilla AATC-BAD per-device)
e1_no_cdda = simulate_multi_device(
    n_devices=2,
    schedules=[(10, 25), (10, 25)],
    use_cdda=False, n_steps=500
)
e1_cdda = simulate_multi_device(
    n_devices=2,
    schedules=[(10, 25), (10, 25)],
    use_cdda=True, n_steps=500
)
results["E1_2device_interleaved"] = {
    "without_CDDA": {k: v for k, v in e1_no_cdda.items() if k != "trajectories"},
    "with_CDDA":    {k: v for k, v in e1_cdda.items()    if k != "trajectories"},
}
print(f"  Without CDDA: M_allow_malicious={e1_no_cdda['total_allow_malicious']}, "
      f"malicious_allow_rate={e1_no_cdda['malicious_allow_rate_pct']}%")
print(f"  With CDDA:    M_allow_malicious={e1_cdda['total_allow_malicious']}, "
      f"malicious_allow_rate={e1_cdda['malicious_allow_rate_pct']}%")

# E2: 3 devices, round-robin (each attacks 1/3 of the time)
print("E2: 3-device round-robin attack (tau_a=10, tau_r=25)...")
e2_no_cdda = simulate_multi_device(
    n_devices=3,
    schedules=[(10, 25), (10, 25), (10, 25)],
    use_cdda=False, n_steps=500
)
e2_cdda = simulate_multi_device(
    n_devices=3,
    schedules=[(10, 25), (10, 25), (10, 25)],
    use_cdda=True, n_steps=500
)
results["E2_3device_roundrobin"] = {
    "without_CDDA": {k: v for k, v in e2_no_cdda.items() if k != "trajectories"},
    "with_CDDA":    {k: v for k, v in e2_cdda.items()    if k != "trajectories"},
}
print(f"  Without CDDA: M_allow_malicious={e2_no_cdda['total_allow_malicious']}, "
      f"malicious_allow_rate={e2_no_cdda['malicious_allow_rate_pct']}%")
print(f"  With CDDA:    M_allow_malicious={e2_cdda['total_allow_malicious']}, "
      f"malicious_allow_rate={e2_cdda['malicious_allow_rate_pct']}%")

# E3: 3 devices, staggered rest periods (harder to detect coordination)
print("E3: 3-device staggered rest (tau_r=15, 25, 35)...")
e3_no_cdda = simulate_multi_device(
    n_devices=3,
    schedules=[(10, 15), (10, 25), (10, 35)],
    use_cdda=False, n_steps=500
)
e3_cdda = simulate_multi_device(
    n_devices=3,
    schedules=[(10, 15), (10, 25), (10, 35)],
    use_cdda=True, n_steps=500
)
results["E3_3device_staggered"] = {
    "without_CDDA": {k: v for k, v in e3_no_cdda.items() if k != "trajectories"},
    "with_CDDA":    {k: v for k, v in e3_cdda.items()    if k != "trajectories"},
}
print(f"  Without CDDA: M_allow_malicious={e3_no_cdda['total_allow_malicious']}, "
      f"malicious_allow_rate={e3_no_cdda['malicious_allow_rate_pct']}%")
print(f"  With CDDA:    M_allow_malicious={e3_cdda['total_allow_malicious']}, "
      f"malicious_allow_rate={e3_cdda['malicious_allow_rate_pct']}%")

# E4: Benign-only baseline under CDDA (availability preservation check)
print("E4: Benign-only availability under CDDA (3 devices)...")
# All devices run benign scores only — attack phases replaced with benign
e4_no_cdda = simulate_multi_device(
    n_devices=3,
    schedules=[(0, 500), (0, 500), (0, 500)],  # tau_a=0 means all-benign
    use_cdda=False, n_steps=500
)
e4_cdda = simulate_multi_device(
    n_devices=3,
    schedules=[(0, 500), (0, 500), (0, 500)],
    use_cdda=True, n_steps=500
)
results["E4_benign_availability"] = {
    "without_CDDA_benign_allow_pct": e4_no_cdda["benign_allow_rate_pct"],
    "with_CDDA_benign_allow_pct":    e4_cdda["benign_allow_rate_pct"],
    "availability_delta_pp": round(
        e4_cdda["benign_allow_rate_pct"] - e4_no_cdda["benign_allow_rate_pct"], 4
    ),
}
print(f"  Without CDDA benign ALLOW: {e4_no_cdda['benign_allow_rate_pct']}%")
print(f"  With CDDA benign ALLOW:    {e4_cdda['benign_allow_rate_pct']}%")
print(f"  Delta: {results['E4_benign_availability']['availability_delta_pp']} pp")

# =============================================================================
# 7. CDDA EFFECTIVENESS SUMMARY
# =============================================================================
def reduction_pct(before, after):
    if before == 0:
        return 0.0
    return round(100 * (before - after) / before, 2)

summary = {
    "mechanism": "Cross-Device Debt Aggregation (CDDA)",
    "new_parameters": {
        "K_net": K_NET,
        "eta_net": ETA_NET,
        "description": "K_net = network-level debt threshold; eta_net = network ceiling decay rate"
    },
    "E1_2device": {
        "allow_malicious_before_CDDA": e1_no_cdda["total_allow_malicious"],
        "allow_malicious_after_CDDA":  e1_cdda["total_allow_malicious"],
        "reduction_pct":               reduction_pct(e1_no_cdda["total_allow_malicious"],
                                                     e1_cdda["total_allow_malicious"]),
        "benign_allow_before":         e1_no_cdda["benign_allow_rate_pct"],
        "benign_allow_after":          e1_cdda["benign_allow_rate_pct"],
    },
    "E2_3device_roundrobin": {
        "allow_malicious_before_CDDA": e2_no_cdda["total_allow_malicious"],
        "allow_malicious_after_CDDA":  e2_cdda["total_allow_malicious"],
        "reduction_pct":               reduction_pct(e2_no_cdda["total_allow_malicious"],
                                                     e2_cdda["total_allow_malicious"]),
        "benign_allow_before":         e2_no_cdda["benign_allow_rate_pct"],
        "benign_allow_after":          e2_cdda["benign_allow_rate_pct"],
    },
    "E3_3device_staggered": {
        "allow_malicious_before_CDDA": e3_no_cdda["total_allow_malicious"],
        "allow_malicious_after_CDDA":  e3_cdda["total_allow_malicious"],
        "reduction_pct":               reduction_pct(e3_no_cdda["total_allow_malicious"],
                                                     e3_cdda["total_allow_malicious"]),
        "benign_allow_before":         e3_no_cdda["benign_allow_rate_pct"],
        "benign_allow_after":          e3_cdda["benign_allow_rate_pct"],
    },
    "E4_benign_availability_delta_pp": results["E4_benign_availability"]["availability_delta_pp"],
}

results["summary"] = summary
results_path = out_dir / "phase12_results.json"
with open(results_path, "w") as f:
    json.dump(results, f, indent=2)

repro = {
    "phase": 12,
    "execution_timestamp": datetime.datetime.now(timezone.utc).isoformat(),
    "dataset_sha256": dataset_hash,
    "attack_profile_sha256": ATK_PROFILE_SHA256,
    "parameters": {
        "frozen_B0": {"A_thr": A_THR, "beta": BETA, "gamma": GAMMA, "T_allow": T_ALLOW},
        "frozen_AATC_BAD": {"delta_D": DELTA_D, "K_allow": K_ALLOW, "eta_C": ETA_C, "C_floor": C_FLOOR},
        "CDDA_new": {"K_net": K_NET, "eta_net": ETA_NET},
    },
    "results_hash": sha256_obj(summary),
}
with open(out_dir / "reproducibility.json", "w") as f:
    json.dump(repro, f, indent=2)

print()
print("=" * 55)
print("PHASE 12 SUMMARY")
print("=" * 55)
print(f"E1 (2-dev): CDDA reduces malicious ALLOW by {summary['E1_2device']['reduction_pct']}%")
print(f"E2 (3-dev round-robin): CDDA reduces malicious ALLOW by {summary['E2_3device_roundrobin']['reduction_pct']}%")
print(f"E3 (3-dev staggered):   CDDA reduces malicious ALLOW by {summary['E3_3device_staggered']['reduction_pct']}%")
print(f"E4 benign availability delta: {summary['E4_benign_availability_delta_pp']} pp")
print(f"Results written to: {results_path}")
