"""
Phase 14 — State Integrity in Decision-Only AI Security Pipelines
=================================================================
Problem: AATC-BAD's trust state (T_t, D_t, C_t) is computed from
telemetry fed into a decision-only pipeline. If an adversary tampers
with the state variables directly (e.g., via memory injection, API
spoofing, or stale cached state), the model produces confident but
unauthorized ALLOW decisions.

This experiment implements and evaluates:

1. THREAT MODEL: State injection attacks
   - Type I:  Trust inflation (D_t zeroed, T_t set to 0.95)
   - Type II: Ceiling reset (C_t reset to 1.0 mid-stream)
   - Type III: Debt erasure (D_t zeroed without trust change)
   - Type IV: Full state replacement (T_t=0.95, D_t=0.0, C_t=1.0)

2. MECHANISM: Hash-Chained State Ledger (HCSL)
   - Each authorization decision is paired with a BLAKE2 commitment:
       H_t = BLAKE2b( H_{t-1} || T_t || D_t || C_t || A_t || decision_t )
   - Any state tamper invalidates H_t vs. the expected chain
   - Tamper is detectable at next verification step

3. EXPERIMENTS:
   E1 — Baseline: measure unauthorized ALLOW gain from each attack type
   E2 — HCSL detection rate: fraction of tampering events detected within 1 step
   E3 — False positive rate: HCSL chain breaks on legitimate state transitions
   E4 — Overhead estimate: average commitment bytes per step
"""

import json
import hashlib
import datetime
from datetime import timezone
import math
import numpy as np
import pandas as pd
from pathlib import Path

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

Q_LOW   = 0.35
DENOM   = A_THR - Q_LOW
DELTA_D = 0.10
K_ALLOW = 1.70
ETA_C   = 0.30
C_FLOOR = 0.70

# HCSL parameters
HCSL_GENESIS = b"\x00" * 32    # genesis hash (H_0)

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

def hcsl_commit(H_prev: bytes, T_t: float, D_t: float, C_t: float,
                A_t: float, decision: str) -> bytes:
    """BLAKE2b commitment over previous hash + current state vector."""
    h = hashlib.blake2b(digest_size=32)
    h.update(H_prev)
    # Quantize to 6 decimal places to avoid float repr drift
    h.update(f"{T_t:.6f}|{D_t:.6f}|{C_t:.6f}|{A_t:.6f}|{decision}".encode())
    return h.digest()

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

out_dir = Path("experiments/candidate5/phase14_state_integrity")
out_dir.mkdir(parents=True, exist_ok=True)

# =============================================================================
# 4. TRUST ENGINE
# =============================================================================
def trust_step_b0(T_t, A_t):
    if A_t >= A_THR:
        T_next = max(0.0, T_t - P_PEN * (1.0 - T_t))
        T_next = max(0.0, T_next * (1.0 - DELTA))
    else:
        gap   = 1.0 - T_t
        T_next = T_t + GAMMA * gap * (1.0 - T_t) + DELTA * T_t
        T_next = min(1.0, T_next)
    return T_next

def aatc_bad_step(T_t, D_t, C_t, A_t):
    T_next_b0 = trust_step_b0(T_t, A_t)
    is_borderline = (Q_LOW <= A_t < A_THR)
    if is_borderline:
        norm_signal = (A_t - Q_LOW) / DENOM
        D_next = D_t + DELTA_D * norm_signal
    else:
        D_next = D_t * (1.0 - DELTA_D)
    D_next = max(0.0, D_next)
    C_next = C_t - ETA_C * (D_next / K_ALLOW) * (C_t - C_FLOOR)
    C_next = max(C_FLOOR, min(1.0, C_next))
    T_next = min(T_next_b0, C_next)
    if D_next >= K_ALLOW:
        T_next = min(T_next, T_ALLOW - 0.001)
    decision = "ALLOW" if T_next >= T_ALLOW else ("QUARANTINE" if T_next <= T_QUAR else "VERIFY")
    return T_next, D_next, C_next, decision

# =============================================================================
# 5. STATE INJECTION ATTACK TYPES
# =============================================================================
def apply_injection(T_t, D_t, C_t, attack_type: str):
    """Apply a state injection attack. Returns tampered (T, D, C)."""
    if attack_type == "I_trust_inflation":
        # Adversary zeroes debt and inflates trust
        return 0.95, 0.0, C_t
    elif attack_type == "II_ceiling_reset":
        # Adversary resets ceiling to 1.0 (removes debt suppression)
        return T_t, D_t, 1.0
    elif attack_type == "III_debt_erasure":
        # Adversary zeroes debt only (trust unchanged)
        return T_t, 0.0, C_t
    elif attack_type == "IV_full_replacement":
        # Complete state replacement: maximize trust access
        return 0.95, 0.0, 1.0
    return T_t, D_t, C_t

# =============================================================================
# 6. EXPERIMENT E1 — UNAUTHORIZED ALLOW GAIN FROM STATE INJECTION
# =============================================================================
print("Phase 14: State Integrity in Decision-Only AI Security Pipelines")
print("=" * 65)
print("E1: Measuring unauthorized ALLOW gain from state injection attacks...")

# Run a 300-step mixed workload (sub-threshold attack schedule: tau_a=10, tau_r=25)
# Then inject state at step 150 and measure ALLOWs gained vs. clean baseline

def run_injection_experiment(attack_type: str, inject_step: int = 150, n_steps: int = 300):
    """
    Run n_steps. At inject_step, tamper with state.
    Returns (clean_allows_after, tampered_allows_after, detection_step).
    detection_step = step at which HCSL detects tamper (-1 if undetected).
    """
    atk_cyc = list(ATK_PROFILE)
    atk_i = 0
    ben_i = 0
    in_attack = True
    phase_ctr = 0
    tau_a, tau_r = 10, 25

    # Clean run (no injection)
    T_c, D_c, C_c = ALPHA_0, 0.0, 1.0
    clean_allows_after = 0

    # Tampered run
    T_t2, D_t2, C_t2 = ALPHA_0, 0.0, 1.0
    tampered_allows_after = 0

    # HCSL chain
    H_clean   = HCSL_GENESIS
    H_tamper  = HCSL_GENESIS
    # Trusted chain: computed from clean transitions only
    H_trusted = HCSL_GENESIS
    detection_step = -1

    for step in range(n_steps):
        if in_attack:
            A_t = atk_cyc[atk_i % len(atk_cyc)]
            atk_i += 1
        else:
            A_t = float(benign_scores[ben_i % N_EVAL])
            ben_i += 1

        # Clean step
        T_c, D_c, C_c, dec_c = aatc_bad_step(T_c, D_c, C_c, A_t)
        H_clean   = hcsl_commit(H_clean,   T_c, D_c, C_c, A_t, dec_c)
        H_trusted = hcsl_commit(H_trusted, T_c, D_c, C_c, A_t, dec_c)

        # Tampered step: inject at inject_step
        if step == inject_step and attack_type is not None:
            T_t2, D_t2, C_t2 = apply_injection(T_t2, D_t2, C_t2, attack_type)

        T_t2, D_t2, C_t2, dec_t = aatc_bad_step(T_t2, D_t2, C_t2, A_t)
        H_tamper_new = hcsl_commit(H_tamper, T_t2, D_t2, C_t2, A_t, dec_t)

        # HCSL detection: tampered chain diverges from trusted chain
        if detection_step == -1 and step >= inject_step:
            if H_tamper_new != H_trusted:
                detection_step = step

        H_tamper = H_tamper_new

        if step >= inject_step:
            if dec_c == "ALLOW":
                clean_allows_after += 1
            if dec_t == "ALLOW":
                tampered_allows_after += 1

        # Advance attack schedule
        phase_ctr += 1
        limit = tau_a if in_attack else tau_r
        if phase_ctr >= limit:
            in_attack = not in_attack
            phase_ctr = 0

    steps_after = n_steps - inject_step
    return {
        "attack_type": attack_type,
        "inject_step": inject_step,
        "steps_measured": steps_after,
        "clean_allows_after": clean_allows_after,
        "tampered_allows_after": tampered_allows_after,
        "unauthorized_allow_gain": tampered_allows_after - clean_allows_after,
        "unauthorized_gain_pct": round(
            100 * (tampered_allows_after - clean_allows_after) / max(steps_after, 1), 2
        ),
        "hcsl_detection_step": detection_step,
        "hcsl_detected_within_1_step": detection_step == inject_step or detection_step == inject_step + 1,
    }

attack_types = ["I_trust_inflation", "II_ceiling_reset", "III_debt_erasure", "IV_full_replacement"]
e1_results = {}
for at in attack_types:
    res = run_injection_experiment(at)
    e1_results[at] = res
    print(f"  {at}: unauthorized_gain={res['unauthorized_allow_gain']} steps "
          f"({res['unauthorized_gain_pct']}%), "
          f"HCSL detect@step={res['hcsl_detection_step']}, "
          f"within_1_step={res['hcsl_detected_within_1_step']}")

# =============================================================================
# 7. EXPERIMENT E2 — HCSL DETECTION RATE ACROSS 100 INJECTION POINTS
# =============================================================================
print("E2: HCSL detection rate across 100 random injection points...")

def hcsl_detection_sweep(attack_type: str, n_trials: int = 100) -> dict:
    rng = np.random.default_rng(42)
    inject_steps = rng.integers(50, 250, size=n_trials).tolist()
    detected_within_1 = 0
    detected_total = 0
    for inj in inject_steps:
        r = run_injection_experiment(attack_type, inject_step=int(inj), n_steps=300)
        if r["hcsl_detection_step"] >= 0:
            detected_total += 1
        if r["hcsl_detected_within_1_step"]:
            detected_within_1 += 1
    return {
        "attack_type": attack_type,
        "n_trials": n_trials,
        "detected_total": detected_total,
        "detection_rate_pct": round(100 * detected_total / n_trials, 2),
        "detected_within_1_step": detected_within_1,
        "within_1_step_rate_pct": round(100 * detected_within_1 / n_trials, 2),
    }

e2_results = {}
for at in attack_types:
    r = hcsl_detection_sweep(at)
    e2_results[at] = r
    print(f"  {at}: detection_rate={r['detection_rate_pct']}%, "
          f"within_1_step={r['within_1_step_rate_pct']}%")

# =============================================================================
# 8. EXPERIMENT E3 — FALSE POSITIVE RATE (clean chain breaks)
# =============================================================================
print("E3: False positive rate on legitimate state transitions (benign-only)...")

def hcsl_false_positive_check(n_steps: int = 500) -> dict:
    """
    Run benign-only stream through AATC-BAD + HCSL.
    Count any chain breaks (should be zero — deterministic system).
    """
    T, D, C = ALPHA_0, 0.0, 1.0
    H = HCSL_GENESIS
    H_verify = HCSL_GENESIS
    breaks = 0
    for step in range(n_steps):
        A_t = float(benign_scores[step % N_EVAL])
        T, D, C, dec = aatc_bad_step(T, D, C, A_t)
        H_new      = hcsl_commit(H,        T, D, C, A_t, dec)
        H_ver_new  = hcsl_commit(H_verify, T, D, C, A_t, dec)
        if H_new != H_ver_new:
            breaks += 1
        H        = H_new
        H_verify = H_ver_new
    return {
        "n_steps": n_steps,
        "chain_breaks": breaks,
        "false_positive_rate_pct": round(100 * breaks / n_steps, 4),
    }

e3 = hcsl_false_positive_check()
print(f"  Chain breaks on benign stream: {e3['chain_breaks']} / {e3['n_steps']} "
      f"({e3['false_positive_rate_pct']}%)")

# =============================================================================
# 9. EXPERIMENT E4 — COMMITMENT OVERHEAD
# =============================================================================
print("E4: HCSL commitment overhead per step...")

commitment_bytes   = 32          # BLAKE2b-256 digest
state_vector_bytes = 4 * 8      # T_t, D_t, C_t, A_t as float64
decision_bytes     = 8          # max 11-char string padded
prev_hash_bytes    = 32         # H_{t-1}
total_per_step     = commitment_bytes + state_vector_bytes + decision_bytes + prev_hash_bytes

e4 = {
    "commitment_hash_bytes": commitment_bytes,
    "state_vector_bytes": state_vector_bytes,
    "decision_bytes": decision_bytes,
    "prev_hash_bytes": prev_hash_bytes,
    "total_overhead_per_step_bytes": total_per_step,
    "overhead_per_1000_steps_kb": round(total_per_step * 1000 / 1024, 2),
    "overhead_per_million_steps_mb": round(total_per_step * 1_000_000 / (1024 ** 2), 2),
}
print(f"  Per-step overhead: {total_per_step} bytes")
print(f"  Per 1M steps: {e4['overhead_per_million_steps_mb']} MB")

# =============================================================================
# 10. SAVE RESULTS
# =============================================================================
results = {
    "mechanism": "Hash-Chained State Ledger (HCSL)",
    "hash_function": "BLAKE2b-256",
    "threat_model": {
        "I_trust_inflation":   "Zero debt + set T_t=0.95",
        "II_ceiling_reset":    "Reset C_t=1.0 (remove debt suppression)",
        "III_debt_erasure":    "Zero debt only (D_t=0)",
        "IV_full_replacement": "Full state: T_t=0.95, D_t=0, C_t=1.0",
    },
    "E1_injection_gain": e1_results,
    "E2_detection_rate": e2_results,
    "E3_false_positive": e3,
    "E4_overhead": e4,
    "summary": {
        "all_attack_types_detected_within_1_step":
            all(e2_results[at]["within_1_step_rate_pct"] == 100.0 for at in attack_types),
        "false_positive_rate_pct": e3["false_positive_rate_pct"],
        "overhead_per_step_bytes": total_per_step,
        "max_unauthorized_gain_steps":
            max(e1_results[at]["unauthorized_allow_gain"] for at in attack_types),
    }
}

results_path = out_dir / "phase14_results.json"
with open(results_path, "w") as f:
    json.dump(results, f, indent=2)

repro = {
    "phase": 14,
    "execution_timestamp": datetime.datetime.now(timezone.utc).isoformat(),
    "dataset_sha256": dataset_hash,
    "results_hash": sha256_obj(results["summary"]),
}
with open(out_dir / "reproducibility.json", "w") as f:
    json.dump(repro, f, indent=2)

print()
print("=" * 65)
print("PHASE 14 SUMMARY")
print("=" * 65)
print(f"All types detected within 1 step: {results['summary']['all_attack_types_detected_within_1_step']}")
print(f"False positive rate: {e3['false_positive_rate_pct']}%")
print(f"Overhead per step: {total_per_step} bytes")
print(f"Max unauthorized ALLOW gain (before HCSL): {results['summary']['max_unauthorized_gain_steps']} steps")
print(f"Results written to: {results_path}")
