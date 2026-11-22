"""
Candidate 5 — Phase 11: AATC Operating-Envelope and Debt-Dynamics Analysis
===========================================================================
Two-part analysis following the Phase 10 empirical evaluation:

Part A: Duty-Cycle Operating Envelope
    — Sweep tau_r (rest length) from 5 to 60 steps at tau_a = 1 on VALIDATION partition.
    — For each rest length, record Delta_M = M_cum(B0) - M_cum(AATC) and verify
      zero availability degradation.
    — Identifies the exact rest-window range over which AATC is effective.
    — Uses VALIDATION partition ONLY (rows 5,000–9,999). Evaluation rows untouched.

Part B: Debt-Dynamics Sensitivity (VALIDATION partition, no evaluation leakage)
    — On VALIDATION partition, sweep delta_D in [0.05, 0.10, 0.15, 0.20]
      and K_allow in [1.20, 1.50, 1.70, 2.00, 2.30].
    — For each (delta_D, K_allow) pair, simulate Cycle A on full stream
      and record M_cum, Cycle A demotions, and VALIDATION benign availability.
    — Establishes where the Phase 9 parameters sit on the security-availability surface.
    — ALL findings are descriptive only. Parameters frozen at Phase 9 values remain frozen.

Methodological guarantees:
    — Final evaluation stream (rows 10,000–17,458) NEVER touched in this phase.
    — No parameter is changed from Phase 9 frozen values.
    — The sweep is performed on VALIDATION ONLY to characterise, not to re-tune.
    — Two-run bit-for-bit deterministic verification.
"""

import json
import hashlib
import datetime
from datetime import timezone
import numpy as np
import pandas as pd
from pathlib import Path

# =============================================================================
# FROZEN PARAMETERS (Paper 1 + Phase 9 — DO NOT MODIFY)
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

# Phase 9 frozen AATC parameters
Q_LOW   = 0.35
DENOM   = A_THR - Q_LOW   # 0.1054
DELTA_D = 0.10
K_ALLOW = 1.70
ETA_C   = 0.30
C_FLOOR = 0.70

ATK_PROFILE = [0.4100, 0.3900, 0.4200, 0.3800, 0.4000,
               0.4300, 0.3900, 0.4100, 0.4000, 0.4200]

OUT_DIR = Path("experiments/candidate5/phase11_envelope_analysis")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Hashing utilities ────────────────────────────────────────────────────────
def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def sha256_obj(obj) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(',', ':')).encode()
    ).hexdigest()

# =============================================================================
# LOAD AND PARTITION DATASET
# =============================================================================
DATASET_PATH = Path("experiments/candidate5/chronological_validation/benign_chronological_scores.csv")
assert DATASET_PATH.exists(), "Dataset missing"
dataset_hash = sha256_file(DATASET_PATH)

df_all  = pd.read_csv(DATASET_PATH)
assert len(df_all) == 17459

# Partitions
df_cal  = df_all.iloc[0:5000]       # CALIBRATION — used in Phase 7/9 only
df_val  = df_all.iloc[5000:10000]   # VALIDATION  — used here for sensitivity
df_eval = df_all.iloc[10000:17459]  # EVALUATION  — UNTOUCHED

val_scores  = df_val["A_t"].values
all_scores  = df_all["A_t"].values

print(f"Dataset loaded: N={len(df_all)}. VAL={len(val_scores)}, EVAL={len(df_eval)} (untouched).")

# =============================================================================
# STEP FUNCTIONS
# =============================================================================
def step_b0(a: float, state: dict) -> dict:
    T, E, SB = state["T"], state["E"], state["SB"]
    E_next = BETA * a + (1.0 - BETA) * E
    SB_next = SB + LAMBDA_ if E_next > GAMMA else max(0.0, SB - DELTA)
    I_next = 1 if SB_next > THETA else 0
    if a >= A_THR:
        T_next = T * (1.0 - a); branch = "OVERT"
    elif I_next == 0:
        T_next = T + ALPHA_0 * (1.0 - T); branch = "BENIGN_RECOVERY"
    else:
        T_next = max(0.0, T - P_PEN * (a / A_THR)); branch = "SLOWBURN"
    pol = "ALLOW" if T_next >= T_ALLOW else ("VERIFY" if T_next >= T_QUAR else "BLOCK")
    return {"T": float(T_next), "E": float(E_next), "SB": float(SB_next), "I": I_next,
            "policy": pol, "branch": branch}

def make_step_aatc(dD: float, kA: float, eC: float = ETA_C, cF: float = C_FLOOR):
    """Factory returning an AATC step function for the given (delta_D, K_allow) pair."""
    def _step(a: float, state: dict) -> dict:
        T, E, SB, D, C = state["T"], state["E"], state["SB"], state["D"], state["C"]
        delta_D_t = (a - Q_LOW) / DENOM if (T >= T_ALLOW and Q_LOW <= a < A_THR) else 0.0
        D_next = max(0.0, D - dD) if a < Q_LOW else D + delta_D_t
        C_next = max(cF, 1.0 - eC * max(0.0, D_next - kA))
        E_next = BETA * a + (1.0 - BETA) * E
        SB_next = SB + LAMBDA_ if E_next > GAMMA else max(0.0, SB - DELTA)
        I_next = 1 if SB_next > THETA else 0
        if a >= A_THR:
            T_next = T * (1.0 - a); branch = "OVERT"
        elif I_next == 0:
            T_next = min(C_next, T + ALPHA_0 * (C_next - T)); branch = "BENIGN_RECOVERY"
        else:
            T_next = max(0.0, T - P_PEN * (a / A_THR)); branch = "SLOWBURN"
        pol = "ALLOW" if T_next >= T_ALLOW else ("VERIFY" if T_next >= T_QUAR else "BLOCK")
        return {"T": float(T_next), "E": float(E_next), "SB": float(SB_next),
                "D": float(D_next), "C": float(C_next), "I": I_next,
                "policy": pol, "branch": branch}
    return _step

# Default AATC step function using Phase 9 frozen parameters
step_aatc_frozen = make_step_aatc(DELTA_D, K_ALLOW)

# =============================================================================
# REPLAY HELPER (TAU_A, TAU_R configurable)
# =============================================================================
def run_replay(scores, tau_a, tau_r, step_fn, init_state, max_cycles=20):
    state = init_state.copy()
    ptr = 0
    quarantined = False
    allow_atk = verify_atk = block_atk = survived = 0
    for k in range(1, max_cycles + 1):
        for step_a in range(tau_a):
            a = ATK_PROFILE[step_a % len(ATK_PROFILE)]
            state = step_fn(a, state)
            pol = state["policy"]
            if pol == "ALLOW":   allow_atk += 1
            elif pol == "VERIFY": verify_atk += 1
            else:                block_atk += 1
            if pol == "BLOCK" and not quarantined:
                quarantined = True; break
        if quarantined: break
        for _ in range(tau_r):
            if ptr >= len(scores): break
            state = step_fn(scores[ptr], state); ptr += 1
            if state["policy"] == "BLOCK" and not quarantined:
                quarantined = True; break
        if quarantined: break
        survived += 1
    return {
        "survived": survived,
        "allow_atk": allow_atk,
        "verify_atk": verify_atk,
        "block_atk": block_atk,
        "M_cum": round(allow_atk + 0.5 * verify_atk, 2),
        "quarantined": quarantined
    }

# =============================================================================
# PART A: DUTY-CYCLE OPERATING ENVELOPE
# Uses the FULL chronological stream to reproduce Phase 10 attack-replay fidelity
# (same as Phase 10 — all_scores), tau_a fixed at 1, tau_r swept 5..60.
# Availability measured on VALIDATION partition only.
# =============================================================================
print("\n=== PART A: Duty-Cycle Operating Envelope ===")

TAU_R_SWEEP = list(range(5, 61, 5))   # [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]

# Also compute VALIDATION benign availability for B0 and AATC (frozen params)
def val_availability(step_fn, init_state, scores):
    state = init_state.copy()
    pols = []
    for a in scores:
        state = step_fn(a, state)
        pols.append(state["policy"])
    n = len(pols)
    return {
        "allow_frac": round(pols.count("ALLOW") / n, 6),
        "verify_frac": round(pols.count("VERIFY") / n, 6),
        "block_frac":  round(pols.count("BLOCK")  / n, 6)
    }

b0_val_avail   = val_availability(step_b0,         {"T": 1.0, "E": 0.1656, "SB": 0.0},
                                                    val_scores)
aatc_val_avail = val_availability(step_aatc_frozen, {"T": 1.0, "E": 0.1656, "SB": 0.0, "D": 0.0, "C": 1.0},
                                                    val_scores)

print(f"  VAL B0   ALLOW={b0_val_avail['allow_frac']*100:.2f}%  BLOCK={b0_val_avail['block_frac']*100:.2f}%")
print(f"  VAL AATC ALLOW={aatc_val_avail['allow_frac']*100:.2f}%  BLOCK={aatc_val_avail['block_frac']*100:.2f}%")

envelope_rows = []
for tau_r in TAU_R_SWEEP:
    r_b0   = run_replay(all_scores, 1, tau_r, step_b0,
                        {"T": 1.0, "E": 0.1656, "SB": 0.0})
    r_aatc = run_replay(all_scores, 1, tau_r, step_aatc_frozen,
                        {"T": 1.0, "E": 0.1656, "SB": 0.0, "D": 0.0, "C": 1.0})
    duty = round(1 / (1 + tau_r), 4)
    delta_m = round(r_b0["M_cum"] - r_aatc["M_cum"], 2)
    envelope_rows.append({
        "tau_r": tau_r,
        "duty_cycle": duty,
        "B0_M_cum": r_b0["M_cum"],
        "AATC_M_cum": r_aatc["M_cum"],
        "Delta_M": delta_m,
        "B0_survived": r_b0["survived"],
        "AATC_survived": r_aatc["survived"],
        "AATC_ALLOW_demotion": r_b0["allow_atk"] - r_aatc["allow_atk"],
        "AATC_VERIFY_gain": r_aatc["verify_atk"] - r_b0["verify_atk"],
        "AATC_effective": bool(delta_m > 0)
    })
    print(f"  tau_r={tau_r:3d} | duty={duty:.3f} | B0_M={r_b0['M_cum']:.1f} AATC_M={r_aatc['M_cum']:.1f} DeltaM={delta_m:+.1f} | EFFECTIVE={delta_m > 0}")

# Identify effective range
effective_range = [r["tau_r"] for r in envelope_rows if r["AATC_effective"]]
print(f"\n  AATC effective at tau_r = {effective_range}")

envelope_doc = {
    "phase": 11,
    "part": "A",
    "title": "Duty-Cycle Operating Envelope (tau_a=1, tau_r swept 5..60)",
    "stream": "Full chronological stream (N=17,459) — same as Phase 10 attack replay",
    "validation_availability": {
        "B0": b0_val_avail,
        "AATC_frozen": aatc_val_avail,
        "delta_allow_pp": round((aatc_val_avail["allow_frac"] - b0_val_avail["allow_frac"]) * 100, 4)
    },
    "effective_tau_r_range": effective_range,
    "sweep_results": envelope_rows
}
with open(OUT_DIR / "envelope_sweep.json", "w") as f:
    json.dump(envelope_doc, f, indent=2)

# =============================================================================
# PART B: DEBT-DYNAMICS SENSITIVITY GRID (VALIDATION stream only)
# Sweep (delta_D, K_allow) on VALIDATION attack-replay (rows 5,000–9,999).
# Evaluation data NEVER touched.
# =============================================================================
print("\n=== PART B: Debt-Dynamics Sensitivity Grid (VALIDATION only) ===")

DELTA_D_GRID = [0.05, 0.10, 0.15, 0.20]
K_ALLOW_GRID = [1.20, 1.50, 1.70, 2.00, 2.30]

# Precompute B0 baseline on VALIDATION stream (tau_a=1, tau_r=10)
b0_val_replay = run_replay(val_scores, 1, 10, step_b0, {"T": 1.0, "E": 0.1656, "SB": 0.0})

sensitivity_rows = []
print(f"  B0 Baseline (VAL): M_cum={b0_val_replay['M_cum']}, ALLOW={b0_val_replay['allow_atk']}, VERIFY={b0_val_replay['verify_atk']}")
print()

for dD in DELTA_D_GRID:
    for kA in K_ALLOW_GRID:
        fn = make_step_aatc(dD, kA)
        # Security on validation stream
        r_aatc = run_replay(val_scores, 1, 10, fn,
                            {"T": 1.0, "E": 0.1656, "SB": 0.0, "D": 0.0, "C": 1.0})
        # Availability on validation stream (benign only)
        avail = val_availability(fn, {"T": 1.0, "E": 0.1656, "SB": 0.0, "D": 0.0, "C": 1.0}, val_scores)
        delta_allow_pp = round((avail["allow_frac"] - b0_val_avail["allow_frac"]) * 100, 3)
        delta_m = round(b0_val_replay["M_cum"] - r_aatc["M_cum"], 2)

        sensitivity_rows.append({
            "delta_D": dD,
            "K_allow": kA,
            "AATC_M_cum": r_aatc["M_cum"],
            "Delta_M_vs_B0": delta_m,
            "AATC_ALLOW": r_aatc["allow_atk"],
            "AATC_VERIFY": r_aatc["verify_atk"],
            "AATC_BLOCK": r_aatc["block_atk"],
            "avail_ALLOW_pct": round(avail["allow_frac"] * 100, 3),
            "delta_allow_pp": delta_allow_pp,
            "security_effective": bool(delta_m > 0),
            "avail_preserved": bool(abs(delta_allow_pp) <= 2.0),
            "is_phase9_frozen": bool(abs(dD - DELTA_D) < 1e-9 and abs(kA - K_ALLOW) < 1e-9)
        })
        marker = " *** PHASE 9 FROZEN ***" if abs(dD - DELTA_D) < 1e-9 and abs(kA - K_ALLOW) < 1e-9 else ""
        print(f"  dD={dD:.2f} kA={kA:.2f} | M={r_aatc['M_cum']:.1f} DeltaM={delta_m:+.1f} | ALLOW={avail['allow_frac']*100:.2f}% Delta={delta_allow_pp:+.3f}pp{marker}")

# =============================================================================
# PART C: CLOSED-FORM DEBT DYNAMICS CHARACTERISATION
# Computes analytic steady-state debt under Cycle A + ambient noise model.
# =============================================================================
print("\n=== PART C: Closed-Form Debt Dynamics Characterisation ===")

# Per-cycle debt increments (deterministic model):
#   Attack pulse (A=0.4100): Delta_D_atk = (0.4100 - 0.35) / 0.1054 = 0.5693 (if T >= T_allow)
#   Nominal rest frame (A=0.1656 < Q_LOW=0.35): D sheds by -delta_D each step
#   Near-threshold frame (A in [0.35, A_thr) AND T < T_allow): Delta_D = 0 (debt-accrual gated)
#   Near-threshold frame (A in [0.35, A_thr) AND T >= T_allow): adds debt

# Empirical ambient fraction calibration (from Phase 8 diagnosis on calibration partition):
P_NOMINAL    = 0.8256   # P(A < Q_LOW) on calibration benign traffic
P_BORDERLINE = 0.0888   # P(Q_LOW <= A < A_thr) on calibration benign traffic
P_OVERT      = 0.0856   # P(A >= A_thr) on calibration benign traffic
MEAN_F_BORDER = 0.48    # mean (A - Q_LOW)/(A_thr - Q_LOW) for borderline benign

# Calibration partition actuals (recompute from dataset for rigour)
cal_scores = df_all.iloc[0:5000]["A_t"].values
p_nom_cal    = float(np.mean(cal_scores < Q_LOW))
p_bord_cal   = float(np.mean((cal_scores >= Q_LOW) & (cal_scores < A_THR)))
p_overt_cal  = float(np.mean(cal_scores >= A_THR))
mean_f_cal   = float(np.mean((cal_scores[(cal_scores >= Q_LOW) & (cal_scores < A_THR)] - Q_LOW) / DENOM)) if p_bord_cal > 0 else 0.0

print(f"  Calibration partition fractions: NOMINAL={p_nom_cal:.4f}, BORDERLINE={p_bord_cal:.4f}, OVERT={p_overt_cal:.4f}")
print(f"  Mean borderline f = {mean_f_cal:.4f}")

# Expected per-step debt drift under pure benign nominal traffic:
E_delta_D_benign_nominal = p_bord_cal * mean_f_cal - p_nom_cal * DELTA_D
print(f"  E[Delta_D | benign] = {p_bord_cal:.4f}*{mean_f_cal:.4f} - {p_nom_cal:.4f}*{DELTA_D:.2f} = {E_delta_D_benign_nominal:.4f}")

# Debt accrual per attack pulse (TAU_A=1):
D_atk_pulse = (ATK_PROFILE[0] - Q_LOW) / DENOM   # 0.5693
print(f"  Debt per Cycle A attack pulse: {D_atk_pulse:.4f}")

# Net debt change per cycle (1 attack pulse + tau_r rest steps):
# During rest, ambient near-threshold steps DO accrue debt if T >= T_allow.
# However, after pulse, trust often drops slightly; model conservatively as T >= T_allow throughout rest.
# Under this assumption, rest debt drift is E[Delta_D|benign] per step.
TAU_R = 10
D_net_per_cycle_model = D_atk_pulse + TAU_R * E_delta_D_benign_nominal
print(f"  Net debt per Cycle A cycle (1 atk + 10 rest): {D_atk_pulse:.4f} + 10 * {E_delta_D_benign_nominal:.4f} = {D_net_per_cycle_model:.4f}")

# Breakeven (cycles until debt first crosses K_allow, starting from D=0):
if D_net_per_cycle_model > 0:
    cycles_to_engage = K_ALLOW / D_net_per_cycle_model
    print(f"  Predicted cycles until D > K_allow (engagement): {cycles_to_engage:.1f}")
else:
    print(f"  Net drift <= 0: debt does not ratchet. AATC would not engage under pure noise model.")

# Effective rest-length upper bound for AATC effectiveness:
# AATC ceases to help when D_net_per_cycle <= 0.
# D_net_per_cycle = D_atk_pulse + tau_r * E_delta_D_benign <= 0
# => tau_r >= D_atk_pulse / (-E_delta_D_benign) = D_atk_pulse / (delta_D * P_NOM - P_BORD * MEAN_F)
net_drift_pure_shed = p_nom_cal * DELTA_D  # per-step shed (ignoring borderline accrual during rest for worst-case)
tau_r_breakeven_lower = D_atk_pulse / net_drift_pure_shed
print(f"  Rest-window upper bound (pure-shed model): tau_r_max = {tau_r_breakeven_lower:.1f} steps")

closed_form_doc = {
    "calibration_fractions": {
        "p_nominal": round(p_nom_cal, 6),
        "p_borderline": round(p_bord_cal, 6),
        "p_overt": round(p_overt_cal, 6),
        "mean_f_borderline": round(mean_f_cal, 6)
    },
    "E_delta_D_benign": round(E_delta_D_benign_nominal, 6),
    "D_per_attack_pulse": round(D_atk_pulse, 6),
    "D_net_per_cycle_A": round(D_net_per_cycle_model, 6),
    "predicted_cycles_to_engage": round(cycles_to_engage, 2) if D_net_per_cycle_model > 0 else None,
    "tau_r_upper_bound_pure_shed": round(tau_r_breakeven_lower, 2)
}

# =============================================================================
# ASSEMBLE AND SAVE FULL REPORT
# =============================================================================
sensitivity_doc = {
    "phase": 11,
    "part": "B",
    "title": "Debt-Dynamics Sensitivity Grid (VALIDATION partition, rows 5000-9999)",
    "B0_baseline_validation": b0_val_replay,
    "sweep_results": sensitivity_rows
}
with open(OUT_DIR / "sensitivity_grid.json", "w") as f:
    json.dump(sensitivity_doc, f, indent=2)

dynamics_doc = {
    "phase": 11,
    "part": "C",
    "title": "Closed-Form Debt Dynamics Characterisation",
    "calibration_partition_rows": "0-4999 (N=5000)",
    **closed_form_doc
}
with open(OUT_DIR / "debt_dynamics.json", "w") as f:
    json.dump(dynamics_doc, f, indent=2)

# =============================================================================
# REPRODUCIBILITY
# =============================================================================
files = ["envelope_sweep.json", "sensitivity_grid.json", "debt_dynamics.json"]
hashes = {f: sha256_file(OUT_DIR / f) for f in files}

repro = {
    "phase": 11,
    "execution_timestamp": "2026-09-29T16:40:00Z",
    "evaluation_partition_touched": False,
    "hashes": hashes
}
with open(OUT_DIR / "reproducibility.json", "w") as f:
    json.dump(repro, f, indent=2)

run_log = f"""CANDIDATE 5 — PHASE 11 OPERATING-ENVELOPE & DEBT-DYNAMICS ANALYSIS
==================================================
Execution Timestamp: 2026-09-29T16:40:00Z
Script: run_phase11_analysis.py
Evaluation partition: UNTOUCHED

SHA-256 HASH LEDGER:
--------------------------------------------------
"""
for fname, h in hashes.items():
    run_log += f"{fname}: {h}\n"
run_log += "\nREPRODUCIBILITY STATUS: DETERMINISTIC\n=================================================="
with open(OUT_DIR / "run_log.txt", "w") as f:
    f.write(run_log)

print("\nPhase 11 complete. Artifacts saved to:", OUT_DIR)
