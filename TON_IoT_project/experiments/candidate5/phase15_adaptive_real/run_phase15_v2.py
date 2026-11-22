#!/usr/bin/env python3
"""
Phase 15 v2 — Adaptive Adversary on Real Benign Telemetry (corrected metrics)
=============================================================================
Changes from v1 (commit aa49870):
  1. Primary yield metric = count of ALLOW'd attack frames ("allows"), NOT allows/attempts.
     allows/attempts kept as secondary column "success_rate_pct".
  2. Wilcoxon computed on "allows" count, both two-sided and one-sided (B0 > AATC).
     Reports W+, N_eff, seeds_aatc_worse, p_twosided, p_onesided.
  3. Added policy A4b "debt_aware_oracle":
       - one-step lookahead using the actual next benign score
       - attacks only if ALLOW now AND predicted D_next <= K_ALLOW - 0.05
       - scans amplitude grid [0.35, 0.4554) step 0.005, picks highest A
         whose predicted D stays under the limit
  4. Same seeds (master=42, offset=99), T=1000, frozen params as aa49870.
  5. Benign scores loaded from committed .npy files:
       edge_benign_eval.npy  (7459 scores, Edge eval rows 10000:17458)
       cic_benign_eval.npy   (10000 scores, CIC Normal class)
  6. Outputs: summary_v2.json, summary_table_v2.tex (LaTeX booktabs table)
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJ_ROOT = Path(__file__).resolve().parents[4] / "TON_IoT_project"
PHASE_DIR = PROJ_ROOT / "experiments/candidate5/phase15_adaptive_real"
PARITY_REF = PROJ_ROOT / "rust-edge/crates/aatc-engine/tests/parity_reference.json"

EDGE_NPY = PHASE_DIR / "edge_benign_eval.npy"
CIC_NPY  = PHASE_DIR / "cic_benign_eval.npy"

# SHA-256 guard – must match committed files
EDGE_NPY_SHA256 = "26a184b95748954aa190f272f221d341c0afb1397545d0eaa62a9ed9b11a594d"
CIC_NPY_SHA256  = "bbc718d787fc71dbdc4ede84c39fe9aa8a7df13d7fa8d56c379553466b623b4f"

sys.path.insert(0, str(PROJ_ROOT))
from experiments.candidate5.adaptive_adversary.run_adaptive_adversary import (
    step_aatc, step_b0,
)

# ---------------------------------------------------------------------------
# Frozen parameters (identical to aa49870)
# ---------------------------------------------------------------------------
A_THR   = 0.4554
Q_LOW   = 0.35
DENOM   = A_THR - Q_LOW       # 0.1054
K_ALLOW = 1.70
DELTA_D = 0.10
ETA_C   = 0.30
C_FLOOR = 0.70
T_ALLOW = 0.80
T_QUAR  = 0.30

ATK_AMP_DEFAULT   = 0.450
ATK_AMP_JITTER_SD = 0.005
N_SEEDS = 100
T_MAX   = 1000

D_SAFE_LIMIT = K_ALLOW - 0.05  # 1.65

# Amplitude grid for oracle (descending order – prefer highest)
AMP_GRID = list(reversed([
    round(0.350 + i * 0.005, 3) for i in range(int((A_THR - 0.350) / 0.005))
]))  # [0.450, 0.445, … , 0.350]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(1 << 16):
            h.update(chunk)
    return h.hexdigest()


def _init(is_aatc: bool) -> dict:
    if is_aatc:
        return {"T": 1.0, "E": 0.0, "SB": 0.0, "D": 0.0, "C": 1.0}
    return {"T": 1.0, "E": 0.0, "SB": 0.0}


def _jitter(rng: np.random.RandomState, amp: float) -> float:
    return float(np.clip(rng.normal(amp, ATK_AMP_JITTER_SD),
                         Q_LOW + 0.005, A_THR - 0.001))


def _next(pool: np.ndarray, ptr: int):
    return float(pool[ptr % len(pool)]), (ptr + 1)


# Predict D after one AATC attack frame, given current state
def _predict_D(a: float, state: dict) -> float:
    T, D = state["T"], state["D"]
    if (T >= T_ALLOW) and (Q_LOW <= a < A_THR):
        return D + (a - Q_LOW) / DENOM
    elif a < Q_LOW:
        return max(0.0, D - DELTA_D)
    return D


# ---------------------------------------------------------------------------
# Parity check
# ---------------------------------------------------------------------------
def run_parity():
    print("PARITY CHECK vs parity_reference.json …", end=" ")
    with open(PARITY_REF) as f:
        ref = json.load(f)
    assert len(ref) == 1000
    state = {"T": 1.0, "E": 0.0, "SB": 0.0, "D": 0.0, "C": 1.0}
    for s in ref:
        state = step_aatc(s["score"], state)
        assert state["policy"] == s["policy"], f"policy @ step {s['step']}"
        assert abs(state["T"] - s["alpha"])   < 1e-4, f"alpha @ step {s['step']}"
        assert abs(state["D"] - s["debt"])    < 1e-4, f"debt  @ step {s['step']}"
        assert abs(state["C"] - s["ceiling"]) < 1e-4, f"ceil  @ step {s['step']}"
    print("PASSED (1000/1000, tol=1e-4)")
    return sha256_file(PARITY_REF)


# ---------------------------------------------------------------------------
# Policy runners — return dict of per-seed metrics
# ---------------------------------------------------------------------------
def run_greedy(step_fn, is_aatc, pool, offset, seed):
    rng = np.random.RandomState(seed)
    state = _init(is_aatc); ptr = offset
    allows = verifies = blocks = attempts = 0
    max_debt = 0.0
    for _ in range(T_MAX):
        if state["T"] >= T_ALLOW:
            a = _jitter(rng, ATK_AMP_DEFAULT); attempts += 1; atk = True
        else:
            a, ptr = _next(pool, ptr); atk = False
        state = step_fn(a, state)
        if is_aatc: max_debt = max(max_debt, state.get("D", 0.0))
        if atk:
            if state["policy"] == "ALLOW":    allows  += 1
            elif state["policy"] == "VERIFY": verifies += 1
            else:                             blocks  += 1
    return dict(policy="greedy", allows=allows, attempts=attempts,
                verifies=verifies, blocks=blocks, max_debt=round(max_debt, 6),
                final=state["policy"])


def run_cooldown(step_fn, is_aatc, pool, offset, seed):
    rng = np.random.RandomState(seed)
    state = _init(is_aatc); ptr = offset
    allows = verifies = blocks = attempts = 0
    max_debt = 0.0; tau_r = 4; t = 0
    while t < T_MAX:
        a = _jitter(rng, ATK_AMP_DEFAULT); attempts += 1; t += 1
        state = step_fn(a, state)
        if is_aatc: max_debt = max(max_debt, state.get("D", 0.0))
        if   state["policy"] == "ALLOW":  allows += 1
        elif state["policy"] == "VERIFY": verifies += 1; tau_r = min(tau_r + 2, 200)
        else:                             blocks  += 1;  tau_r = min(tau_r + 2, 200)
        for _ in range(tau_r):
            if t >= T_MAX: break
            a_r, ptr = _next(pool, ptr); t += 1
            state = step_fn(a_r, state)
            if is_aatc: max_debt = max(max_debt, state.get("D", 0.0))
    return dict(policy="cooldown", allows=allows, attempts=attempts,
                verifies=verifies, blocks=blocks, max_debt=round(max_debt, 6),
                final=state["policy"], final_tau_r=tau_r)


def run_amplitude(step_fn, is_aatc, pool, offset, seed):
    rng = np.random.RandomState(seed)
    state = _init(is_aatc); ptr = offset
    allows = verifies = blocks = attempts = 0
    max_debt = 0.0; amp = ATK_AMP_DEFAULT; tau_r = 6; t = 0
    while t < T_MAX:
        a = _jitter(rng, amp); attempts += 1; t += 1
        state = step_fn(a, state)
        if is_aatc: max_debt = max(max_debt, state.get("D", 0.0))
        if   state["policy"] == "ALLOW":  allows += 1
        elif state["policy"] == "VERIFY": verifies += 1; amp = max(0.340, amp - 0.02)
        else:                             blocks  += 1;  amp = max(0.340, amp - 0.02)
        for _ in range(tau_r):
            if t >= T_MAX: break
            a_r, ptr = _next(pool, ptr); t += 1
            state = step_fn(a_r, state)
            if is_aatc: max_debt = max(max_debt, state.get("D", 0.0))
    return dict(policy="amplitude", allows=allows, attempts=attempts,
                verifies=verifies, blocks=blocks, max_debt=round(max_debt, 6),
                final=state["policy"], final_amp=round(amp, 4))


def run_debt_pacing(step_fn, is_aatc, pool, offset, seed):
    """Debt-aware pacing: attack only when D + delta_atk <= 1.65 (from v1)."""
    rng = np.random.RandomState(seed)
    state = _init(is_aatc); ptr = offset
    allows = verifies = blocks = attempts = 0
    max_debt = 0.0; delta_atk = (ATK_AMP_DEFAULT - Q_LOW) / DENOM
    for _ in range(T_MAX):
        curr_D = state.get("D", 0.0) if is_aatc else 0.0
        if state["T"] >= T_ALLOW and (curr_D + delta_atk <= D_SAFE_LIMIT):
            a = _jitter(rng, ATK_AMP_DEFAULT); attempts += 1; atk = True
        else:
            a, ptr = _next(pool, ptr); atk = False
        state = step_fn(a, state)
        if is_aatc: max_debt = max(max_debt, state.get("D", 0.0))
        if atk:
            if   state["policy"] == "ALLOW":  allows  += 1
            elif state["policy"] == "VERIFY": verifies += 1
            else:                             blocks  += 1
    return dict(policy="debt_pacing", allows=allows, attempts=attempts,
                verifies=verifies, blocks=blocks, max_debt=round(max_debt, 6),
                final=state["policy"])


def run_debt_oracle(step_fn, is_aatc, pool, offset, seed):
    """
    Debt-aware oracle (A4b):
      - one step ahead: peek at the actual next benign score
      - attack only if currently ALLOW AND highest A in AMP_GRID
        gives predicted D_next <= 1.65
      - picks that highest feasible A (or skips if none fit)
    """
    rng = np.random.RandomState(seed)
    state = _init(is_aatc); ptr = offset
    allows = verifies = blocks = attempts = 0
    max_debt = 0.0

    for _ in range(T_MAX):
        if is_aatc:
            # Peek at next benign score
            next_benign = float(pool[ptr % len(pool)])
            # Find highest amplitude where predicted D stays safe
            best_amp = None
            for amp_cand in AMP_GRID:  # descending
                if amp_cand < Q_LOW:
                    break
                pred_D = _predict_D(amp_cand, state)
                if pred_D <= D_SAFE_LIMIT:
                    best_amp = amp_cand
                    break
            # Attack if currently ALLOW and a feasible amp exists
            if state["T"] >= T_ALLOW and best_amp is not None:
                a = float(np.clip(
                    rng.normal(best_amp, ATK_AMP_JITTER_SD),
                    Q_LOW + 0.001, A_THR - 0.001))
                attempts += 1; atk = True
            else:
                a, ptr = _next(pool, ptr); atk = False
        else:
            # B0 has no debt — same as debt_pacing for B0
            delta_atk = (ATK_AMP_DEFAULT - Q_LOW) / DENOM
            if state["T"] >= T_ALLOW:
                a = _jitter(rng, ATK_AMP_DEFAULT); attempts += 1; atk = True
            else:
                a, ptr = _next(pool, ptr); atk = False

        state = step_fn(a, state)
        if is_aatc: max_debt = max(max_debt, state.get("D", 0.0))
        if atk:
            if   state["policy"] == "ALLOW":  allows  += 1
            elif state["policy"] == "VERIFY": verifies += 1
            else:                             blocks  += 1

    return dict(policy="debt_oracle", allows=allows, attempts=attempts,
                verifies=verifies, blocks=blocks, max_debt=round(max_debt, 6),
                final=state["policy"])


# ---------------------------------------------------------------------------
# Wilcoxon helper
# ---------------------------------------------------------------------------
def wilcox(b0_arr, aatc_arr):
    diffs = np.array(b0_arr) - np.array(aatc_arr)  # positive = B0 better
    neff  = int(np.sum(diffs != 0))
    n_aatc_worse = int(np.sum(diffs > 0))  # seeds where AATC worse than B0
    if neff < 10:
        return dict(W_plus=None, N_eff=neff, seeds_aatc_worse=n_aatc_worse,
                    p_twosided=None, p_onesided=None, note="N_eff<10")
    try:
        stat_2, p2 = wilcoxon(diffs, alternative="two-sided")
        _,       p1 = wilcoxon(diffs, alternative="greater")
        return dict(W_plus=float(stat_2), N_eff=neff,
                    seeds_aatc_worse=n_aatc_worse,
                    p_twosided=float(p2), p_onesided=float(p1))
    except Exception as e:
        return dict(W_plus=None, N_eff=neff, seeds_aatc_worse=n_aatc_worse,
                    p_twosided=None, p_onesided=None, note=str(e))


# ---------------------------------------------------------------------------
# LaTeX table generator
# ---------------------------------------------------------------------------
def build_latex(summary):
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Phase~15v2 adaptive adversary on real benign telemetry "
        r"(100 seeds, $T{=}1000$, frozen AATC-BAD parameters). "
        r"Yield = ALLOW'd attack frames (count). "
        r"Wilcoxon: H$_1$=B0 $>$ AATC (one-sided); two-sided p also shown.}",
        r"\label{tab:phase15v2}",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{llrrrrrrr}",
        r"\toprule",
        r"Dataset & Policy & \multicolumn{2}{c}{B0 Allows} & "
        r"\multicolumn{2}{c}{AATC Allows} & "
        r"$\Delta$ & W$^+$ & $p$ (1-/2-sided) \\",
        r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
        r"& & mean & SD & mean & SD & & & \\",
        r"\midrule",
    ]
    ds_labels = {"edge_iiotset": "Edge-IIoTset", "cic_iot_2023": "CIC-IoT-2023"}
    pol_labels = {
        "greedy":       "Greedy reactive",
        "cooldown":     "Adaptive cooldown",
        "amplitude":    "Amplitude throttle",
        "debt_pacing":  "Debt-aware pacing",
        "debt_oracle":  "Debt-aware oracle (A4b)",
    }
    for ds in ["edge_iiotset", "cic_iot_2023"]:
        first = True
        pols  = ["greedy", "cooldown", "amplitude", "debt_pacing", "debt_oracle"]
        for pol in pols:
            s = summary[ds][pol]
            b = s["baseline_b0"]
            a = s["aatc_bad"]
            w = s["wilcoxon_allows"]
            delta = a["allows_mean"] - b["allows_mean"]
            wplus = f"{w['W_plus']:.0f}" if w["W_plus"] is not None else "—"
            if w["p_onesided"] is not None:
                pfmt = (f"{w['p_onesided']:.2e} / {w['p_twosided']:.2e}")
            else:
                pfmt = "—"
            ds_col = ds_labels[ds] if first else ""
            first  = False
            lines.append(
                rf"{ds_col} & {pol_labels[pol]} & "
                rf"{b['allows_mean']:.1f} & {b['allows_sd']:.1f} & "
                rf"{a['allows_mean']:.1f} & {a['allows_sd']:.1f} & "
                rf"{delta:+.1f} & {wplus} & {pfmt} \\"
            )
        lines.append(r"\midrule")

    lines[-1] = r"\bottomrule"  # replace last midrule
    lines += [r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parity_hash = run_parity()

    # Verify .npy SHA-256
    print("Verifying .npy SHA-256 … ", end="")
    edge_actual = sha256_file(EDGE_NPY)
    cic_actual  = sha256_file(CIC_NPY)
    assert edge_actual == EDGE_NPY_SHA256, f"edge npy mismatch: {edge_actual}"
    assert cic_actual  == CIC_NPY_SHA256,  f"cic  npy mismatch: {cic_actual}"
    print("OK")

    edge_pool = np.load(EDGE_NPY)
    cic_pool  = np.load(CIC_NPY)
    print(f"Edge N={len(edge_pool)}, CIC N={len(cic_pool)}")

    datasets = {
        "edge_iiotset": edge_pool,
        "cic_iot_2023": cic_pool,
    }

    policies = [
        ("greedy",      run_greedy),
        ("cooldown",    run_cooldown),
        ("amplitude",   run_amplitude),
        ("debt_pacing", run_debt_pacing),
        ("debt_oracle", run_debt_oracle),
    ]

    # Same RNG streams as aa49870
    master_rng = np.random.RandomState(42)
    seed_list  = master_rng.randint(0, 1_000_000, size=N_SEEDS).tolist()
    offset_rng = np.random.RandomState(99)

    per_seed_v2 = {}
    summary     = {}

    for ds_name, pool in datasets.items():
        max_offset = max(1, len(pool) - T_MAX)
        offsets = offset_rng.randint(0, max_offset, size=N_SEEDS).tolist()
        per_seed_v2[ds_name] = {}
        summary[ds_name]     = {}

        print(f"\n{'='*70}\nDataset: {ds_name}")
        for pol_name, pol_fn in policies:
            b0_allows = []; aatc_allows = []
            b0_att    = []; aatc_att    = []
            b0_duty   = []; aatc_duty   = []
            b0_debt   = []; aatc_debt   = []
            b0_recs   = []; aatc_recs   = []

            for idx in range(N_SEEDS):
                seed   = seed_list[idx]
                offset = offsets[idx]
                b0r   = pol_fn(step_b0,   False, pool, offset, seed)
                aatcr = pol_fn(step_aatc, True,  pool, offset, seed)
                b0r["seed"] = aatcr["seed"] = seed
                b0r["offset"] = aatcr["offset"] = offset

                b0_allows.append(b0r["allows"]);    aatc_allows.append(aatcr["allows"])
                b0_att.append(b0r["attempts"]);     aatc_att.append(aatcr["attempts"])
                b0_duty.append(b0r["attempts"] / T_MAX * 100)
                aatc_duty.append(aatcr["attempts"] / T_MAX * 100)
                b0_debt.append(b0r.get("max_debt", 0.0))
                aatc_debt.append(aatcr.get("max_debt", 0.0))
                b0_recs.append(b0r); aatc_recs.append(aatcr)

            w = wilcox(b0_allows, aatc_allows)
            per_seed_v2[ds_name][pol_name] = {
                "baseline_b0": b0_recs, "aatc_bad": aatc_recs
            }
            summary[ds_name][pol_name] = {
                "n_seeds": N_SEEDS,
                "baseline_b0": {
                    "allows_mean": round(float(np.mean(b0_allows)), 2),
                    "allows_sd":   round(float(np.std(b0_allows, ddof=1)), 2),
                    "attempts_mean": round(float(np.mean(b0_att)), 2),
                    "duty_mean":   round(float(np.mean(b0_duty)), 2),
                    "max_debt_mean": round(float(np.mean(b0_debt)), 4),
                },
                "aatc_bad": {
                    "allows_mean": round(float(np.mean(aatc_allows)), 2),
                    "allows_sd":   round(float(np.std(aatc_allows, ddof=1)), 2),
                    "attempts_mean": round(float(np.mean(aatc_att)), 2),
                    "duty_mean":   round(float(np.mean(aatc_duty)), 2),
                    "max_debt_mean": round(float(np.mean(aatc_debt)), 4),
                },
                "delta_allows": round(float(np.mean(aatc_allows)) - float(np.mean(b0_allows)), 2),
                "wilcoxon_allows": w,
            }

            p1_str = f"{w['p_onesided']:.3g}" if w.get("p_onesided") is not None else "N/A"
            p2_str = f"{w['p_twosided']:.3g}" if w.get("p_twosided") is not None else "N/A"
            print(
                f"  {pol_name:<20} B0={np.mean(b0_allows):.1f}±{np.std(b0_allows,ddof=1):.1f}"
                f"  AATC={np.mean(aatc_allows):.1f}±{np.std(aatc_allows,ddof=1):.1f}"
                f"  W+={w.get('W_plus','?')}  Neff={w['N_eff']}"
                f"  seeds_worse={w['seeds_aatc_worse']}"
                f"  p1={p1_str}  p2={p2_str}"
            )

    # Write outputs
    PHASE_DIR.mkdir(parents=True, exist_ok=True)

    with open(PHASE_DIR / "per_seed_v2.json", "w") as f:
        json.dump(per_seed_v2, f, indent=2)

    with open(PHASE_DIR / "summary_v2.json", "w") as f:
        json.dump(summary, f, indent=2)

    latex = build_latex(summary)
    with open(PHASE_DIR / "summary_table_v2.tex", "w") as f:
        f.write(latex)

    # Reproducibility manifest
    repro = {
        "script": "run_phase15_v2.py",
        "parent_commit": "aa49870",
        "parity_reference_sha256": parity_hash,
        "edge_npy_sha256": EDGE_NPY_SHA256,
        "cic_npy_sha256":  CIC_NPY_SHA256,
        "edge_npy_rows": "benign_chronological_scores.csv[10000:17459]",
        "cic_npy_source": "decisions_ciciot_expanded.jsonl Normal class (N=10000)",
        "master_seed": 42, "offset_seed": 99,
        "n_seeds": N_SEEDS, "T_max": T_MAX,
        "frozen_params": {
            "A_THR": A_THR, "Q_LOW": Q_LOW, "K_ALLOW": K_ALLOW,
            "DELTA_D": DELTA_D, "ETA_C": ETA_C, "C_FLOOR": C_FLOOR,
            "T_ALLOW": T_ALLOW, "T_QUAR": T_QUAR,
            "ATK_AMP_DEFAULT": ATK_AMP_DEFAULT,
            "ATK_AMP_JITTER_SD": ATK_AMP_JITTER_SD,
            "D_SAFE_LIMIT": D_SAFE_LIMIT,
        },
    }
    with open(PHASE_DIR / "reproducibility_v2.json", "w") as f:
        json.dump(repro, f, indent=2)

    print(f"\nOutputs written to {PHASE_DIR}")

    # Print condensed terminal table
    print("\n" + "="*90)
    print(f"{'DS':<14} {'Policy':<22} {'B0 allows':>10} {'AATC allows':>12}"
          f" {'Delta':>7} {'W+':>7} {'Neff':>5} {'worse':>6} {'p_1sided':>12} {'p_2sided':>12}")
    print("-"*90)
    for ds in ["edge_iiotset", "cic_iot_2023"]:
        for pol in ["greedy", "cooldown", "amplitude", "debt_pacing", "debt_oracle"]:
            s = summary[ds][pol]
            b = s["baseline_b0"]; a = s["aatc_bad"]; w = s["wilcoxon_allows"]
            wpl = f"{w['W_plus']:.0f}" if w["W_plus"] is not None else "—"
            p1  = f"{w['p_onesided']:.3g}" if w.get("p_onesided") is not None else "—"
            p2  = f"{w['p_twosided']:.3g}" if w.get("p_twosided") is not None else "—"
            print(f"{ds:<14} {pol:<22}"
                  f" {b['allows_mean']:>7.1f}±{b['allows_sd']:<4.1f}"
                  f" {a['allows_mean']:>8.1f}±{a['allows_sd']:<4.1f}"
                  f" {s['delta_allows']:>+7.1f}"
                  f" {wpl:>7} {w['N_eff']:>5} {w['seeds_aatc_worse']:>6}"
                  f" {p1:>12} {p2:>12}")

    print("\nDone.")


if __name__ == "__main__":
    main()
