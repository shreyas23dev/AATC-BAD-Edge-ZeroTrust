#!/usr/bin/env python3
"""
Phase 15 — Adaptive Adversary on Real Benign Telemetry
======================================================
Runs all four adaptive adversary policies against Baseline CDTM (B0) and
AATC-BAD on REAL benign telemetry rest epochs from two datasets:

  Edge-IIoTset  : benign_chronological_scores.csv  eval split [10000:17459]
  CIC-IoT-2023  : decisions_ciciot_expanded.jsonl  Normal class (10,000 frames)

Design decisions
----------------
  * step_b0 / step_aatc imported UNCHANGED from run_adaptive_adversary.py
  * Parity against rust-edge/crates/aatc-engine/tests/parity_reference.json
    checked FIRST; script stops immediately on failure.
  * Frozen parameters only: K_ALLOW 1.70, DELTA_D 0.10, ETA_C 0.30 etc.
  * Rest epoch = next real benign score drawn sequentially from the eval split
    (starting at a random seed-dependent offset). Attack frames use the same
    amp as before (0.450) with ±Gaussian jitter σ=0.005.
  * Policies:
      1. greedy        – attack whenever ALLOW; rest on first non-ALLOW
      2. cooldown      – adaptive tau_r backoff (+2 on every demotion)
      3. amplitude     – throttle A_t down by 0.02 on every demotion (floor 0.340)
      4. debt_aware    – attack only when D + delta_atk ≤ K_ALLOW − 0.05
  * 100 seeds, T = 1000 steps each
  * Statistics: mean ± SD, paired Wilcoxon signed-rank (W+, N_eff, p)
  * Outputs:
      per_seed.json   – every seed result for every policy × engine × dataset
      summary.json    – mean/SD/Wilcoxon tables
  * Commits to: experiments/candidate5/phase15_adaptive_real/

Parity check
------------
  Python step_aatc is replayed against all 1,000 reference frames in
  parity_reference.json. Any policy, state, or ceiling mismatch aborts.

Usage
-----
  cd <repo root>
  python TON_IoT_project/experiments/candidate5/phase15_adaptive_real/run_phase15_adaptive_real.py
"""

import hashlib
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[4]  # <repo>/TON_IoT_project/../..
PROJ_ROOT = REPO_ROOT / "TON_IoT_project"

ADVERSARY_SCRIPT = (
    PROJ_ROOT
    / "experiments/candidate5/adaptive_adversary/run_adaptive_adversary.py"
)
PARITY_REF = (
    PROJ_ROOT
    / "rust-edge/crates/aatc-engine/tests/parity_reference.json"
)
EDGE_SCORES_CSV = (
    PROJ_ROOT
    / "experiments/candidate5/chronological_validation/benign_chronological_scores.csv"
)
CIC_EVAL_CSV = PROJ_ROOT / "data/ciciot2023/ciciot_eval_expanded_44.csv"
CIC_JSONL = PROJ_ROOT / "rust-edge/decisions_ciciot_expanded.jsonl"
OUT_DIR = PROJ_ROOT / "experiments/candidate5/phase15_adaptive_real"

# ---------------------------------------------------------------------------
# Import step functions UNCHANGED from run_adaptive_adversary.py
# ---------------------------------------------------------------------------
sys.path.insert(0, str(PROJ_ROOT))
from experiments.candidate5.adaptive_adversary.run_adaptive_adversary import (
    step_aatc,
    step_b0,
)

# ---------------------------------------------------------------------------
# Frozen parameters (must match parity_reference.json provenance)
# ---------------------------------------------------------------------------
A_THR = 0.4554
Q_LOW = 0.35
DENOM = A_THR - Q_LOW  # 0.1054
K_ALLOW = 1.70
DELTA_D = 0.10
T_ALLOW = 0.80
T_QUAR = 0.30
ATK_AMP_DEFAULT = 0.450
ATK_AMP_JITTER_SD = 0.005
N_SEEDS = 100
T_MAX = 1000

# ---------------------------------------------------------------------------
# SHA-256 helpers
# ---------------------------------------------------------------------------
def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(1 << 16):
            h.update(chunk)
    return h.hexdigest()


def sha256_array(arr: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(arr.tobytes())
    return h.hexdigest()


# ---------------------------------------------------------------------------
# 1. PARITY CHECK
# ---------------------------------------------------------------------------
def run_parity_check():
    print("=" * 70)
    print("STEP 1: Parity check — Python step_aatc vs parity_reference.json")
    print("=" * 70)

    with open(PARITY_REF) as f:
        ref = json.load(f)
    assert len(ref) == 1000, f"Expected 1000 ref steps, got {len(ref)}"

    state = {"T": 1.0, "E": 0.0, "SB": 0.0, "D": 0.0, "C": 1.0}
    failures = []
    for s in ref:
        state = step_aatc(s["score"], state)
        if state["policy"] != s["policy"]:
            failures.append(
                f"step {s['step']}: policy {state['policy']} != {s['policy']}"
            )
        if abs(state["T"] - s["alpha"]) > 1e-4:
            failures.append(
                f"step {s['step']}: alpha {state['T']:.6f} != {s['alpha']:.6f}"
            )
        if abs(state["D"] - s["debt"]) > 1e-4:
            failures.append(
                f"step {s['step']}: debt {state['D']:.6f} != {s['debt']:.6f}"
            )
        if abs(state["C"] - s["ceiling"]) > 1e-4:
            failures.append(
                f"step {s['step']}: ceiling {state['C']:.6f} != {s['ceiling']:.6f}"
            )

    if failures:
        print(f"PARITY FAILED — {len(failures)} mismatch(es):")
        for msg in failures[:10]:
            print("  ", msg)
        sys.exit(1)

    print(f"  PARITY PASSED: 1,000 / 1,000 steps matched exactly (tol=1e-4)")
    print()
    return sha256_file(PARITY_REF)


# ---------------------------------------------------------------------------
# 2. LOAD BENIGN SCORES
# ---------------------------------------------------------------------------
def load_edge_scores() -> np.ndarray:
    df = pd.read_csv(EDGE_SCORES_CSV)
    scores = df["A_t"].values[10000:17459].astype(np.float64)
    assert len(scores) == 7459, f"Expected 7459, got {len(scores)}"
    return scores


def load_cic_scores() -> np.ndarray:
    df_meta = pd.read_csv(CIC_EVAL_CSV, usecols=["Attack_type"])
    norm_idx = set(df_meta[df_meta["Attack_type"] == "Normal"].index.values)
    raw = []
    with open(CIC_JSONL) as f:
        for i, line in enumerate(f):
            if i in norm_idx:
                rec = json.loads(line)
                phi = rec.get("phi")
                if phi is not None:
                    raw.append(float(np.mean(phi)))
                else:
                    raw.append(0.0)
    scores = np.clip(
        np.array(raw, dtype=np.float64) / 0.0360 * A_THR, 0.0, 1.0
    )
    assert len(scores) == 10000, f"Expected 10000, got {len(scores)}"
    return scores


# ---------------------------------------------------------------------------
# 3. POLICY RUNNERS
#    Each runner takes (step_fn, is_aatc, benign_pool, start_offset, seed)
#    and returns a dict of per-seed metrics.
#    Attack frames use the fixed amplitude with optional Gaussian jitter.
#    Rest frames draw the NEXT real benign score (cycling if needed).
# ---------------------------------------------------------------------------
def _init_state(is_aatc: bool) -> dict:
    if is_aatc:
        return {"T": 1.0, "E": 0.0, "SB": 0.0, "D": 0.0, "C": 1.0}
    return {"T": 1.0, "E": 0.0, "SB": 0.0}


def _atk_score(rng: np.random.RandomState, amp: float) -> float:
    return float(
        np.clip(rng.normal(amp, ATK_AMP_JITTER_SD), Q_LOW + 0.005, A_THR - 0.001)
    )


def _next_benign(pool: np.ndarray, ptr: int) -> tuple:
    score = float(pool[ptr % len(pool)])
    return score, (ptr + 1)


# --- Policy 1: Greedy Reactive ---
def run_greedy(
    step_fn, is_aatc: bool, pool: np.ndarray, offset: int, seed: int
) -> dict:
    rng = np.random.RandomState(seed)
    state = _init_state(is_aatc)
    ptr = offset
    allows = verifies = blocks = attempts = 0
    max_debt = 0.0
    t_quench = None

    for t in range(T_MAX):
        curr_allow = state["T"] >= T_ALLOW
        if curr_allow:
            a = _atk_score(rng, ATK_AMP_DEFAULT)
            was_attack = True
            attempts += 1
        else:
            a, ptr = _next_benign(pool, ptr)
            was_attack = False

        state = step_fn(a, state)

        if is_aatc:
            max_debt = max(max_debt, state.get("D", 0.0))
            if t_quench is None and state.get("C", 1.0) <= T_ALLOW:
                t_quench = t

        if was_attack:
            if state["policy"] == "ALLOW":
                allows += 1
            elif state["policy"] == "VERIFY":
                verifies += 1
            else:
                blocks += 1

    final_outcome = state["policy"]
    return {
        "policy": "greedy",
        "attempts": attempts,
        "allows": allows,
        "verifies": verifies,
        "blocks": blocks,
        "yield_pct": round(100.0 * allows / max(1, attempts), 4),
        "duty_cycle_pct": round(100.0 * attempts / T_MAX, 4),
        "max_debt": round(max_debt, 6),
        "final_outcome": final_outcome,
        "t_quench": t_quench,
    }


# --- Policy 2: Adaptive Cooldown Backoff ---
def run_cooldown(
    step_fn, is_aatc: bool, pool: np.ndarray, offset: int, seed: int
) -> dict:
    rng = np.random.RandomState(seed)
    state = _init_state(is_aatc)
    ptr = offset
    allows = verifies = blocks = attempts = 0
    max_debt = 0.0
    tau_r = 4
    t = 0

    while t < T_MAX:
        # 1 attack frame
        a = _atk_score(rng, ATK_AMP_DEFAULT)
        attempts += 1
        t += 1
        state = step_fn(a, state)
        if is_aatc:
            max_debt = max(max_debt, state.get("D", 0.0))

        if state["policy"] == "ALLOW":
            allows += 1
        elif state["policy"] == "VERIFY":
            verifies += 1
            tau_r = min(tau_r + 2, 200)
        else:
            blocks += 1
            tau_r = min(tau_r + 2, 200)

        # Rest frames from real benign pool
        for _ in range(tau_r):
            if t >= T_MAX:
                break
            a_rest, ptr = _next_benign(pool, ptr)
            t += 1
            state = step_fn(a_rest, state)
            if is_aatc:
                max_debt = max(max_debt, state.get("D", 0.0))

    return {
        "policy": "cooldown",
        "attempts": attempts,
        "allows": allows,
        "verifies": verifies,
        "blocks": blocks,
        "yield_pct": round(100.0 * allows / max(1, attempts), 4),
        "duty_cycle_pct": round(100.0 * attempts / T_MAX, 4),
        "max_debt": round(max_debt, 6),
        "final_outcome": state["policy"],
        "final_tau_r": tau_r,
    }


# --- Policy 3: Online Amplitude Throttling ---
def run_amplitude(
    step_fn, is_aatc: bool, pool: np.ndarray, offset: int, seed: int
) -> dict:
    rng = np.random.RandomState(seed)
    state = _init_state(is_aatc)
    ptr = offset
    allows = verifies = blocks = attempts = 0
    max_debt = 0.0
    amp = ATK_AMP_DEFAULT
    tau_r_fixed = 6
    t = 0

    while t < T_MAX:
        # 1 attack frame
        a = _atk_score(rng, amp)
        attempts += 1
        t += 1
        state = step_fn(a, state)
        if is_aatc:
            max_debt = max(max_debt, state.get("D", 0.0))

        if state["policy"] == "ALLOW":
            allows += 1
        elif state["policy"] == "VERIFY":
            verifies += 1
            amp = max(0.340, amp - 0.02)
        else:
            blocks += 1
            amp = max(0.340, amp - 0.02)

        # Rest frames from real benign pool
        for _ in range(tau_r_fixed):
            if t >= T_MAX:
                break
            a_rest, ptr = _next_benign(pool, ptr)
            t += 1
            state = step_fn(a_rest, state)
            if is_aatc:
                max_debt = max(max_debt, state.get("D", 0.0))

    return {
        "policy": "amplitude",
        "attempts": attempts,
        "allows": allows,
        "verifies": verifies,
        "blocks": blocks,
        "yield_pct": round(100.0 * allows / max(1, attempts), 4),
        "duty_cycle_pct": round(100.0 * attempts / T_MAX, 4),
        "max_debt": round(max_debt, 6),
        "final_outcome": state["policy"],
        "final_amp": round(amp, 4),
    }


# --- Policy 4: Debt-Aware (keeps D ≤ K_ALLOW − 0.05) ---
def run_debt_aware(
    step_fn, is_aatc: bool, pool: np.ndarray, offset: int, seed: int
) -> dict:
    rng = np.random.RandomState(seed)
    state = _init_state(is_aatc)
    ptr = offset
    allows = verifies = blocks = attempts = 0
    max_debt = 0.0
    # pre-compute delta per attack frame (worst-case amp=0.450)
    delta_atk = (ATK_AMP_DEFAULT - Q_LOW) / DENOM  # ≈ 0.9487
    D_safe_limit = K_ALLOW - 0.05  # 1.65

    for t in range(T_MAX):
        curr_allow = state["T"] >= T_ALLOW
        # Attack only if ALLOW AND projected debt stays safe
        curr_D = state.get("D", 0.0) if is_aatc else 0.0
        if curr_allow and (curr_D + delta_atk <= D_safe_limit):
            a = _atk_score(rng, ATK_AMP_DEFAULT)
            was_attack = True
            attempts += 1
        else:
            a, ptr = _next_benign(pool, ptr)
            was_attack = False

        state = step_fn(a, state)
        if is_aatc:
            max_debt = max(max_debt, state.get("D", 0.0))

        if was_attack:
            if state["policy"] == "ALLOW":
                allows += 1
            elif state["policy"] == "VERIFY":
                verifies += 1
            else:
                blocks += 1

    return {
        "policy": "debt_aware",
        "attempts": attempts,
        "allows": allows,
        "verifies": verifies,
        "blocks": blocks,
        "yield_pct": round(100.0 * allows / max(1, attempts), 4),
        "duty_cycle_pct": round(100.0 * attempts / T_MAX, 4),
        "max_debt": round(max_debt, 6),
        "final_outcome": state["policy"],
    }


# ---------------------------------------------------------------------------
# 4. WILCOXON HELPER
# ---------------------------------------------------------------------------
def wilcoxon_paired(b0_vals, aatc_vals):
    diffs = np.array(b0_vals) - np.array(aatc_vals)
    n = len(diffs)
    nonzero = diffs[diffs != 0]
    n_eff = len(nonzero)
    if n_eff < 10:
        return {"W_plus": None, "N_eff": n_eff, "p_value": None, "note": "N_eff<10"}
    try:
        stat, p = wilcoxon(diffs, alternative="greater")
        return {"W_plus": float(stat), "N_eff": n_eff, "p_value": float(p)}
    except Exception as e:
        return {"W_plus": None, "N_eff": n_eff, "p_value": None, "note": str(e)}


# ---------------------------------------------------------------------------
# 5. MAIN
# ---------------------------------------------------------------------------
def main():
    parity_hash = run_parity_check()

    # -- Load datasets
    print("Loading Edge-IIoTset eval split (rows 10,000–17,458)...")
    edge_scores = load_edge_scores()
    edge_hash = sha256_array(edge_scores)

    print("Loading CIC-IoT-2023 Normal frames (10,000 frames)...")
    cic_scores = load_cic_scores()
    cic_hash = sha256_array(cic_scores)

    print(f"  Edge: N={len(edge_scores)}, mean={np.mean(edge_scores):.4f}")
    print(f"  CIC : N={len(cic_scores)},  mean={np.mean(cic_scores):.4f}")
    print()

    datasets = {
        "edge_iiotset": edge_scores,
        "cic_iot_2023": cic_scores,
    }

    policies = [
        ("greedy", run_greedy),
        ("cooldown", run_cooldown),
        ("amplitude", run_amplitude),
        ("debt_aware", run_debt_aware),
    ]

    # Seed sequence reproducible from master seed 42
    master_rng = np.random.RandomState(42)
    seed_list = master_rng.randint(0, 1_000_000, size=N_SEEDS).tolist()

    per_seed_all = {}  # dataset → policy → engine → [per-seed dicts]
    summary_all = {}

    for ds_name, pool in datasets.items():
        max_offset = max(1, len(pool) - T_MAX)
        offset_rng = np.random.RandomState(99)
        offsets = offset_rng.randint(0, max_offset, size=N_SEEDS).tolist()

        per_seed_all[ds_name] = {}
        summary_all[ds_name] = {}

        print(f"{'=' * 70}")
        print(f"Dataset: {ds_name}  (pool N={len(pool)})")
        print(f"{'=' * 70}")

        for pol_name, pol_fn in policies:
            b0_yields, aatc_yields = [], []
            b0_duties, aatc_duties = [], []
            b0_debts, aatc_debts = [], []
            b0_records, aatc_records = [], []

            print(f"\n  Policy: {pol_name}")
            print(
                f"    {'Seed':>6}  {'B0 yield%':>10}  {'AATC yield%':>11}"
                f"  {'B0 duty%':>9}  {'AATC duty%':>10}"
                f"  {'MaxDebt':>8}  {'AATC out':>9}"
            )

            for idx in range(N_SEEDS):
                seed = seed_list[idx]
                offset = offsets[idx]

                b0_res = pol_fn(step_b0, False, pool, offset, seed)
                aatc_res = pol_fn(step_aatc, True, pool, offset, seed)

                b0_res["seed"] = seed
                b0_res["offset"] = offset
                aatc_res["seed"] = seed
                aatc_res["offset"] = offset

                b0_yields.append(b0_res["yield_pct"])
                aatc_yields.append(aatc_res["yield_pct"])
                b0_duties.append(b0_res["duty_cycle_pct"])
                aatc_duties.append(aatc_res["duty_cycle_pct"])
                b0_debts.append(b0_res.get("max_debt", 0.0))
                aatc_debts.append(aatc_res.get("max_debt", 0.0))
                b0_records.append(b0_res)
                aatc_records.append(aatc_res)

                if idx % 20 == 0:
                    print(
                        f"    {seed:>6}  {b0_res['yield_pct']:>10.2f}  {aatc_res['yield_pct']:>11.2f}"
                        f"  {b0_res['duty_cycle_pct']:>9.2f}  {aatc_res['duty_cycle_pct']:>10.2f}"
                        f"  {aatc_res['max_debt']:>8.4f}  {aatc_res['final_outcome']:>9}"
                    )

            # Wilcoxon: H0 B0_yield == AATC_yield, H1 B0_yield > AATC_yield
            wilc = wilcoxon_paired(b0_yields, aatc_yields)

            mean_b0_y = float(np.mean(b0_yields))
            sd_b0_y = float(np.std(b0_yields, ddof=1))
            mean_aatc_y = float(np.mean(aatc_yields))
            sd_aatc_y = float(np.std(aatc_yields, ddof=1))
            reduction = mean_b0_y - mean_aatc_y

            print(
                f"\n    SUMMARY  B0 yield: {mean_b0_y:.2f} ± {sd_b0_y:.2f}%"
                f"  AATC yield: {mean_aatc_y:.2f} ± {sd_aatc_y:.2f}%"
                f"  Δ={reduction:+.2f}pp"
            )
            if wilc["p_value"] is not None:
                print(
                    f"    Wilcoxon  W+={wilc['W_plus']:.0f}  N_eff={wilc['N_eff']}"
                    f"  p={wilc['p_value']:.4g}"
                )

            per_seed_all[ds_name][pol_name] = {
                "baseline_b0": b0_records,
                "aatc_bad": aatc_records,
            }
            summary_all[ds_name][pol_name] = {
                "n_seeds": N_SEEDS,
                "baseline_b0": {
                    "yield_mean": round(mean_b0_y, 4),
                    "yield_sd": round(sd_b0_y, 4),
                    "duty_mean": round(float(np.mean(b0_duties)), 4),
                    "duty_sd": round(float(np.std(b0_duties, ddof=1)), 4),
                    "max_debt_mean": round(float(np.mean(b0_debts)), 4),
                    "max_debt_sd": round(float(np.std(b0_debts, ddof=1)), 4),
                },
                "aatc_bad": {
                    "yield_mean": round(mean_aatc_y, 4),
                    "yield_sd": round(sd_aatc_y, 4),
                    "duty_mean": round(float(np.mean(aatc_duties)), 4),
                    "duty_sd": round(float(np.std(aatc_duties, ddof=1)), 4),
                    "max_debt_mean": round(float(np.mean(aatc_debts)), 4),
                    "max_debt_sd": round(float(np.std(aatc_debts, ddof=1)), 4),
                },
                "yield_reduction_pp": round(reduction, 4),
                "wilcoxon": wilc,
            }

    # -- Write outputs
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    per_seed_path = OUT_DIR / "per_seed.json"
    with open(per_seed_path, "w") as f:
        json.dump(per_seed_all, f, indent=2)

    summary_path = OUT_DIR / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary_all, f, indent=2)

    # Reproducibility manifest
    repro = {
        "script": str(Path(__file__).name),
        "parity_reference_sha256": parity_hash,
        "edge_scores_sha256": edge_hash,
        "cic_scores_sha256": cic_hash,
        "edge_scores_path": str(EDGE_SCORES_CSV),
        "cic_jsonl_path": str(CIC_JSONL),
        "cic_csv_path": str(CIC_EVAL_CSV),
        "edge_eval_rows": "10000:17459  (N=7459)",
        "cic_normal_frames": 10000,
        "master_seed": 42,
        "offset_seed": 99,
        "n_seeds": N_SEEDS,
        "T_max": T_MAX,
        "frozen_params": {
            "A_THR": A_THR,
            "Q_LOW": Q_LOW,
            "K_ALLOW": K_ALLOW,
            "DELTA_D": DELTA_D,
            "ETA_C": 0.30,
            "C_FLOOR": 0.70,
            "T_ALLOW": T_ALLOW,
            "T_QUAR": T_QUAR,
            "ATK_AMP_DEFAULT": ATK_AMP_DEFAULT,
            "ATK_AMP_JITTER_SD": ATK_AMP_JITTER_SD,
        },
    }
    repro_path = OUT_DIR / "reproducibility.json"
    with open(repro_path, "w") as f:
        json.dump(repro, f, indent=2)

    print("\n" + "=" * 70)
    print("OUTPUTS")
    print(f"  {per_seed_path}")
    print(f"  {summary_path}")
    print(f"  {repro_path}")

    # -- Print final summary table
    print("\n" + "=" * 70)
    print("SUMMARY TABLE (mean ± SD over 100 seeds, T=1000 steps each)")
    print("=" * 70)
    hdr = (
        f"{'Dataset':<14} {'Policy':<14} {'Engine':<8} "
        f"{'Yield%':>8} {'±':>2} {'SD':>6}  "
        f"{'Duty%':>7} {'MaxD':>7}  {'Wilcoxon p':>12}"
    )
    print(hdr)
    print("-" * 90)
    for ds in ["edge_iiotset", "cic_iot_2023"]:
        for pol in ["greedy", "cooldown", "amplitude", "debt_aware"]:
            s = summary_all[ds][pol]
            w = s["wilcoxon"]
            p_str = f"{w['p_value']:.4g}" if w.get("p_value") is not None else "N/A"
            for eng, key in [("B0", "baseline_b0"), ("AATC", "aatc_bad")]:
                e = s[key]
                wilc_col = p_str if eng == "B0" else ""
                print(
                    f"{ds:<14} {pol:<14} {eng:<8} "
                    f"{e['yield_mean']:>8.2f} {'±':>2} {e['yield_sd']:>6.2f}  "
                    f"{e['duty_mean']:>7.2f} {e['max_debt_mean']:>7.4f}  {wilc_col:>12}"
                )

    print("\nDone.")


if __name__ == "__main__":
    main()
