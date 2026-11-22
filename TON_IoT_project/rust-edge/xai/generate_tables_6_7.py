"""
generate_tables_6_7.py
Re-runs the Monte Carlo pulsing experiment and generates Tables 6 & 7 directly
from per-seed arrays. No hand-typed values.

Output:
  - monte_carlo_per_seed.json  (per-seed B0/AATC arrays for Wilcoxon)
  - table6_monte_carlo.tex     (LaTeX Table 6 rows)
  - table7_wilcoxon.tex        (LaTeX Table 7 rows)
  - tables_67_summary.txt      (human-readable)
"""

import json, os
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import wilcoxon

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE = Path(r"c:\Users\Shreyas A\PROJECTS\BAZT-TON_IoT\TON_IoT_project")
RUST = BASE / "rust-edge"
XAI  = RUST / "xai"
DATA = BASE / "data"

EDGE_CSV       = BASE / "experiments/candidate5/chronological_validation/benign_chronological_scores.csv"
CICIOT_CSV     = DATA / "ciciot2023/ciciot_eval_expanded_44.csv"
CICIOT_JSONL   = RUST / "decisions_ciciot_expanded.jsonl"

OUT_PER_SEED = XAI / "monte_carlo_per_seed.json"
OUT_T6       = XAI / "table6_monte_carlo.tex"
OUT_T7       = XAI / "table7_wilcoxon.tex"
OUT_SUMMARY  = XAI / "tables_67_summary.txt"

# ─── Frozen Paper Parameters (match benchmark_monte_carlo_pulsing.py exactly) ─
A_THR   = 0.4554
Q_LOW   = 0.35
DENOM   = A_THR - Q_LOW
DELTA_D = 0.10
K_ALLOW = 1.70
ETA_C   = 0.30
C_FLOOR = 0.70
ALPHA_0 = 0.2057
T_ALLOW = 0.80
T_QUAR  = 0.30
BETA    = 0.40
GAMMA   = 0.335
THETA   = 1.0
LAMBDA  = 0.55
DELTA   = 0.05
P_PEN   = 0.25

# Attack pulse profile (from paper §6.4 / benchmark script)
ATK_PROFILE = [0.41, 0.39, 0.42, 0.38, 0.40, 0.43, 0.39, 0.41, 0.40, 0.42]

# ─── B0 baseline step function ──────────────────────────────────────────────
def step_b0(T, E, S, a):
    """Baseline CDTM (no debt, no ceiling)."""
    E_new = BETA * a + (1.0 - BETA) * E
    if E_new > GAMMA:
        S_new = S + LAMBDA
    else:
        S_new = max(0.0, S - DELTA)
    slow_burn = (S_new > THETA)

    if a >= A_THR:
        T_new = max(0.0, T * (1.0 - a))
    elif slow_burn:
        T_new = max(0.0, T - P_PEN * (a / A_THR))
    else:
        T_new = T + ALPHA_0 * (1.0 - T)   # recovers to 1.0 (no ceiling)

    pol = "ALLOW" if T_new >= T_ALLOW else ("VERIFY" if T_new >= T_QUAR else "BLOCK")
    return T_new, E_new, S_new, pol


# ─── AATC-BAD step function (corrected) ──────────────────────────────────────
def step_aatc(T, D, C, E, S, a):
    """AATC-BAD step (Equations 15-18)."""
    if (T >= T_ALLOW) and (Q_LOW <= a < A_THR):
        D_new = D + (a - Q_LOW) / DENOM
    elif a < Q_LOW:
        D_new = max(0.0, D - DELTA_D)
    else:
        D_new = D

    C_new = max(C_FLOOR, 1.0 - ETA_C * max(0.0, D_new - K_ALLOW))

    E_new = BETA * a + (1.0 - BETA) * E
    if E_new > GAMMA:
        S_new = S + LAMBDA
    else:
        S_new = max(0.0, S - DELTA)
    slow_burn = (S_new > THETA)

    if a >= A_THR:
        T_new = max(0.0, T * (1.0 - a))
    elif slow_burn:
        T_new = max(0.0, T - P_PEN * (a / A_THR))
    else:
        rec = T + ALPHA_0 * (C_new - T)
        T_new = min(C_new, rec)

    pol = "ALLOW" if T_new >= T_ALLOW else ("VERIFY" if T_new >= T_QUAR else "BLOCK")
    return T_new, D_new, C_new, E_new, S_new, pol


# ─── Single-run simulation ────────────────────────────────────────────────────
def simulate_single_run(scores, start_idx, tau_a, tau_r, jitter_range, seed, max_cycles=25):
    rng = np.random.RandomState(seed)
    n = len(scores)

    # Initial states
    T_b0, E_b0, S_b0 = 1.0, 0.0, 0.0
    T_aa, D_aa, C_aa, E_aa, S_aa = 1.0, 0.0, 1.0, 0.0, 0.0

    b0_mcum = 0.0
    aatc_mcum = 0.0
    b0_allow = 0
    aatc_allow = 0

    t = start_idx
    for _ in range(max_cycles):
        # Attack phase: tau_a steps of borderline-attack telemetry
        actual_tau_a = tau_a + rng.randint(-jitter_range, jitter_range + 1)
        actual_tau_a = max(1, actual_tau_a)
        for _ in range(actual_tau_a):
            if t >= n:
                t = rng.randint(0, min(500, n))
            atk_a = float(np.clip(ATK_PROFILE[rng.randint(0, len(ATK_PROFILE))], Q_LOW + 0.01, A_THR - 0.005))
            T_b0, E_b0, S_b0, pol_b0 = step_b0(T_b0, E_b0, S_b0, atk_a)
            T_aa, D_aa, C_aa, E_aa, S_aa, pol_aa = step_aatc(T_aa, D_aa, C_aa, E_aa, S_aa, atk_a)
            if pol_b0 == "ALLOW":
                b0_mcum += 1
                b0_allow += 1
            if pol_aa == "ALLOW":
                aatc_mcum += 1
                aatc_allow += 1
            t += 1

        # Rest phase: tau_r steps of real benign telemetry
        actual_tau_r = tau_r + rng.randint(-jitter_range, jitter_range + 1)
        actual_tau_r = max(1, actual_tau_r)
        for _ in range(actual_tau_r):
            if t >= n:
                t = rng.randint(0, min(500, n))
            benign_a = float(scores[t % n])
            T_b0, E_b0, S_b0, pol_b0 = step_b0(T_b0, E_b0, S_b0, benign_a)
            T_aa, D_aa, C_aa, E_aa, S_aa, pol_aa = step_aatc(T_aa, D_aa, C_aa, E_aa, S_aa, benign_a)
            t += 1

    pct_red = (b0_mcum - aatc_mcum) / b0_mcum * 100 if b0_mcum > 0 else 0.0
    return {
        "b0_mcum": b0_mcum,
        "aatc_mcum": aatc_mcum,
        "pct_mcum_reduction": pct_red,
        "b0_allow": b0_allow,
        "aatc_allow": aatc_allow,
        "delta_allow": b0_allow - aatc_allow,
    }


# ─── Full Monte Carlo run ─────────────────────────────────────────────────────
SCHEDULES = [
    {"name": "Pacing (1, 8)",  "tau_a": 1, "tau_r": 8,  "jitter": 1},
    {"name": "Pacing (1, 10)", "tau_a": 1, "tau_r": 10, "jitter": 2},
    {"name": "Pacing (1, 12)", "tau_a": 1, "tau_r": 12, "jitter": 2},
    {"name": "Pacing (1, 15)", "tau_a": 1, "tau_r": 15, "jitter": 2},
    {"name": "Pacing (2, 12)", "tau_a": 2, "tau_r": 12, "jitter": 2},
    {"name": "Pacing (2, 18)", "tau_a": 2, "tau_r": 18, "jitter": 3},
]
N_SEEDS = 100


def run_monte_carlo(dataset_name, scores, n_seeds=N_SEEDS):
    master_rng = np.random.RandomState(42)
    max_start = max(100, len(scores) - 1500)

    per_schedule = []
    for sched in SCHEDULES:
        b0_arr   = []
        aatc_arr = []
        pct_arr  = []
        cut_arr  = []

        for _ in range(n_seeds):
            seed      = master_rng.randint(0, 1_000_000)
            start_idx = master_rng.randint(0, max_start)
            res = simulate_single_run(
                scores=scores,
                start_idx=start_idx,
                tau_a=sched["tau_a"],
                tau_r=sched["tau_r"],
                jitter_range=sched["jitter"],
                seed=seed,
                max_cycles=25,
            )
            b0_arr.append(res["b0_mcum"])
            aatc_arr.append(res["aatc_mcum"])
            pct_arr.append(res["pct_mcum_reduction"])
            cut_arr.append(res["delta_allow"])

        per_schedule.append({
            "schedule":   sched["name"],
            "tau_a":      sched["tau_a"],
            "tau_r":      sched["tau_r"],
            "b0_mcum":    b0_arr,
            "aatc_mcum":  aatc_arr,
            "pct_reduction": pct_arr,
            "allow_cut":  cut_arr,
        })
        print(f"  {sched['name']}: mean red {np.mean(pct_arr):.2f}% ± {np.std(pct_arr):.2f}%  "
              f"B0={np.mean(b0_arr):.2f}  AATC={np.mean(aatc_arr):.2f}")

    return per_schedule


# ─── Wilcoxon analysis ────────────────────────────────────────────────────────
def wilcoxon_analysis(per_schedule_data, label):
    rows = []
    for sched in per_schedule_data:
        b0   = np.array(sched["b0_mcum"])
        aatc = np.array(sched["aatc_mcum"])
        diff = b0 - aatc

        # Drop zero-difference seeds (tied)
        nonzero = diff[diff != 0]
        n_total = len(diff)
        n_ties  = n_total - len(nonzero)
        n_eff   = len(nonzero)

        if n_eff == 0:
            W, p = 0.0, 1.0
        else:
            W, p = wilcoxon(nonzero, alternative="greater")

        rows.append({
            "dataset":  label,
            "schedule": sched["schedule"],
            "n_total":  n_total,
            "n_ties":   n_ties,
            "n_eff":    n_eff,
            "W":        W,
            "p":        p,
            "b0_mean":    float(np.mean(b0)),
            "b0_std":     float(np.std(b0)),
            "aatc_mean":  float(np.mean(aatc)),
            "aatc_std":   float(np.std(aatc)),
            "pct_mean":   float(np.mean(sched["pct_reduction"])),
            "pct_std":    float(np.std(sched["pct_reduction"])),
            "cut_mean":   float(np.mean(sched["allow_cut"])),
            "cut_std":    float(np.std(sched["allow_cut"])),
        })
    return rows


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    # 1. Edge-IIoTset benign scores (eval split rows 10000-17458)
    print("Loading Edge-IIoTset benign chronological scores...")
    df_edge = pd.read_csv(EDGE_CSV)
    edge_scores = df_edge["A_t"].values[10000:17459]
    print(f"  Edge eval scores: {len(edge_scores)} frames")

    # 2. CIC-IoT-2023 benign scores (Normal rows, rescaled anomaly_score)
    print("Loading CIC-IoT-2023 benign scores...")
    df_cic = pd.read_csv(CICIOT_CSV, usecols=["Attack_type"])
    normal_indices = df_cic[df_cic["Attack_type"] == "Normal"].index.values

    A_THR_BINARY = 0.7042
    scale = A_THR / A_THR_BINARY
    cic_scores_all = []
    with open(CICIOT_JSONL) as f:
        for i, line in enumerate(f):
            if line.strip():
                rec = json.loads(line)
                cic_scores_all.append(rec["anomaly_score"])
    cic_scores_all = np.array(cic_scores_all)
    cic_a_t = np.clip(cic_scores_all[normal_indices] * scale, 0.0, 1.0)
    print(f"  CIC Normal scores: {len(cic_a_t)} frames")

    # 3. Run Monte Carlo
    print("\n=== EDGE-IIoTset Monte Carlo ===")
    edge_per_sched = run_monte_carlo("Edge-IIoTset", edge_scores)

    print("\n=== CIC-IoT-2023 Monte Carlo ===")
    cic_per_sched = run_monte_carlo("CIC-IoT-2023", cic_a_t)

    # 4. Save per-seed data (stripped of large arrays for JSON compatibility)
    per_seed_out = {
        "edge_iiotset": edge_per_sched,
        "cic_iot_2023": cic_per_sched,
    }
    with open(OUT_PER_SEED, "w") as f:
        json.dump(per_seed_out, f, indent=2)
    print(f"\nPer-seed data saved to {OUT_PER_SEED.name}")

    # 5. Wilcoxon analysis
    edge_wilcox = wilcoxon_analysis(edge_per_sched, "Edge-IIoTset")
    cic_wilcox  = wilcoxon_analysis(cic_per_sched,  "CIC-IoT-2023")
    all_wilcox  = edge_wilcox + cic_wilcox

    # 6. Human-readable summary (Tables 6 & 7)
    summary_lines = []

    summary_lines.append("\nTable 6 — Monte Carlo M_cum Reduction Results (100 seeds)")
    summary_lines.append("=" * 100)
    summary_lines.append(f"{'Dataset':<18} {'Schedule':<18} {'B0 M_cum':>14} {'AATC M_cum':>14} {'Red %':>10} {'Pulses Cut':>12}")
    summary_lines.append("-" * 100)
    for r in all_wilcox:
        summary_lines.append(
            f"{r['dataset']:<18} {r['schedule']:<18} "
            f"{r['b0_mean']:>6.2f}±{r['b0_std']:>5.2f}  "
            f"{r['aatc_mean']:>6.2f}±{r['aatc_std']:>5.2f}  "
            f"{r['pct_mean']:>7.2f}%  "
            f"{r['cut_mean']:>5.2f}±{r['cut_std']:>5.2f}"
        )

    summary_lines.append("\n\nTable 7 — Wilcoxon Signed-Rank Test (W is positive-rank sum, zero-diff seeds excluded)")
    summary_lines.append("=" * 90)
    summary_lines.append(f"{'Dataset':<18} {'Schedule':<18} {'N_eff':>6} {'Ties':>5} {'W':>8} {'p-value':>12} {'Sig?':>6}")
    summary_lines.append("-" * 90)
    for r in all_wilcox:
        sig = "***" if r['p'] < 0.001 else ("**" if r['p'] < 0.01 else ("*" if r['p'] < 0.05 else "ns"))
        summary_lines.append(
            f"{r['dataset']:<18} {r['schedule']:<18} "
            f"{r['n_eff']:>6} {r['n_ties']:>5} "
            f"{r['W']:>8.1f} {r['p']:>12.3e} {sig:>6}"
        )
    summary_text = "\n".join(summary_lines)
    print(summary_text)
    with open(OUT_SUMMARY, "w") as f:
        f.write(summary_text)
    print(f"\nSummary saved to {OUT_SUMMARY.name}")

    # 7. LaTeX Table 6
    t6_lines = []
    t6_lines.append("% Table 6 — Auto-generated by generate_tables_6_7.py")
    t6_lines.append(r"\begin{table}[htbp]")
    t6_lines.append(r"\caption{Monte Carlo M\textsubscript{cum} Reduction (100 seeds each)}")
    t6_lines.append(r"\begin{tabular}{llrrrr}")
    t6_lines.append(r"\toprule")
    t6_lines.append(r"Dataset & Schedule & $\bar{M}^{B_0}$ & $\bar{M}^{AATC}$ & Red.\% & Pulses Cut \\")
    t6_lines.append(r"\midrule")
    cur_ds = None
    for r in all_wilcox:
        ds_cell = r["dataset"] if r["dataset"] != cur_ds else ""
        cur_ds  = r["dataset"]
        t6_lines.append(
            f"{ds_cell} & {r['schedule']} & "
            f"${r['b0_mean']:.2f}\\pm{r['b0_std']:.2f}$ & "
            f"${r['aatc_mean']:.2f}\\pm{r['aatc_std']:.2f}$ & "
            f"{r['pct_mean']:.2f}\\% & "
            f"${r['cut_mean']:.2f}\\pm{r['cut_std']:.2f}$ \\\\"
        )
        if ds_cell == "" and r == next((x for x in all_wilcox if x["dataset"] == "CIC-IoT-2023"), None):
            t6_lines.append(r"\midrule")
    t6_lines.append(r"\bottomrule")
    t6_lines.append(r"\end{tabular}")
    t6_lines.append(r"\end{table}")
    with open(OUT_T6, "w") as f:
        f.write("\n".join(t6_lines))

    # 8. LaTeX Table 7
    t7_lines = []
    t7_lines.append("% Table 7 — Auto-generated by generate_tables_6_7.py")
    t7_lines.append(r"\begin{table}[htbp]")
    t7_lines.append(r"\caption{Wilcoxon Signed-Rank Test ($W^+$, zero-diff seeds excluded)}")
    t7_lines.append(r"\begin{tabular}{llrrrr}")
    t7_lines.append(r"\toprule")
    t7_lines.append(r"Dataset & Schedule & $N_{\text{eff}}$ & Ties & $W^+$ & $p$-value \\")
    t7_lines.append(r"\midrule")
    cur_ds = None
    for r in all_wilcox:
        ds_cell = r["dataset"] if r["dataset"] != cur_ds else ""
        cur_ds  = r["dataset"]
        p_str = f"$<0.001$" if r["p"] < 0.001 else f"${r['p']:.3f}$"
        t7_lines.append(
            f"{ds_cell} & {r['schedule']} & "
            f"{r['n_eff']} & {r['n_ties']} & "
            f"${r['W']:.1f}$ & {p_str} \\\\"
        )
    t7_lines.append(r"\bottomrule")
    t7_lines.append(r"\end{tabular}")
    t7_lines.append(r"\end{table}")
    with open(OUT_T7, "w") as f:
        f.write("\n".join(t7_lines))

    print(f"Table 6 LaTeX saved to {OUT_T6.name}")
    print(f"Table 7 LaTeX saved to {OUT_T7.name}")

    return all_wilcox


if __name__ == "__main__":
    results = main()
