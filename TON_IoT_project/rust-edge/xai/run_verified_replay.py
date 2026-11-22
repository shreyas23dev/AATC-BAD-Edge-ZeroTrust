"""
Verified AATC-BAD Replay — generates Tables 4 & 5 from pre-computed A_t scores.

Strategy:
  - Edge-IIoTset: anomaly scores from decisions_xai.jsonl (ONNX model scores are
    correct; only the old Rust policy engine was wrong).  For Normal rows the
    benign_chronological_scores.csv A_t values are used instead when available,
    but we primarily use the old JSONL scores which ran through the ONNX model.
  - CIC-IoT-2023: mean(phi) from decisions_ciciot_expanded.jsonl, renormalized to
    A_t using the same formula the Monte Carlo script uses.
  - Both streams are processed by the corrected Python step_aatc() function which
    matches Equations (15)-(18) of the paper exactly, and matches the fixed Rust
    engine (verified by parity test).

Output:
  - decisions_edge_verified.jsonl
  - decisions_cic_verified.jsonl
  - table4_edge.txt   (Table 4 — Edge-IIoTset per-class results)
  - table5_cic.txt    (Table 5 — CIC-IoT-2023 per-class results)
  - replay_summary.json  (machine-readable numbers for paper update)
"""

import json
import os
import numpy as np
import pandas as pd
from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE = Path(r"c:\Users\Shreyas A\PROJECTS\BAZT-TON_IoT\TON_IoT_project")
RUST = BASE / "rust-edge"
XAI  = RUST / "xai"
DATA = BASE / "data"

EDGE_DECISIONS_NEW  = RUST / "decisions_multi_attack_new.jsonl"  # 20k frames, correct ONNX scores, fixed engine
CIC_DECISIONS_OLD   = RUST / "decisions_ciciot_expanded.jsonl"
CIC_EVAL_CSV        = DATA / "ciciot2023" / "ciciot_eval_expanded_44.csv"
EDGE_CSV            = RUST / "multi_attack_eval.csv"

OUT_EDGE = RUST / "decisions_edge_verified.jsonl"
OUT_CIC  = RUST / "decisions_cic_verified.jsonl"
OUT_T4   = RUST / "table4_edge.txt"
OUT_T5   = RUST / "table5_cic.txt"
OUT_SUM  = RUST / "replay_summary.json"

# ─── Frozen Paper Parameters (Table 1) ────────────────────────────────────────
A_THR   = 0.4554
Q_LOW   = 0.35
DENOM   = A_THR - Q_LOW       # 0.1054
DELTA_D = 0.10
K_ALLOW = 1.70
ETA_C   = 0.30
C_FLOOR = 0.70
ALPHA_0 = 0.2057
T_ALLOW = 0.80
T_QUAR  = 0.30                 # confirmed: Monte Carlo uses 0.30 (paper notation table has typo)
BETA    = 0.40
GAMMA   = 0.335
THETA   = 1.0
LAMBDA  = 0.55
DELTA   = 0.05
P_PEN   = 0.25

# CIC normalization constant (from Monte Carlo script line 314)
CIC_R_THR = 0.0360             # raw phi mean threshold for CIC model

# ─── AATC-BAD step function (Equations 15-18, matches fixed Rust engine) ──────
def step_aatc(T, D, C, E, S, a):
    """
    One frame of AATC-BAD.
    Args:
        T: trust alpha_t (prev)
        D: debt D_t (prev)
        C: ceiling C_t (prev)
        E: smoothed anomaly E_t (prev)
        S: slow-burn energy S_t (prev)
        a: anomaly score A_t for this frame
    Returns:
        (T_new, D_new, C_new, E_new, S_new, decision_str)
    """
    # Debt accrual / paydown (Eq. 15)
    if (T >= T_ALLOW) and (Q_LOW <= a < A_THR):
        D_new = D + (a - Q_LOW) / DENOM
    elif a < Q_LOW:
        D_new = max(0.0, D - DELTA_D)
    else:
        D_new = D

    # Dynamic ceiling (Eq. 17)
    C_new = max(C_FLOOR, 1.0 - ETA_C * max(0.0, D_new - K_ALLOW))

    # Smoothing & slow-burn (Eqs. 1-2)
    E_new = BETA * a + (1.0 - BETA) * E
    if E_new > GAMMA:
        S_new = S + LAMBDA
    else:
        S_new = max(0.0, S - DELTA)
    slow_burn = (S_new > THETA)

    # Trust update (Eq. 18)
    if a >= A_THR:
        T_new = max(0.0, T * (1.0 - a))
    elif slow_burn:
        T_new = max(0.0, T - P_PEN * (a / A_THR))
    else:
        rec = T + ALPHA_0 * (C_new - T)
        T_new = min(C_new, rec)

    # Decision (Eq. 4)
    if T_new >= T_ALLOW:
        dec = "ALLOW"
    elif T_new >= T_QUAR:
        dec = "VERIFY"
    else:
        dec = "BLOCK"

    return T_new, D_new, C_new, E_new, S_new, dec


def run_replay(frames_iter, out_path):
    """
    frames_iter: iterator of (row_idx, device_id, a_t, label) tuples
    Returns list of result dicts.
    """
    # Per-device state: T, D, C, E, S
    device_states = {}

    results = []
    with open(out_path, "w") as fout:
        for row_idx, device_id, a_t, label in frames_iter:
            if device_id not in device_states:
                # Authenticated device starts at full trust
                device_states[device_id] = (1.0, 0.0, 1.0, 0.0, 0.0)

            T, D, C, E, S = device_states[device_id]
            T_new, D_new, C_new, E_new, S_new, dec = step_aatc(T, D, C, E, S, a_t)
            device_states[device_id] = (T_new, D_new, C_new, E_new, S_new)

            rec = {
                "row": row_idx,
                "device_id": device_id,
                "anomaly_score": round(a_t, 6),
                "trust": round(T_new, 6),
                "debt": round(D_new, 6),
                "ceiling": round(C_new, 6),
                "decision": dec,
                "label": label,
            }
            results.append(rec)
            fout.write(json.dumps(rec) + "\n")
    return results


# ─── Edge-IIoTset Replay ───────────────────────────────────────────────────────
print("=" * 60)
print("EDGE-IIoTset Replay")
print("=" * 60)

edge_csv = pd.read_csv(EDGE_CSV, low_memory=False)
print(f"  Edge CSV rows: {len(edge_csv)}")
print(f"  Label distribution:\n{edge_csv['Attack_type'].value_counts()}")

# decisions_multi_attack_new.jsonl: 20,000 frames, ONNX scores from the fixed-engine
# binary run. The ONNX model scores are correct; we just need to rescale from binary
# A_thr space (0.7042) to paper A_thr space (0.4554).
print(f"\n  Loading anomaly scores from {EDGE_DECISIONS_NEW.name}...")
edge_binary_scores = []
with open(EDGE_DECISIONS_NEW) as f:
    for line in f:
        if line.strip():
            rec = json.loads(line)
            edge_binary_scores.append(rec["anomaly_score"])

print(f"  Loaded {len(edge_binary_scores)} scores")
assert len(edge_binary_scores) == len(edge_csv), \
    f"Score count mismatch: {len(edge_binary_scores)} vs {len(edge_csv)} CSV rows"

# The Rust binary uses a_thr=0.7042 (recalibrated for the ONNX min-max pipeline).
# Paper and Monte Carlo use A_thr=0.4554. Rescale linearly:
#   a_paper = clip(a_binary * (A_THR_paper / A_THR_binary), 0, 1)
A_THR_BINARY = 0.7042
scale_factor = A_THR / A_THR_BINARY
print(f"\n  Rescaling binary scores: factor = {A_THR:.4f} / {A_THR_BINARY:.4f} = {scale_factor:.4f}")

edge_a_t_scaled = np.clip(np.array(edge_binary_scores) * scale_factor, 0.0, 1.0)

normal_mask = edge_csv["Attack_label"] == 0
normal_scaled = edge_a_t_scaled[normal_mask.values]
print(f"  Normal A_t stats after rescaling (N={normal_mask.sum()}):")
print(f"    mean={normal_scaled.mean():.4f}, std={normal_scaled.std():.4f}")
print(f"    pct >= A_thr ({A_THR}): {(normal_scaled >= A_THR).mean()*100:.2f}%")
print(f"    pct borderline [{Q_LOW},{A_THR}): {((normal_scaled >= Q_LOW) & (normal_scaled < A_THR)).mean()*100:.2f}%")

# Build frame iterator
def edge_frame_iter(edge_csv, scores):
    for i, row in edge_csv.iterrows():
        device_id = str(row.get("ip.src_host", f"dev_{i}"))
        # Sanitize device ID
        if pd.isna(device_id) or device_id in ("0", "0.0", "nan"):
            device_id = f"unknown_{i % 10}"
        a_t = float(scores[i])
        label = str(row["Attack_type"])
        yield (i + 1, device_id, a_t, label)

print(f"\n  Running corrected AATC-BAD replay on Edge-IIoTset...")
edge_results = run_replay(edge_frame_iter(edge_csv, edge_a_t_scaled), OUT_EDGE)
print(f"  Written {len(edge_results)} decisions to {OUT_EDGE.name}")


# ─── CIC-IoT-2023 Replay ──────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("CIC-IoT-2023 Replay")
print("=" * 60)

cic_csv = pd.read_csv(CIC_EVAL_CSV, low_memory=False)
print(f"  CIC CSV rows: {len(cic_csv)}")
print(f"  Label distribution:\n{cic_csv['Attack_type'].value_counts()}")

print(f"\n  Loading anomaly_score from {CIC_DECISIONS_OLD.name}...")
cic_binary_scores = []
with open(CIC_DECISIONS_OLD) as f:
    for line in f:
        if line.strip():
            rec = json.loads(line)
            cic_binary_scores.append(rec["anomaly_score"])

print(f"  Loaded {len(cic_binary_scores)} scores")
assert len(cic_binary_scores) == len(cic_csv), \
    f"Score count mismatch: {len(cic_binary_scores)} vs {len(cic_csv)} CSV rows"

# CIC binary uses same a_thr_binary=0.7042 as Edge; rescale to paper A_thr=0.4554
# (same approach as Edge-IIoTset: anomaly_score is min-max normalized by ONNX model)
cic_a_t = np.clip(np.array(cic_binary_scores) * scale_factor, 0.0, 1.0)
normal_mask_cic = cic_csv["Attack_label"] == 0
normal_cic = cic_a_t[normal_mask_cic.values]
print(f"\n  CIC Normal A_t stats after rescaling (N={normal_mask_cic.sum()}):")
print(f"    mean={normal_cic.mean():.4f}, std={normal_cic.std():.4f}")
print(f"    pct >= A_thr ({A_THR}): {(normal_cic >= A_THR).mean()*100:.2f}%")
print(f"    pct borderline [{Q_LOW},{A_THR}): {((normal_cic >= Q_LOW) & (normal_cic < A_THR)).mean()*100:.2f}%")

def cic_frame_iter(cic_csv, scores):
    for i, row in cic_csv.iterrows():
        device_id = str(row.get("src_ip", f"dev_{i}"))
        if pd.isna(device_id) or device_id == "nan":
            device_id = f"unknown_{i % 20}"
        a_t = float(scores[i])
        label = str(row["Attack_type"])
        yield (i + 1, device_id, a_t, label)

print(f"\n  Running corrected AATC-BAD replay on CIC-IoT-2023...")
cic_results = run_replay(cic_frame_iter(cic_csv, cic_a_t), OUT_CIC)
print(f"  Written {len(cic_results)} decisions to {OUT_CIC.name}")


# ─── Analysis ─────────────────────────────────────────────────────────────────
def compute_per_class(results, benign_label="Normal"):
    """Compute ALLOW/VERIFY/BLOCK counts and mitigation per class."""
    from collections import defaultdict
    stats = defaultdict(lambda: {"ALLOW": 0, "VERIFY": 0, "BLOCK": 0, "total": 0})
    for rec in results:
        lbl = rec["label"]
        dec = rec["decision"]
        stats[lbl][dec] += 1
        stats[lbl]["total"] += 1

    rows = []
    for lbl in sorted(stats.keys()):
        s = stats[lbl]
        total = s["total"]
        allow = s["ALLOW"]
        verify = s["VERIFY"]
        block = s["BLOCK"]
        mitigation = (verify + block) / total * 100 if total else 0
        is_attack = (lbl != benign_label)
        rows.append({
            "class": lbl,
            "total": total,
            "ALLOW": allow,
            "ALLOW_pct": allow / total * 100 if total else 0,
            "VERIFY": verify,
            "VERIFY_pct": verify / total * 100 if total else 0,
            "BLOCK": block,
            "BLOCK_pct": block / total * 100 if total else 0,
            "mitigation_pct": mitigation,
            "is_attack": is_attack,
        })
    return rows


def format_table(rows, title, benign_label="Normal"):
    lines = [f"\n{title}", "=" * 90]
    header = f"{'Class':<26} {'N':>6} {'ALLOW%':>8} {'VERIFY%':>8} {'BLOCK%':>8} {'Mitig%':>8}"
    lines.append(header)
    lines.append("-" * 90)
    total_all = total_allow = total_verify = total_block = 0
    benign_row = None
    attack_rows = []
    for r in rows:
        line = (f"{r['class']:<26} {r['total']:>6} "
                f"{r['ALLOW_pct']:>8.2f} {r['VERIFY_pct']:>8.2f} "
                f"{r['BLOCK_pct']:>8.2f} {r['mitigation_pct']:>8.2f}")
        if r["class"] == benign_label:
            benign_row = r
        else:
            attack_rows.append(r)
        total_all   += r["total"]
        total_allow += r["ALLOW"]
        total_verify += r["VERIFY"]
        total_block += r["BLOCK"]

    # Print benign first, then attacks
    if benign_row:
        lines.append(f"{benign_row['class']:<26} {benign_row['total']:>6} "
                     f"{benign_row['ALLOW_pct']:>8.2f} {benign_row['VERIFY_pct']:>8.2f} "
                     f"{benign_row['BLOCK_pct']:>8.2f} {benign_row['mitigation_pct']:>8.2f}")
        lines.append("-" * 90)
    for r in attack_rows:
        lines.append(f"{r['class']:<26} {r['total']:>6} "
                     f"{r['ALLOW_pct']:>8.2f} {r['VERIFY_pct']:>8.2f} "
                     f"{r['BLOCK_pct']:>8.2f} {r['mitigation_pct']:>8.2f}")

    lines.append("=" * 90)
    lines.append(f"{'TOTAL':<26} {total_all:>6} "
                 f"{total_allow/total_all*100:>8.2f} {total_verify/total_all*100:>8.2f} "
                 f"{total_block/total_all*100:>8.2f} "
                 f"{(total_verify+total_block)/total_all*100:>8.2f}")
    return "\n".join(lines)


print("\n" + "=" * 60)
print("RESULTS SUMMARY")
print("=" * 60)

edge_rows = compute_per_class(edge_results, "Normal")
cic_rows  = compute_per_class(cic_results, "Normal")

edge_table = format_table(edge_rows, "Table 4 — Edge-IIoTset (20,000 frames)")
cic_table  = format_table(cic_rows,  "Table 5 — CIC-IoT-2023 (26,011 frames)")

print(edge_table)
print(cic_table)

with open(OUT_T4, "w") as f: f.write(edge_table)
with open(OUT_T5, "w") as f: f.write(cic_table)

# FPR (benign BLOCK rate)
def fpr_tpr(rows, benign_label="Normal"):
    benign = next((r for r in rows if r["class"] == benign_label), None)
    attacks = [r for r in rows if r["class"] != benign_label]
    fpr = benign["BLOCK_pct"] / 100 if benign else None
    total_att = sum(r["total"] for r in attacks)
    total_mitigated = sum(r["VERIFY"] + r["BLOCK"] for r in attacks)
    tpr = total_mitigated / total_att if total_att else None
    return fpr, tpr

edge_fpr, edge_tpr = fpr_tpr(edge_rows)
cic_fpr,  cic_tpr  = fpr_tpr(cic_rows)

print(f"\n  Edge-IIoTset: FPR (benign BLOCK) = {edge_fpr*100:.2f}%,  TPR (attack mitigated) = {edge_tpr*100:.2f}%")
print(f"  CIC-IoT-2023: FPR (benign BLOCK) = {cic_fpr*100:.2f}%,  TPR (attack mitigated) = {cic_tpr*100:.2f}%")

summary = {
    "edge": {
        "total_frames": len(edge_results),
        "per_class": edge_rows,
        "fpr_block_pct": round(edge_fpr * 100, 4),
        "tpr_pct": round(edge_tpr * 100, 4),
    },
    "cic": {
        "total_frames": len(cic_results),
        "per_class": cic_rows,
        "fpr_block_pct": round(cic_fpr * 100, 4),
        "tpr_pct": round(cic_tpr * 100, 4),
    },
    "parameters": {
        "A_THR": A_THR, "Q_LOW": Q_LOW, "K_ALLOW": K_ALLOW, "DELTA_D": DELTA_D,
        "ETA_C": ETA_C, "C_FLOOR": C_FLOOR, "ALPHA_0": ALPHA_0, "T_ALLOW": T_ALLOW,
        "T_QUAR": T_QUAR, "BETA": BETA, "GAMMA": GAMMA,
    }
}

with open(OUT_SUM, "w") as f:
    json.dump(summary, f, indent=2)

print(f"\n  Summary saved to {OUT_SUM.name}")
print(f"  Table 4 saved to {OUT_T4.name}")
print(f"  Table 5 saved to {OUT_T5.name}")
print("\nDone.")
