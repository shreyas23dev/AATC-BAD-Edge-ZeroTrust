#!/usr/bin/env python3
"""
run_cross_attack_validation.py — Multi-Attack Stress Test & Explainability Fingerprinting
=========================================================================================
Constructs a balanced multi-attack evaluation stream across 5 distinct threat families:
  1. DDoS_UDP (Layer 4 Volumetric Flood)
  2. DDoS_ICMP (Layer 3 Protocol Flood)
  3. Vulnerability_scanner (Stealthy Network Probe)
  4. Port_Scanning (Reconnaissance)
  5. Password (Application Credential Attack)

Runs through `aatc-edge replay --xai` and computes:
  - Per-attack TPR (Pure BLOCK vs VERIFY vs ALLOW)
  - Per-attack top-3 feature attributions Phi_t proving distinct XAI fingerprints.
"""

import sys, os, subprocess, json, collections, csv
from pathlib import Path
import pandas as pd
import numpy as np

DATASET = Path("/mnt/c/Users/Shreyas A/PROJECTS/Behavior-Adaptive-Zero-Trust-IoT/edge-iiot-anomaly-detection/data/raw/DNN-EdgeIIoT-dataset.csv")
PROJ = Path("/mnt/c/Users/Shreyas A/PROJECTS/BAZT-TON_IoT/TON_IoT_project/rust-edge")
XAI_DIR = PROJ / "xai"
MODELS_DIR = XAI_DIR / "models"
MODEL_INT8 = MODELS_DIR / "autoencoder_int8.onnx"
MULTI_CSV = PROJ / "multi_attack_eval.csv"
OUTPUT_JSONL = PROJ / "decisions_multi_attack.jsonl"
META_PATH = XAI_DIR / "xai_meta.json"

print("=" * 75)
print("AATC-BAD XAI Extension — Cross-Attack Validation & Attribution Fingerprinting")
print("=" * 75)

# ── 1. Build Multi-Attack Evaluation Dataset ─────────────────────────────────
print("\n[1/4] Building balanced multi-attack evaluation stream...")

# 1a. Benign frames (rows 10,000 to 20,000)
print("  Extracting 10,000 benign frames (rows 10k-20k)...")
df_benign = pd.read_csv(DATASET, nrows=20_000, low_memory=False).iloc[10_000:20_000].copy()
df_benign["Attack_category"] = "Normal"

# 1b. Sample 2,000 frames each from 5 attack families
ATTACK_TARGETS = {
    "DDoS_UDP": 2000,
    "DDoS_ICMP": 2000,
    "Vulnerability_scanner": 2000,
    "Port_Scanning": 2000,
    "Password": 2000,
}

attack_collected = {k: [] for k in ATTACK_TARGETS}
total_needed = sum(ATTACK_TARGETS.values())
total_got = 0

CHUNK = 100_000
cursor = 0
START_SCAN = 1_800_000

print("  Scanning dataset chunks from row ~1.8M for target attack classes...")
for chunk in pd.read_csv(DATASET, chunksize=CHUNK, low_memory=False):
    cursor += len(chunk)
    if cursor < START_SCAN:
        continue

    for atype, needed in ATTACK_TARGETS.items():
        curr_len = sum(len(d) for d in attack_collected[atype])
        if curr_len < needed:
            sub = chunk[chunk["Attack_type"] == atype]
            if len(sub) > 0:
                take = min(needed - curr_len, len(sub))
                sub_taken = sub.iloc[:take].copy()
                sub_taken["Attack_category"] = atype
                attack_collected[atype].append(sub_taken)

    total_got = sum(sum(len(d) for d in attack_collected[k]) for k in ATTACK_TARGETS)
    if total_got >= total_needed:
        break

attack_dfs = [pd.concat(attack_collected[k]) for k in ATTACK_TARGETS]
print("  Collected attack samples:")
for k in ATTACK_TARGETS:
    count = sum(len(d) for d in attack_collected[k])
    print(f"    - {k:25s} : {count:5d} frames")

# Combine benign + all attack categories
multi_df = pd.concat([df_benign] + attack_dfs).reset_index(drop=True)
print(f"\n  Multi-attack evaluation set: {len(multi_df):,d} frames total.")
multi_df.to_csv(MULTI_CSV, index=False)
print(f"  Saved -> {MULTI_CSV}")

# ── 2. Run Replay via Rust Native Binary ─────────────────────────────────────
print("\n[2/4] Executing native Rust replay with Single-Pass INT8 Autoencoder XAI...")
binary = PROJ / "target" / "release" / "aatc-edge"

cmd = [
    str(binary), "replay",
    "--csv", str(MULTI_CSV),
    "--model", str(MODEL_INT8),
    "--output", str(OUTPUT_JSONL),
    "--xai",
]

env = os.environ.copy()
env["RUST_LOG"] = "warn"

t0 = pd.Timestamp.now()
res = subprocess.run(cmd, env=env, capture_output=True, text=True)
t1 = pd.Timestamp.now()
duration = (t1 - t0).total_seconds()

if res.returncode != 0:
    print("ERROR in Rust execution:")
    print(res.stderr)
    sys.exit(1)

fps = len(multi_df) / duration
print(f"  Processed {len(multi_df):,d} frames in {duration:.3f} s ({fps:.0f} frames/sec).")

# ── 3. Analyze Decision Matrix per Attack Category ───────────────────────────
print("\n[3/4] Analyzing Detection Performance per Attack Category...")

with open(META_PATH) as f:
    meta = json.load(f)
feature_cols = meta["feature_cols"]

# Map row index to Attack_category
cat_by_row = {}
with open(MULTI_CSV) as f:
    reader = csv.DictReader(f)
    for i, row in enumerate(reader, 1):
        cat_by_row[i] = row["Attack_category"]

decisions_by_cat = collections.defaultdict(collections.Counter)
phi_by_cat = collections.defaultdict(list)

with open(OUTPUT_JSONL) as f:
    for line in f:
        d = json.loads(line)
        row = d["row"]
        cat = cat_by_row.get(row, "Unknown")
        dec = d["decision"]
        decisions_by_cat[cat][dec] += 1
        if d.get("phi"):
            phi_by_cat[cat].append(d["phi"])

print("\n" + "=" * 80)
print(f"{'Category':25s} | {'Total':6s} | {'ALLOW':6s} | {'VERIFY':6s} | {'BLOCK':6s} | {'Pure BLOCK %':12s} | {'TPR/FPR':8s}")
print("-" * 80)

summary_table = []
for cat in ["Normal", "DDoS_UDP", "DDoS_ICMP", "Vulnerability_scanner", "Port_Scanning", "Password"]:
    c = decisions_by_cat[cat]
    total = sum(c.values())
    if total == 0: continue
    allow_cnt = c["ALLOW"]
    ver_cnt = c["VERIFY"]
    block_cnt = c["BLOCK"]
    pure_block_pct = (block_cnt / total) * 100
    
    if cat == "Normal":
        rate_str = f"FPR={pure_block_pct:4.1f}%"
    else:
        rate_str = f"TPR={pure_block_pct:4.1f}%"
        
    print(f"{cat:25s} | {total:6d} | {allow_cnt:6d} | {ver_cnt:6d} | {block_cnt:6d} | {pure_block_pct:11.1f}% | {rate_str:8s}")
    summary_table.append({
        "category": cat,
        "total": total,
        "allow": allow_cnt,
        "verify": ver_cnt,
        "block": block_cnt,
        "pure_block_pct": pure_block_pct,
    })

print("=" * 80)

# ── 4. Extract Top-3 XAI Attribution Fingerprints ────────────────────────────
print("\n[4/4] Extracting Attack-Specific Attribution Fingerprints (Phi_t)...")
print("Showing top 3 distinguishing features (mean squared error) per attack family:\n")

fingerprint_results = {}
for cat in ["DDoS_UDP", "DDoS_ICMP", "Vulnerability_scanner", "Port_Scanning", "Password"]:
    phis = np.array(phi_by_cat[cat])
    if len(phis) == 0: continue
    mean_phi = np.mean(phis, axis=0)
    top_indices = np.argsort(mean_phi)[::-1][:4]

    print(f"  * Threat Family: {cat} (Average Attribution Fingerprint)")
    cat_fingerprint = []
    for idx in top_indices:
        if idx < len(feature_cols):
            feat_name = f"mean({feature_cols[idx]})"
        else:
            feat_name = f"std({feature_cols[idx - len(feature_cols)]})"
        err = float(mean_phi[idx])
        print(f"      - {feat_name:30s} : attribution error = {err:.6f}")
        cat_fingerprint.append({"feature": feat_name, "error": err})
    print()
    fingerprint_results[cat] = cat_fingerprint

# Save results to json
final_output = {
    "summary_table": summary_table,
    "attribution_fingerprints": fingerprint_results,
    "throughput_fps": fps,
}

out_path = XAI_DIR / "cross_attack_results.json"
with open(out_path, "w") as f:
    json.dump(final_output, f, indent=2)

print(f"Saved complete cross-attack validation results -> {out_path}")
print("=" * 75)
