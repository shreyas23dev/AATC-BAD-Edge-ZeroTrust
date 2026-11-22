#!/usr/bin/env python3
"""
Find attack rows in the full DNN-EdgeIIoT-dataset.csv and build a
balanced evaluation set for the Rust replay pipeline.
"""
import sys, json
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, "/tmp/aatc-venv/lib/python3.12/site-packages")

DATASET  = Path("/mnt/c/Users/Shreyas A/PROJECTS/Behavior-Adaptive-Zero-Trust-IoT/edge-iiot-anomaly-detection/data/raw/DNN-EdgeIIoT-dataset.csv")
PROJ     = Path("/mnt/c/Users/Shreyas A/PROJECTS/BAZT-TON_IoT/TON_IoT_project/rust-edge")
META     = PROJ / "model_meta.json"
EVAL_CSV = PROJ / "eval_set.csv"  # output: mixed benign + attack rows

# ── Sample the full CSV in chunks to find attack distribution ────────────────
print("Scanning dataset for attack label distribution...")
CHUNK = 50_000
label_counts = {}
first_attack_row = None
chunk_num = 0
total_rows = 0

for chunk in pd.read_csv(DATASET, chunksize=CHUNK, low_memory=False):
    chunk_num += 1
    if "Attack_label" not in chunk.columns and "Attack_type" not in chunk.columns:
        print("ERROR: No Attack_label or Attack_type column found!")
        # Show columns
        print("Columns:", list(chunk.columns[:10]))
        break

    # Try Attack_type (string) or Attack_label (int)
    if "Attack_label" in chunk.columns:
        col = chunk["Attack_label"]
    else:
        col = chunk["Attack_type"]

    vc = col.value_counts()
    for k, v in vc.items():
        label_counts[k] = label_counts.get(k, 0) + v

    # Find first attack row index
    if first_attack_row is None:
        if "Attack_label" in chunk.columns:
            attack_mask = chunk["Attack_label"] != 0
        else:
            attack_mask = chunk["Attack_type"] != "Normal"
        if attack_mask.any():
            first_attack_row = total_rows + attack_mask.idxmax()

    total_rows += len(chunk)
    if chunk_num % 5 == 0:
        print(f"  Scanned {total_rows:,} rows so far...")
    if total_rows >= 500_000:  # scan first 500k rows only
        break

print(f"\nScanned {total_rows:,} rows")
print(f"Label distribution:")
for k, v in sorted(label_counts.items(), key=lambda x: -x[1]):
    print(f"  {str(k):30s}: {v:8,}")
print(f"First attack row index: {first_attack_row}")

# ── Build evaluation set: 10000 benign + 7459 attack rows ───────────────────
# (matching paper's train/eval split)
print(f"\nBuilding evaluation set CSV...")

meta = json.loads(META.read_text())
feature_cols = meta["feature_cols"]

# Read full data up to 2 * first_attack_row + 10000 to get a good mix
if first_attack_row is not None and first_attack_row < 500_000:
    read_rows = min(first_attack_row + 50_000, 500_000)
else:
    read_rows = 500_000

df = pd.read_csv(DATASET, nrows=read_rows, low_memory=False)
print(f"Read {len(df):,} rows")

label_col = "Attack_label" if "Attack_label" in df.columns else "Attack_type"
if label_col == "Attack_label":
    benign_mask = df[label_col] == 0
    attack_mask = df[label_col] != 0
else:
    benign_mask = df[label_col] == "Normal"
    attack_mask = df[label_col] != "Normal"

print(f"Benign: {benign_mask.sum():,}  Attack: {attack_mask.sum():,}")

# Sample 10000 benign from AFTER train split (rows 10000+), 7459 attack
benign_eval = df[benign_mask].iloc[10_000:20_000]   # eval benign (not training rows)
attack_eval  = df[attack_mask].head(7_459)

print(f"Eval benign sampled : {len(benign_eval):,}")
print(f"Eval attack sampled : {len(attack_eval):,}")

eval_df = pd.concat([benign_eval, attack_eval]).sample(frac=1, random_state=42).reset_index(drop=True)
print(f"Eval set total      : {len(eval_df):,} rows (shuffled)")

eval_df.to_csv(EVAL_CSV, index=False)
print(f"\nSaved eval set → {EVAL_CSV}")
print(f"Label breakdown:")
print(eval_df[label_col].value_counts().to_string())
