#!/usr/bin/env bash
set -e
source /tmp/aatc-venv/bin/activate 2>/dev/null || { python3 -m venv /tmp/aatc-venv && source /tmp/aatc-venv/bin/activate && pip install -q onnxruntime pandas numpy; }

python3 - << 'PYEOF'
import json, pandas as pd, numpy as np
from pathlib import Path

CSV     = Path("/mnt/c/Users/Shreyas A/PROJECTS/Behavior-Adaptive-Zero-Trust-IoT/edge-iiot-anomaly-detection/data/raw/DNN-EdgeIIoT-dataset.csv")
PROJ    = Path("/mnt/c/Users/Shreyas A/PROJECTS/BAZT-TON_IoT/TON_IoT_project/rust-edge")
META    = PROJ / "model_meta.json"
EVAL_CSV = PROJ / "eval_set.csv"

meta         = json.loads(META.read_text())
feature_cols = meta["feature_cols"]

# ── 1. Benign eval rows: rows 10000-20000 (not in training split) ────────────
print("Reading benign eval rows (10k-20k)...")
df_benign_all = pd.read_csv(CSV, nrows=20_000, low_memory=False)
df_benign = df_benign_all.iloc[10_000:20_000].copy()
print(f"  Benign rows: {len(df_benign)}  labels: {df_benign['Attack_label'].value_counts().to_dict()}")

# ── 2. Attack rows: scan via chunks, collect from ~2M mark ──────────────────
print("Scanning for attack rows (chunked, targeting row ~2M)...")
CHUNK       = 100_000
TARGET_ROW  = 1_950_000   # start collecting shortly before 2M
N_ATTACK    = 7_459
attack_dfs  = []
row_cursor  = 0

for chunk in pd.read_csv(CSV, chunksize=CHUNK, low_memory=False):
    row_cursor += len(chunk)
    if row_cursor < TARGET_ROW:
        continue   # skip early chunks fast
    # Collect attack rows from this chunk
    atk = chunk[chunk["Attack_label"] == 1]
    if len(atk) > 0:
        attack_dfs.append(atk)
        collected = sum(len(d) for d in attack_dfs)
        print(f"  Collected {collected} attack rows at cursor ~{row_cursor:,} (type: {atk['Attack_type'].iloc[0]})")
        if collected >= N_ATTACK:
            break

if attack_dfs:
    df_attack = pd.concat(attack_dfs).head(N_ATTACK).copy()
else:
    df_attack = pd.DataFrame(columns=df_benign.columns)
print(f"  Attack rows collected: {len(df_attack)}")
if len(df_attack) > 0:
    print(f"  Attack types: {df_attack['Attack_type'].value_counts().to_dict()}")

# ── 3. Merge + shuffle ───────────────────────────────────────────────────────
eval_df = pd.concat([df_benign, df_attack]).sample(frac=1, random_state=42).reset_index(drop=True)
print(f"\nEval set: {len(eval_df)} rows total")
print(f"  Label breakdown: {eval_df['Attack_label'].value_counts().to_dict()}")

eval_df.to_csv(EVAL_CSV, index=False)
print(f"Saved → {EVAL_CSV}")
PYEOF
