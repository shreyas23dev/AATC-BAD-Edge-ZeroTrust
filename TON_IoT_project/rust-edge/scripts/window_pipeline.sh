#!/usr/bin/env bash
set -e
export PATH="$HOME/.cargo/bin:$PATH"
source /tmp/aatc-venv/bin/activate 2>/dev/null || {
  python3 -m venv /tmp/aatc-venv
  source /tmp/aatc-venv/bin/activate
  pip install -q skl2onnx onnx onnxruntime scikit-learn pandas numpy
}

PROJ="/mnt/c/Users/Shreyas A/PROJECTS/BAZT-TON_IoT/TON_IoT_project/rust-edge"

# ── Step 1: Export windowed pipeline ONNX ───────────────────────────────────
echo "════ Step 1: Windowed ONNX export ════"
cd "$PROJ"
python retrain_and_export.py

# ── Step 2: Build chronological eval set (benign 10k-20k + attack ~2M) ──────
echo ""
echo "════ Step 2: Build chronological eval_set.csv ════"
python3 - << 'PYEOF'
import json, pandas as pd, numpy as np
from pathlib import Path

CSV      = Path("/mnt/c/Users/Shreyas A/PROJECTS/Behavior-Adaptive-Zero-Trust-IoT/edge-iiot-anomaly-detection/data/raw/DNN-EdgeIIoT-dataset.csv")
PROJ     = Path("/mnt/c/Users/Shreyas A/PROJECTS/BAZT-TON_IoT/TON_IoT_project/rust-edge")
EVAL_CSV = PROJ / "eval_set.csv"

# Benign eval rows: 10000-20000 (chronological, NOT shuffled)
print("Reading benign eval rows (10k-20k, chronological)...")
df_benign = pd.read_csv(CSV, nrows=20_000, low_memory=False).iloc[10_000:]
print(f"  Benign: {len(df_benign)}, label={df_benign['Attack_label'].value_counts().to_dict()}")

# Attack rows from ~2M (chronological, NOT shuffled)
print("Collecting attack rows (chunked)...")
attack_dfs = []
row_cursor  = 0
CHUNK       = 100_000
TARGET      = 1_950_000
N_ATTACK    = 7_459

for chunk in pd.read_csv(CSV, chunksize=CHUNK, low_memory=False):
    row_cursor += len(chunk)
    if row_cursor < TARGET:
        continue
    atk = chunk[chunk["Attack_label"] == 1]
    if len(atk) > 0:
        attack_dfs.append(atk)
        collected = sum(len(d) for d in attack_dfs)
        print(f"  Collected {collected} attack rows at cursor ~{row_cursor:,}")
        if collected >= N_ATTACK:
            break

df_attack = pd.concat(attack_dfs).head(N_ATTACK) if attack_dfs else pd.DataFrame()
print(f"  Attack: {len(df_attack)}, type={df_attack['Attack_type'].value_counts().to_dict() if len(df_attack) else {}}")

# Concatenate chronologically: benign first, then attack (NOT shuffled)
# This preserves per-device window coherence
eval_df = pd.concat([df_benign, df_attack]).reset_index(drop=True)
print(f"\nEval set: {len(eval_df)} rows (chronological — benign then attack)")
eval_df.to_csv(EVAL_CSV, index=False)
print(f"Saved → {EVAL_CSV}")
PYEOF

echo ""
echo "════ Step 3: cargo build --release ════"
cd "$PROJ"
cargo build --release 2>&1 | grep -E '(Compiling|Finished|error|warning:.*unused)'

echo ""
echo "════ Step 4: Replay (windowed) ════"
BINARY="$PROJ/target/release/aatc-edge"
MODEL="$PROJ/model_pipeline.onnx"
EVAL_CSV="$PROJ/eval_set.csv"
OUTPUT="$PROJ/decisions_eval.jsonl"

RUST_LOG=warn "$BINARY" replay \
  --csv    "$EVAL_CSV" \
  --model  "$MODEL" \
  --output "$OUTPUT" 2>&1 | grep -v '^\s*$'

echo ""
echo "════ Results ════"
wc -l < "$OUTPUT"
python3 - << 'PYEOF'
import json, collections, csv
from pathlib import Path

PROJ   = Path("/mnt/c/Users/Shreyas A/PROJECTS/BAZT-TON_IoT/TON_IoT_project/rust-edge")
OUTPUT = PROJ / "decisions_eval.jsonl"
EVAL   = PROJ / "eval_set.csv"

labels = {}
with open(EVAL) as f:
    for i, row in enumerate(csv.DictReader(f), 1):
        labels[i] = int(row.get("Attack_label", 0))

by_label = {0: collections.Counter(), 1: collections.Counter()}
with open(OUTPUT) as f:
    for line in f:
        d = json.loads(line)
        lbl = labels.get(d["row"], -1)
        if lbl in by_label:
            by_label[lbl][d["decision"]] += 1

print()
for lbl, name in [(0, "BENIGN"), (1, "ATTACK")]:
    c = by_label[lbl]
    total = sum(c.values())
    if total == 0: continue
    print(f"  {name} ({total} rows):")
    for dec in ["ALLOW", "VERIFY", "BLOCK"]:
        print(f"    {dec:6s}: {c[dec]:5d}  ({c[dec]/total*100:5.1f}%)")

bt = sum(by_label[0].values()); at = sum(by_label[1].values())
fpr    = by_label[0]["BLOCK"] / bt if bt else 0
tpr_bk = by_label[1]["BLOCK"] / at if at else 0
tpr_bv = (by_label[1]["BLOCK"] + by_label[1]["VERIFY"]) / at if at else 0
fnr    = by_label[1]["ALLOW"] / at if at else 0
print(f"\n  FPR (benign BLOCK)         : {fpr*100:.1f}%")
print(f"  TPR (attack BLOCK only)    : {tpr_bk*100:.1f}%")
print(f"  TPR (attack BLOCK+VERIFY)  : {tpr_bv*100:.1f}%")
print(f"  FNR (attack ALLOW)         : {fnr*100:.2f}%")
PYEOF
