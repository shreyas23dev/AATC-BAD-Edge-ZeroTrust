#!/usr/bin/env bash
set -e
export PATH="$HOME/.cargo/bin:$PATH"

PROJ="/mnt/c/Users/Shreyas A/PROJECTS/BAZT-TON_IoT/TON_IoT_project/rust-edge"
XAI_DIR="$PROJ/xai"
MODELS_DIR="$XAI_DIR/models"
BINARY="$PROJ/target/release/aatc-edge"
MODEL_INT8="$MODELS_DIR/autoencoder_int8.onnx"
EVAL_CSV="$PROJ/eval_set.csv"
OUTPUT_XAI="$PROJ/decisions_xai.jsonl"

echo "==== 1. Building aatc-edge with Single-Pass XAI ===="
cd "$PROJ"
cargo build --release 2>&1

echo ""
echo "==== 2. Replay with Single-Pass INT8 Autoencoder XAI ===="
RUST_LOG=warn "$BINARY" replay \
  --csv    "$EVAL_CSV" \
  --model  "$MODEL_INT8" \
  --output "$OUTPUT_XAI" \
  --xai 2>&1

echo ""
echo "==== 3. Decision Breakdown & Attribution Verification ===="
source /home/shreyas/aatc-venv/bin/activate 2>/dev/null || true
python3 - << 'PYEOF'
import json, collections, csv
from pathlib import Path

PROJ = Path("/mnt/c/Users/Shreyas A/PROJECTS/BAZT-TON_IoT/TON_IoT_project/rust-edge")
OUTPUT = PROJ / "decisions_xai.jsonl"
EVAL = PROJ / "eval_set.csv"
META = PROJ / "xai" / "xai_meta.json"

with open(META) as f:
    meta = json.load(f)
feature_cols = meta["feature_cols"]

labels = {}
with open(EVAL) as f:
    for i, row in enumerate(csv.DictReader(f), 1):
        labels[i] = int(row.get("Attack_label", 0))

by_label = {0: collections.Counter(), 1: collections.Counter()}
sample_phi_benign = None
sample_phi_attack = None

with open(OUTPUT) as f:
    for line in f:
        d = json.loads(line)
        lbl = labels.get(d["row"], -1)
        if lbl in by_label:
            by_label[lbl][d["decision"]] += 1
        if lbl == 0 and sample_phi_benign is None and d.get("phi"):
            sample_phi_benign = d["phi"]
        if lbl == 1 and sample_phi_attack is None and d.get("phi"):
            sample_phi_attack = d["phi"]

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

print(f"\n  Detection Metrics (Single-Pass INT8 Autoencoder XAI):")
print(f"  FPR (benign BLOCK)         : {fpr*100:.1f}%")
print(f"  TPR (attack BLOCK only)    : {tpr_bk*100:.1f}%")
print(f"  TPR (attack BLOCK+VERIFY)  : {tpr_bv*100:.1f}%")
print(f"  FNR (attack ALLOW)         : {fnr*100:.2f}%")

if sample_phi_attack:
    # Top 5 most anomalous features for attack sample
    import numpy as np
    phi_arr = np.array(sample_phi_attack)
    top_indices = np.argsort(phi_arr)[::-1][:5]
    print(f"\n  Sample Attribution Vector Phi_t for Attack Frame:")
    for idx in top_indices:
        if idx < len(feature_cols):
            feat_name = f"mean({feature_cols[idx]})"
        else:
            feat_name = f"std({feature_cols[idx - len(feature_cols)]})"
        print(f"    - {feat_name:30s} : squared_error = {phi_arr[idx]:.6f}")

PYEOF
