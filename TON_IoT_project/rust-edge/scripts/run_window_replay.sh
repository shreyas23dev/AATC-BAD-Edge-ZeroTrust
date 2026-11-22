#!/usr/bin/env bash
set -e
export PATH="$HOME/.cargo/bin:$PATH"
PROJ="/mnt/c/Users/Shreyas A/PROJECTS/BAZT-TON_IoT/TON_IoT_project/rust-edge"

echo "==== Cargo Build Release ===="
cd "$PROJ"
cargo build --release 2>&1

echo "==== Replay (Windowed) ===="
BINARY="$PROJ/target/release/aatc-edge"
MODEL="$PROJ/model_pipeline.onnx"
EVAL_CSV="$PROJ/eval_set.csv"
OUTPUT="$PROJ/decisions_eval.jsonl"

RUST_LOG=warn "$BINARY" replay \
  --csv    "$EVAL_CSV" \
  --model  "$MODEL" \
  --output "$OUTPUT" 2>&1

echo "==== Evaluating Decisions ===="
source /tmp/aatc-venv/bin/activate 2>/dev/null || true
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
