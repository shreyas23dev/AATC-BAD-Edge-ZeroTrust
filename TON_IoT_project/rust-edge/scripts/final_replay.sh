#!/usr/bin/env bash
# final_replay.sh — replay the pre-built eval_set.csv (10k benign + 7459 attacks)
# and report per-label breakdown.
set -e
export PATH="$HOME/.cargo/bin:$PATH"
source /tmp/aatc-venv/bin/activate 2>/dev/null || { python3 -m venv /tmp/aatc-venv && source /tmp/aatc-venv/bin/activate && pip install -q pandas numpy; }

PROJ="/mnt/c/Users/Shreyas A/PROJECTS/BAZT-TON_IoT/TON_IoT_project/rust-edge"
BINARY="$PROJ/target/release/aatc-edge"
MODEL="$PROJ/model_pipeline.onnx"
EVAL_CSV="$PROJ/eval_set.csv"
OUTPUT="$PROJ/decisions_eval.jsonl"

echo "Eval set row count: $(wc -l < "$EVAL_CSV") (incl. header)"
echo ""

RUST_LOG=warn "$BINARY" replay \
  --csv    "$EVAL_CSV" \
  --model  "$MODEL" \
  --output "$OUTPUT" 2>&1

echo ""
echo "════ Decision breakdown (all rows) ════"
wc -l < "$OUTPUT"

python3 - << 'PYEOF'
import json, collections, csv
from pathlib import Path

PROJ   = Path("/mnt/c/Users/Shreyas A/PROJECTS/BAZT-TON_IoT/TON_IoT_project/rust-edge")
OUTPUT = PROJ / "decisions_eval.jsonl"
EVAL   = PROJ / "eval_set.csv"

# Load ground-truth labels by row index
labels = {}
with open(EVAL) as f:
    reader = csv.DictReader(f)
    for i, row in enumerate(reader, 1):
        labels[i] = int(row.get("Attack_label", 0))

# Load decisions
by_label = {0: collections.Counter(), 1: collections.Counter()}
with open(OUTPUT) as f:
    for line in f:
        d = json.loads(line)
        lbl = labels.get(d["row"], -1)
        if lbl in by_label:
            by_label[lbl][d["decision"]] += 1

print()
for lbl, name in [(0, "BENIGN"), (1, "ATTACK")]:
    c     = by_label[lbl]
    total = sum(c.values())
    print(f"  {name} ({total} rows):")
    for dec in ["ALLOW", "VERIFY", "BLOCK"]:
        n = c[dec]
        pct = n / total * 100 if total else 0
        print(f"    {dec:6s}: {n:5d}  ({pct:5.1f}%)")

# Key metrics
benign_total = sum(by_label[0].values())
attack_total = sum(by_label[1].values())
fpr = by_label[0]["BLOCK"] / benign_total if benign_total else 0
tpr = by_label[1]["BLOCK"] / attack_total if attack_total else 0
print()
print(f"  FPR (benign BLOCKed) : {fpr:.3f}  ({fpr*100:.1f}%)")
print(f"  TPR (attack BLOCKed) : {tpr:.3f}  ({tpr*100:.1f}%)")
print(f"  Note: VERIFY = quarantine/step-up — also indicates suspicion")
tpr_bv = (by_label[1]["BLOCK"] + by_label[1]["VERIFY"]) / attack_total if attack_total else 0
fpr_bv = (by_label[0]["BLOCK"] + by_label[0]["VERIFY"]) / benign_total if benign_total else 0
print(f"  TPR (BLOCK+VERIFY)   : {tpr_bv:.3f}  ({tpr_bv*100:.1f}%)")
print(f"  FPR (BLOCK+VERIFY)   : {fpr_bv:.3f}  ({fpr_bv*100:.1f}%)")
PYEOF
