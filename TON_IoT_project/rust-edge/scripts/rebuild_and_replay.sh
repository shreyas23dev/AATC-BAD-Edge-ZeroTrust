#!/usr/bin/env bash
set -e
export PATH="$HOME/.cargo/bin:$PATH"
source /tmp/aatc-venv/bin/activate 2>/dev/null || { python3 -m venv /tmp/aatc-venv && source /tmp/aatc-venv/bin/activate && pip install -q pandas numpy onnxruntime; }

PROJ="/mnt/c/Users/Shreyas A/PROJECTS/BAZT-TON_IoT/TON_IoT_project/rust-edge"

echo "════ Step 1: Rebuild eval_set.csv (10k benign + 7459 attacks) ════"
bash "$PROJ/scripts/build_eval_set2.sh"

echo ""
echo "Rows in eval_set.csv: $(wc -l < "$PROJ/eval_set.csv") (incl. header)"

echo ""
echo "════ Step 2: Final replay ════"
bash "$PROJ/scripts/final_replay.sh"
