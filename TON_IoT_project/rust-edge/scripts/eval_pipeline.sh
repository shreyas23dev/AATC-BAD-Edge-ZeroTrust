#!/usr/bin/env bash
set -e
source /tmp/aatc-venv/bin/activate 2>/dev/null || {
  python3 -m venv /tmp/aatc-venv
  source /tmp/aatc-venv/bin/activate
  pip install -q onnxruntime scikit-learn pandas numpy
}

PROJ="/mnt/c/Users/Shreyas A/PROJECTS/BAZT-TON_IoT/TON_IoT_project/rust-edge"

echo "════ Step 1: Build eval set (find attacks in CSV) ════"
python "$PROJ/scripts/build_eval_set.py"

echo ""
echo "════ Step 2: cargo build --release ════"
export PATH="$HOME/.cargo/bin:$PATH"
cd "$PROJ"
cargo build --release 2>&1

echo ""
echo "════ Step 3: Replay on eval set ════"
BINARY="$PROJ/target/release/aatc-edge"
MODEL="$PROJ/model_pipeline.onnx"
EVAL_CSV="$PROJ/eval_set.csv"
OUTPUT="$PROJ/decisions_eval.jsonl"

RUST_LOG=info "$BINARY" replay \
  --csv    "$EVAL_CSV" \
  --model  "$MODEL" \
  --output "$OUTPUT" 2>&1

echo ""
echo "════ Decision breakdown ════"
wc -l < "$OUTPUT"
python3 -c "
import json, collections
counts = collections.Counter()
with open('$OUTPUT') as f:
    for line in f:
        d = json.loads(line)
        counts[d['decision']] += 1
total = sum(counts.values())
for k,v in sorted(counts.items()):
    print(f'  {k:8s}: {v:6d}  ({v/total*100:.1f}%)')
print(f'  Total   : {total:6d}')
"
