#!/usr/bin/env bash
# replay.sh — run the full Edge-IIoTset replay using the pipeline ONNX
# Usage: bash .../scripts/replay.sh
set -e

PROJ="/mnt/c/Users/Shreyas A/PROJECTS/BAZT-TON_IoT/TON_IoT_project/rust-edge"
BINARY="$PROJ/target/release/aatc-edge"
MODEL="$PROJ/model_pipeline.onnx"
CSV="/mnt/c/Users/Shreyas A/PROJECTS/Behavior-Adaptive-Zero-Trust-IoT/edge-iiot-anomaly-detection/data/raw/DNN-EdgeIIoT-dataset.csv"
OUTPUT="$PROJ/decisions.jsonl"

export PATH="$HOME/.cargo/bin:$PATH"

echo "Binary : $BINARY"
echo "Model  : $MODEL"
echo "CSV    : $CSV"
echo "Output : $OUTPUT"
echo ""

RUST_LOG=info "$BINARY" replay \
  --csv      "$CSV" \
  --model    "$MODEL" \
  --output   "$OUTPUT" \
  --max-rows 17459

echo ""
echo "── Quick stats ──────────────────────────────────────"
echo "Total rows written:"
wc -l < "$OUTPUT"
echo "Decision breakdown:"
python3 -c "
import json, collections, sys
counts = collections.Counter()
with open('$OUTPUT') as f:
    for line in f:
        d = json.loads(line)
        counts[d['decision']] += 1
for k,v in sorted(counts.items()):
    print(f'  {k:8s}: {v:6d}')
print(f'  Total   : {sum(counts.values()):6d}')
"
