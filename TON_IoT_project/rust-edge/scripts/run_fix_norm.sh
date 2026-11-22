#!/usr/bin/env bash
set -e
source /tmp/aatc-venv/bin/activate 2>/dev/null || {
  python3 -m venv /tmp/aatc-venv
  source /tmp/aatc-venv/bin/activate
  pip install -q onnxruntime scikit-learn pandas numpy
}
python "/mnt/c/Users/Shreyas A/PROJECTS/BAZT-TON_IoT/TON_IoT_project/rust-edge/scripts/fix_normalization.py" 2>&1
