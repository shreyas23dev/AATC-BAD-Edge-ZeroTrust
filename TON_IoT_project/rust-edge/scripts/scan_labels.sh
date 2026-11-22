#!/usr/bin/env bash
set -e
source /tmp/aatc-venv/bin/activate 2>/dev/null || { python3 -m venv /tmp/aatc-venv && source /tmp/aatc-venv/bin/activate && pip install -q pandas numpy; }

python3 - << 'PYEOF'
import pandas as pd
from pathlib import Path

CSV = Path("/mnt/c/Users/Shreyas A/PROJECTS/Behavior-Adaptive-Zero-Trust-IoT/edge-iiot-anomaly-detection/data/raw/DNN-EdgeIIoT-dataset.csv")

# Check columns from first row
header = pd.read_csv(CSV, nrows=1, low_memory=False)
print("Columns:", list(header.columns))
print()

# Check Attack_label and Attack_type unique values across deep slices
for skip in [500_000, 1_000_000, 1_500_000, 2_000_000]:
    try:
        df = pd.read_csv(CSV, skiprows=range(1, skip), nrows=2000, low_memory=False)
        lbl_vc = df["Attack_label"].value_counts().to_dict() if "Attack_label" in df.columns else {}
        typ_vc = df["Attack_type"].value_counts().head(5).to_dict() if "Attack_type" in df.columns else {}
        print(f"Row ~{skip:>9,}: Attack_label={lbl_vc}  Attack_type={typ_vc}")
    except Exception as e:
        print(f"Row ~{skip:>9,}: ERROR {e}")

PYEOF
