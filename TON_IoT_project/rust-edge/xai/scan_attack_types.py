#!/usr/bin/env python3
"""
scan_attack_types.py — Discover all attack categories and row distributions
in DNN-EdgeIIoT-dataset.csv beyond row 1.9M.
"""

import pandas as pd
from pathlib import Path

DATASET = Path("/mnt/c/Users/Shreyas A/PROJECTS/Behavior-Adaptive-Zero-Trust-IoT/edge-iiot-anomaly-detection/data/raw/DNN-EdgeIIoT-dataset.csv")

print("Scanning DNN-EdgeIIoT-dataset.csv for unique Attack_type labels...")

CHUNK = 100_000
START_ROW = 1_900_000
cursor = 0
attack_type_counts = {}
attack_type_samples = {}

for chunk in pd.read_csv(DATASET, chunksize=CHUNK, low_memory=False):
    cursor += len(chunk)
    if cursor < START_ROW:
        continue

    if "Attack_type" in chunk.columns:
        counts = chunk["Attack_type"].value_counts().to_dict()
        for k, v in counts.items():
            attack_type_counts[k] = attack_type_counts.get(k, 0) + v
            if k not in attack_type_samples and k != "Normal":
                attack_type_samples[k] = cursor - len(chunk)

print(f"\nCompleted scan up to row {cursor:,}.")
print("\nUnique Attack Types Discovered & Frequency:")
for k, v in sorted(attack_type_counts.items(), key=lambda x: -x[1]):
    first_seen = attack_type_samples.get(k, 0)
    print(f"  - {k:25s} : {v:8,d} frames (first seen around row ~{first_seen:,})")
