#!/usr/bin/env python3
"""
Fix: re-derive the correct normalization and A_thr for the ONNX pipeline.
The IF was trained correctly but the normalization formula needs adjustment.
score_samples for benign data: ≈ -0.47 (normal range for a healthy IF model)
"""
import sys, json, pickle
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, "/tmp/aatc-venv/lib/python3.12/site-packages")
import onnxruntime as rt

PROJ    = Path("/mnt/c/Users/Shreyas A/PROJECTS/BAZT-TON_IoT/TON_IoT_project/rust-edge")
DATASET = Path("/mnt/c/Users/Shreyas A/PROJECTS/Behavior-Adaptive-Zero-Trust-IoT/edge-iiot-anomaly-detection/data/raw/DNN-EdgeIIoT-dataset.csv")
META    = PROJ / "model_meta.json"

meta         = json.loads(META.read_text())
feature_cols = meta["feature_cols"]
TRAIN_ROWS   = meta["train_rows"]  # 10000
TOTAL_ROWS   = 17459

# Load full evaluation set
df     = pd.read_csv(DATASET, nrows=TOTAL_ROWS, low_memory=False)
X_raw  = df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0).values.astype(np.float32)
labels = df["Attack_label"].values if "Attack_label" in df.columns else np.zeros(len(df))

sess       = rt.InferenceSession(str(PROJ / "model_pipeline.onnx"))
input_name = sess.get_inputs()[0].name

# Run all 17459 rows through ONNX pipeline
print("Running ONNX inference on 17,459 rows...")
all_raw = []
for i in range(len(X_raw)):
    out = sess.run(None, {input_name: X_raw[i:i+1]})
    all_raw.append(float(out[2][0][0]))
all_raw = np.array(all_raw)

benign_mask = (labels == 0)
attack_mask = (labels == 1)

print(f"\nDataset split:")
print(f"  Benign : {benign_mask.sum()}")
print(f"  Attack : {attack_mask.sum()}")

print(f"\nRaw score_samples (from ONNX):")
print(f"  Benign → min={all_raw[benign_mask].min():.4f}  max={all_raw[benign_mask].max():.4f}  mean={all_raw[benign_mask].mean():.4f}")
if attack_mask.any():
    print(f"  Attack → min={all_raw[attack_mask].min():.4f}  max={all_raw[attack_mask].max():.4f}  mean={all_raw[attack_mask].mean():.4f}")

# The correct normalization: use the clf's decision_function-style offset
# sklearn IF: score_samples() = -(mean path length normalised)
# More negative = MORE anomalous (shorter average path → isolated quickly)
# So: anomaly_score = -score_samples  (flip sign: higher = more anomalous)
# Then min-max normalise to [0,1] using training distribution

# Approach A: simple sign flip, normalise by observed range on TRAINING data
raw_train = all_raw[:TRAIN_ROWS]
raw_min, raw_max = raw_train.min(), raw_train.max()
print(f"\nTraining split raw range: [{raw_min:.4f}, {raw_max:.4f}]")

# Normalise: anomaly_score = (raw_max - raw) / (raw_max - raw_min)  → higher = anomalous
def normalise(raw, rmin, rmax):
    return np.clip((rmax - raw) / (rmax - rmin + 1e-9), 0, 1)

norm_scores = normalise(all_raw, raw_min, raw_max)

print(f"\nNormalised scores (range-normalised, higher=anomalous):")
print(f"  Benign → min={norm_scores[benign_mask].min():.4f}  max={norm_scores[benign_mask].max():.4f}  mean={norm_scores[benign_mask].mean():.4f}")
if attack_mask.any():
    print(f"  Attack → min={norm_scores[attack_mask].min():.4f}  max={norm_scores[attack_mask].max():.4f}  mean={norm_scores[attack_mask].mean():.4f}")

# Calibrate threshold: 90th percentile of training benign scores
benign_train_mask = (labels[:TRAIN_ROWS] == 0)
train_norm = norm_scores[:TRAIN_ROWS]
a_thr_new = np.percentile(train_norm[benign_train_mask], 90)
print(f"\n  90th-pct threshold (benign train) = {a_thr_new:.4f}")
print(f"  Paper A_thr = 0.4554")

# Decision breakdown with new threshold
for thr in [a_thr_new, 0.4554, 0.5, 0.6, 0.7]:
    block_b  = (norm_scores[benign_mask] > thr).sum()
    if attack_mask.any():
        block_a  = (norm_scores[attack_mask] > thr).sum()
        total_a  = attack_mask.sum()
        tpr      = block_a / total_a
    else:
        block_a, tpr = 0, 0
    fpr = block_b / benign_mask.sum()
    print(f"  thr={thr:.4f}: BLOCK benign={block_b}/{benign_mask.sum()} (FPR={fpr:.2%}), BLOCK attack={block_a}/{attack_mask.sum()} (TPR={tpr:.2%})")

# Rust-ready constants
print(f"\n=== RUST CONSTANTS ===")
print(f"  SCORE_MIN  : {raw_min:.6f}_f32")
print(f"  SCORE_MAX  : {raw_max:.6f}_f32")
print(f"  // anomaly = (SCORE_MAX - raw) / (SCORE_MAX - SCORE_MIN)")
print(f"  // A_thr (recalibrated) = {a_thr_new:.4f}")
