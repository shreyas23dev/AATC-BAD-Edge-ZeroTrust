#!/usr/bin/env python3
"""
Diagnostic: check what score_samples values the ONNX pipeline actually
returns for benign vs attack rows, and determine the correct normalization.
"""
import sys, json
import numpy as np
import pandas as pd
from pathlib import Path

VENV_SITE = "/tmp/aatc-venv/lib/python3.12/site-packages"
if VENV_SITE not in sys.path:
    sys.path.insert(0, VENV_SITE)

import onnxruntime as rt
import pickle

PROJ    = Path("/mnt/c/Users/Shreyas A/PROJECTS/BAZT-TON_IoT/TON_IoT_project/rust-edge")
DATASET = Path("/mnt/c/Users/Shreyas A/PROJECTS/Behavior-Adaptive-Zero-Trust-IoT/edge-iiot-anomaly-detection/data/raw/DNN-EdgeIIoT-dataset.csv")
META    = PROJ / "model_meta.json"

meta = json.loads(META.read_text())
feature_cols = meta["feature_cols"]
n_features   = meta["n_features"]
A_THR        = meta["paper_a_thr"]   # 0.4554

# Load 100 rows for diagnosis
df = pd.read_csv(DATASET, nrows=200, low_memory=False)
X_raw = df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0).values.astype(np.float32)
labels = df["Attack_label"].values if "Attack_label" in df.columns else np.zeros(len(df))

# Load ONNX pipeline
print("Loading ONNX pipeline...")
sess = rt.InferenceSession(str(PROJ / "model_pipeline.onnx"))
input_name = sess.get_inputs()[0].name
print(f"Input name: {input_name!r}")
print(f"Outputs:")
for i, o in enumerate(sess.get_outputs()):
    print(f"  [{i}] {o.name:30s} shape={o.shape} dtype={o.type}")

# Also load sklearn model for comparison
with open(PROJ / "isolation_forest.pkl", "rb") as f:
    pkl = pickle.load(f)
sklearn_clf    = pkl["model"]
sklearn_scaler = pkl["scaler"]

print(f"\n{'Row':>4}  {'Label':>5}  {'ORT[2]':>12}  {'sklearn_raw':>12}  {'norm_ort':>10}  {'norm_skl':>10}  {'Decision'}")
print("-" * 85)

for i in range(min(20, len(X_raw))):
    row = X_raw[i:i+1]
    
    # ONNX pipeline output
    outs = sess.run(None, {input_name: row})
    ort_score = float(outs[2][0][0])  # output[2] = score_samples
    
    # sklearn reference (apply scaler manually then IF)
    row_scaled = sklearn_scaler.transform(row)
    skl_raw    = float(sklearn_clf.score_samples(row_scaled)[0])
    
    # Current normalization in inference.rs: (0.5 - raw).clip(0,1)
    norm_ort = max(0.0, min(1.0, 0.5 - ort_score))
    norm_skl = max(0.0, min(1.0, 0.5 - skl_raw))
    
    dec_ort = "ALLOW" if norm_ort < A_THR else "BLOCK"
    
    print(f"{i:>4}  {int(labels[i]):>5}  {ort_score:>12.6f}  {skl_raw:>12.6f}  {norm_ort:>10.4f}  {norm_skl:>10.4f}  {dec_ort}")

print()
# Full-range stats across all 200 rows
all_ort = []
all_skl = []
for i in range(len(X_raw)):
    row = X_raw[i:i+1]
    outs = sess.run(None, {input_name: row})
    all_ort.append(float(outs[2][0][0]))
    row_scaled = sklearn_scaler.transform(row)
    all_skl.append(float(sklearn_clf.score_samples(row_scaled)[0]))

all_ort = np.array(all_ort)
all_skl = np.array(all_skl)

print(f"ORT  score_samples → min={all_ort.min():.4f}  max={all_ort.max():.4f}  mean={all_ort.mean():.4f}")
print(f"SKL  score_samples → min={all_skl.min():.4f}  max={all_skl.max():.4f}  mean={all_skl.mean():.4f}")

# Find correct normalization
print(f"\nIf we use norm = (0.5 - ort_raw).clip(0,1):")
norm_ort_arr = np.clip(0.5 - all_ort, 0, 1)
print(f"  norm range: [{norm_ort_arr.min():.4f}, {norm_ort_arr.max():.4f}]")
print(f"  % above A_thr {A_THR}: {(norm_ort_arr > A_THR).mean()*100:.1f}%")

print(f"\nIf we use norm = ort_raw directly (already [0,1]):")
print(f"  range: [{all_ort.min():.4f}, {all_ort.max():.4f}]")
print(f"  % above A_thr {A_THR}: {(all_ort > A_THR).mean()*100:.1f}%")

print(f"\nIf we use norm = (-ort_raw).clip(0,1):")
norm_neg = np.clip(-all_ort, 0, 1)
print(f"  range: [{norm_neg.min():.4f}, {norm_neg.max():.4f}]")
print(f"  % above A_thr {A_THR}: {(norm_neg > A_THR).mean()*100:.1f}%")
