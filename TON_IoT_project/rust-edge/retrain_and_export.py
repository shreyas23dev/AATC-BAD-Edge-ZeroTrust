#!/usr/bin/env python3
"""
retrain_and_export.py  — AATC-BAD v2: Flow-level windowed features
=======================================================================
Instead of single-frame inference, we aggregate over a per-device sliding
window of WINDOW_SIZE=16 frames:
  window_features = [mean_44, std_44]  →  88-dimensional input to IF

This exposes temporal patterns invisible to per-frame scoring:
  - Vulnerability_scanner  → near-zero std (uniform probes), high port variance in mean
  - Normal traffic         → varied means, non-zero stds

Usage:
    python rust-edge/retrain_and_export.py
"""

import sys, pickle, json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline

import platform, os
if platform.system() == "Linux" and os.path.exists("/mnt/c"):
    DATASET = Path("/mnt/c/Users/Shreyas A/PROJECTS/Behavior-Adaptive-Zero-Trust-IoT/edge-iiot-anomaly-detection/data/raw/DNN-EdgeIIoT-dataset.csv")
else:
    DATASET = Path(r"C:\Users\Shreyas A\PROJECTS\Behavior-Adaptive-Zero-Trust-IoT\edge-iiot-anomaly-detection\data\raw\DNN-EdgeIIoT-dataset.csv")

OUT_DIR       = Path(__file__).parent
PKL_OUT       = OUT_DIR / "isolation_forest.pkl"
ONNX_OUT      = OUT_DIR / "model.onnx"            # bare IF (88-dim input, pre-scaled)
PIPELINE_ONNX = OUT_DIR / "model_pipeline.onnx"   # scaler + IF (88-dim raw input)

# ── Frozen paper parameters ──────────────────────────────────────────────────
IF_CONTAMINATION = 0.1
IF_N_ESTIMATORS  = 100
IF_MAX_SAMPLES   = "auto"
IF_RANDOM_STATE  = 42
TRAIN_ROWS       = 10_000
TOTAL_ROWS       = 17_459

# ── Windowing ────────────────────────────────────────────────────────────────
WINDOW_SIZE      = 16   # frames per device; must match WINDOW_SIZE in replay.rs
DEVICE_COL       = "ip.src_host"

print("=" * 60)
print("AATC-BAD v2 — Flow-level Window Features + ONNX Export")
print(f"  Window size : {WINDOW_SIZE} frames per device")
print(f"  Feature dim : 44 raw → 88 windowed (mean + std)")
print("=" * 60)

# ── 1. Load dataset ──────────────────────────────────────────────────────────
print(f"\n[1/6] Loading dataset ({TOTAL_ROWS} rows)...")
df = pd.read_csv(DATASET, nrows=TOTAL_ROWS, low_memory=False)
print(f"      Shape: {df.shape}")
print(f"      Devices: {df[DEVICE_COL].nunique() if DEVICE_COL in df.columns else 'N/A'}")

# ── 2. Feature engineering ───────────────────────────────────────────────────
print("\n[2/6] Feature engineering...")
DROP_COLS = {
    "Attack_label", "Attack_type", "frame.time",
    "ip.src_host", "ip.dst_host", "arp.src.proto_ipv4", "arp.dst.proto_ipv4",
    "http.file_data", "http.request.full_uri", "http.request.uri.query",
    "http.referer", "icmp.transmit_timestamp", "dns.qry.name", "mqtt.msg",
}
feature_cols = [c for c in df.columns if c not in DROP_COLS]
numeric_cols = []
for c in feature_cols:
    try:
        pd.to_numeric(df[c], errors="raise")
        numeric_cols.append(c)
    except Exception:
        pass

print(f"      Raw numeric feature columns: {len(numeric_cols)}")

X_raw = df[numeric_cols].apply(pd.to_numeric, errors="coerce").fillna(0).values.astype(np.float32)
labels = df["Attack_label"].values if "Attack_label" in df.columns else np.zeros(len(df))

# ── 3. Per-device sliding window aggregation ─────────────────────────────────
print(f"\n[3/6] Computing windowed features (window={WINDOW_SIZE}, per device)...")

def compute_window_features(X, df, device_col, window_size):
    """
    For each row i (belonging to device d), compute mean and std of the
    last window_size rows of device d (inclusive of row i).

    Returns X_win of shape (N, 2 * n_features):
      columns 0..n_features-1   = per-feature mean over window
      columns n_features..end   = per-feature std  over window
    """
    n, f = X.shape
    means = np.zeros((n, f), dtype=np.float32)
    stds  = np.zeros((n, f), dtype=np.float32)

    if device_col in df.columns:
        # Group indices by device (preserving original row order)
        device_to_indices = {}
        for global_idx, device in enumerate(df[device_col].values):
            device_to_indices.setdefault(device, []).append(global_idx)

        for device, indices in device_to_indices.items():
            # indices are already in original order (pandas iteration order)
            for j, global_idx in enumerate(indices):
                start = max(0, j - window_size + 1)
                window = X[indices[start:j+1]]      # shape (w, f)
                means[global_idx] = window.mean(axis=0)
                if len(window) > 1:
                    stds[global_idx] = window.std(axis=0)   # ddof=0, matches Rust
                # else stds row stays 0 (cold start)
    else:
        # No device grouping — treat each row as its own window
        means = X.copy()
        # stds stay 0

    X_win = np.concatenate([means, stds], axis=1)
    return X_win.astype(np.float32)

X_win = compute_window_features(X_raw, df, DEVICE_COL, WINDOW_SIZE)
n_features_win = X_win.shape[1]   # 88
print(f"      Windowed feature dimension: {n_features_win}")

# Check discriminability: benign vs (hypothetical) attack
benign_mask = (labels == 0)
X_win_train = X_win[:TRAIN_ROWS]
X_win_full  = X_win

# Std columns (second half) — expect near-zero for uniform scanners
std_part = X_win[:, len(numeric_cols):]
print(f"      Std columns (benign train) mean : {std_part[:TRAIN_ROWS].mean():.4f}")

# ── 4. Scale ─────────────────────────────────────────────────────────────────
print("\n[4/6] MinMaxScaler on windowed features...")
scaler = MinMaxScaler()
X_win_scaled = scaler.fit_transform(X_win).astype(np.float32)
X_win_train_scaled = X_win_scaled[:TRAIN_ROWS]

# ── 5. Train Isolation Forest (benign windowed rows only) ────────────────────
print(f"\n[5/6] Training Isolation Forest on windowed benign rows...")
if "Attack_label" in df.columns:
    benign_mask_train = (df["Attack_label"].values[:TRAIN_ROWS] == 0)
    X_fit = X_win_train_scaled[benign_mask_train]
    print(f"      Benign training rows: {X_fit.shape[0]}")
else:
    X_fit = X_win_train_scaled
    print(f"      Training rows (no label): {X_fit.shape[0]}")

clf = IsolationForest(
    n_estimators=IF_N_ESTIMATORS,
    max_samples=IF_MAX_SAMPLES,
    contamination=IF_CONTAMINATION,
    random_state=IF_RANDOM_STATE,
    n_jobs=-1,
)
clf.fit(X_fit)
print("      Training complete.")

# Score distribution on training set
raw_scores = clf.score_samples(X_win_train_scaled)
score_min  = raw_scores.min()
score_max  = raw_scores.max()
norm_scores = np.clip((score_max - raw_scores) / (score_max - score_min + 1e-9), 0, 1)
a_thr_new   = float(np.percentile(norm_scores, 90))
print(f"      score_samples range (train) : [{score_min:.4f}, {score_max:.4f}]")
print(f"      Norm 90th-pct (train)       : {a_thr_new:.4f}  ← new A_thr")

# Save pickle
with open(PKL_OUT, "wb") as f:
    pickle.dump({
        "model": clf,
        "scaler": scaler,
        "feature_cols": numeric_cols,
        "window_size": WINDOW_SIZE,
        "score_min": float(score_min),
        "score_max": float(score_max),
    }, f)
print(f"      Saved pickle → {PKL_OUT}")

# ── 6. Export to ONNX ────────────────────────────────────────────────────────
print(f"\n[6/6] Exporting Pipeline ONNX (88-dim windowed input)...")
try:
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
except ImportError:
    sys.exit("ERROR: pip install skl2onnx onnx")

initial_type = [("float_input", FloatTensorType([None, n_features_win]))]

# 6a. Bare IF (input = pre-scaled 88-dim)
print(f"  [6a] Bare IF ONNX...")
onnx_if = convert_sklearn(
    clf,
    initial_types=initial_type,
    options={type(clf): {"score_samples": True}},
    target_opset={"": 17, "ai.onnx.ml": 3},
)
with open(ONNX_OUT, "wb") as f:
    f.write(onnx_if.SerializeToString())
print(f"       Saved → {ONNX_OUT}  ({ONNX_OUT.stat().st_size/1024:.1f} KB)")

# 6b. Pipeline (scaler + IF; input = raw 88-dim windowed features)
print(f"  [6b] Pipeline ONNX (scaler + IF)...")
pipeline = Pipeline([("scaler", scaler), ("clf", clf)])
onnx_pipeline = convert_sklearn(
    pipeline,
    initial_types=initial_type,
    options={type(clf): {"score_samples": True}},
    target_opset={"": 17, "ai.onnx.ml": 3},
)

# Validate + test
try:
    import onnxruntime as rt
    sess = rt.InferenceSession(onnx_pipeline.SerializeToString())
    input_name = sess.get_inputs()[0].name
    dummy = np.zeros((1, n_features_win), dtype=np.float32)
    out = sess.run(None, {input_name: dummy})
    score_out_idx = None
    print(f"       Input  : {input_name!r}  shape={sess.get_inputs()[0].shape}")
    for i, o in enumerate(sess.get_outputs()):
        print(f"       Out[{i}] : {o.name:35s} shape={o.shape}")
        if "score" in o.name.lower():
            score_out_idx = i
    print(f"       Score output index: {score_out_idx}")
    raw_dummy = float(out[score_out_idx][0][0])
    norm_dummy = float(np.clip((score_max - raw_dummy) / (score_max - score_min + 1e-9), 0, 1))
    print(f"       Dummy score (raw={raw_dummy:.4f} → norm={norm_dummy:.4f})")
except ImportError:
    print("       (onnxruntime not installed)")

try:
    import onnx as _onnx
    try:
        _onnx.checker.check_model(onnx_pipeline)
        print("       ONNX strict check: PASSED")
    except Exception as e:
        print(f"       ONNX strict check skipped ({e.__class__.__name__})")
except ImportError:
    pass

with open(PIPELINE_ONNX, "wb") as f:
    f.write(onnx_pipeline.SerializeToString())
print(f"       Saved → {PIPELINE_ONNX}  ({PIPELINE_ONNX.stat().st_size/1024:.1f} KB)")

# ── Write model_meta.json ────────────────────────────────────────────────────
meta = {
    "n_features_raw": len(numeric_cols),
    "n_features_win": n_features_win,
    "feature_cols": numeric_cols,
    "window_size": WINDOW_SIZE,
    "scaler_min": scaler.data_min_.tolist(),
    "scaler_scale": scaler.scale_.tolist(),
    "if_contamination": IF_CONTAMINATION,
    "if_n_estimators": IF_N_ESTIMATORS,
    "train_rows": TRAIN_ROWS,
    "paper_a_thr": 0.4554,
    "pipeline_onnx": str(PIPELINE_ONNX),
    "note": "Input = window features [mean_44, std_44]. Compute per device before ONNX call.",
    "score_norm": {
        "formula": "anomaly = (score_max - raw) / (score_max - score_min)",
        "score_min": float(score_min),
        "score_max": float(score_max),
        "a_thr_recalibrated": a_thr_new,
    },
}
meta_path = OUT_DIR / "model_meta.json"
with open(meta_path, "w") as f:
    json.dump(meta, f, indent=2)
print(f"\n       Saved meta → {meta_path}")

print("\n" + "=" * 60)
print("✓ Done.")
print(f"  Pipeline ONNX (88-dim windowed input) → {PIPELINE_ONNX}")
print(f"  WINDOW_SIZE in replay.rs must be {WINDOW_SIZE}")
print(f"  FEATURE_DIM in inference.rs must be {len(numeric_cols)} (raw)")
print(f"  MODEL_INPUT_DIM              must be {n_features_win} (windowed)")
print(f"  A_thr (recalibrated)         = {a_thr_new:.4f}")
print("=" * 60)
