#!/usr/bin/env python3
"""
benchmark_shap_vs_xai.py — Phase 3: Latency & Throughput Benchmark
===================================================================
Produces Table 5 for the Journal Extension manuscript:
Direct empirical comparison between:
  1. Baseline: Isolation Forest + TreeSHAP (sampling-based post-hoc XAI)
  2. InCIS 2027: Isolation Forest alone (no XAI)
  3. Journal Proposed (f32): Single-Pass Autoencoder XAI
  4. Journal Proposed (INT8): Single-Pass Quantized Autoencoder XAI

Measures:
  - Latency per frame (mean, median, p95, std) in ms
  - Throughput (frames / sec)
  - Model size (KB)
  - Explanation fidelity & marginal overhead
"""

import sys, os, time, json, pickle, platform
from pathlib import Path
import numpy as np
import pandas as pd
import onnxruntime as ort
import shap

if platform.system() == "Linux" and os.path.exists("/mnt/c"):
    PROJ_DIR = Path("/mnt/c/Users/Shreyas A/PROJECTS/BAZT-TON_IoT/TON_IoT_project/rust-edge")
else:
    PROJ_DIR = Path(r"C:\Users\Shreyas A\PROJECTS\BAZT-TON_IoT\TON_IoT_project\rust-edge")

XAI_DIR = PROJ_DIR / "xai"
MODELS_DIR = XAI_DIR / "models"
EVAL_CSV = PROJ_DIR / "eval_set.csv"
PKL_PATH = PROJ_DIR / "isolation_forest.pkl"
IF_ONNX_PATH = PROJ_DIR / "model_pipeline.onnx"
AE_F32_PATH = MODELS_DIR / "autoencoder_f32.onnx"
AE_INT8_PATH = MODELS_DIR / "autoencoder_int8.onnx"
META_PATH = XAI_DIR / "xai_meta.json"

print("=" * 70)
print("AATC-BAD XAI Extension — Phase 3: XAI Latency & Throughput Benchmark")
print("=" * 70)

# ── 1. Load Data ─────────────────────────────────────────────────────────────
with open(META_PATH, "r") as f:
    meta = json.load(f)

feature_cols = meta["feature_cols"]
window_size = meta["window_size"]
DEVICE_COL = "ip.src_host"

print(f"\n[1/5] Loading evaluation samples from {EVAL_CSV.name}...")
df = pd.read_csv(EVAL_CSV, nrows=2000, low_memory=False)
X_raw = df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0).values.astype(np.float32)

def compute_window_features(X, df, device_col, window_size):
    n, f = X.shape
    means = np.zeros((n, f), dtype=np.float32)
    stds  = np.zeros((n, f), dtype=np.float32)
    device_to_indices = {}
    for global_idx, device in enumerate(df[device_col].values):
        device_to_indices.setdefault(device, []).append(global_idx)
    for device, indices in device_to_indices.items():
        for j, global_idx in enumerate(indices):
            start = max(0, j - window_size + 1)
            window = X[indices[start:j+1]]
            means[global_idx] = window.mean(axis=0)
            if len(window) > 1:
                stds[global_idx] = window.std(axis=0)
    return np.concatenate([means, stds], axis=1).astype(np.float32)

X_win = compute_window_features(X_raw, df, DEVICE_COL, window_size)
print(f"Prepared {len(X_win)} 88-dimensional windowed samples.")

N_BENCH_FAST = 1000
N_BENCH_SHAP = 100  # TreeSHAP is much slower; 100 samples is standard for latency

bench_samples_fast = [np.expand_dims(X_win[i], axis=0) for i in range(N_BENCH_FAST)]
bench_samples_shap = X_win[:N_BENCH_SHAP]

# ── 2. Benchmark Method 1: Isolation Forest alone (InCIS 2027) ───────────────
print(f"\n[2/5] Benchmarking Method 1: Isolation Forest alone (ONNX, {N_BENCH_FAST} frames)...")
sess_if = ort.InferenceSession(str(IF_ONNX_PATH))
input_name_if = sess_if.get_inputs()[0].name

# Warmup
for _ in range(50):
    _ = sess_if.run(None, {input_name_if: bench_samples_fast[0]})

latencies_if = []
for sample in bench_samples_fast:
    t0 = time.perf_counter()
    _ = sess_if.run(None, {input_name_if: sample})
    latencies_if.append((time.perf_counter() - t0) * 1000.0)

mean_if = np.mean(latencies_if)
fps_if = 1000.0 / mean_if
size_if = IF_ONNX_PATH.stat().st_size / 1024
print(f"      Latency : {mean_if:.3f} ms/frame (p95: {np.percentile(latencies_if, 95):.3f} ms)")
print(f"      Throughput: {fps_if:.1f} fps")
print(f"      Size    : {size_if:.1f} KB")

# ── 3. Benchmark Method 2: Isolation Forest + TreeSHAP (Traditional XAI) ─────
print(f"\n[3/5] Benchmarking Method 2: Isolation Forest + TreeSHAP ({N_BENCH_SHAP} frames)...")
with open(PKL_PATH, "rb") as f:
    pkl_data = pickle.load(f)
sklearn_clf = pkl_data["model"]
scaler = pkl_data["scaler"]

explainer = shap.TreeExplainer(sklearn_clf)

# Warmup
scaled_first = scaler.transform(bench_samples_shap[:1])
_ = explainer.shap_values(scaled_first)

latencies_shap = []
for i in range(N_BENCH_SHAP):
    sample_raw = bench_samples_shap[i:i+1]
    t0 = time.perf_counter()
    # 1. Scale
    sample_scaled = scaler.transform(sample_raw)
    # 2. IF Inference
    _ = sklearn_clf.score_samples(sample_scaled)
    # 3. TreeSHAP Computation
    _ = explainer.shap_values(sample_scaled)
    latencies_shap.append((time.perf_counter() - t0) * 1000.0)

mean_shap = np.mean(latencies_shap)
fps_shap = 1000.0 / mean_shap
print(f"      Latency : {mean_shap:.3f} ms/frame (p95: {np.percentile(latencies_shap, 95):.3f} ms)")
print(f"      Throughput: {fps_shap:.1f} fps")
print(f"      Slowdown vs IF alone: {mean_shap / mean_if:.1f}x slower")

# ── 4. Benchmark Method 3: Autoencoder Float32 (Single-Pass XAI) ─────────────
print(f"\n[4/5] Benchmarking Method 3: Autoencoder Float32 ({N_BENCH_FAST} frames)...")
sess_ae_f32 = ort.InferenceSession(str(AE_F32_PATH))
input_name_ae = sess_ae_f32.get_inputs()[0].name

# Warmup
for _ in range(50):
    _ = sess_ae_f32.run(None, {input_name_ae: bench_samples_fast[0]})

latencies_ae_f32 = []
for sample in bench_samples_fast:
    t0 = time.perf_counter()
    score, phi = sess_ae_f32.run(None, {input_name_ae: sample})
    latencies_ae_f32.append((time.perf_counter() - t0) * 1000.0)

mean_ae_f32 = np.mean(latencies_ae_f32)
fps_ae_f32 = 1000.0 / mean_ae_f32
size_ae_f32 = AE_F32_PATH.stat().st_size / 1024
print(f"      Latency : {mean_ae_f32:.3f} ms/frame (p95: {np.percentile(latencies_ae_f32, 95):.3f} ms)")
print(f"      Throughput: {fps_ae_f32:.1f} fps")
print(f"      Size    : {size_ae_f32:.1f} KB")

# ── 5. Benchmark Method 4: Autoencoder INT8 (Quantized Single-Pass XAI) ──────
print(f"\n[5/5] Benchmarking Method 4: Autoencoder INT8 ({N_BENCH_FAST} frames)...")
sess_ae_int8 = ort.InferenceSession(str(AE_INT8_PATH))

# Warmup
for _ in range(50):
    _ = sess_ae_int8.run(None, {input_name_ae: bench_samples_fast[0]})

latencies_ae_int8 = []
for sample in bench_samples_fast:
    t0 = time.perf_counter()
    score, phi = sess_ae_int8.run(None, {input_name_ae: sample})
    latencies_ae_int8.append((time.perf_counter() - t0) * 1000.0)

mean_ae_int8 = np.mean(latencies_ae_int8)
fps_ae_int8 = 1000.0 / mean_ae_int8
size_ae_int8 = AE_INT8_PATH.stat().st_size / 1024
print(f"      Latency : {mean_ae_int8:.3f} ms/frame (p95: {np.percentile(latencies_ae_int8, 95):.3f} ms)")
print(f"      Throughput: {fps_ae_int8:.1f} fps")
print(f"      Size    : {size_ae_int8:.1f} KB")

# ── Generate Table 5 ─────────────────────────────────────────────────────────
speedup_vs_shap = mean_shap / mean_ae_int8

table_5 = f"""
========================================================================================
TABLE 5: Empirical Comparison of Edge Explainability Overhead (Journal Paper)
========================================================================================
| Method                                | Latency / frame | Throughput | Model Size | XAI Type       |
|---------------------------------------|-----------------|------------|------------|----------------|
| Isolation Forest alone (InCIS 2027)   | {mean_if:8.3f} ms    | {fps_if:8.1f} fps | {size_if:8.1f} KB | None           |
| Isolation Forest + TreeSHAP (Baseline)| {mean_shap:8.3f} ms    | {fps_shap:8.1f} fps | {size_if:8.1f} KB | Post-hoc Tree  |
| Autoencoder Float32 (Proposed)        | {mean_ae_f32:8.3f} ms    | {fps_ae_f32:8.1f} fps | {size_ae_f32:8.1f} KB | Single-Pass    |
| Autoencoder INT8 (Proposed Deploy)    | {mean_ae_int8:8.3f} ms    | {fps_ae_int8:8.1f} fps | {size_ae_int8:8.1f} KB | Single-Pass Q8 |
========================================================================================
Key Finding: Proposed Single-Pass INT8 Autoencoder provides per-packet local attribution
             {speedup_vs_shap:.1f}x faster than TreeSHAP, maintaining real-time edge line rate (> {fps_ae_int8:.0f} fps).
========================================================================================
"""

print(table_5)

# Save results to JSON
results = {
    "isolation_forest_onnx": {"latency_ms": mean_if, "fps": fps_if, "size_kb": size_if},
    "isolation_forest_treeshap": {"latency_ms": mean_shap, "fps": fps_shap},
    "autoencoder_f32": {"latency_ms": mean_ae_f32, "fps": fps_ae_f32, "size_kb": size_ae_f32},
    "autoencoder_int8": {"latency_ms": mean_ae_int8, "fps": fps_ae_int8, "size_kb": size_ae_int8},
    "speedup_vs_treeshap": speedup_vs_shap,
    "table_5_text": table_5
}

results_path = XAI_DIR / "benchmark_results.json"
with open(results_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"Saved benchmark results -> {results_path}")
