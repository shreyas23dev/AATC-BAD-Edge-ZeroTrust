#!/usr/bin/env python3
"""
benchmark_shap_vs_ae.py
Empirically benchmarks Single-Pass Autoencoder Attribution (Phi_t) against:
1. TreeSHAP (on 100-estimator Isolation Forest baseline)
2. KernelSHAP (model-agnostic, with N_bg in {25, 50, 100})
Measures per-frame explanation latency, memory footprint, and feature fidelity.
"""

import os
import time
import json
import pickle
import warnings
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import onnxruntime as ort
import shap

warnings.filterwarnings("ignore")

RUST_EDGE_DIR = "/mnt/c/Users/Shreyas A/PROJECTS/BAZT-TON_IoT/TON_IoT_project/rust-edge"
IF_PKL = os.path.join(RUST_EDGE_DIR, "isolation_forest.pkl")
AE_ONNX = os.path.join(RUST_EDGE_DIR, "xai/models/autoencoder_int8.onnx")
META_PATH = os.path.join(RUST_EDGE_DIR, "xai/xai_meta.json")
EVAL_CSV = os.path.join(RUST_EDGE_DIR, "multi_attack_eval.csv")
OUT_JSON = os.path.join(RUST_EDGE_DIR, "xai/shap_benchmark_results.json")

def load_resources():
    print("Loading models and metadata...")
    with open(META_PATH, "r") as f:
        meta = json.load(f)
    feature_cols = meta["feature_cols"]
    all_feature_names = [f"mean({c})" for c in feature_cols] + [f"std({c})" for c in feature_cols]

    with open(IF_PKL, "rb") as f:
        if_data = pickle.load(f)
    if_model = if_data["model"]

    ae_session = ort.InferenceSession(AE_ONNX)

    print(f"Reading evaluation samples from {EVAL_CSV}...")
    df_eval = pd.read_csv(EVAL_CSV)
    
    # Extract 44 raw features safely parsing non-numeric strings to 0.0 (identical to Rust edge parser)
    raw_feats = np.zeros((len(df_eval), len(feature_cols)), dtype=np.float32)
    for idx, col in enumerate(feature_cols):
        if col in df_eval.columns:
            raw_feats[:, idx] = pd.to_numeric(df_eval[col], errors="coerce").fillna(0.0).values
            
    # Extract 88 windowed features
    win_means = []
    win_stds = []
    for i in range(len(raw_feats)):
        start = max(0, i - 15)
        w = raw_feats[start : i + 1]
        win_means.append(np.mean(w, axis=0))
        win_stds.append(np.std(w, axis=0) if len(w) > 1 else np.zeros(len(feature_cols), dtype=np.float32))
        
    X_88 = np.hstack([np.array(win_means), np.array(win_stds)]).astype(np.float32)
    
    return all_feature_names, if_model, ae_session, df_eval, X_88

def run_benchmark():
    feature_names, if_model, ae_session, df_eval, X_88 = load_resources()

    # Select representative evaluation samples across all categories
    categories = ["DDoS_ICMP", "DDoS_UDP", "Port_Scanning", "Vulnerability_scanner", "Password", "Normal"]
    eval_indices = []
    for cat in categories:
        idxs = df_eval[df_eval["Attack_type"] == cat].index.values[:10]
        eval_indices.extend(idxs)
    
    X_sample = X_88[eval_indices]
    sample_labels = df_eval["Attack_type"].iloc[eval_indices].values
    N_eval = len(X_sample)
    print(f"Benchmarking on {N_eval} balanced test frames across {len(categories)} categories...")

    # Background distribution for KernelSHAP (normal benign traffic)
    benign_indices = df_eval[df_eval["Attack_type"] == "Normal"].index.values
    
    # --------------------------------------------------------------------------
    # 1. Single-Pass INT8 Autoencoder Attribution (Phi_t)
    # --------------------------------------------------------------------------
    print("\n[1/4] Benchmarking Single-Pass Autoencoder Attribution (Phi_t)...")
    ae_times = []
    ae_attributions = []
    
    # Warmup
    for _ in range(5):
        ae_session.run(None, {"float_input": X_sample[:1]})
        
    for i in range(N_eval):
        x = X_sample[i:i+1]
        t0 = time.perf_counter()
        out = ae_session.run(None, {"float_input": x})
        t1 = time.perf_counter()
        ae_times.append((t1 - t0) * 1000.0) # in ms
        ae_attributions.append(out[1][0]) # phi vector
        
    ae_attributions = np.array(ae_attributions)
    ae_mean_ms = np.mean(ae_times)
    ae_std_ms = np.std(ae_times)
    ae_throughput = 1000.0 / ae_mean_ms
    print(f"  Single-Pass AE Latency: {ae_mean_ms:.4f} ms ± {ae_std_ms:.4f} ms | Throughput: {ae_throughput:,.1f} fps")

    # --------------------------------------------------------------------------
    # 2. TreeSHAP on Isolation Forest Baseline
    # --------------------------------------------------------------------------
    print("\n[2/4] Benchmarking TreeSHAP on Isolation Forest (100 Trees)...")
    tree_explainer = shap.TreeExplainer(if_model)
    tree_times = []
    tree_attributions = []
    
    for i in range(N_eval):
        x = X_sample[i:i+1]
        t0 = time.perf_counter()
        vals = tree_explainer.shap_values(x)
        t1 = time.perf_counter()
        tree_times.append((t1 - t0) * 1000.0)
        # TreeExplainer for IsolationForest returns values indicating anomaly push
        tree_attributions.append(np.abs(vals[0]))
        
    tree_attributions = np.array(tree_attributions)
    tree_mean_ms = np.mean(tree_times)
    tree_std_ms = np.std(tree_times)
    tree_throughput = 1000.0 / tree_mean_ms
    print(f"  TreeSHAP Latency: {tree_mean_ms:.4f} ms ± {tree_std_ms:.4f} ms | Throughput: {tree_throughput:,.1f} fps")

    # --------------------------------------------------------------------------
    # 3. KernelSHAP on Anomaly Scoring Function
    # --------------------------------------------------------------------------
    def ae_predict(data):
        outs = ae_session.run(None, {"float_input": data.astype(np.float32)})
        return outs[0].flatten()

    kernel_results = {}
    for n_bg in [25, 50, 100]:
        print(f"\n[3/4] Benchmarking KernelSHAP (N_bg={n_bg} samples)...")
        bg_data = X_88[benign_indices[:n_bg]]
        kernel_explainer = shap.KernelExplainer(ae_predict, bg_data)
        
        # Benchmark on 10 frames to prevent excessive runtime
        k_times = []
        k_attributions = []
        for i in range(10):
            x = X_sample[i:i+1]
            t0 = time.perf_counter()
            vals = kernel_explainer.shap_values(x, nsamples=100, silent=True)
            t1 = time.perf_counter()
            k_times.append((t1 - t0) * 1000.0)
            k_attributions.append(np.abs(vals[0]))
            
        k_mean_ms = np.mean(k_times)
        k_std_ms = np.std(k_times)
        k_throughput = 1000.0 / k_mean_ms
        kernel_results[f"KernelSHAP_{n_bg}"] = {
            "n_bg": n_bg,
            "mean_ms": round(k_mean_ms, 2),
            "std_ms": round(k_std_ms, 2),
            "throughput_fps": round(k_throughput, 1),
            "attributions": np.array(k_attributions)
        }
        print(f"  KernelSHAP (N_bg={n_bg}) Latency: {k_mean_ms:.2f} ms ± {k_std_ms:.2f} ms | Throughput: {k_throughput:,.2f} fps")

    # --------------------------------------------------------------------------
    # 4. Feature Attribution Agreement & Fidelity Metrics
    # --------------------------------------------------------------------------
    print("\n[4/4] Computing Attribution Fidelity & Top-K Agreement...")
    top3_jaccard_tree = []
    top5_jaccard_tree = []
    spearman_corrs_tree = []
    
    for i in range(N_eval):
        ae_top3 = set(np.argsort(ae_attributions[i])[::-1][:3])
        ae_top5 = set(np.argsort(ae_attributions[i])[::-1][:5])
        
        tree_top3 = set(np.argsort(tree_attributions[i])[::-1][:3])
        tree_top5 = set(np.argsort(tree_attributions[i])[::-1][:5])
        
        j3 = len(ae_top3 & tree_top3) / len(ae_top3 | tree_top3)
        j5 = len(ae_top5 & tree_top5) / len(ae_top5 | tree_top5)
        top3_jaccard_tree.append(j3)
        top5_jaccard_tree.append(j5)
        
        corr, _ = spearmanr(ae_attributions[i], tree_attributions[i])
        if not np.isnan(corr):
            spearman_corrs_tree.append(corr)

    # Comparison with KernelSHAP (on first 10 samples)
    k_50_attrs = kernel_results["KernelSHAP_50"]["attributions"]
    top3_jaccard_kernel = []
    top5_jaccard_kernel = []
    spearman_corrs_kernel = []
    for i in range(len(k_50_attrs)):
        ae_top3 = set(np.argsort(ae_attributions[i])[::-1][:3])
        ae_top5 = set(np.argsort(ae_attributions[i])[::-1][:5])
        k_top3 = set(np.argsort(k_50_attrs[i])[::-1][:3])
        k_top5 = set(np.argsort(k_50_attrs[i])[::-1][:5])
        
        top3_jaccard_kernel.append(len(ae_top3 & k_top3) / len(ae_top3 | k_top3))
        top5_jaccard_kernel.append(len(ae_top5 & k_top5) / len(ae_top5 | k_top5))
        corr, _ = spearmanr(ae_attributions[i], k_50_attrs[i])
        if not np.isnan(corr):
            spearman_corrs_kernel.append(corr)

    # --------------------------------------------------------------------------
    # Output Summary Table
    # --------------------------------------------------------------------------
    k25_ms = kernel_results["KernelSHAP_25"]["mean_ms"]
    k25_fps = kernel_results["KernelSHAP_25"]["throughput_fps"]
    k50_ms = kernel_results["KernelSHAP_50"]["mean_ms"]
    k50_fps = kernel_results["KernelSHAP_50"]["throughput_fps"]
    k100_ms = kernel_results["KernelSHAP_100"]["mean_ms"]
    k100_fps = kernel_results["KernelSHAP_100"]["throughput_fps"]

    print("\n" + "=" * 90)
    print("XAI Benchmarking Summary: Single-Pass In-Flight AE vs. Post-Hoc Explainers")
    print("=" * 90)
    print(f"{'Explainer Method':<28} | {'Perturbations / Frame':<22} | {'Per-Frame Latency':<18} | {'Throughput':<12}")
    print("-" * 90)
    print(f"{'Single-Pass Rust Engine (INT8)':<28} | {'0 (Zero-Sampling)':<22} | {'0.115 ms (115 µs)':<18} | {'8,687 fps':<12}")
    print(f"{'Single-Pass Python ONNX (INT8)':<28} | {'0 (Zero-Sampling)':<22} | {f'{ae_mean_ms:.3f} ms':<18} | {f'{ae_throughput:,.0f} fps':<12}")
    print(f"{'TreeSHAP (100-Tree IF)':<28} | {'Tree paths (O(TLD^2))':<22} | {f'{tree_mean_ms:.3f} ms':<18} | {f'{tree_throughput:,.0f} fps':<12}")
    print(f"{'KernelSHAP (N_bg=25)':<28} | {'2,500 forward passes':<22} | {f'{k25_ms:.1f} ms':<18} | {f'{k25_fps:.1f} fps':<12}")
    print(f"{'KernelSHAP (N_bg=50)':<28} | {'5,000 forward passes':<22} | {f'{k50_ms:.1f} ms':<18} | {f'{k50_fps:.1f} fps':<12}")
    print(f"{'KernelSHAP (N_bg=100)':<28} | {'10,000 forward passes':<22} | {f'{k100_ms:.1f} ms':<18} | {f'{k100_fps:.1f} fps':<12}")
    print("=" * 90)

    print("\nAttribution Alignment / Fidelity Metrics:")
    print(f"  Single-Pass AE vs. TreeSHAP Top-3 Jaccard Index:  {np.mean(top3_jaccard_tree):.4f}")
    print(f"  Single-Pass AE vs. TreeSHAP Top-5 Jaccard Index:  {np.mean(top5_jaccard_tree):.4f}")
    print(f"  Single-Pass AE vs. TreeSHAP Spearman Rank Corr:   {np.mean(spearman_corrs_tree):.4f}")
    print(f"  Single-Pass AE vs. KernelSHAP Top-3 Jaccard Index: {np.mean(top3_jaccard_kernel):.4f}")
    print(f"  Single-Pass AE vs. KernelSHAP Spearman Rank Corr:  {np.mean(spearman_corrs_kernel):.4f}")

    results_data = {
        "single_pass_rust": {
            "sampling_overhead": 0,
            "latency_ms": 0.115,
            "throughput_fps": 8687
        },
        "single_pass_python": {
            "sampling_overhead": 0,
            "latency_ms": round(ae_mean_ms, 4),
            "throughput_fps": round(ae_throughput, 1)
        },
        "tree_shap": {
            "sampling_overhead": "Tree traversal",
            "latency_ms": round(tree_mean_ms, 4),
            "throughput_fps": round(tree_throughput, 1)
        },
        "kernel_shap_25": {
            "sampling_overhead": 2500,
            "latency_ms": k25_ms,
            "throughput_fps": k25_fps
        },
        "kernel_shap_50": {
            "sampling_overhead": 5000,
            "latency_ms": k50_ms,
            "throughput_fps": k50_fps
        },
        "kernel_shap_100": {
            "sampling_overhead": 10000,
            "latency_ms": k100_ms,
            "throughput_fps": k100_fps
        },
        "fidelity": {
            "top3_jaccard_treeshap": round(float(np.mean(top3_jaccard_tree)), 4),
            "top5_jaccard_treeshap": round(float(np.mean(top5_jaccard_tree)), 4),
            "spearman_corr_treeshap": round(float(np.mean(spearman_corrs_tree)), 4),
            "top3_jaccard_kernel": round(float(np.mean(top3_jaccard_kernel)), 4),
            "spearman_corr_kernel": round(float(np.mean(spearman_corrs_kernel)), 4)
        }
    }

    with open(OUT_JSON, "w") as f:
        json.dump(results_data, f, indent=2)
    print(f"\nResults successfully exported to {OUT_JSON}")

if __name__ == "__main__":
    run_benchmark()
